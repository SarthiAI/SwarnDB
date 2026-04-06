#!/usr/bin/env python3
"""
SwarnDB Persistence Test: File Locking
=======================================
Verifies that file locking prevents two server instances from using the
same data directory simultaneously.

Usage:
    python test_persistence_file_lock.py

The primary server must already be running at http://localhost:8080.
This script will attempt to start a second server instance on port 8081
pointing to the same data directory. The second instance should fail
with a lock error.

REST API: http://localhost:8080 (primary), http://localhost:8081 (second)
Data directory: ./data  (override with SWARNDB_DATA_DIR)
Server binary: vf-server (override with SWARNDB_BINARY)

NOTE: This test requires the ProcessLock from vf-storage to be wired into
the server startup. If the lock is not yet enforced, the second instance
may start successfully (which this test will report as a FAIL).
"""

import os
import signal
import subprocess
import sys
import time

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL_PRIMARY = os.environ.get("SWARNDB_BASE_URL", "http://localhost:8080")
SECOND_REST_PORT = 8081
SECOND_GRPC_PORT = 50052
BASE_URL_SECOND = f"http://localhost:{SECOND_REST_PORT}"
DATA_DIR = os.environ.get("SWARNDB_DATA_DIR", "./data")
SERVER_BINARY = os.environ.get("SWARNDB_BINARY", "vf-server")

# For Docker mode: if set, we use docker run instead of a local binary
DOCKER_IMAGE = os.environ.get("SWARNDB_DOCKER_IMAGE", "")

STARTUP_TIMEOUT = 10  # seconds to wait for the second instance to start/fail

passed = 0
failed = 0


def report(name: str, ok: bool, detail: str = ""):
    global passed, failed
    tag = "PASS" if ok else "FAIL"
    if ok:
        passed += 1
    else:
        failed += 1
    suffix = f" -- {detail}" if detail else ""
    print(f"  [{tag}] {name}{suffix}")


def server_healthy(base_url: str, timeout: float = 3.0) -> bool:
    """Check if a server is up and responding at the given URL."""
    try:
        resp = requests.get(f"{base_url}/health", timeout=timeout)
        return resp.status_code == 200
    except (requests.ConnectionError, requests.Timeout):
        return False


def start_second_server_native():
    """Start a second server instance using the native binary."""
    env = os.environ.copy()
    env["SWARNDB_REST_PORT"] = str(SECOND_REST_PORT)
    env["SWARNDB_GRPC_PORT"] = str(SECOND_GRPC_PORT)
    env["SWARNDB_DATA_DIR"] = DATA_DIR
    env["SWARNDB_LOG_LEVEL"] = "debug"

    try:
        proc = subprocess.Popen(
            [SERVER_BINARY],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return proc
    except FileNotFoundError:
        print(f"  ERROR: Server binary '{SERVER_BINARY}' not found.")
        print("  Set SWARNDB_BINARY to the path of the vf-server binary.")
        return None


def start_second_server_docker():
    """Start a second server instance using Docker."""
    container_name = "swarndb_lock_test"

    # Clean up any leftover container
    subprocess.run(
        ["docker", "rm", "-f", container_name],
        capture_output=True, timeout=5
    )

    # Get the data volume from the running container
    try:
        result = subprocess.run(
            ["docker", "inspect", "--format",
             "{{range .Mounts}}{{.Source}}{{end}}", "swarndb"],
            capture_output=True, text=True, timeout=5,
        )
        host_data_dir = result.stdout.strip()
    except Exception:
        host_data_dir = ""

    if not host_data_dir:
        # Fall back: use a Docker volume name
        host_data_dir = "swarndb_data"

    cmd = [
        "docker", "run", "--rm",
        "--name", container_name,
        "-p", f"{SECOND_REST_PORT}:8080",
        "-p", f"{SECOND_GRPC_PORT}:50051",
        "-v", f"{host_data_dir}:/data",
        "-e", "SWARNDB_DATA_DIR=/data",
        "-e", "SWARNDB_LOG_LEVEL=debug",
        DOCKER_IMAGE,
    ]

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return proc
    except FileNotFoundError:
        print("  ERROR: Docker not found.")
        return None


# ---------------------------------------------------------------------------
# Main test flow
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("SwarnDB Persistence Test: File Locking")
    print(f"Primary: {BASE_URL_PRIMARY}")
    print(f"Second:  {BASE_URL_SECOND}")
    print(f"Data dir: {DATA_DIR}")
    if DOCKER_IMAGE:
        print(f"Mode: Docker ({DOCKER_IMAGE})")
    else:
        print(f"Mode: Native binary ({SERVER_BINARY})")
    print("=" * 60)
    print()

    # Step 1: Verify the primary server is healthy
    primary_ok = server_healthy(BASE_URL_PRIMARY)
    report("Primary server is healthy", primary_ok)
    if not primary_ok:
        print("  The primary server must be running on port 8080.")
        print("  Start it first, then re-run this test.")
        sys.exit(1)

    # Step 2: Verify the second port is not already in use
    second_up = server_healthy(BASE_URL_SECOND, timeout=2.0)
    if second_up:
        report("Port 8081 is free", False,
               "Something is already running on port 8081")
        print("  Stop whatever is running on port 8081 first.")
        sys.exit(1)
    report("Port 8081 is free", True)

    # Step 3: Try to start a second server on the same data directory
    print(f"\n  Starting second server instance (port {SECOND_REST_PORT}, same data dir)...")

    if DOCKER_IMAGE:
        proc = start_second_server_docker()
    else:
        proc = start_second_server_native()

    if proc is None:
        report("Start second server", False, "Failed to launch process")
        sys.exit(1)

    # Step 4: Wait and check the outcome
    print(f"  Waiting up to {STARTUP_TIMEOUT}s for the second instance to start or fail...")

    second_started = False
    lock_error_found = False
    process_exited = False
    exit_code = None
    stderr_output = ""
    stdout_output = ""

    start_time = time.time()
    while time.time() - start_time < STARTUP_TIMEOUT:
        # Check if the process has exited
        ret = proc.poll()
        if ret is not None:
            process_exited = True
            exit_code = ret
            try:
                stdout_output = proc.stdout.read().decode("utf-8", errors="replace")
                stderr_output = proc.stderr.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            break

        # Check if the second server became healthy
        if server_healthy(BASE_URL_SECOND, timeout=1.0):
            second_started = True
            break

        time.sleep(1)

    # If the process is still running but not healthy, give it a bit more time
    if not process_exited and not second_started:
        try:
            stdout_output, stderr_output = proc.communicate(timeout=3)
            stdout_output = stdout_output.decode("utf-8", errors="replace")
            stderr_output = stderr_output.decode("utf-8", errors="replace")
            process_exited = True
            exit_code = proc.returncode
        except subprocess.TimeoutExpired:
            pass

    # Check for lock-related error messages
    combined_output = stdout_output + stderr_output
    lock_keywords = ["lock", "Lock", "LOCK", "LockHeld", "another instance",
                     "data directory", "exclusive"]
    for keyword in lock_keywords:
        if keyword.lower() in combined_output.lower():
            lock_error_found = True
            break

    # Step 5: Report results
    print()
    if process_exited:
        print(f"  Second server process exited with code: {exit_code}")
        if stderr_output:
            # Print last 10 lines of stderr for context
            lines = stderr_output.strip().split("\n")
            print(f"  Last lines of stderr ({len(lines)} total):")
            for line in lines[-10:]:
                print(f"    {line}")
    elif second_started:
        print("  Second server started successfully (UNEXPECTED).")
    else:
        print("  Second server neither started nor cleanly exited.")
        if combined_output:
            print(f"  Output so far:")
            for line in combined_output.strip().split("\n")[-5:]:
                print(f"    {line}")

    # The second server SHOULD have failed with a lock error
    report("Second server failed to start (lock held)",
           process_exited and not second_started,
           f"exited={process_exited} exit_code={exit_code}")

    report("Lock error in output",
           lock_error_found,
           "Found lock-related error message" if lock_error_found
           else "No lock error message found")

    # Step 6: Verify the primary server is still healthy
    primary_still_ok = server_healthy(BASE_URL_PRIMARY)
    report("Primary server still healthy after lock test", primary_still_ok)

    # Step 7: Clean up the second server if it somehow started
    if second_started or (not process_exited):
        print("\n  Cleaning up second server instance...")
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
                proc.wait(timeout=3)
            except Exception:
                pass

        if DOCKER_IMAGE:
            subprocess.run(
                ["docker", "rm", "-f", "swarndb_lock_test"],
                capture_output=True, timeout=5
            )

    # Step 8: Verify a new instance CAN start after the primary stops
    #         (This is informational -- we don't actually stop the primary here
    #          because that would disrupt any other tests running.)
    print()
    print("  NOTE: To fully verify lock release, you would need to:")
    print("    1. Stop the primary server")
    print("    2. Start a new instance on the same data directory")
    print("    3. Verify it starts successfully")
    print("  This step is left as a manual verification to avoid disrupting")
    print("  the primary server.")

    # Summary
    print()
    print("=" * 60)
    print(f"RESULT: {passed} PASSED, {failed} FAILED")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
