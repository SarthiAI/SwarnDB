#!/usr/bin/env bash
# Generate Python protobuf and gRPC stubs from SwarnDB proto definitions.
#
# Usage:
#   cd sdk/python && bash proto/generate.sh
#
# Prerequisites:
#   pip install grpcio-tools protobuf

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SDK_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROTO_ROOT="$(cd "$SDK_ROOT/../../proto" && pwd)"
OUT_DIR="$SDK_ROOT/src/swarndb/_proto"

echo "Proto root:  $PROTO_ROOT"
echo "Output dir:  $OUT_DIR"
echo ""

# Clean previously generated files
find "$OUT_DIR" -name "*_pb2*" -delete 2>/dev/null || true
rm -rf "$OUT_DIR/swarndb" 2>/dev/null || true

# Generate all stubs in one pass
echo "Generating stubs for all proto files..."
python -m grpc_tools.protoc \
    --proto_path="$PROTO_ROOT" \
    --python_out="$OUT_DIR" \
    --pyi_out="$OUT_DIR" \
    --grpc_python_out="$OUT_DIR" \
    "$PROTO_ROOT"/swarndb/v1/*.proto

# Move generated files from nested swarndb/v1/ to _proto/ directly
echo "Flattening output directory..."
mv "$OUT_DIR"/swarndb/v1/*_pb2*.py "$OUT_DIR"/
mv "$OUT_DIR"/swarndb/v1/*_pb2*.pyi "$OUT_DIR"/ 2>/dev/null || true
rm -rf "$OUT_DIR/swarndb"

# Fix imports to use relative imports within the _proto package.
# The generated code uses absolute imports like:
#   from swarndb.v1 import common_pb2
# We convert them to relative imports:
#   from . import common_pb2
echo "Fixing imports in generated files..."

if [[ "$(uname)" == "Darwin" ]]; then
    SED_INPLACE=(-i '')
else
    SED_INPLACE=(-i)
fi

find "$OUT_DIR" -name "*.py" -exec sed "${SED_INPLACE[@]}" \
    's/^from swarndb\.v1 import/from . import/g' {} +

find "$OUT_DIR" -name "*.pyi" -exec sed "${SED_INPLACE[@]}" \
    's/^from swarndb\.v1 import/from . import/g' {} +

echo ""
echo "Done. Generated stubs in $OUT_DIR"
