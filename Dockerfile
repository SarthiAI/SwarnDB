# Stage 1: Builder
FROM rust:1.85-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config libssl-dev protobuf-compiler cmake make g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY crates/ crates/
COPY proto/ proto/

RUN cargo build --release --bin vf-server \
    && strip target/release/vf-server

# Stage 2: Runtime
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libssl3 wget \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 1001 swarndb \
    && useradd --uid 1001 --gid swarndb --shell /bin/false --create-home swarndb

COPY --from=builder /build/target/release/vf-server /usr/local/bin/vf-server

RUN mkdir -p /data && chown swarndb:swarndb /data

ENV SWARNDB_HOST=0.0.0.0
ENV SWARNDB_GRPC_PORT=50051
ENV SWARNDB_REST_PORT=8080
ENV SWARNDB_DATA_DIR=/data
ENV SWARNDB_LOG_LEVEL=info

EXPOSE 50051 8080
VOLUME /data

USER swarndb

ENTRYPOINT ["vf-server"]
