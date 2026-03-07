fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::configure()
        .build_server(true)
        .build_client(false)
        .compile_protos(
            &[
                "../../proto/swarndb/v1/collection.proto",
                "../../proto/swarndb/v1/vector.proto",
                "../../proto/swarndb/v1/search.proto",
                "../../proto/swarndb/v1/graph.proto",
                "../../proto/swarndb/v1/vector_math.proto",
            ],
            &["../../proto"],
        )?;
    Ok(())
}
