fn main() -> Result<(), Box<dyn std::error::Error>> {
    let proto_file = "../../proto/engine/v1/engine.proto";
    tonic_build::configure()
        .build_server(false)
        .compile(&[proto_file], &["../../proto"]) ?;
    println!("cargo:rerun-if-changed={}", proto_file);
    println!("cargo:rerun-if-changed=../../proto");
    Ok(())
}

