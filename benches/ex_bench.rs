use std::{fs::File, io::Write};

use criterion::{criterion_group, criterion_main, Criterion};

use fusion::auth::FusionCredentials;
use pyo3::Python;
use serde_json::json;
use tempdir::TempDir;

fn bench_fusion_creds(c: &mut Criterion) {
    pyo3::prepare_freethreaded_python();
    let temp_dir =
        TempDir::new("test_fusion_credentials_from_file").expect("Failed to create temp dir");

    // Path to the temporary file
    let temp_file_path = temp_dir.path().join("credentials.json");

    // JSON content to be written to the file
    let json_content = json!({
        "grant_type": "client_credentials",
        "client_id": "my_client_id",
        "client_secret": "my_client_secret",
        "resource": "my_resource",
        "auth_url": "my_auth_url",
    });

    // Write the JSON content to the file
    let mut file = File::create(&temp_file_path).expect("Failed to create temp file");
    file.write_all(json_content.to_string().as_bytes())
        .expect("Failed to write to temp file");

    c.bench_function("my_function", |b| {
        b.iter(|| {
            Python::with_gil(|py| {
                let _creds = FusionCredentials::from_file(
                    &py.get_type_bound::<FusionCredentials>(),
                    temp_file_path.clone(),
                )
                .unwrap();
            })
        })
    });
}

criterion_group!(benches, bench_fusion_creds);
criterion_main!(benches);
