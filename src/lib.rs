use pyo3::prelude::*;

/// Prints a message.
#[pyfunction]
fn rust_ok() -> PyResult<String> {
    Ok("Rust OK".into())
}

/// A Python module implemented in Rust.
#[pymodule]
fn _fusion(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rust_ok, m)?)?;
    Ok(())
}