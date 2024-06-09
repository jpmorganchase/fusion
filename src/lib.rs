use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

mod auth;

fn rust_ok_impl() -> bool {
    true
}

/// Prints a message.
#[pyfunction]
fn rust_ok() -> PyResult<bool> {
    Ok(rust_ok_impl())
}


/// A Python module implemented in Rust.
#[pymodule]
fn _fusion(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rust_ok, m)?)?;
    m.add_class::<auth::FusionCredentials>()?;
    Ok(())
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rust_ok_rs() {
        assert!(rust_ok_impl());
    }

}