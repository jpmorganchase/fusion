use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

mod auth;

fn rust_ok_impl() -> bool {
    true
}

/// Prints a message.
#[pyfunction]
fn rust_ok() -> PyResult<bool> {
    Ok(rust_ok_impl())
}

#[pyclass(module = "fusion._fusion")]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TestRust {
    #[pyo3(get, set)]
    pub name: String,
    #[pyo3(get, set)]
    pub age: i32,
    #[pyo3(get, set)]
    pub map: HashMap<String, String>,
}

/// A Python module implemented in Rust.
#[pymodule]
fn _fusion(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rust_ok, m)?)?;
    m.add_class::<auth::FusionCredentials>()?;
    m.add_class::<auth::AuthToken>()?;
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
