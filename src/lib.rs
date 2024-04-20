use pyo3::prelude::*;

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
    Ok(())
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rust_ok_rs() {
        assert!(rust_ok_impl());
    }

    #[test]
    fn test_rust_ok() -> Result<(), Box<dyn std::error::Error>> {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            // Create a new Python module
            let module = PyModule::new_bound(py, "test_module").unwrap();
            
            // Create the function within the Python context
            let rust_ok_function = wrap_pyfunction!(rust_ok, module).unwrap();
            
            let result = rust_ok_function.call0()?.extract::<bool>()?;
            assert!(result);
            Ok(())
        })
    }
}