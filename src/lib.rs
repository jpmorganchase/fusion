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

    #[test]
    fn test_rusk_ok() -> Result<(), PyErr> {
        let res = rust_ok();
        assert!(res.unwrap());
        Ok(())
    }

    #[test]
    fn test_load_module() -> Result<(), PyErr> {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let _module = PyModule::new_bound(py, "_fusion")?;
            _fusion(py, &PyModule::new_bound(py, "fusion")?)?;
            Ok(())
        })
    }
}
