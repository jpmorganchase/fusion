use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3_log::{Caching, Logger};

#[cfg(feature = "experiments")]
mod experiments;

// Re-export the experiments module if the feature is enabled
#[cfg(feature = "experiments")]
pub use experiments::*;

pub mod auth;
mod utils;

fn rust_ok_impl() -> bool {
    true
}

/// Prints a message.
#[pyfunction]
fn rust_ok() -> PyResult<bool> {
    Ok(rust_ok_impl())
}

#[pyclass]
pub(crate) struct TokioRuntime(tokio::runtime::Runtime);

/// A Python module implemented in Rust.
#[pymodule]
fn _fusion(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let _ = Logger::new(_py, Caching::LoggersAndLevels)?.install();
    m.add(
        "runtime",
        TokioRuntime(tokio::runtime::Runtime::new().unwrap()),
    )?;
    m.add_function(wrap_pyfunction!(rust_ok, m)?)?;
    m.add_class::<auth::FusionCredentials>()?;
    m.add_class::<auth::AuthToken>()?;
    #[cfg(feature = "experiments")]
    m.add_class::<experiments::RustTestClass>()?;
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
