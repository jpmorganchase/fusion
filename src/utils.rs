use crate::TokioRuntime;
use pyo3::prelude::*;

/// Utility to get the Tokio Runtime from Python
pub(crate) fn get_tokio_runtime(py: Python) -> PyRef<TokioRuntime> {
    let fusion = py.import_bound("fusion._fusion").unwrap();
    let tmp = fusion.getattr("runtime").unwrap();
    match tmp.extract::<PyRef<TokioRuntime>>() {
        Ok(runtime) => runtime,
        Err(_e) => {
            let rt = TokioRuntime(tokio::runtime::Runtime::new().unwrap());
            let obj = Py::new(py, rt).unwrap().into_bound(py);
            obj.extract().unwrap()
        }
    }
}
