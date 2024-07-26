// Benchmark scratchpad
use pyo3::exceptions::{PyFileNotFoundError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyType;
use serde::{Deserialize, Serialize};
use std::env;
use std::fs::File;
use std::io::prelude::*;
use std::path::{Path, PathBuf};
use std::str::FromStr;

#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RustTestClass {
    #[pyo3(get, set)]
    i64_1: Option<i64>,

    #[pyo3(get, set)]
    i64_2: Option<i64>,

    #[pyo3(get, set)]
    str_1: Option<String>,

    #[pyo3(get, set)]
    str_2: Option<String>,

    #[pyo3(get, set)]
    str_3: Option<String>,

    #[pyo3(get, set)]
    str_4: Option<String>,
}

#[pymethods]
impl RustTestClass {
    #[new]
    fn new(
        i64_1: Option<i64>,
        i64_2: Option<i64>,
        str_1: Option<String>,
        str_2: Option<String>,
        str_3: Option<String>,
        str_4: Option<String>,
    ) -> Self {
        RustTestClass {
            i64_1,
            i64_2,
            str_1,
            str_2,
            str_3,
            str_4,
        }
    }

    #[classmethod]
    fn factory(
        _cls: &Bound<'_, PyType>,
        i64_1: Option<i64>,
        i64_2: Option<i64>,
        str_1: Option<String>,
        str_2: Option<String>,
        str_3: Option<String>,
        str_4: Option<String>,
    ) -> Self {
        RustTestClass {
            i64_1,
            i64_2,
            str_1,
            str_2,
            str_3,
            str_4,
        }
    }

    #[classmethod]
    fn factory_with_file(
        _cls: &Bound<'_, PyType>,
        file_path: PathBuf,
        i64_1: Option<i64>,
        i64_2: Option<i64>,
        str_1: Option<String>,
        str_2: Option<String>,
        str_3: Option<String>,
        str_4: Option<String>,
    ) -> PyResult<RustTestClass> {
        let mut file =
            File::open(file_path).map_err(|_| PyFileNotFoundError::new_err("File not found"))?;

        let mut contents = String::new();
        file.read_to_string(&mut contents)?;

        Ok(RustTestClass {
            i64_1,
            i64_2,
            str_1,
            str_2,
            str_3,
            str_4,
        })
    }

    #[classmethod]
    fn factory_with_file_serde(
        _cls: &Bound<'_, PyType>,
        file_path: PathBuf,
    ) -> PyResult<RustTestClass> {
        let file =
            File::open(file_path).map_err(|_| PyFileNotFoundError::new_err("File not found"))?;

        let res: RustTestClass = serde_json::from_reader(file).map_err(|err| {
            PyValueError::new_err(format!("Invalid JSON: {}\nContents:\n TBD", err))
        })?;

        let i64_1 = res.i64_1.or_else(|| {
            std::env::var("TEST_I64_1_VAR")
                .ok()
                .map(|v| v.parse().unwrap())
        });
        let i64_2 = res.i64_1.or_else(|| {
            std::env::var("TEST_I64_2_VAR")
                .ok()
                .map(|v| v.parse().unwrap())
        });

        let str_1 = res.str_1.or_else(|| std::env::var("TEST_STR_1_VAR").ok());
        let str_2 = res.str_2.or_else(|| std::env::var("TEST_STR_2_VAR").ok());
        let str_3 = res.str_3.or_else(|| std::env::var("TEST_STR_3_VAR").ok());
        let str_4 = res.str_4.or_else(|| std::env::var("TEST_STR_4_VAR").ok());

        Ok(RustTestClass {
            i64_1,
            i64_2,
            str_1,
            str_2,
            str_3,
            str_4,
        })
    }
}
