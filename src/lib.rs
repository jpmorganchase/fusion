use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use reqwest::{Client, Error, Response, Body, Method, Url};
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, USER_AGENT};
use serde::{Deserialize, Serialize};
use serde_json;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use anyhow::{anyhow, Result};
use tokio::runtime::Runtime;
extern crate serde_urlencoded;

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
    m.add_class::<FusionSession>()?;
    m.add_class::<PyFusionCredentials>()?;
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


#[derive(Debug, Serialize, Deserialize)]
struct FusionCredentials {
    client_id: Option<String>,
    client_secret: Option<String>,
    username: Option<String>,
    password: Option<String>,
    resource: Option<String>,
    auth_url: Option<String>,
    bearer_token: Option<String>,
    bearer_token_expiry: Option<u64>,
    is_bearer_token_expirable: Option<bool>,
    proxies: Option<HashMap<String, String>>,
    grant_type: String,
    fusion_e2e: Option<String>,
}

impl FusionCredentials {
    async fn refresh_token(&mut self) -> Result<(), reqwest::Error> {
//         let payload = if self.grant_type == "client_credentials" {
//             serde_json::json!({
//                 "grant_type": "client_credentials",
//                 "client_id": self.client_id.as_ref().unwrap(),
//                 "client_secret": self.client_secret.as_ref().unwrap(),
//                 "aud": self.resource.as_ref().unwrap(),
//             })
//         } else {
//             serde_json::json!({
//                 "grant_type": "password",
//                 "client_id": self.client_id.as_ref().unwrap(),
//                 "username": self.username.as_ref().unwrap(),
//                 "password": self.password.as_ref().unwrap(),
//                 "resource": self.resource.as_ref().unwrap(),
//             })
//         };


        let client = Client::new();
        let params = if self.grant_type == "client_credentials" {
            vec![
                ("grant_type", "client_credentials"),
                ("client_id", self.client_id.as_ref().unwrap()),
                ("client_secret", self.client_secret.as_ref().unwrap()),
                ("aud", self.resource.as_ref().unwrap())
            ]
        } else {
            vec![
                ("grant_type", "password"),
                ("client_id", self.client_id.as_ref().unwrap()),
                ("username", self.username.as_ref().unwrap()),
                ("password", self.password.as_ref().unwrap()),
                ("resource", self.resource.as_ref().unwrap())
            ]
        };

        let client = Client::new();
        let res = client
            .post(self.auth_url.as_ref().unwrap())
            .form(&params)
            .send()
            .await?;

        let res_json: HashMap<String, serde_json::Value> = res.json().await?;
        self.bearer_token = res_json.get("access_token").map(|v| v.as_str().unwrap().to_string());
        self.bearer_token_expiry = res_json.get("expires_in").map(|v| {
            let expiry_duration = v.as_u64().unwrap();
            let current_time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
            current_time + expiry_duration
        });
        Ok(())
    }
}

#[pyclass]
struct PyFusionCredentials {
    inner: Arc<Mutex<FusionCredentials>>,
}

#[pymethods]
impl PyFusionCredentials {
    #[new]
    #[pyo3(signature = (
        client_id = None,
        client_secret = None,
        username = None,
        password = None,
        resource = None,
        auth_url = None,
        bearer_token = None,
        bearer_token_expiry = 0,
        is_bearer_token_expirable = None,
        proxies = None,
        grant_type = "client_credentials".to_string(),
        fusion_e2e = None
    ))]
    fn new(
        client_id: Option<String>,
        client_secret: Option<String>,
        username: Option<String>,
        password: Option<String>,
        resource: Option<String>,
        auth_url: Option<String>,
        bearer_token: Option<String>,
        bearer_token_expiry: Option<u64>,
        is_bearer_token_expirable: Option<bool>,
        proxies: Option<HashMap<String, String>>,
        grant_type: String,
        fusion_e2e: Option<String>,
    ) -> Self {
        PyFusionCredentials {
            inner: Arc::new(Mutex::new(FusionCredentials {
                client_id,
                client_secret,
                username,
                password,
                resource,
                auth_url,
                bearer_token,
                bearer_token_expiry,
                is_bearer_token_expirable,
                proxies,
                grant_type,
                fusion_e2e,
            })),
        }
    }
}


#[pyclass]
struct FusionSession {
    credentials: Arc<Mutex<FusionCredentials>>,
    client: Client,
    refresh_within_seconds: u64,
}

#[pymethods]
impl FusionSession {
    #[new]
    fn new(credentials: &PyFusionCredentials, refresh_within_seconds: u64) -> Self {
        FusionSession {
            credentials: Arc::clone(&credentials.inner),
            client: Client::new(),
            refresh_within_seconds,
        }
    }

    fn send_request(&self, method: String, url: String, body: Option<String>) -> PyResult<String> {
        let mut credentials = self.credentials.lock().unwrap();
        let token_expires_in = match credentials.bearer_token_expiry {
        Some(expiry) => expiry as i64 - SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() as i64,
        None => return Err(pyo3::exceptions::PyException::new_err("Bearer token expiry is None")),
        };
        let rt = Runtime::new().unwrap();
        if credentials.is_bearer_token_expirable.unwrap_or(true) && token_expires_in < self.refresh_within_seconds as i64 {
            rt.block_on(credentials.refresh_token()).unwrap();
        }
        let mut headers = HeaderMap::new();
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {}", credentials.bearer_token.as_ref().unwrap())).unwrap(),
        );
        headers.insert(USER_AGENT, HeaderValue::from_str("fusion-rust-sdk").unwrap());
        if let Some(e2e) = &credentials.fusion_e2e {
            headers.insert("fusion-e2e", HeaderValue::from_str(e2e).unwrap());
        }

        let request_builder = self.client.request(
            method.parse::<Method>().unwrap(),
            url.parse::<Url>().unwrap(),
        ).headers(headers);

        let request_builder = if let Some(body) = body {
            request_builder.body(body)
        } else {
            request_builder
        };

        let response = rt.block_on(request_builder.send()).map_err(|e| pyo3::exceptions::PyException::new_err(e.to_string()))?;
        let response_text = rt.block_on(response.text()).map_err(|e| pyo3::exceptions::PyException::new_err(e.to_string()))?;

        Ok(response_text)
    }
}

#[pymodule]
fn fusion_session(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<FusionSession>()?;
    m.add_class::<PyFusionCredentials>()?;
    Ok(())
}
