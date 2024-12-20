use bincode::{deserialize, serialize};
use chrono::{NaiveDate, Utc};
use json;
use jsonwebtoken::{encode, Algorithm, EncodingKey, Header};
use pyo3::exceptions::{PyFileNotFoundError, PyValueError};
use pyo3::import_exception;
use pyo3::prelude::*;
use pyo3::types::{PyDate, PyDateAccess, PyType};
use reqwest::Proxy;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use std::fmt::Display;
use std::fs::File;
use std::io::prelude::*;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use url::Url;

#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};

use crate::utils::get_tokio_runtime;

import_exception!(fusion.exceptions, APIResponseError);
import_exception!(fusion.exceptions, APIRequestError);
import_exception!(fusion.exceptions, APIConnectError);
import_exception!(fusion.exceptions, UnrecognizedFormatError);
import_exception!(fusion.exceptions, CredentialError);

#[inline]
fn default_grant_type() -> String {
    "client_credentials".to_string()
}

#[inline]
fn default_auth_url() -> String {
    "https://authe.jpmorgan.com/as/token.oauth2".to_string()
}

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProxyType {
    http,
    https,
}

impl FromStr for ProxyType {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "http" => Ok(ProxyType::http),
            "https" => Ok(ProxyType::https),
            _ => Err(()),
        }
    }
}

impl Display for ProxyType {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            ProxyType::http => write!(f, "http"),
            ProxyType::https => write!(f, "https"),
        }
    }
}

fn untyped_proxies(proxies: HashMap<ProxyType, String>) -> HashMap<String, String> {
    let mut untyped_proxies = HashMap::new();
    for (key, value) in proxies {
        untyped_proxies.insert(key.to_string(), value);
    }
    untyped_proxies
}

fn client_builder_from_proxies(proxies: &HashMap<String, String>) -> reqwest::ClientBuilder {
    let mut client_builder = reqwest::Client::builder();
    for (key, value) in proxies {
        match ProxyType::from_str(key) {
            Ok(ProxyType::http) => {
                client_builder = client_builder.proxy(Proxy::http(value).unwrap());
            }
            Ok(ProxyType::https) => {
                client_builder = client_builder.proxy(Proxy::https(value).unwrap());
            }
            Err(_) => {
                warn!("Unrecognized proxy type: {}", key);
            }
        }
    }
    client_builder
}

fn find_cfg_file(file_path: &Path) -> PyResult<PathBuf> {
    let current_path = file_path.to_path_buf();

    if current_path.is_file() {
        debug!(
            "Found file at the provided path: {}",
            current_path.display()
        );
        return Ok(current_path);
    }
    let cwd = env::current_dir()?;
    let cfg_file_name = "client_credentials.json";
    let cfg_folder_name = "config";
    let start_dir = match current_path.parent() {
        Some(parent) => match parent.exists() {
            true => parent.to_path_buf(),
            false => cwd,
        },
        None => cwd,
    };
    let mut start_dir_abs = start_dir.canonicalize()?;
    let start_dir_init = start_dir_abs.clone();
    loop {
        let full_path = start_dir_abs.join(cfg_folder_name).join(cfg_file_name);
        if full_path.is_file() {
            debug!("Found file at: {}", full_path.display());
            return Ok(full_path);
        }

        // Move to the parent directory
        if let Some(parent) = start_dir_abs.parent() {
            start_dir_abs = parent.to_path_buf().canonicalize()?;
        } else {
            // Reached the root directory
            let error_message = format!(
                "File {} not found in {} or any of its parents. Current parent: {}",
                cfg_file_name,
                start_dir_init.display(),
                start_dir.display()
            );
            return Err(PyFileNotFoundError::new_err(error_message));
        }
    }
}

fn fusion_url_to_auth_url(url: String) -> PyResult<Option<(String, String, String)>> {
    debug!("Trying to form fusion auth url from: {}", url);
    let url_parsed = Url::parse(&url)
        .map_err(|e| CredentialError::new_err(format!("Could not parse URL: {:?}", e)))?;

    let path = url_parsed.path();
    let segments: Vec<&str> = path.split('/').collect();

    // Not a distribution request. No need to authorize
    if !segments.contains(&"distributions") {
        debug!("Not a distribution request. No need to authorize");
        return Ok(None);
    }

    let catalog_name = segments
        .iter()
        .position(|&s| s == "catalogs")
        .and_then(|i| segments.get(i + 1))
        .map(|&s| s.to_string())
        .ok_or_else(|| {
            CredentialError::new_err(
                "'catalogs' segment not found or catalog name missing in the path",
            )
        })?;

    let dataset_name = segments
        .iter()
        .position(|&s| s == "datasets")
        .and_then(|i| segments.get(i + 1))
        .map(|&s| s.to_string())
        .ok_or_else(|| {
            CredentialError::new_err(
                "'datasets' segment not found or dataset name missing in the path",
            )
        })?;

    debug!("Found Catalog: {}, Dataset: {}", catalog_name, dataset_name);
    let new_path = format!(
        "{}/authorize/token",
        segments[..=segments.iter().position(|&s| s == "datasets").unwrap() + 1].join("/")
    );

    let fusion_tk_url = format!(
        "{}://{}{}{}",
        url_parsed.scheme(),
        url_parsed.host_str().unwrap_or_default(),
        url_parsed
            .port()
            .map_or(String::new(), |p| format!(":{}", p)),
        new_path
    );
    debug!("Fusion token URL: {}", fusion_tk_url);
    Ok(Some((fusion_tk_url, catalog_name, dataset_name)))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FusionCredsPersistent {
    client_id: Option<String>,
    client_secret: Option<String>,
    username: Option<String>,
    password: Option<String>,
    resource: Option<String>,
    auth_url: Option<String>,
    root_url: Option<String>,
    #[serde(default)]
    proxies: HashMap<ProxyType, String>,
    #[serde(default = "default_grant_type")]
    grant_type: String,
    //#[serde(deserialize_with = "deserialize_fusion_e2e")]
    fusion_e2e: Option<String>,
    headers: Option<HashMap<String, String>>,
    kid: Option<String>,
    private_key: Option<String>,
}

#[pyclass(module = "fusion._fusion")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthToken {
    #[pyo3(get)]
    token: String,
    #[pyo3(get)]
    expiry: Option<i64>,
}

#[pymethods]
impl AuthToken {
    pub fn is_expirable(&self) -> bool {
        self.expiry.is_some()
    }

    pub fn expires_in_secs(&self) -> Option<i64> {
        match self.expiry {
            Some(expiry) => {
                let current_time = Utc::now().timestamp();
                Some(expiry - current_time)
            }
            None => None,
        }
    }

    #[staticmethod]
    #[pyo3(signature = (token, expires_in_secs=None))]
    pub fn from_token(token: String, expires_in_secs: Option<i64>) -> Self {
        let expiry = expires_in_secs.map(|secs| {
            let current_time = Utc::now().timestamp();
            current_time + secs
        });
        AuthToken { token, expiry }
    }

    fn __getstate__(&self) -> PyResult<Vec<u8>> {
        Ok(serialize(&self).unwrap())
    }

    fn __setstate__(&mut self, state: Vec<u8>) -> PyResult<()> {
        *self = deserialize(&state).unwrap();
        Ok(())
    }

    fn __getnewargs__(&self) -> PyResult<(String, Option<i64>)> {
        Ok((self.token.clone(), self.expiry))
    }

    #[new]
    #[pyo3(signature = (token, expires_in_secs=None))]
    fn new(token: String, expires_in_secs: Option<i64>) -> Self {
        AuthToken::from_token(token, expires_in_secs)
    }

    fn as_bearer_header(&self) -> PyResult<(String, String)> {
        Ok(("Authorization".to_owned(), format!("Bearer {}", self.token)))
    }

    fn as_fusion_header(&self) -> PyResult<(String, String)> {
        Ok((
            "Fusion-Authorization".to_owned(),
            format!("Bearer {}", self.token),
        ))
    }
}

impl Default for AuthToken {
    fn default() -> Self {
        AuthToken {
            token: "".to_string(),
            expiry: None,
        }
    }
}

fn build_client(proxies: &Option<HashMap<String, String>>) -> PyResult<reqwest::Client> {
    client_builder_from_proxies(proxies.as_ref().unwrap_or(&HashMap::new()))
        .use_rustls_tls()
        .tls_built_in_native_certs(true)
        .build()
        .map_err(|err| CredentialError::new_err(format!("Error creating HTTP client: {}", err)))
}

#[pyclass(module = "fusion._fusion")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionCredentials {
    #[pyo3(get, set)]
    client_id: Option<String>,
    #[pyo3(get, set)]
    client_secret: Option<String>,

    #[pyo3(get, set)]
    username: Option<String>,

    #[pyo3(get, set)]
    password: Option<String>,

    #[pyo3(get, set)]
    resource: Option<String>,

    #[pyo3(get, set)]
    auth_url: Option<String>,

    #[pyo3(get, set)]
    grant_type: String,

    #[pyo3(get, set)]
    fusion_e2e: Option<String>,

    #[pyo3(get, set)]
    headers: HashMap<String, String>,

    #[pyo3(get)]
    proxies: HashMap<String, String>,

    #[pyo3(get, set)]
    bearer_token: Option<AuthToken>,

    #[pyo3(get, set)]
    fusion_token: HashMap<String, AuthToken>,

    #[serde(skip)]
    http_client: Option<reqwest::Client>,

    #[pyo3(get, set)]
    kid: Option<String>,

    #[pyo3(get, set)]
    private_key: Option<String>,
}

impl Default for FusionCredentials {
    fn default() -> Self {
        FusionCredentials {
            client_id: None,
            client_secret: None,
            username: None,
            password: None,
            resource: None,
            auth_url: Some(default_auth_url()),
            bearer_token: None,
            fusion_token: HashMap::new(),
            proxies: HashMap::new(),
            grant_type: "client_credentials".to_string(),
            fusion_e2e: None,
            headers: HashMap::new(),
            kid: None,
            private_key: None,
            http_client: None,
        }
    }
}

#[derive(Serialize, Debug)]
struct Claims {
    iss: String,
    aud: String,
    sub: String,
    iat: i64,
    exp: i64,
    jti: String,
}

#[pymethods]
impl FusionCredentials {
    fn __getstate__(&self) -> PyResult<Vec<u8>> {
        Ok(serialize(&self).unwrap())
    }

    fn __setstate__(&mut self, state: Vec<u8>) -> PyResult<()> {
        *self = deserialize(&state).unwrap();
        Ok(())
    }

    fn __getnewargs__(
        &self,
    ) -> PyResult<(
        Option<String>,
        Option<String>,
        Option<String>,
        Option<String>,
        Option<String>,
        Option<String>,
        Option<AuthToken>,
        Option<HashMap<String, String>>,
        Option<String>,
        Option<String>,
        Option<HashMap<String, String>>,
    )> {
        Ok((
            self.client_id.clone(),
            self.client_secret.clone(),
            self.username.clone(),
            self.password.clone(),
            self.resource.clone(),
            self.auth_url.clone(),
            self.bearer_token.clone(),
            Some(self.proxies.clone()),
            Some(self.grant_type.clone()),
            self.fusion_e2e.clone(),
            Some(self.headers.clone()),
        ))
    }

    #[classmethod]
    #[pyo3(signature = (client_id=None, client_secret=None, resource=None, auth_url=None, proxies=None, fusion_e2e=None, headers=None, kid=None, private_key=None))]
    fn from_client_id(
        _cls: &Bound<'_, PyType>,
        client_id: Option<String>,
        client_secret: Option<String>,
        resource: Option<String>,
        auth_url: Option<String>,
        proxies: Option<HashMap<String, String>>,
        fusion_e2e: Option<String>,
        headers: Option<HashMap<String, String>>,
        kid: Option<String>,
        private_key: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            client_id,
            client_secret,
            resource,
            auth_url: Some(auth_url.unwrap_or_else(default_auth_url)),
            proxies: proxies.unwrap_or_default(),
            grant_type: "client_credentials".to_string(),
            fusion_e2e,
            headers: headers.unwrap_or_default(),
            kid,
            private_key,
            fusion_token: HashMap::new(),
            bearer_token: None,
            username: None,
            password: None,
            http_client: None,
        })
    }

    fn _ensure_http_client(&mut self) -> PyResult<()> {
        if self.http_client.is_none() {
            debug!("Creating HTTP client");
            self.http_client = Some(build_client(&Some(self.proxies.clone()))?);
        }
        Ok(())
    }

    #[classmethod]
    #[pyo3(signature = (client_id=None, username=None, password=None, resource=None, auth_url=None, proxies=None, fusion_e2e=None, headers=None, kid=None, private_key=None))]
    fn from_user_id(
        _cls: &Bound<'_, PyType>,
        client_id: Option<String>,
        username: Option<String>,
        password: Option<String>,
        resource: Option<String>,
        auth_url: Option<String>,
        proxies: Option<HashMap<String, String>>,
        fusion_e2e: Option<String>,
        headers: Option<HashMap<String, String>>,
        kid: Option<String>,
        private_key: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            client_id,
            username,
            password,
            resource,
            auth_url: Some(auth_url.unwrap_or_else(default_auth_url)),
            proxies: proxies.unwrap_or_default(),
            grant_type: "password".to_string(),
            fusion_e2e,
            headers: headers.unwrap_or_default(),
            kid,
            private_key,
            fusion_token: HashMap::new(),
            bearer_token: None,
            client_secret: None,
            http_client: None,
        })
    }

    #[classmethod]
    #[pyo3(signature = (bearer_token=None, bearer_token_expiry=None, proxies=None, fusion_e2e=None, headers=None))]
    fn from_bearer_token(
        _cls: &Bound<'_, PyType>,
        bearer_token: Option<String>,
        bearer_token_expiry: Option<&Bound<PyDate>>,
        proxies: Option<HashMap<String, String>>,
        fusion_e2e: Option<String>,
        headers: Option<HashMap<String, String>>,
    ) -> PyResult<Self> {
        Ok(Self {
            resource: None,
            auth_url: None,
            proxies: proxies.unwrap_or_default(),
            grant_type: "bearer".to_string(),
            fusion_e2e,
            headers: headers.unwrap_or_default(),
            fusion_token: HashMap::new(),
            bearer_token: Some(AuthToken {
                token: match bearer_token {
                    Some(token) => token,
                    None => return Err(PyValueError::new_err("Bearer token not provided")),
                },
                expiry: bearer_token_expiry.map(|date| {
                    let year = date.get_year();
                    let month = date.get_month() as u32;
                    let day = date.get_day() as u32;
                    NaiveDate::from_ymd_opt(year, month, day)
                        .and_then(|naive_date| naive_date.and_hms_opt(0, 0, 0))
                        .map(|datetime| datetime.and_utc().timestamp())
                        .unwrap_or(0)
                }),
            }),
            client_id: None,
            client_secret: None,
            username: None,
            password: None,
            http_client: None,
            kid: None,
            private_key: None,
        })
    }

    // #[allow(clippy::too_many_arguments)]
    #[new]
    #[pyo3(signature = (client_id=None, client_secret=None, username=None, password=None, resource=None, auth_url=None, bearer_token=None, proxies=None, grant_type=None, fusion_e2e=None, headers=None, kid=None, private_key=None))]
    fn new(
        client_id: Option<String>,
        client_secret: Option<String>,
        username: Option<String>,
        password: Option<String>,
        resource: Option<String>,
        auth_url: Option<String>,
        bearer_token: Option<AuthToken>,
        proxies: Option<HashMap<String, String>>,
        grant_type: Option<String>,
        fusion_e2e: Option<String>,
        headers: Option<HashMap<String, String>>,
        kid: Option<String>,
        private_key: Option<String>,
    ) -> PyResult<Self> {
        Ok(FusionCredentials {
            client_id,
            client_secret,
            username,
            password,
            resource,
            auth_url: Some(auth_url.unwrap_or_else(default_auth_url)),
            bearer_token,
            fusion_token: HashMap::new(),
            proxies: proxies.unwrap_or_default(),
            grant_type: grant_type.unwrap_or_else(|| "client_credentials".to_string()),
            fusion_e2e,
            headers: headers.unwrap_or_default(),
            kid,
            private_key,
            http_client: None,
        })
    }

    #[pyo3(signature = (force=true, max_remain_secs=30))]
    fn _refresh_bearer_token(
        &mut self,
        py: Python,
        force: bool,
        max_remain_secs: u32,
    ) -> PyResult<bool> {
        if !force {
            if let Some(token) = self.bearer_token.as_ref() {
                if !token.is_expirable() {
                    return Ok(false);
                }
                if let Some(expires_in_secs) = token.expires_in_secs() {
                    if expires_in_secs > max_remain_secs as i64 {
                        return Ok(false);
                    }
                }
            }
        }
        self._ensure_http_client()?;
        let client = self.http_client.as_ref().ok_or_else(|| {
            CredentialError::new_err(
                "HTTP client not initialized. Use from_* methods to create credentials",
            )
        })?;

        let payload = if let (Some(_kid), Some(private_key)) = (&self.kid, &self.private_key) {
            // Create JWT claims
            let claims = Claims {
                iss: self.client_id.clone().unwrap_or_default(),
                aud: self.auth_url.clone().unwrap_or_default(),
                sub: self.client_id.clone().unwrap_or_default(),
                iat: Utc::now().timestamp(),
                exp: Utc::now().timestamp() + 3600,
                jti: "id001".to_string(),
            };
            // Encode the JWT
            let private_key_bytes = private_key.as_bytes();
            let encoding_key =
                EncodingKey::from_rsa_pem(private_key_bytes).expect("Invalid RSA private key");
            let mut header = Header::new(Algorithm::RS256);
            header.kid = Some(self.kid.clone().unwrap_or_default());
            let private_key_jwt =
                encode(&header, &claims, &encoding_key).expect("Failed to encode JWT");

            // Build the payload vector
            vec![
                ("grant_type".to_string(), self.grant_type.clone()),
                (
                    "client_id".to_string(),
                    self.client_id.clone().unwrap_or_default(),
                ),
                (
                    "client_assertion_type".to_string(),
                    "urn:ietf:params:oauth:client-assertion-type:jwt-bearer".to_string(),
                ),
                ("client_assertion".to_string(), private_key_jwt),
                (
                    "resource".to_string(),
                    self.resource.clone().unwrap_or_default(),
                ),
            ]
        } else {
            match self.grant_type.as_str() {
                "client_credentials" => vec![
                    ("grant_type".to_string(), self.grant_type.clone()),
                    (
                        "client_id".to_string(),
                        self.client_id.clone().unwrap_or_default(),
                    ),
                    (
                        "client_secret".to_string(),
                        self.client_secret.clone().unwrap_or_default(),
                    ),
                    ("aud".to_string(), self.resource.clone().unwrap_or_default()),
                ],
                "password" => vec![
                    ("grant_type".to_string(), self.grant_type.clone()),
                    (
                        "client_id".to_string(),
                        self.client_id.clone().unwrap_or_default(),
                    ),
                    (
                        "username".to_string(),
                        self.username.clone().unwrap_or_default(),
                    ),
                    (
                        "password".to_string(),
                        self.password.clone().unwrap_or_default(),
                    ),
                    (
                        "resource".to_string(),
                        self.resource.clone().unwrap_or_default(),
                    ),
                ],
                "bearer" => {
                    // Nothing to do
                    return Ok(true);
                }
                _ => {
                    return Err(PyValueError::new_err("Unrecognized grant type"));
                }
            }
        };

        let rt = &get_tokio_runtime(py).0;

        let response_res: PyResult<json::JsonValue> = rt.block_on(async {
            let res = client
                .post(self.auth_url.as_ref().unwrap())
                .header(
                    "User-Agent",
                    format!("fusion-python-sdk {}", env!("CARGO_PKG_VERSION")),
                )
                .form(&payload)
                .send()
                .await
                .map_err(|e| {
                    CredentialError::new_err(format!("Could not post request: {:?}", e))
                })?;

            let res_text = res.text().await.map_err(|e| {
                CredentialError::new_err(format!("Could not get response text: {:?}", e))
            })?;
            let res_json = json::parse(&res_text).map_err(|e| {
                CredentialError::new_err(format!("Could not parse test to json: {:?}", e))
            })?;
            debug!("Called for Bearer token. Response: {:?}", res_json);
            Ok(res_json)
        });
        let response = response_res?;
        let token = response["access_token"].as_str().unwrap().to_string();
        let expires_in_secs = response["expires_in"].as_i64();
        match expires_in_secs {
            Some(expires_in_secs) => {
                debug!("Got Bearer token, expires in: {}", expires_in_secs);
            }
            None => {
                debug!("Got Bearer token, no expiration");
            }
        }
        self.put_bearer_token(token, expires_in_secs);
        Ok(true)
    }

    #[pyo3(signature = (bearer_token, expires_in_secs=None))]
    fn put_bearer_token(&mut self, bearer_token: String, expires_in_secs: Option<i64>) {
        self.bearer_token = Some(AuthToken::from_token(bearer_token, expires_in_secs));
    }

    #[pyo3(signature = (token_key, token, expires_in_secs=None))]
    fn put_fusion_token(&mut self, token_key: String, token: String, expires_in_secs: Option<i64>) {
        self.fusion_token
            .insert(token_key, AuthToken::from_token(token, expires_in_secs));
    }

    fn _gen_fusion_token(&mut self, py: Python, url: String) -> PyResult<(String, Option<i64>)> {
        let rt = &get_tokio_runtime(py).0;
        self._ensure_http_client()?;
        let client = self.http_client.as_ref().ok_or_else(|| {
            CredentialError::new_err(
                "HTTP client not initialized. Use from_* methods to create credentials",
            )
        })?;

        let response_res: PyResult<json::JsonValue> = rt.block_on(async {
            let token = self
                .bearer_token
                .as_ref()
                .ok_or_else(|| CredentialError::new_err("Bearer token is missing".to_string()))?;

            let req = client
                .get(&url)
                .header("Authorization", format!("Bearer {}", token.token))
                .header(
                    "User-Agent",
                    format!("fusion-python-sdk {}", env!("CARGO_PKG_VERSION")),
                );

            debug!("Calling for Fusion token: {:?}", req);

            let res_maybe = req.send().await.map_err(|e| {
                CredentialError::new_err(format!("Could not post request: {:?}", e))
            })?;

            let res = res_maybe
                .error_for_status()
                .map_err(|e| CredentialError::new_err(format!("Error from endpoint: {:?}", e)))?;

            match res.text().await {
                Ok(text) => {
                    let res_json = json::parse(&text).map_err(|e| {
                        CredentialError::new_err(format!(
                            "Could not parse response to json: {:?}",
                            e
                        ))
                    })?;
                    Ok(res_json)
                }
                Err(e) => Err(CredentialError::new_err(format!(
                    "Could not get response text: {:?}",
                    e
                ))),
            }
        });
        let response = response_res?;
        let token = response["access_token"].as_str().unwrap().to_string();
        let expires_in_secs = response["expires_in"].as_i64();
        debug!("Got Fusion token, expires in: {:?}", expires_in_secs);
        Ok((token, expires_in_secs))
    }

    fn get_fusion_token_headers(
        &mut self,
        py: Python,
        url: String,
    ) -> PyResult<HashMap<String, String>> {
        // Ensure bearer token is valid, don't force refresh, and refresh if it expires in 30 seconds
        let mut ret = HashMap::new();
        ret.insert(
            "User-Agent".into(),
            format!("fusion-python-sdk {}", env!("CARGO_PKG_VERSION")),
        );
        if self.fusion_e2e.is_some() {
            ret.insert(
                "fusion-e2e".into(),
                self.fusion_e2e.as_ref().unwrap().clone(),
            );
        }
        // if headers are not empty add each key value pair to the headers
        if !self.headers.is_empty() {
            for (key, value) in self.headers.iter() {
                ret.insert(key.clone(), value.clone());
            }
        }

        let is_bearer_refreshed = self._refresh_bearer_token(py, false, 15 * 60)?;
        let bearer_token_tup = self
            .bearer_token
            .as_ref()
            .ok_or_else(|| CredentialError::new_err("No bearer token set"))?
            .as_bearer_header()?;

        // Check if the URL is a distribution request, and if so, get the Fusion token auth url, dataset name, and catalog name
        let fusion_tk_url_info = fusion_url_to_auth_url(url)?;

        if fusion_tk_url_info.is_none() {
            ret.insert(bearer_token_tup.0, bearer_token_tup.1);
            debug!("Headers are {:?}", ret);
            return Ok(ret);
        }

        let (fusion_tk_url, catalog_name, dataset_name) = fusion_tk_url_info.unwrap();

        let token_key = format!("{}_{}", catalog_name, dataset_name);

        // Check if the token exists and if it's expired
        let token_entry = self.fusion_token.entry(token_key.clone());
        let (fusion_token_tup, new_token) = match token_entry {
            std::collections::hash_map::Entry::Occupied(mut entry) => {
                let token = entry.get_mut();
                if let Some(expires_in_secs) = token.expires_in_secs() {
                    if expires_in_secs < 15 * 60 || is_bearer_refreshed {
                        (None, Some(self._gen_fusion_token(py, fusion_tk_url)?))
                    } else {
                        (Some(token.as_fusion_header()?), None)
                    }
                } else {
                    (Some(token.as_fusion_header()?), None)
                }
            }
            std::collections::hash_map::Entry::Vacant(_) => {
                (None, Some(self._gen_fusion_token(py, fusion_tk_url)?))
            }
        };

        // If we need to generate a new token, insert it into the HashMap
        let fusion_token_tup = if let Some(new_token) = new_token {
            let token = AuthToken::from_token(new_token.0, new_token.1);
            self.fusion_token.insert(token_key, token.clone());
            token.as_fusion_header()?
        } else {
            fusion_token_tup.unwrap()
        };

        ret.insert(bearer_token_tup.0, bearer_token_tup.1);
        ret.insert(fusion_token_tup.0, fusion_token_tup.1);
        debug!("Headers are {:?}", ret);
        Ok(ret)
    }

    fn get_fusion_token_expires_in(&self, token_key: String) -> PyResult<Option<i64>> {
        Ok(self
            .fusion_token
            .get(&token_key)
            .and_then(|token| token.expires_in_secs()))
    }

    #[classmethod]
    pub fn from_file(cls: &Bound<'_, PyType>, file_path: PathBuf) -> PyResult<FusionCredentials> {
        let found_path = find_cfg_file(&file_path)?;
        let file =
            File::open(&found_path).map_err(|_| PyFileNotFoundError::new_err("File not found"))?;

        let credentials: FusionCredsPersistent = match serde_json::from_reader(file) {
            Ok(credentials) => credentials,
            Err(err) => {
                let mut contents = String::new();
                let mut file = File::open(&found_path)
                    .map_err(|_| PyFileNotFoundError::new_err("File not found"))?;
                file.read_to_string(&mut contents)
                    .map_err(|_| PyFileNotFoundError::new_err("Could not read file contents"))?;
                return Err(CredentialError::new_err(format!(
                    "Invalid JSON: {}\nContents:\n{}",
                    err, contents
                )));
            }
        };
        let client_id = credentials
            .client_id
            .or_else(|| std::env::var("FUSION_CLIENT_ID").ok())
            .ok_or_else(|| CredentialError::new_err("Missing client ID"))?;

        let full_creds = match credentials.grant_type.as_str() {
            "client_credentials" => FusionCredentials::from_client_id(
                cls,
                Some(client_id),
                Some(
                    credentials
                        .client_secret
                        .or_else(|| std::env::var("FUSION_CLIENT_SECRET").ok())
                        .ok_or_else(|| CredentialError::new_err("Missing client secret"))?,
                ),
                credentials.resource,
                credentials.auth_url,
                Some(untyped_proxies(credentials.proxies)),
                credentials.fusion_e2e,
                credentials.headers,
                credentials.kid,
                credentials.private_key,
            )?,
            "bearer" => FusionCredentials::from_bearer_token(
                cls,
                None,
                None,
                Some(untyped_proxies(credentials.proxies)),
                credentials.fusion_e2e,
                credentials.headers,
            )?,
            "password" => FusionCredentials::from_user_id(
                cls,
                Some(client_id),
                credentials.username,
                credentials.password,
                credentials.resource,
                credentials.auth_url,
                Some(untyped_proxies(credentials.proxies)),
                credentials.fusion_e2e,
                credentials.headers,
                credentials.kid,
                credentials.private_key,
            )?,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Unrecognized grant type",
                ))
            }
        };
        Ok(full_creds)
    }
}
// Tests

#[cfg(test)]
mod tests {
    use super::*;
    use bincode::{deserialize, serialize};
    use serde_json::json;
    use std::fs;
    use std::fs::File;
    use std::io::Write;
    use tempdir::TempDir;

    #[test]
    fn test_default_grant_type() {
        let expected = "client_credentials".to_string();
        let result = default_grant_type();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_proxy_type_from_str() {
        assert_eq!(ProxyType::from_str("http"), Ok(ProxyType::http));
        assert_eq!(ProxyType::from_str("https"), Ok(ProxyType::https));
        assert_eq!(ProxyType::from_str("ftp"), Err(()));
    }

    #[test]
    fn test_proxy_type_display() {
        assert_eq!(format!("{}", ProxyType::http), "http");
        assert_eq!(format!("{}", ProxyType::https), "https");
    }

    #[test]
    fn test_untyped_proxies() {
        let mut proxies = HashMap::new();
        proxies.insert(ProxyType::http, "http://example.com".to_string());
        proxies.insert(ProxyType::https, "https://example.com".to_string());

        let untyped = untyped_proxies(proxies);
        assert_eq!(untyped.get("http"), Some(&"http://example.com".to_string()));
        assert_eq!(
            untyped.get("https"),
            Some(&"https://example.com".to_string())
        );
    }

    #[test]
    fn test_find_cfg_file_in_current_directory() {
        let temp_dir = TempDir::new("test_find_cfg_file").unwrap();
        let cfg_file_path = temp_dir.path().join("client_credentials.json");

        // Create the config file in the current directory
        let mut file = File::create(&cfg_file_path).unwrap();
        writeln!(file, "{{\"key\": \"value\"}}").unwrap();

        let result = find_cfg_file(&cfg_file_path).unwrap();
        assert_eq!(result, cfg_file_path);
    }

    #[test]
    fn test_find_cfg_file_in_parent_directory() {
        let temp_dir = TempDir::new("test_find_cfg_file").unwrap();
        let parent_dir = temp_dir.path().join("parent");
        let child_dir = parent_dir.join("child");
        let dir_path = Path::new("config");

        fs::create_dir_all(&child_dir).unwrap();
        let cfg_file_path = parent_dir.join("config").join("client_credentials.json");

        // Create the config file in the parent directory
        fs::create_dir_all(parent_dir.join(dir_path)).expect("Failed to create directory");
        let mut file = File::create(&cfg_file_path).unwrap();
        writeln!(file, "{{\"key\": \"value\"}}").unwrap();

        let result = find_cfg_file(&child_dir.join("dummy_file")).unwrap();
        assert_eq!(result, cfg_file_path);
    }

    #[test]
    fn test_file_not_found() {
        pyo3::prepare_freethreaded_python();
        let temp_dir = TempDir::new("test_find_cfg_file").unwrap();
        let child_dir = temp_dir.path().join("child");

        fs::create_dir_all(&child_dir).unwrap();

        let result = find_cfg_file(&child_dir.join("dummy_file"));
        assert!(result.is_err());
        if let Err(e) = result {
            let error_message = e.to_string();
            assert!(error_message.contains("File client_credentials.json not found in"));
            assert!(error_message.contains("or any of its parents"));
        } else {
            panic!("Expected an error, but got Ok");
        }
    }

    #[test]
    fn test_auth_token_creation() {
        let token = AuthToken::new("test_token".to_string(), Some(3600));
        assert_eq!(token.token, "test_token");
        assert!(token.expiry.is_some());
    }

    #[test]
    fn test_auth_token_default() {
        let token = AuthToken::default();
        assert_eq!(token.token, "");
        assert!(token.expiry.is_none());
    }

    #[test]
    fn test_auth_token_is_expirable() {
        let token_with_expiry = AuthToken::new("test_token".to_string(), Some(3600));
        assert!(token_with_expiry.is_expirable());

        let token_without_expiry = AuthToken::new("test_token".to_string(), None);
        assert!(!token_without_expiry.is_expirable());
    }

    #[test]
    fn test_auth_token_expires_in_secs() {
        let token = AuthToken::new("test_token".to_string(), Some(3600));
        assert!(token.expires_in_secs().is_some());
        assert!(token.expires_in_secs().unwrap() <= 3600);

        let token_without_expiry = AuthToken::new("test_token".to_string(), None);
        assert!(token_without_expiry.expires_in_secs().is_none());
    }

    #[test]
    fn test_auth_token_from_token() {
        let token = AuthToken::from_token("test_token".to_string(), Some(3600));
        assert_eq!(token.token, "test_token");
        assert!(token.expiry.is_some());

        let token_without_expiry = AuthToken::from_token("test_token".to_string(), None);
        assert_eq!(token_without_expiry.token, "test_token");
        assert!(token_without_expiry.expiry.is_none());
    }

    #[test]
    fn test_auth_token_serialization() {
        let token = AuthToken::new("test_token".to_string(), Some(3600));
        let serialized_token = token.__getstate__().unwrap();
        let deserialized_token: AuthToken = deserialize(&serialized_token).unwrap();
        assert_eq!(token.token, deserialized_token.token);
        assert_eq!(token.expiry, deserialized_token.expiry);
    }

    #[test]
    fn test_auth_token_deserialization() {
        let token = AuthToken::new("test_token".to_string(), Some(3600));
        let serialized_token = serialize(&token).unwrap();
        let mut deserialized_token = AuthToken::default();
        deserialized_token.__setstate__(serialized_token).unwrap();
        assert_eq!(token.token, deserialized_token.token);
        assert_eq!(token.expiry, deserialized_token.expiry);
    }

    #[test]
    fn test_auth_token_getnewargs() {
        let token = AuthToken::new("test_token".to_string(), Some(3600));
        let (token_str, expiry) = token.__getnewargs__().unwrap();
        assert_eq!(token_str, token.token);
        assert_eq!(expiry, token.expiry);
    }

    #[test]
    fn test_auth_token_pymethods() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let token = Py::new(py, AuthToken::new("test_token".to_string(), Some(3600))).unwrap();
            let token_ref = token.borrow(py);
            assert_eq!(token_ref.token, "test_token");
            assert!(token_ref.expiry.is_some());

            let token_dict = token_ref.__getstate__().unwrap();
            assert!(!token_dict.is_empty());

            let new_args = token_ref.__getnewargs__().unwrap();
            assert_eq!(new_args.0, "test_token");
            assert!(new_args.1.is_some());
        });
    }

    #[test]
    fn test_fusion_credentials_creation() {
        let creds = FusionCredentials::new(
            Some("client_id".to_string()),
            Some("client_secret".to_string()),
            Some("username".to_string()),
            Some("password".to_string()),
            Some("resource".to_string()),
            Some("auth_url".to_string()),
            None,
            Some(HashMap::new()),
            Some("grant_type".to_string()),
            Some("fusion_e2e".to_string()),
            Some(HashMap::new()),
            Some("kid".to_string()),
            Some("private_key".to_string()),
        )
        .unwrap();

        assert_eq!(creds.client_id, Some("client_id".to_string()));
        assert_eq!(creds.client_secret, Some("client_secret".to_string()));
        assert_eq!(creds.username, Some("username".to_string()));
        assert_eq!(creds.password, Some("password".to_string()));
        assert_eq!(creds.resource, Some("resource".to_string()));
        assert_eq!(creds.auth_url, Some("auth_url".to_string()));
        assert_eq!(creds.grant_type, "grant_type".to_string());
        assert_eq!(creds.fusion_e2e, Some("fusion_e2e".to_string()));
    }

    #[test]
    fn test_fusion_credentials_default() {
        let creds = FusionCredentials::default();
        assert!(creds.client_id.is_none());
        assert!(creds.client_secret.is_none());
        assert!(creds.username.is_none());
        assert!(creds.password.is_none());
        assert!(creds.resource.is_none());
        assert_eq!(creds.auth_url, Some(default_auth_url()));
        assert_eq!(creds.grant_type, "client_credentials".to_string());
        assert!(creds.fusion_e2e.is_none());
        assert!(creds.bearer_token.is_none());
        assert!(creds.proxies.is_empty());
        assert!(creds.headers.is_empty());
        assert!(creds.fusion_token.is_empty());
    }

    #[test]
    fn test_fusion_credentials_serialization() {
        let creds = FusionCredentials::new(
            Some("client_id".to_string()),
            Some("client_secret".to_string()),
            Some("username".to_string()),
            Some("password".to_string()),
            Some("resource".to_string()),
            Some("auth_url".to_string()),
            None,
            Some(HashMap::new()),
            Some("grant_type".to_string()),
            Some("fusion_e2e".to_string()),
            Some(HashMap::new()),
            Some("kid".to_string()),
            Some("private_key".to_string()),
        )
        .unwrap();

        let serialized_creds = creds.__getstate__().unwrap();
        let deserialized_creds: FusionCredentials = deserialize(&serialized_creds).unwrap();

        assert_eq!(creds.client_id, deserialized_creds.client_id);
        assert_eq!(creds.client_secret, deserialized_creds.client_secret);
        assert_eq!(creds.username, deserialized_creds.username);
        assert_eq!(creds.password, deserialized_creds.password);
        assert_eq!(creds.resource, deserialized_creds.resource);
        assert_eq!(creds.auth_url, deserialized_creds.auth_url);
        assert_eq!(creds.grant_type, deserialized_creds.grant_type);
        assert_eq!(creds.fusion_e2e, deserialized_creds.fusion_e2e);
    }

    #[test]
    fn test_fusion_credentials_deserialization() {
        let creds = FusionCredentials::new(
            Some("client_id".to_string()),
            Some("client_secret".to_string()),
            Some("username".to_string()),
            Some("password".to_string()),
            Some("resource".to_string()),
            Some("auth_url".to_string()),
            None,
            Some(HashMap::new()),
            Some("grant_type".to_string()),
            Some("fusion_e2e".to_string()),
            Some(HashMap::new()),
            Some("kid".to_string()),
            Some("private_key".to_string()),
        )
        .unwrap();

        let serialized_creds = serialize(&creds).unwrap();
        let mut deserialized_creds = FusionCredentials::default();
        deserialized_creds.__setstate__(serialized_creds).unwrap();

        assert_eq!(creds.client_id, deserialized_creds.client_id);
        assert_eq!(creds.client_secret, deserialized_creds.client_secret);
        assert_eq!(creds.username, deserialized_creds.username);
        assert_eq!(creds.password, deserialized_creds.password);
        assert_eq!(creds.resource, deserialized_creds.resource);
        assert_eq!(creds.auth_url, deserialized_creds.auth_url);
        assert_eq!(creds.grant_type, deserialized_creds.grant_type);
        assert_eq!(creds.fusion_e2e, deserialized_creds.fusion_e2e);
    }

    #[test]
    fn test_fusion_credentials_getnewargs() {
        let creds = FusionCredentials::new(
            Some("client_id".to_string()),
            Some("client_secret".to_string()),
            Some("username".to_string()),
            Some("password".to_string()),
            Some("resource".to_string()),
            Some("auth_url".to_string()),
            None,
            Some(HashMap::new()),
            Some("grant_type".to_string()),
            Some("fusion_e2e".to_string()),
            Some(HashMap::new()),
            Some("kid".to_string()),
            Some("private_key".to_string()),
        )
        .unwrap();

        let (
            client_id,
            client_secret,
            username,
            password,
            resource,
            auth_url,
            bearer_token,
            proxies,
            grant_type,
            fusion_e2e,
            headers,
        ) = creds.__getnewargs__().unwrap();

        assert_eq!(client_id, Some("client_id".to_string()));
        assert_eq!(client_secret, Some("client_secret".to_string()));
        assert_eq!(username, Some("username".to_string()));
        assert_eq!(password, Some("password".to_string()));
        assert_eq!(resource, Some("resource".to_string()));
        assert_eq!(auth_url, Some("auth_url".to_string()));
        assert!(bearer_token.is_none());
        assert!(proxies.is_some());
        assert!(headers.is_some());
        assert_eq!(grant_type, Some("grant_type".to_string()));
        assert_eq!(fusion_e2e, Some("fusion_e2e".to_string()));
        // assert_eq!(kid, Some("kid".to_string()));
        // assert_eq!(private_key, Some("private_key".to_string()));
    }

    #[test]
    fn test_fusion_credentials_from_client_id() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let creds = FusionCredentials::from_client_id(
                &py.get_type_bound::<FusionCredentials>(),
                Some("client_id".to_string()),
                Some("client_secret".to_string()),
                Some("resource".to_string()),
                Some("auth_url".to_string()),
                None,
                None,
                None,
                None,
                None,
            )
            .unwrap();

            assert_eq!(creds.client_id, Some("client_id".to_string()));
            assert_eq!(creds.client_secret, Some("client_secret".to_string()));
            assert_eq!(creds.resource, Some("resource".to_string()));
            assert_eq!(creds.auth_url, Some("auth_url".to_string()));
            assert_eq!(creds.grant_type, "client_credentials".to_string());
        });
    }

    #[test]
    fn test_fusion_credentials_from_user_id() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let creds = FusionCredentials::from_user_id(
                &py.get_type_bound::<FusionCredentials>(),
                Some("client_id".to_string()),
                Some("username".to_string()),
                Some("password".to_string()),
                Some("resource".to_string()),
                Some("auth_url".to_string()),
                None,
                None,
                None,
                None,
                None,
            )
            .unwrap();

            assert_eq!(creds.username, Some("username".to_string()));
            assert_eq!(creds.password, Some("password".to_string()));
            assert_eq!(creds.resource, Some("resource".to_string()));
            assert_eq!(creds.auth_url, Some("auth_url".to_string()));
            assert_eq!(creds.grant_type, "password".to_string());
        });
    }

    #[test]
    fn test_fusion_credentials_from_bearer_token() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let expiry_date = PyDate::new_bound(py, 2023, 11, 1).unwrap();
            let mut creds = FusionCredentials::from_bearer_token(
                &py.get_type_bound::<FusionCredentials>(),
                Some("token".to_string()),
                Some(&expiry_date),
                None,
                None,
                None,
            )
            .unwrap();

            assert_eq!(creds.resource, None);
            assert_eq!(creds.auth_url, None);
            let token = creds.bearer_token.clone().unwrap();
            assert_eq!(token.token, "token".to_string());
            assert!(token.expires_in_secs().is_some());

            let new_token = "new_token".to_string();
            let new_expiry = 100;
            creds.put_bearer_token(new_token, Some(new_expiry));
            assert_eq!(
                creds.bearer_token.clone().unwrap().token,
                "new_token".to_string()
            );
            assert!(creds.bearer_token.unwrap().expires_in_secs().is_some());
        });
    }

    #[test]
    fn test_fusion_credentials_put_bearer_token() {
        let mut creds = FusionCredentials::default();
        creds.put_bearer_token("new_token".to_string(), Some(3600));

        assert!(creds.bearer_token.is_some());
        assert_eq!(creds.bearer_token.unwrap().token, "new_token".to_string());
    }

    #[test]
    fn test_fusion_credentials_put_fusion_token() {
        let mut creds = FusionCredentials::default();
        creds.put_fusion_token("key".to_string(), "token".to_string(), Some(3600));

        assert!(creds.fusion_token.contains_key("key"));
        assert_eq!(
            creds.fusion_token.get("key").unwrap().token,
            "token".to_string()
        );
    }

    #[test]
    fn test_fusion_credentials_get_fusion_token_expires_in() {
        let mut creds = FusionCredentials::default();
        creds.put_fusion_token("key".to_string(), "token".to_string(), Some(3600));

        let expires_in = creds
            .get_fusion_token_expires_in("key".to_string())
            .unwrap();
        assert!(expires_in.is_some());
        assert!(expires_in.unwrap() <= 3600);
    }

    #[test]
    fn test_fusion_credentials_from_file() {
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

        Python::with_gil(|py| {
            let creds = FusionCredentials::from_file(
                &py.get_type_bound::<FusionCredentials>(),
                temp_file_path.clone(),
            )
            .unwrap();

            assert_eq!(creds.client_id, Some("my_client_id".to_string()));
            assert_eq!(creds.client_secret, Some("my_client_secret".to_string()));
            assert_eq!(creds.resource, Some("my_resource".to_string()));
            assert_eq!(creds.auth_url, Some("my_auth_url".to_string()));
            assert_eq!(creds.grant_type, "client_credentials".to_string());
        });
    }

    #[test]
    fn test_fusion_credentials_from_file_with_env_vars() {
        pyo3::prepare_freethreaded_python();
        // Set environment variables
        env::set_var("FUSION_CLIENT_ID", "env_client_id");
        env::set_var("FUSION_CLIENT_SECRET", "env_client_secret");

        // Create a temporary directory
        let temp_dir = TempDir::new("test_fusion_credentials_from_file_with_env_vars")
            .expect("Failed to create temp dir");

        // Path to the temporary file
        let temp_file_path = temp_dir.path().join("credentials.json");

        // JSON content to be written to the file
        let json_content = json!({
            "grant_type": "client_credentials",
            "resource": "my_resource",
            "auth_url": "my_auth_url",
        });

        // Write the JSON content to the file
        let mut file = File::create(&temp_file_path).expect("Failed to create temp file");
        file.write_all(json_content.to_string().as_bytes())
            .expect("Failed to write to temp file");

        // Test the from_file method
        Python::with_gil(|py| {
            let creds = FusionCredentials::from_file(
                &py.get_type_bound::<FusionCredentials>(),
                temp_file_path.clone(),
            )
            .expect("Failed to create FusionCredentials from file");

            assert_eq!(creds.client_id, Some("env_client_id".to_string()));
            assert_eq!(creds.client_secret, Some("env_client_secret".to_string()));
            assert_eq!(creds.resource, Some("my_resource".to_string()));
            assert_eq!(creds.auth_url, Some("my_auth_url".to_string()));
            assert_eq!(creds.grant_type, "client_credentials".to_string());
        });

        // Cleanup is handled automatically by TempDir

        // Unset environment variables
        env::remove_var("FUSION_CLIENT_ID");
        env::remove_var("FUSION_CLIENT_SECRET");
    }

    #[test]
    fn test_fusion_url_to_auth_url_valid_url() {
        let url =
            "http://example.com/catalogs/my_catalog/datasets/my_dataset/distributions".to_string();
        let result = fusion_url_to_auth_url(url).unwrap();
        assert!(result.is_some());
        let (fusion_tk_url, catalog_name, dataset_name) = result.unwrap();
        assert_eq!(
            fusion_tk_url,
            "http://example.com/catalogs/my_catalog/datasets/my_dataset/authorize/token"
        );
        assert_eq!(catalog_name, "my_catalog");
        assert_eq!(dataset_name, "my_dataset");
    }

    #[test]
    fn test_fusion_url_to_auth_url_no_distributions_segment() {
        let url = "http://example.com/catalogs/my_catalog/datasets/my_dataset".to_string();
        let result = fusion_url_to_auth_url(url).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_fusion_url_to_auth_url_missing_catalogs_segment() {
        let url = "http://example.com/datasets/my_dataset/distributions".to_string();
        let result = fusion_url_to_auth_url(url);
        assert!(result.is_err());
    }

    #[test]
    fn test_fusion_url_to_auth_url_missing_datasets_segment() {
        let url = "http://example.com/catalogs/my_catalog/distributions".to_string();
        let result = fusion_url_to_auth_url(url);
        assert!(result.is_err());
    }

    #[test]
    fn test_fusion_url_to_auth_url_invalid_url() {
        let url = "not a valid url".to_string();
        let result = fusion_url_to_auth_url(url);
        assert!(result.is_err());
    }

    #[test]
    fn test_fusion_url_to_auth_url_with_port() {
        let url = "http://example.com:8080/catalogs/my_catalog/datasets/my_dataset/distributions"
            .to_string();
        let result = fusion_url_to_auth_url(url).unwrap();
        assert!(result.is_some());
        let (fusion_tk_url, catalog_name, dataset_name) = result.unwrap();
        assert_eq!(
            fusion_tk_url,
            "http://example.com:8080/catalogs/my_catalog/datasets/my_dataset/authorize/token"
        );
        assert_eq!(catalog_name, "my_catalog");
        assert_eq!(dataset_name, "my_dataset");
    }
}
