use chrono::{NaiveDate, Utc};
use pyo3::exceptions::{PyFileNotFoundError, PyValueError};
use pyo3::import_exception;
use pyo3::prelude::*;
use pyo3::types::{PyDate, PyDateAccess, PyTuple, PyType};
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::HashMap;
use std::env;
use std::fmt::Display;
use std::fs::File;
use std::io::prelude::*;
use std::path::{Path, PathBuf};
use std::str::FromStr;

//use std::time::{SystemTime, UNIX_EPOCH};

import_exception!(fusion.exceptions, APIResponseError);
import_exception!(fusion.exceptions, APIRequestError);
import_exception!(fusion.exceptions, APIConnectError);
import_exception!(fusion.exceptions, UnrecognizedFormatError);
import_exception!(fusion.exceptions, CredentialError);

fn default_grant_type() -> String {
    "client_credentials".to_string()
}

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

fn typed_proxies(proxies: Option<HashMap<String, String>>) -> PyResult<HashMap<ProxyType, String>> {
    match proxies {
        Some(proxies) => {
            let mut mapped_proxies = HashMap::new();
            for (key, value) in proxies {
                let key = key
                    .parse::<ProxyType>()
                    .map_err(|_| PyValueError::new_err("Invalid proxy type"))?;
                mapped_proxies.insert(key, value);
            }
            Ok(mapped_proxies)
        }
        None => Ok(HashMap::new()),
    }
}

fn untyped_proxies(proxies: HashMap<ProxyType, String>) -> HashMap<String, String> {
    let mut untyped_proxies = HashMap::new();
    for (key, value) in proxies {
        untyped_proxies.insert(key.to_string(), value);
    }
    untyped_proxies
}

fn deserialize_with_env<'de, D>(deserializer: D, env_var: &str) -> Result<Option<String>, D::Error>
where
    D: Deserializer<'de>,
{
    let opt = Option::<String>::deserialize(deserializer)?;
    println!("Deser Option: {:?}. Env {:?}", opt, env::var(env_var));
    let res = match opt {
        Some(value) => Some(value),
        None => match env::var(env_var) {
            Ok(value) => Some(value),
            Err(_) => None,
        },
    };
    Ok(res)
}


fn deserialize_client_id<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: Deserializer<'de>,
{
    deserialize_with_env(deserializer, "FUSION_CLIENT_ID")
}

fn deserialize_client_secret<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: Deserializer<'de>,
{
    deserialize_with_env(deserializer, "FUSION_CLIENT_SECRET")
}

#[allow(dead_code)]
fn deserialize_fusion_e2e<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: Deserializer<'de>,
{
    deserialize_with_env(deserializer, "FUSION_E2E")
}

fn find_cfg_file(file_path: &Path) -> PyResult<PathBuf> {
    let current_path = file_path.to_path_buf();

    if current_path.is_file() {
        return Ok(current_path);
    }

    let cwd = env::current_dir()?;
    println!("The current directory is {}", cwd.display());

    let cfg_file_name = "client_credentials.json";
    let mut start_dir = match file_path.parent() {
        Some(parent) => match parent.exists() {
            true => parent.to_path_buf(),
            false => cwd,
        },
        None => cwd,
    };
    let start_dir_init = start_dir.clone();

    let mut count = 0;
    loop {
        count += 1;
        let full_path = start_dir.join(cfg_file_name);
        println!("{}: {}", count, full_path.display());
        if full_path.is_file() {
            println!("Found file: {}", full_path.display());
            return Ok(full_path);
        }

        // Move to the parent directory
        if let Some(parent) = start_dir.parent() {
            start_dir = parent.to_path_buf();
        } else {
            // Reached the root directory
            println!("Root directory reached");
            let error_message = format!(
                "File {} not found in {} or any of its parents",
                cfg_file_name,
                start_dir_init.display()
            );
            return Err(PyFileNotFoundError::new_err(error_message.clone()));
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FusionCredsPersistent {
    #[serde(deserialize_with = "deserialize_client_id")]
    client_id: Option<String>,
    #[serde(deserialize_with = "deserialize_client_secret")]
    client_secret: Option<String>,
    username: Option<String>,
    password: Option<String>,
    resource: Option<String>,
    auth_url: Option<String>,
    #[serde(default)]
    proxies: HashMap<ProxyType, String>,
    #[serde(default = "default_grant_type")]
    grant_type: String,
    //#[serde(deserialize_with = "deserialize_fusion_e2e")]
    fusion_e2e: Option<String>,
}

#[pyclass]
#[derive(Debug, Clone)]
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
    pub fn from_token(token: String, expires_in_secs: Option<i64>) -> Self {
        let expiry = expires_in_secs.map(|secs| {
            let current_time = Utc::now().timestamp();
            current_time + secs
        });
        AuthToken { token, expiry }
    }
}

#[pyclass]
#[derive(Debug, Clone)]
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

    #[pyo3(get)]
    proxies: HashMap<String, String>,

    #[pyo3(get, set)]
    bearer_token: Option<AuthToken>,

    #[pyo3(get)]
    fusion_token: HashMap<String, AuthToken>,
}

impl Default for FusionCredentials {
    fn default() -> Self {
        FusionCredentials {
            client_id: None,
            client_secret: None,
            username: None,
            password: None,
            resource: None,
            auth_url: None,
            bearer_token: None,
            fusion_token: HashMap::new(),
            proxies: HashMap::new(),
            grant_type: "client_credentials".to_string(),
            fusion_e2e: None,
        }
    }
}

#[pymethods]
impl FusionCredentials {
    #[classmethod]
    fn from_client_id(
        _cls: &Bound<'_, PyType>,
        client_id: Option<String>,
        client_secret: Option<String>,
        resource: Option<String>,
        auth_url: Option<String>,
        proxies: Option<HashMap<String, String>>,
        fusion_e2e: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            client_id,
            client_secret,
            resource,
            auth_url,
            proxies: proxies.unwrap_or_default(),
            grant_type: "client_credentials".to_string(),
            fusion_e2e,
            fusion_token: HashMap::new(),
            bearer_token: None,
            username: None,
            password: None,
        })
    }

    #[classmethod]
    fn from_user_id(
        _cls: &Bound<'_, PyType>,
        username: Option<String>,
        password: Option<String>,
        resource: Option<String>,
        auth_url: Option<String>,
        proxies: Option<HashMap<String, String>>,
        fusion_e2e: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            username,
            password,
            resource,
            auth_url,
            proxies: proxies.unwrap_or_default(),
            grant_type: "password".to_string(),
            fusion_e2e,
            fusion_token: HashMap::new(),
            bearer_token: None,
            client_id: None,
            client_secret: None,
        })
    }

    #[classmethod]
    fn from_bearer_token(
        _cls: &Bound<'_, PyType>,
        resource: Option<String>,
        auth_url: Option<String>,
        bearer_token: Option<String>,
        bearer_token_expiry: Option<&Bound<PyDate>>,
        proxies: Option<HashMap<String, String>>,
        fusion_e2e: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            resource,
            auth_url,
            proxies: proxies.unwrap_or_default(),
            grant_type: "password".to_string(),
            fusion_e2e,
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
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[new]
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
    ) -> PyResult<Self> {
        Ok(FusionCredentials {
            client_id,
            client_secret,
            username,
            password,
            resource,
            auth_url,
            bearer_token,
            fusion_token: HashMap::new(),
            proxies: proxies.unwrap_or_default(),
            grant_type: grant_type.unwrap_or_else(|| "client_credentials".to_string()),
            fusion_e2e,
        })
    }

    fn put_bearer_token(&mut self, bearer_token: String, expires_in_secs: Option<i64>) {
        self.bearer_token = Some(AuthToken::from_token(bearer_token, expires_in_secs));
    }

    fn put_fusion_token(&mut self, token_key: String, token: String, expires_in_secs: Option<i64>) {
        self.fusion_token
            .insert(token_key, AuthToken::from_token(token, expires_in_secs));
    }

    fn get_bearer_token_header<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let token_tup =
            self.bearer_token
                .as_ref()
                .map_or(Vec::<(String, String)>::new(), |token| {
                    vec![(
                        "Authorization".to_owned(),
                        format!("Bearer {}", token.token),
                    )]
                });
        Ok(PyTuple::new_bound(py, &token_tup))
    }

    fn get_fusion_token_header<'py>(
        &self,
        py: Python<'py>,
        token_key: String,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let token_tup = self.fusion_token.get(&token_key).as_ref().map_or(
            Vec::<(String, String)>::new(),
            |token| {
                vec![(
                    "Fusion-Authorization".to_owned(),
                    format!("Bearer {}", token.token),
                )]
            },
        );
        Ok(PyTuple::new_bound(py, &token_tup))
    }

    fn get_fusion_token_expires_in<'py>(
        &self,
        py: Python<'py>,
        token_key: String,
    ) -> PyResult<Option<i64>> {
        Ok(self.fusion_token.get(&token_key).and_then(|token| token.expires_in_secs()))
    }

    #[classmethod]
    fn from_file(cls: &Bound<'_, PyType>, file_path: PathBuf) -> PyResult<FusionCredentials> {
        let found_path = find_cfg_file(&file_path)
            .map_err(|_| PyFileNotFoundError::new_err("File not found"))?;

        let mut file =
            File::open(found_path).map_err(|_| PyFileNotFoundError::new_err("File not found"))?;

        let mut contents = String::new();
        file.read_to_string(&mut contents)?;

        let credentials: FusionCredsPersistent =
            serde_json::from_str(&contents).map_err(|err| {
                CredentialError::new_err(format!("Invalid JSON: {}\nContents:\n{}", err, &contents))
            })?;

        let full_creds = match credentials.grant_type.as_str() {
            "client_credentials" => FusionCredentials::from_client_id(
                cls,
                credentials.client_id,
                credentials.client_secret,
                credentials.resource,
                credentials.auth_url,
                Some(untyped_proxies(credentials.proxies)),
                credentials.fusion_e2e,
            )?,
            "bearer" => FusionCredentials::from_bearer_token(
                cls,
                credentials.resource,
                credentials.auth_url,
                None,
                None,
                Some(untyped_proxies(credentials.proxies)),
                credentials.fusion_e2e,
            )?,
            "password" => FusionCredentials::from_user_id(
                cls,
                credentials.username,
                credentials.password,
                credentials.resource,
                credentials.auth_url,
                Some(untyped_proxies(credentials.proxies)),
                credentials.fusion_e2e,
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
