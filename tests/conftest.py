import pytest


@pytest.fixture
def example_client_id():
    return "vf3tdjK0jdp7MdY3"


@pytest.fixture
def example_client_secret():
    return "vswag2iet7Merdkdwe64YcI9gxbemjMsh5jgimrwpcghsqc2mnj4w4qQffrfhtKz0ba3u48tqJrbp1y"


@pytest.fixture
def example_http_proxy():
    return "http://myproxy.com:8080"


@pytest.fixture
def example_https_proxy():
    return "https://myproxy.com:8080"


@pytest.fixture
def example_proxy_http_dict(example_http_proxy):
    return {'http': example_http_proxy}


@pytest.fixture
def example_proxy_https_dict(example_https_proxy):
    return {'https': example_https_proxy}


@pytest.fixture
def example_proxy_both_dict(example_http_proxy, example_https_proxy):
    return {'http': example_http_proxy, 'https': example_https_proxy}


@pytest.fixture
def example_proxy_both_alt_dict(example_http_proxy, example_https_proxy):
    return {'http_proxy': example_http_proxy, 'https_proxy': example_https_proxy}


@pytest.fixture
def example_proxy_str1(example_http_proxy):
    return example_http_proxy


@pytest.fixture
def example_proxy_str_bad():
    return 'not_a_proxy'


@pytest.fixture
def example_creds_dict(example_client_id, example_client_secret):
    # Mocked creds info
    return {
        "client_id": example_client_id,
        "client_secret": example_client_secret,
        "resource": "JPMC:URI:RS-97834-Fusion-PROD",
        "auth_url": "https://authe.mysite.com/as/token.oauth2",
        "proxies": {
            "http": "http://myproxy.com:8080",
            "https": "http://myproxy.com:8081",
        },
    }


@pytest.fixture
def example_creds_dict_no_pxy(example_creds_dict):
    example_creds_dict.pop('proxies')
    return example_creds_dict


@pytest.fixture
def example_creds_dict_empty_pxy(example_creds_dict):
    example_creds_dict['proxies'].pop('http')
    example_creds_dict['proxies'].pop('https')
    return example_creds_dict


@pytest.fixture
def good_json():
    return """{
        "client_id": "vf3tdjK0jdp7MdY3",
        "client_secret": "vswag2iet7Merdkdwe64YcI9gxbemjMsh5jgimrwpcghsqc2mnj4w4qQffrfhtKz0ba3u48tqJrbp1y",
        "resource": "JPMC:URI:RS-97834-Fusion-PROD",
        "auth_url": "https://authe.mysite.com/as/token.oauth2",
        "proxies": {
            "http": "http://myproxy.com:8080",
            "https": "http://myproxy.com:8081",
        }"""


@pytest.fixture
def bad_json1():
    return """{
        "client_id" "vf3tdjK0jdp7MdY3",
        "client_secret": "vswag2iet7Merdkdwe64YcI9gxbemjMsh5jgimrwpcghsqc2mnj4w4qQffrfhtKz0ba3u48tqJrbp1y",
        "resource": "JPMC:URI:RS-97834-Fusion-PROD",
        "auth_url": "https://authe.mysite.com/as/token.oauth2",
        "proxies": {
            "http": "http://myproxy.com:8080",
            "https": "http://myproxy.com:8081",
        }"""


@pytest.fixture
def bad_json2():
    return """{
        "client_id", vf3tdjK0jdp7MdY3,
        "client_secret": "vswag2iet7Merdkdwe64YcI9gxbemjMsh5jgimrwpcghsqc2mnj4w4qQffrfhtKz0ba3u48tqJrbp1y",
        "resource": "JPMC:URI:RS-97834-Fusion-PROD",
        "auth_url": "https://authe.mysite.com/as/token.oauth2",
        "proxies": {
            "http": "http://myproxy.com:8080",
            "https": "http://myproxy.com:8081",
        }"""


@pytest.fixture
def bad_json3():
    return """{
        "client_id", "vf3tdjK0jdp7MdY3",
        "client_secret": "vswag2iet7Merdkdwe64YcI9gxbemjMsh5jgimrwpcghsqc2mnj4w4qQffrfhtKz0ba3u48tqJrbp1y",
        "resource": "JPMC:URI:RS-97834-Fusion-PROD",
        "auth_url": "https://authe.mysite.com/as/token.oauth2",
        "proxies":
            "http": "http://myproxy.com:8080",
            "https": "http://myproxy.com:8081",
        }"""
