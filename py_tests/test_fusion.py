import pandas as pd
import pytest
import requests

from fusion.fusion import Fusion


def test_rust_ok():
    from fusion import rust_ok

    assert rust_ok()


def test__get_canonical_root_url():
    from fusion.utils import _get_canonical_root_url

    some_url = "https://fusion-api.jpmorgan.com/fusion/v1/a_given_resource"
    root_url = "https://fusion-api.jpmorgan.com"
    assert root_url == _get_canonical_root_url(some_url)


def test_FusionCredentials(example_creds_dict):
    from fusion.authentication import FusionCredentials

    FusionCredentials.from_dict(example_creds_dict)


def test_FusionCredentials_no_pxy(example_creds_dict_no_pxy):
    from fusion.authentication import FusionCredentials

    FusionCredentials.from_dict(example_creds_dict_no_pxy)


def test_FusionCredentials_empty_pxy(example_creds_dict_empty_pxy):
    from fusion.fusion import FusionCredentials

    FusionCredentials.from_dict(example_creds_dict_empty_pxy)


def test_FusionCredentials_from_empty(example_client_id, example_client_secret):
    from fusion.authentication import FusionCredentials

    creds = FusionCredentials.generate_credentials_file(
        client_id=example_client_id, client_secret=example_client_secret, proxies={}
    )

    assert creds.proxies == {}


def test_FusionCredentials_from_str(example_client_id, example_client_secret, example_http_proxy):
    from fusion.authentication import FusionCredentials

    creds = FusionCredentials.generate_credentials_file(
        client_id=example_client_id,
        client_secret=example_client_secret,
        proxies=example_http_proxy,
    )

    assert creds.proxies["http"] == example_http_proxy


def test_FusionCredentials_from_http_dict(
    example_client_id,
    example_client_secret,
    example_proxy_http_dict,
    example_http_proxy,
):
    from fusion.authentication import FusionCredentials

    creds = FusionCredentials.generate_credentials_file(
        client_id=example_client_id,
        client_secret=example_client_secret,
        proxies=example_proxy_http_dict,
    )

    assert creds.proxies["http"] == example_http_proxy


def test_FusionCredentials_from_https_dict(
    example_client_id,
    example_client_secret,
    example_proxy_https_dict,
    example_https_proxy,
):
    from fusion.authentication import FusionCredentials

    creds = FusionCredentials.generate_credentials_file(
        client_id=example_client_id,
        client_secret=example_client_secret,
        proxies=example_proxy_https_dict,
    )

    assert creds.proxies["https"] == example_https_proxy


def test_FusionCredentials_from_both_dict(
    example_client_id,
    example_client_secret,
    example_proxy_both_dict,
    example_https_proxy,
    example_http_proxy,
):
    from fusion.authentication import FusionCredentials

    creds = FusionCredentials.generate_credentials_file(
        client_id=example_client_id,
        client_secret=example_client_secret,
        proxies=example_proxy_both_dict,
    )

    assert creds.proxies["https"] == example_https_proxy
    assert creds.proxies["http"] == example_http_proxy


def test_FusionCredentials_from_both_alt_dict(
    example_client_id,
    example_client_secret,
    example_proxy_both_alt_dict,
    example_https_proxy,
    example_http_proxy,
):
    from fusion.authentication import FusionCredentials

    creds = FusionCredentials.generate_credentials_file(
        client_id=example_client_id,
        client_secret=example_client_secret,
        proxies=example_proxy_both_alt_dict,
    )

    assert creds.proxies["https"] == example_https_proxy
    assert creds.proxies["http"] == example_http_proxy


def test_date_parsing():
    import datetime

    from fusion.utils import _normalise_dt_param

    assert _normalise_dt_param(20201212) == "2020-12-12"
    assert _normalise_dt_param("20201212") == "2020-12-12"
    assert _normalise_dt_param("2020-12-12") == "2020-12-12"
    assert _normalise_dt_param(datetime.date(2020, 12, 12)) == "2020-12-12"
    assert _normalise_dt_param(datetime.datetime(2020, 12, 12, 23, 55, 59, 342380)) == "2020-12-12"


@pytest.mark.parametrize("ref_int", [-1, 0, 1, 2])
@pytest.mark.parametrize("pluraliser", [None, "s", "es"])
def test_res_plural(ref_int, pluraliser):
    from fusion.authentication import _res_plural

    res = _res_plural(ref_int, pluraliser)
    if abs(ref_int) == 1:
        assert res == ""
    else:
        assert res == pluraliser


def test_is_json_positive(good_json):
    from fusion.authentication import _is_json

    _is_json(good_json)


def test_is_json_negative1(bad_json1):
    from fusion.authentication import _is_json

    _is_json(bad_json1)


def test_is_json_negative2(bad_json2):
    from fusion.authentication import _is_json

    _is_json(bad_json2)


def test_is_json_negative3(bad_json3):
    from fusion.authentication import _is_json

    _is_json(bad_json3)


def test_fusion_class():
    from fusion.fusion import Fusion

    fusion_obj = Fusion()
    assert fusion_obj
    assert repr(fusion_obj)
    assert fusion_obj.default_catalog == "common"
    fusion_obj.default_catalog = "other"
    assert fusion_obj.default_catalog == "other"


def test_get_fusion_filesystem():
    fusion_obj = Fusion()
    filesystem = fusion_obj.get_fusion_filesystem()
    assert filesystem is not None


def test__call_for_dataframe_success(requests_mock):
    # Mock the response from the API endpoint
    url = "https://fusion-api.jpmorgan.com/fusion/v1/a_given_resource"
    expected_data = {"resources": [{"id": 1, "name": "Resource 1"}, {"id": 2, "name": "Resource 2"}]}
    requests_mock.get(url, json=expected_data)

    # Create a mock session
    session = requests.Session()

    # Call the _call_for_dataframe function
    df = Fusion._call_for_dataframe(url, session)

    # Check if the dataframe is created correctly
    expected_df = pd.DataFrame(expected_data["resources"])
    pd.testing.assert_frame_equal(df, expected_df)


def test__call_for_dataframe_error(requests_mock):
    # Mock the response from the API endpoint with an error status code
    url = "https://fusion-api.jpmorgan.com/fusion/v1/a_given_resource"
    requests_mock.get(url, status_code=500)

    # Create a mock session
    session = requests.Session()

    # Call the _call_for_dataframe function and expect an exception to be raised
    with pytest.raises(requests.exceptions.HTTPError):
        Fusion._call_for_dataframe(url, session)


def test__call_for_bytes_object_success(requests_mock):
    # Mock the response from the API endpoint
    url = "https://fusion-api.jpmorgan.com/fusion/v1/a_given_resource"
    expected_data = b"some binary data"
    requests_mock.get(url, content=expected_data)

    # Create a mock session
    session = requests.Session()

    # Call the _call_for_bytes_object function
    data = Fusion._call_for_bytes_object(url, session)

    # Check if the data is returned correctly
    assert data.getbuffer() == expected_data


def test__call_for_bytes_object_fail(requests_mock):
    # Mock the response from the API endpoint with an error status code
    url = "https://fusion-api.jpmorgan.com/fusion/v1/a_given_resource"
    requests_mock.get(url, status_code=500)

    # Create a mock session
    session = requests.Session()

    # Call the _call_for_dataframe function and expect an exception to be raised
    with pytest.raises(requests.exceptions.HTTPError):
        Fusion._call_for_bytes_object(url, session)


def test_list_catalogs_success(requests_mock):
    # Mock the response from the API endpoint
    url = "https://fusion-api.jpmorgan.com/fusion/v1/catalogs/"
    expected_data = {"resources": [{"id": 1, "name": "Catalog 1"}, {"id": 2, "name": "Catalog 2"}]}
    requests_mock.get(url, json=expected_data)

    # Create a mock Fusion object
    fusion_obj = Fusion()

    # Call the list_catalogs method
    df = fusion_obj.list_catalogs()

    # Check if the dataframe is created correctly
    expected_df = pd.DataFrame(expected_data["resources"])
    pd.testing.assert_frame_equal(df, expected_df)
