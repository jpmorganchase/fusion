import pytest


def test__get_canonical_root_url():
    from fusion.fusion import _get_canonical_root_url

    some_url = "https://fusion-api.jpmorgan.com/fusion/v1/a_given_resource"
    root_url = "https://fusion-api.jpmorgan.com"
    assert root_url == _get_canonical_root_url(some_url)


def test_FusionCredentials(example_creds_dict):
    from fusion.fusion import FusionCredentials

    FusionCredentials.from_dict(example_creds_dict)


def test_FusionCredentials_no_pxy(example_creds_dict_no_pxy):
    from fusion.fusion import FusionCredentials

    FusionCredentials.from_dict(example_creds_dict_no_pxy)


def test_FusionCredentials_empty_pxy(example_creds_dict_empty_pxy):
    from fusion.fusion import FusionCredentials

    FusionCredentials.from_dict(example_creds_dict_empty_pxy)


@pytest.mark.parametrize('ref_int', [-1, 0, 1, 2])
@pytest.mark.parametrize('pluraliser', [None, 's', 'es'])
def test_res_plural(ref_int, pluraliser):
    from fusion.fusion import _res_plural

    res = _res_plural(ref_int, pluraliser)
    if abs(ref_int) == 1:
        assert res == ''
    else:
        assert res == pluraliser


def test_is_json_positive(good_json):
    from fusion.fusion import _is_json

    _is_json(good_json)


def test_is_json_negative1(bad_json1):
    from fusion.fusion import _is_json

    _is_json(bad_json1)


def test_is_json_negative2(bad_json2):
    from fusion.fusion import _is_json

    _is_json(bad_json2)


def test_is_json_negative3(bad_json3):
    from fusion.fusion import _is_json

    _is_json(bad_json3)
