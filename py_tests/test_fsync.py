from fusion.fs_sync import _url_to_path


def test__url_to_path() -> None:
    # The function this is testing looks a bit broken. The test is here just to make sure we don't break it further.

    catalog = "my_catalog"
    dataset = "my_dataset"
    dt_str = "20200101"
    file_format = "csv"

    url = f"{catalog}/datasets/{dataset}/dataseries/{dt_str}/distributions/{file_format}"
    path = _url_to_path(url)
    exp_res = f"{catalog}/my_dataset/{dt_str}//{dataset}__{catalog}__{dt_str}.csv"
    assert path == exp_res
