"""Test cases for the product module."""

import json
from collections.abc import Generator
from typing import Any

import pandas as pd
import pytest
import requests
import requests_mock

from fusion import Fusion
from fusion.product import Product


def test_product_class() -> None:
    """Test the Product class."""
    test_product = Product(title="Test Product", identifier="Test Product", release_date="May 5, 2020")
    assert test_product.title == "Test Product"
    assert test_product.identifier == "TEST_PRODUCT"
    assert test_product.category is None
    assert test_product.shortAbstract == "Test Product"
    assert test_product.description == "Test Product"
    assert test_product.isActive is True
    assert test_product.isRestricted is None
    assert test_product.maintainer is None
    assert test_product.region == ["Global"]
    assert test_product.publisher == "J.P. Morgan"
    assert test_product.subCategory is None
    assert test_product.tag is None
    assert test_product.deliveryChannel == ["API"]
    assert test_product.theme is None
    assert test_product.releaseDate == "2020-05-05"
    assert test_product.language == "English"
    assert test_product.status == "Available"
    assert test_product.image == ""
    assert test_product.logo == ""
    assert test_product.dataset is None


def test_product_client_value_error() -> None:
    """Test product client value error."""
    my_product = Product(identifier="Test Product")
    with pytest.raises(ValueError, match="A Fusion client object is required.") as error_info:
        my_product._use_client(client=None)
    assert str(error_info.value) == "A Fusion client object is required."


def test_product_class_from_series() -> None:
    """Test the Product class."""
    test_product = Product._from_series(
        pd.Series(
            {
                "title": "Test Product",
                "identifier": "Test Product",
                "releaseDate": "May 5, 2020",
            }
        )
    )
    assert test_product.title == "Test Product"
    assert test_product.identifier == "TEST_PRODUCT"
    assert test_product.category is None
    assert test_product.shortAbstract == "Test Product"
    assert test_product.description == "Test Product"
    assert test_product.isActive is True
    assert test_product.isRestricted is None
    assert test_product.maintainer is None
    assert test_product.region == ["Global"]
    assert test_product.publisher == "J.P. Morgan"
    assert test_product.subCategory is None
    assert test_product.tag is None
    assert test_product.deliveryChannel == ["API"]
    assert test_product.theme is None
    assert test_product.releaseDate == "2020-05-05"
    assert test_product.language == "English"
    assert test_product.status == "Available"
    assert test_product.image == ""
    assert test_product.logo == ""
    assert test_product.dataset is None


def test_product_class_from_dict() -> None:
    """Test the Product class."""
    test_product = Product._from_dict(
        {
            "title": "Test Product",
            "identifier": "TEST_PRODUCT",
            "releaseDate": "May 5, 2020",
        }
    )
    assert test_product.title == "Test Product"
    assert test_product.identifier == "TEST_PRODUCT"
    assert test_product.category is None
    assert test_product.shortAbstract == "Test Product"
    assert test_product.description == "Test Product"
    assert test_product.isActive is True
    assert test_product.isRestricted is None
    assert test_product.maintainer is None
    assert test_product.region == ["Global"]
    assert test_product.publisher == "J.P. Morgan"
    assert test_product.subCategory is None
    assert test_product.tag is None
    assert test_product.deliveryChannel == ["API"]
    assert test_product.theme is None
    assert test_product.releaseDate == "2020-05-05"
    assert test_product.language == "English"
    assert test_product.status == "Available"
    assert test_product.image == ""
    assert test_product.logo == ""
    assert test_product.dataset is None


def test_product_class_from_csv(mock_product_pd_read_csv: Generator[pd.DataFrame, Any, None]) -> None:  # noqa: ARG001
    """Test the Product class."""
    test_product = Product._from_csv("products.csv")
    assert test_product.title == "Test Product"
    assert test_product.identifier == "TEST_PRODUCT"
    assert test_product.category is None
    assert test_product.shortAbstract == "Test Product"
    assert test_product.description == "Test Product"
    assert test_product.isActive is True
    assert test_product.isRestricted is None
    assert test_product.maintainer is None
    assert test_product.region == ["Global"]
    assert test_product.publisher == "J.P. Morgan"
    assert test_product.subCategory is None
    assert test_product.tag is None
    assert test_product.deliveryChannel == ["API"]
    assert test_product.theme is None
    assert test_product.releaseDate is None
    assert test_product.language == "English"
    assert test_product.status == "Available"
    assert test_product.image == ""
    assert test_product.logo == ""
    assert test_product.dataset is None


def test_product_from_catalog(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    """Test list Product from_catalog method."""
    catalog = "my_catalog"
    url = f"{fusion_obj.root_url}catalogs/{catalog}/products"

    expected_data = {
        "resources": [
            {
                "catalog": {
                    "@id": "my_catalog/",
                    "description": "my catalog",
                    "title": "my catalog",
                    "identifier": "my_catalog",
                },
                "title": "Test Product",
                "identifier": "TEST_PRODUCT",
                "category": ["category"],
                "shortAbstract": "short abstract",
                "description": "description",
                "isActive": True,
                "isRestricted": False,
                "maintainer": "maintainer",
                "region": ["region"],
                "publisher": "publisher",
                "subCategory": ["subCategory"],
                "tag": ["tag1", "tag2"],
                "deliveryChannel": ["API"],
                "theme": "theme",
                "releaseDate": "2020-05-05",
                "language": "English",
                "status": "Available",
                "datasetCount": 1,
                "@id": "TEST_PRODUCT/",
            },
        ],
    }
    requests_mock.get(url, json=expected_data)

    my_product = Product(identifier="TEST_PRODUCT").from_catalog(client=fusion_obj, catalog=catalog)
    assert isinstance(my_product, Product)
    assert my_product.title == "Test Product"
    assert my_product.identifier == "TEST_PRODUCT"
    assert my_product.category == ["category"]
    assert my_product.shortAbstract == "short abstract"
    assert my_product.description == "description"
    assert my_product.isActive is True
    assert my_product.isRestricted is False
    assert my_product.maintainer == ["maintainer"]
    assert my_product.region == ["region"]
    assert my_product.publisher == "publisher"
    assert my_product.subCategory == ["subCategory"]
    assert my_product.tag == ["tag1", "tag2"]
    assert my_product.deliveryChannel == ["API"]
    assert my_product.theme == "theme"
    assert my_product.releaseDate == "2020-05-05"
    assert my_product.language == "English"
    assert my_product.status == "Available"
    assert my_product.image == ""
    assert my_product.logo == ""
    assert isinstance(my_product._client, Fusion)


def test_product_class_from_object_product() -> None:
    """Test the Product class."""
    test_product_input = Product(identifier="TEST_PRODUCT", title="Test Product", release_date="May 5, 2020")

    test_product = Product(identifier="TEST_PRODUCT").from_object(test_product_input)
    assert test_product.title == "Test Product"
    assert test_product.identifier == "TEST_PRODUCT"
    assert test_product.category is None
    assert test_product.shortAbstract == "Test Product"
    assert test_product.description == "Test Product"
    assert test_product.isActive is True
    assert test_product.isRestricted is None
    assert test_product.maintainer is None
    assert test_product.region == ["Global"]
    assert test_product.publisher == "J.P. Morgan"
    assert test_product.subCategory is None
    assert test_product.tag is None
    assert test_product.deliveryChannel == ["API"]
    assert test_product.theme is None
    assert test_product.releaseDate == "2020-05-05"
    assert test_product.language == "English"
    assert test_product.status == "Available"
    assert test_product.image == ""
    assert test_product.logo == ""
    assert test_product.dataset is None


def test_product_class_from_object_dict() -> None:
    """Test the Product class."""
    test_product = Product(identifier="TEST_PRODUCT").from_object(
        {
            "title": "Test Product",
            "identifier": "TEST_PRODUCT",
            "releaseDate": "May 5, 2020",
        }
    )
    assert test_product.title == "Test Product"
    assert test_product.identifier == "TEST_PRODUCT"
    assert test_product.category is None
    assert test_product.shortAbstract == "Test Product"
    assert test_product.description == "Test Product"
    assert test_product.isActive is True
    assert test_product.isRestricted is None
    assert test_product.maintainer is None
    assert test_product.region == ["Global"]
    assert test_product.publisher == "J.P. Morgan"
    assert test_product.subCategory is None
    assert test_product.tag is None
    assert test_product.deliveryChannel == ["API"]
    assert test_product.theme is None
    assert test_product.releaseDate == "2020-05-05"
    assert test_product.language == "English"
    assert test_product.status == "Available"
    assert test_product.image == ""
    assert test_product.logo == ""
    assert test_product.dataset is None


def test_product_class_from_object_series() -> None:
    """Test the Product class."""
    test_product = Product(identifier="TEST_PRODUCT").from_object(
        pd.Series(
            {
                "title": "Test Product",
                "identifier": "Test Product",
                "releaseDate": "May 5, 2020",
            }
        )
    )
    assert test_product.title == "Test Product"
    assert test_product.identifier == "TEST_PRODUCT"
    assert test_product.category is None
    assert test_product.shortAbstract == "Test Product"
    assert test_product.description == "Test Product"
    assert test_product.isActive is True
    assert test_product.isRestricted is None
    assert test_product.maintainer is None
    assert test_product.region == ["Global"]
    assert test_product.publisher == "J.P. Morgan"
    assert test_product.subCategory is None
    assert test_product.tag is None
    assert test_product.deliveryChannel == ["API"]
    assert test_product.theme is None
    assert test_product.releaseDate == "2020-05-05"
    assert test_product.language == "English"
    assert test_product.status == "Available"
    assert test_product.image == ""
    assert test_product.logo == ""
    assert test_product.dataset is None


def test_product_class_from_object_csv(mock_product_pd_read_csv: Generator[pd.DataFrame, Any, None]) -> None:  # noqa: ARG001
    """Test the Product class."""
    test_product = Product(identifier="TEST_PRODUCT").from_object("products.csv")
    assert test_product.title == "Test Product"
    assert test_product.identifier == "TEST_PRODUCT"
    assert test_product.category is None
    assert test_product.shortAbstract == "Test Product"
    assert test_product.description == "Test Product"
    assert test_product.isActive is True
    assert test_product.isRestricted is None
    assert test_product.maintainer is None
    assert test_product.region == ["Global"]
    assert test_product.publisher == "J.P. Morgan"
    assert test_product.subCategory is None
    assert test_product.tag is None
    assert test_product.deliveryChannel == ["API"]
    assert test_product.theme is None
    assert test_product.releaseDate is None
    assert test_product.language == "English"
    assert test_product.status == "Available"
    assert test_product.image == ""
    assert test_product.logo == ""
    assert test_product.dataset is None


def test_product_class_from_object_json() -> None:
    """Test the Product class."""
    product_json = json.dumps(
        {
            "title": "Test Product",
            "identifier": "TEST_PRODUCT",
            "releaseDate": "May 5, 2020",
        }
    )
    test_product = Product(identifier="TEST_PRODUCT").from_object(product_json)

    assert test_product.title == "Test Product"
    assert test_product.identifier == "TEST_PRODUCT"
    assert test_product.category is None
    assert test_product.shortAbstract == "Test Product"
    assert test_product.description == "Test Product"
    assert test_product.isActive is True
    assert test_product.isRestricted is None
    assert test_product.maintainer is None
    assert test_product.region == ["Global"]
    assert test_product.publisher == "J.P. Morgan"
    assert test_product.subCategory is None
    assert test_product.tag is None
    assert test_product.deliveryChannel == ["API"]
    assert test_product.theme is None
    assert test_product.releaseDate == "2020-05-05"
    assert test_product.language == "English"
    assert test_product.status == "Available"
    assert test_product.image == ""
    assert test_product.logo == ""
    assert test_product.dataset is None


def test_product_class_type_error() -> None:
    """Test the Product class."""
    unsupported_obj = 123
    with pytest.raises(TypeError) as error_info:
        Product(identifier="TEST_PRODUCT").from_object(unsupported_obj)  # type: ignore
    assert str(error_info.value) == f"Could not resolve the object provided: {unsupported_obj}"


def test_create_product(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    """Test create Product method."""
    catalog = "my_catalog"
    url = f"{fusion_obj.root_url}catalogs/{catalog}/products/TEST_PRODUCT"
    expected_data = {
        "title": "Test Product",
        "identifier": "TEST_PRODUCT",
        "category": ["category"],
        "shortAbstract": "short abstract",
        "description": "description",
        "isActive": True,
        "isRestricted": False,
        "maintainer": ["maintainer"],
        "region": ["region"],
        "publisher": "publisher",
        "subCategory": ["subCategory"],
        "tag": ["tag1", "tag2"],
        "deliveryChannel": ["API"],
        "theme": "theme",
        "releaseDate": "2020-05-05",
        "language": "English",
        "status": "Available",
        "image": "",
        "logo": "",
    }
    requests_mock.post(url, json=expected_data)

    my_product = Product(
        title="Test Product",
        identifier="TEST_PRODUCT",
        category=["category"],
        short_abstract="short abstract",
        description="description",
        is_active=True,
        is_restricted=False,
        maintainer=["maintainer"],
        region=["region"],
        publisher="publisher",
        sub_category=["subCategory"],
        tag=["tag1", "tag2"],
        delivery_channel=["API"],
        theme="theme",
        release_date="2020-05-05",
        language="English",
        status="Available",
        image="",
        logo="",
    )
    status_code = 200
    resp = my_product.create(catalog=catalog, client=fusion_obj, return_resp_obj=True)
    assert isinstance(resp, requests.Response)
    assert resp.status_code == status_code


def test_update_product(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    """Test update Product method."""
    catalog = "my_catalog"
    url = f"{fusion_obj.root_url}catalogs/{catalog}/products/TEST_PRODUCT"
    expected_data = {
        "title": "Test Product",
        "identifier": "TEST_PRODUCT",
        "category": ["category"],
        "shortAbstract": "short abstract",
        "description": "description",
        "isActive": True,
        "isRestricted": False,
        "maintainer": ["maintainer"],
        "region": ["region"],
        "publisher": "publisher",
        "subCategory": ["subCategory"],
        "tag": ["tag1", "tag2"],
        "deliveryChannel": ["API"],
        "theme": "theme",
        "releaseDate": "2020-05-05",
        "language": "English",
        "status": "Available",
        "image": "",
        "logo": "",
    }
    requests_mock.put(url, json=expected_data)

    my_product = Product(
        title="Test Product",
        identifier="TEST_PRODUCT",
        category=["category"],
        short_abstract="short abstract",
        description="description",
        is_active=True,
        is_restricted=False,
        maintainer=["maintainer"],
        region=["region"],
        publisher="publisher",
        sub_category=["subCategory"],
        tag=["tag1", "tag2"],
        delivery_channel=["API"],
        theme="theme",
        release_date="2020-05-05",
        language="English",
        status="Available",
        image="",
        logo="",
    )
    status_code = 200
    resp = my_product.update(client=fusion_obj, catalog=catalog, return_resp_obj=True)
    assert isinstance(resp, requests.Response)
    assert resp.status_code == status_code


def test_delete_product(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    """Test delete Product method."""
    catalog = "my_catalog"
    url = f"{fusion_obj.root_url}catalogs/{catalog}/products/TEST_PRODUCT"
    status_code = 204
    requests_mock.delete(url, status_code=status_code)

    resp = Product(identifier="TEST_PRODUCT").delete(client=fusion_obj, catalog=catalog, return_resp_obj=True)
    assert isinstance(resp, requests.Response)
    assert resp.status_code == status_code


def test_copy_product(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    """Test copy Product method."""
    catalog_from = "my_catalog"
    url = f"{fusion_obj.root_url}catalogs/{catalog_from}/products"
    expected_get_data = {
        "resources": [
            {
                "catalog": {
                    "@id": "my_catalog/",
                    "description": "my catalog",
                    "title": "my catalog",
                    "identifier": "my_catalog",
                },
                "title": "Test Product",
                "identifier": "TEST_PRODUCT",
                "category": ["category"],
                "shortAbstract": "short abstract",
                "description": "description",
                "isActive": True,
                "isRestricted": False,
                "maintainer": ["maintainer"],
                "region": ["region"],
                "publisher": "publisher",
                "subCategory": ["subCategory"],
                "tag": ["tag1", "tag2"],
                "deliveryChannel": ["API"],
                "theme": "theme",
                "releaseDate": "2020-05-05",
                "language": "English",
                "status": "Available",
                "image": "",
                "logo": "",
            },
        ]
    }
    requests_mock.get(url, json=expected_get_data)

    new_catalog = "new_catalog"
    post_url = f"{fusion_obj.root_url}catalogs/{new_catalog}/products/TEST_PRODUCT"
    expected_post_data = {
        "title": "Test Product",
        "identifier": "TEST_PRODUCT",
        "category": ["category"],
        "shortAbstract": "short abstract",
        "description": "description",
        "isActive": True,
        "isRestricted": False,
        "maintainer": ["maintainer"],
        "region": ["region"],
        "publisher": "publisher",
        "subCategory": ["subCategory"],
        "tag": ["tag1", "tag2"],
        "deliveryChannel": ["API"],
        "theme": "theme",
        "releaseDate": "2020-05-05",
        "language": "English",
        "status": "Available",
        "image": "",
        "logo": "",
    }

    requests_mock.post(post_url, json=expected_post_data)

    status_code = 200
    resp = Product(identifier="TEST_PRODUCT").copy(
        client=fusion_obj,
        catalog_from=catalog_from,
        catalog_to=new_catalog,
        return_resp_obj=True,
    )
    assert isinstance(resp, requests.Response)
    assert resp.status_code == status_code


def test_product_case_switching() -> None:
    """Test the product case switching."""
    my_product = Product(
        title="Test Product",
        identifier="Test Product",
        release_date="May 5, 2020",
        short_abstract="Short Abstract",
        description="Description",
        is_active=True,
        is_restricted=False,
        maintainer="Maintainer",
        region="Region",
        publisher="Publisher",
        sub_category="Sub Category",
        tag="Tag",
        delivery_channel="API",
        theme="Theme",
        language="English",
        status="Available",
    )

    camel_case_dict = my_product.to_dict()

    assert camel_case_dict == {
        "identifier": "TEST_PRODUCT",
        "title": "Test Product",
        "category": None,
        "shortAbstract": "Short Abstract",
        "description": "Description",
        "isActive": True,
        "isRestricted": False,
        "maintainer": ["Maintainer"],
        "region": ["Region"],
        "publisher": "Publisher",
        "subCategory": ["Sub Category"],
        "tag": ["Tag"],
        "deliveryChannel": ["API"],
        "theme": "Theme",
        "releaseDate": "2020-05-05",
        "language": "English",
        "status": "Available",
        "image": "",
        "logo": "",
        "dataset": None,
    }

    product_from_camel_dict = Product("TEST_PRODUCT").from_object(camel_case_dict)

    assert product_from_camel_dict == my_product

    assert product_from_camel_dict.short_abstract == product_from_camel_dict.shortAbstract
    assert product_from_camel_dict.sub_category == product_from_camel_dict.subCategory
    assert product_from_camel_dict.delivery_channel == product_from_camel_dict.deliveryChannel
    assert product_from_camel_dict.release_date == product_from_camel_dict.releaseDate
    assert product_from_camel_dict.is_active == product_from_camel_dict.isActive
    assert product_from_camel_dict.is_restricted == product_from_camel_dict.isRestricted


def test_product_repr() -> None:
    """Test the __repr__ method of the Product class."""
    test_product = Product(
        title="Test Product",
        identifier="Test Product",
        release_date="May 5, 2020",
        short_abstract="Short Abstract",
        description="Description",
        is_active=True,
        is_restricted=False,
        maintainer="Maintainer",
        region="Region",
        publisher="Publisher",
        sub_category="Sub Category",
        tag="Tag",
        delivery_channel="API",
        theme="Theme",
        language="English",
        status="Available",
    )

    expected_repr = (
        "Product(\n"
        "identifier='TEST_PRODUCT',\n "
        "title='Test Product',\n "
        "category=None,\n "
        "short_abstract='Short Abstract',\n "
        "description='Description',\n "
        "is_active=True,\n "
        "is_restricted=False,\n "
        "maintainer=['Maintainer'],\n "
        "region=['Region'],\n "
        "publisher='Publisher',\n "
        "sub_category=['Sub Category'],\n "
        "tag=['Tag'],\n "
        "delivery_channel=['API'],\n "
        "theme='Theme',\n "
        "release_date='2020-05-05',\n "
        "language='English',\n "
        "status='Available',\n "
        "image='',\n "
        "logo='',\n "
        "dataset=None\n"
        ")"
    )

    assert repr(test_product) == expected_repr


def test_product_getattr_existing_attribute() -> None:
    """Test the __getattr__ method for an existing attribute."""
    test_product = Product(title="Test Product", identifier="Test Product", release_date="May 5, 2020")
    assert test_product.shortAbstract == "Test Product"
    assert test_product.isActive is True
    assert test_product.releaseDate == "2020-05-05"


def test_product_getattr_non_existing_attribute() -> None:
    """Test the __getattr__ method for a non-existing attribute."""
    test_product = Product(title="Test Product", identifier="Test Product", release_date="May 5, 2020")
    with pytest.raises(AttributeError) as error_info:
        _ = test_product.nonExistingAttribute
    assert str(error_info.value) == "'Product' object has no attribute 'nonExistingAttribute'"


def test_product_client_property(fusion_obj: Fusion) -> None:
    """Test the client property getter."""
    test_product = Product(identifier="Test Product")
    assert test_product.client is None

    test_product.client = fusion_obj
    assert test_product.client is fusion_obj
