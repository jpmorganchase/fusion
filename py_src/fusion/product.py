"""Fusion Product class and functions."""

from __future__ import annotations

import json as js
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any

import pandas as pd

from fusion.utils import (
    CamelCaseMeta,
    _is_json,
    camel_to_snake,
    convert_date_format,
    make_bool,
    make_list,
    requests_raise_for_status,
    snake_to_camel,
    tidy_string,
)

if TYPE_CHECKING:
    import requests

    from fusion import Fusion


@dataclass
class Product(metaclass=CamelCaseMeta):
    """Fusion Product class for managing product metadata in a Fusion catalog.

    Attributes:
        identifier (str): A unique identifier for the product.
        title (str, optional): Product title. Defaults to "".
        category (str | list[str] | None, optional): Product category. Defaults to None.
        short_abstract (str, optional): Short abstract of the product. Defaults to "".
        description (str, optional): Product description. If not provided, defaults to identifier.
        is_active (bool, optional): Boolean for Active status. Defaults to True.
        is_restricted (bool | None, optional): Flag for restricted products. Defaults to None.
        maintainer (str | list[str] | None, optional): Product maintainer. Defaults to None.
        region (str | list[str] | None, optional): Product region. Defaults to None.
        publisher (str | None, optional): Name of vendor that publishes the data. Defaults to None.
        sub_category (str | list[str] | None, optional): Product sub-category. Defaults to None.
        tag (str | list[str] | None, optional): Tags used for search purposes. Defaults to None.
        delivery_channel (str | list[str], optional): Product delivery channel. Defaults to ["API"].
        theme (str | None, optional): Product theme. Defaults to None.
        release_date (str | None, optional): Product release date. Defaults to None.
        language (str, optional): Product language. Defaults to "English".
        status (str, optional): Product status. Defaults to "Available".
        image (str, optional): Product image. Defaults to "".
        logo (str, optional): Product logo. Defaults to "".
        dataset (str | list[str] | None, optional): Product datasets. Defaults to None.
        _client (Any, optional): Fusion client object. Defaults to None.

    """

    identifier: str
    title: str = ""
    category: str | list[str] | None = None
    short_abstract: str = ""
    description: str = ""
    is_active: bool = True
    is_restricted: bool | None = None
    maintainer: str | list[str] | None = None
    region: str | list[str] = field(default_factory=lambda: ["Global"])
    publisher: str = "J.P. Morgan"
    sub_category: str | list[str] | None = None
    tag: str | list[str] | None = None
    delivery_channel: str | list[str] = field(default_factory=lambda: ["API"])
    theme: str | None = None
    release_date: str | None = None
    language: str = "English"
    status: str = "Available"
    image: str = ""
    logo: str = ""
    dataset: str | list[str] | None = None

    _client: Fusion | None = field(init=False, repr=False, compare=False, default=None)

    def __repr__(self: Product) -> str:
        """Return an object representation of the Product object.

        Returns:
            str: Object representaiton of the product.

        """
        attrs = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        return f"Product(\n" + ",\n ".join(f"{k}={v!r}" for k, v in attrs.items()) + "\n)"

    def __post_init__(self: Product) -> None:
        """Format Product metadata fields after object instantiation."""
        self.identifier = tidy_string(self.identifier).upper().replace(" ", "_")
        self.title = tidy_string(self.title) if self.title != "" else self.identifier.replace("_", " ").title()
        self.description = tidy_string(self.description) if self.description != "" else self.title
        self.short_abstract = tidy_string(self.short_abstract) if self.short_abstract != "" else self.title
        self.description = tidy_string(self.description)
        self.category = (
            self.category if isinstance(self.category, list) or self.category is None else make_list(self.category)
        )
        self.tag = self.tag if isinstance(self.tag, list) or self.tag is None else make_list(self.tag)
        self.dataset = (
            self.dataset if isinstance(self.dataset, list) or self.dataset is None else make_list(self.dataset)
        )
        self.sub_category = (
            self.sub_category
            if isinstance(self.sub_category, list) or self.sub_category is None
            else make_list(self.sub_category)
        )
        self.is_active = self.is_active if isinstance(self.is_active, bool) else make_bool(self.is_active)
        self.is_restricted = (
            self.is_restricted
            if isinstance(self.is_restricted, bool) or self.is_restricted is None
            else make_bool(self.is_restricted)
        )
        self.maintainer = (
            self.maintainer
            if isinstance(self.maintainer, list) or self.maintainer is None
            else make_list(self.maintainer)
        )
        self.region = self.region if isinstance(self.region, list) or self.region is None else make_list(self.region)
        self.delivery_channel = (
            self.delivery_channel if isinstance(self.delivery_channel, list) else make_list(self.delivery_channel)
        )
        self.release_date = convert_date_format(self.release_date) if self.release_date else None

    def __getattr__(self, name: str) -> Any:
        # Redirect attribute access to the snake_case version
        snake_name = camel_to_snake(name)
        if snake_name in self.__dict__:
            return self.__dict__[snake_name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "client":
            # Use the property setter for client
            object.__setattr__(self, name, value)
        else:
            snake_name = camel_to_snake(name)
            self.__dict__[snake_name] = value

    @property
    def client(self) -> Fusion | None:
        """Return the client."""
        return self._client

    @client.setter
    def client(self, client: Fusion | None) -> None:
        """Set the client for the Product. Set automatically, if the Product is instantiated from a Fusion object.

        Args:
            client (Any): Fusion client object.

        Examples:
            >>> from fusion import Fusion
            >>> fusion = Fusion()
            >>> product = fusion.product("my_product")
            >>> product.client = fusion

        """
        self._client = client

    def _use_client(self, client: Fusion | None) -> Fusion:
        """Determine client."""

        res = self._client if client is None else client
        if res is None:
            raise ValueError("A Fusion client object is required.")
        return res

    @classmethod
    def _from_series(cls: type[Product], series: pd.Series[Any]) -> Product:
        """Instantiate a Product object from a pandas Series.

        Args:
            series (pd.Series[Any]): Product metadata as a pandas Series.

        Returns:
            Product: Product object.

        """
        series = series.rename(lambda x: x.replace(" ", "").replace("_", "").lower())
        series = series.rename({"tag": "tags", "dataset": "datasets"})
        short_abstract = series.get("abstract", "")
        short_abstract = series.get("shortabstract", "") if short_abstract is None else short_abstract

        return cls(
            title=series.get("title", ""),
            identifier=series.get("identifier", ""),
            category=series.get("category", None),
            short_abstract=short_abstract,
            description=series.get("description", ""),
            theme=series.get("theme", None),
            release_date=series.get("releasedate", None),
            is_active=series.get("isactive", True),
            is_restricted=series.get("isrestricted", None),
            maintainer=series.get("maintainer", None),
            region=series.get("region", "Global"),
            publisher=series.get("publisher", "J.P. Morgan"),
            sub_category=series.get("subcategory", None),
            tag=series.get("tags", None),
            delivery_channel=series.get("deliverychannel", "API"),
            language=series.get("language", "English"),
            status=series.get("status", "Available"),
            dataset=series.get("datasets", None),
        )

    @classmethod
    def _from_dict(cls: type[Product], data: dict[str, Any]) -> Product:
        """Instantiate a Product object from a dictionary.

        Args:
            data (dict[str, Any]): Product metadata as a dictionary.

        Returns:
            Product: Product object.

        """
        keys = [f.name for f in fields(cls)]
        data = {camel_to_snake(k): v for k, v in data.items()}
        data = {k: v for k, v in data.items() if k in keys}
        return cls(**data)

    @classmethod
    def _from_csv(cls: type[Product], file_path: str, identifier: str | None = None) -> Product:
        """Instantiate a Product object from a CSV file.

        Args:
            file_path (str): Path to the CSV file.
            identifier (str | None, optional): Product identifer for filtering if multipler products are defined in csv.
                Defaults to None.

        Returns:
            Product: Product object.

        """
        data = pd.read_csv(file_path)

        return (
            Product._from_series(data[data["identifier"] == identifier].reset_index(drop=True).iloc[0])
            if identifier
            else Product._from_series(data.reset_index(drop=True).iloc[0])
        )

    def from_object(
        self,
        product_source: Product | dict[str, Any] | str | pd.Series[Any],
    ) -> Product:
        """Instantiate a Product object from a Product object, dictionary, path to CSV, JSON string, or pandas Series.

        Args:
            product_source (Product | dict[str, Any] | str | pd.Series[Any]): Product metadata source.

        Raises:
            TypeError: If the object provided is not a Product, dictionary, path to CSV file, JSON string,
            or pandas Series.

        Returns:
            Product: Product object.

        Examples:
            Instantiating a Product object from a dictionary:

            >>> from fusion import Fusion
            >>> from fusion.product import Product
            >>> fusion = Fusion()
            >>> product_dict = {
            ...     "identifier": "my_product",
            ...     "title": "My Product",
            ...     "category": "Data",
            ...     "short_abstract": "My product is awesome",
            ...     "description": "My product is very awesome",
            ...     "is_active": True,
            ...     "is_restricted": False,
            ...     "maintainer": "My Company",
            ...     "region": "Global",
            ...     "publisher": "My Company",
            ...     "sub_category": "Data",
            ...     "tag": "My Company",
            ...     "delivery_channel": "API",
            ...     "theme": "Data",
            ...     "release_date": "2021-01-01",
            ...     "language": "English",
            ...     "status": "Available"
            ... }
            >>> product = fusion.product("my_product").from_object(product_dict)

            Instantiating a Product object from a JSON string:

            >>> from fusion import Fusion
            >>> from fusion.product import Product
            >>> fusion = Fusion()
            >>> product_json = '{
            ...     "identifier": "my_product",
            ...     "title": "My Product",
            ...     "category": "Data",
            ...     "short_abstract": "My product is awesome",
            ...     "description": "My product is very awesome",
            ...     "is_active": True,
            ...     "is_restricted": False,
            ...     "maintainer": "My Company",
            ...     "region": "Global",
            ...     "publisher": "My Company",
            ...     "sub_category": "Data",
            ...     "tag": "My Company",
            ...     "delivery_channel": "API",
            ...     "theme": "Data",
            ...     "release_date": "2021-01-01",
            ...     "language": "English",
            ...     "status": "Available",
            ... }'
            >>> product = fusion.product("my_product").from_object(product_json)

            Instantiating a Product object from a CSV file:

            >>> from fusion import Fusion
            >>> from fusion.product import Product
            >>> fusion = Fusion()
            >>> product = fusion.product("my_product").from_object("path/to/product.csv")

            Instantiating a Product object from a pandas Series:

            >>> from fusion import Fusion
            >>> from fusion.product import Product
            >>> fusion = Fusion()
            >>> product_series = pd.Series({
            ...     "identifier": "my_product",
            ...     "title": "My Product",
            ...     "category": "Data",
            ...     "short_abstract": "My product is awesome",
            ...     "description": "My product is very awesome",
            ...     "is_active": True,
            ...     "is_restricted": False,
            ...     "maintainer": "My Company",
            ...     "region": "Global",
            ...     "publisher": "My Company",
            ...     "sub_category": "Data",
            ...     "tag": "My Company",
            ...     "delivery_channel": "API",
            ...     "theme": "Data",
            ...     "release_date": "2021-01-01",
            ...     "language": "English",
            ...     "status": "Available",
            ... })
            >>> product = fusion.product("my_product").from_object(product_series)

        """
        if isinstance(product_source, Product):
            product = product_source
        elif isinstance(product_source, dict):
            product = Product._from_dict(product_source)
        elif isinstance(product_source, str):
            if _is_json(product_source):
                product = Product._from_dict(js.loads(product_source))
            else:
                product = Product._from_csv(product_source)
        elif isinstance(product_source, pd.Series):
            product = Product._from_series(product_source)
        else:
            raise TypeError(f"Could not resolve the object provided: {product_source}")
        product.client = self._client
        return product

    def from_catalog(self, catalog: str | None = None, client: Fusion | None = None) -> Product:
        """Instantiate a Product object from a Fusion catalog.

        Args:
            catalog (str | None, optional): Catalog identifer. Defaults to None.
            client (Fusion | None, optional): Fusion session. Defaults to None.
                If instantiated from a Fusion object, then the client is set automatically.

        Returns:
            Product: Product object.

        Examples:
            >>> from fusion import Fusion
            >>> fusion = Fusion()
            >>> product = fusion.product("my_product").from_catalog(catalog="my_catalog")

        """
        client = self._use_client(client)
        catalog = client._use_catalog(catalog)

        resp = client.session.get(f"{client.root_url}catalogs/{catalog}/products")
        requests_raise_for_status(resp)
        list_products = resp.json()["resources"]
        dict_ = [dict_ for dict_ in list_products if dict_["identifier"] == self.identifier][0]
        product_obj = Product._from_dict(dict_)
        product_obj.client = client

        return product_obj

    def to_dict(self: Product) -> dict[str, Any]:
        """Convert the Product instance to a dictionary.

        Returns:
            dict[str, Any]: Product metadata as a dictionary.

        Examples:
            >>> from fusion import Fusion
            >>> fusion = Fusion()
            >>> product = fusion.product("my_product")
            >>> product_dict = product.to_dict()

        """
        product_dict = {snake_to_camel(k): v for k, v in self.__dict__.items() if not k.startswith("_")}
        return product_dict

    def create(
        self,
        catalog: str | None = None,
        client: Fusion | None = None,
        return_resp_obj: bool = False,
    ) -> requests.Response | None:
        """Upload a new product to a Fusion catalog.

        Args:
            client (Fusion, optional): A Fusion client object. Defaults to the instance's _client.
                If instantiated from a Fusion object, then the client is set automatically.
            catalog (str, optional): A catalog identifier. Defaults to None.
            return_resp_obj (bool, optional): If True then return the response object. Defaults to False.

        Returns:
            requests.Response | None: The response object from the API call if return_resp_obj is True, otherwise None.

        Examples:

            From scratch:

            >>> from fusion import Fusion
            >>> fusion = Fusion()
            >>> product = fusion.product(
            ...     identifer="my_product"
            ...     title="My Product",
            ...     category="Data",
            ...     short_abstract="My product is awesome",
            ...     description="My product is very awesome",
            ...     )
            >>> product.create(catalog="my_catalog")

            From a dictionary:

            >>> product_dict = {
            ...     "identifier": "my_product",
            ...     "title": "My Product",
            ...     "category": "Data"
            ...     }
            >>> product = fusion.product("my_product").from_object(product_dict)
            >>> product.create(catalog="my_catalog")

            From a JSON string:

            >>> from fusion import Fusion
            >>> fusion = Fusion()
            >>> product_json = '{
            ...     "identifier": "my_product",
            ...     "title": "My Product",
            ...     "category": "Data"
            ...     }'
            >>> product = fusion.product("my_product").from_object(product_json)
            >>> product.create(catalog="my_catalog")

            From a CSV file:

            >>> from fusion import Fusion
            >>> fusion = Fusion()
            >>> product = fusion.product("my_product").from_object("path/to/product.csv")
            >>> product.create(catalog="my_catalog")

            From a pandas Series:

            >>> from fusion import Fusion
            >>> fusion = Fusion()
            >>> product_series = pd.Series({
            ...     "identifier": "my_product",
            ...     "title": "My Product",
            ...     "category": "Data"
            ...     })
            >>> product = fusion.product("my_product").from_object(product_series)
            >>> product.create(catalog="my_catalog")

            From existing product in a catalog:

            >>> from fusion import Fusion
            >>> fusion = Fusion()
            >>> product = fusion.product("my_product").from_catalog()
            >>> product.identifier = "my_new_product"
            >>> product.create(catalog="my_catalog")

        """
        client = self._use_client(client)
        catalog = client._use_catalog(catalog)

        release_date = self.release_date if self.release_date else pd.Timestamp("today").strftime("%Y-%m-%d")
        delivery_channel = self.delivery_channel if self.delivery_channel else ["API"]

        self.release_date = release_date
        self.delivery_channel = delivery_channel

        data = self.to_dict()

        url = f"{client.root_url}catalogs/{catalog}/products/{self.identifier}"
        resp: requests.Response = client.session.post(url, json=data)
        requests_raise_for_status(resp)
        return resp if return_resp_obj else None

    def update(
        self,
        catalog: str | None = None,
        client: Fusion | None = None,
        return_resp_obj: bool = False,
    ) -> requests.Response | None:
        """Update an existing product in a Fusion catalog.

        Args:
            client (Fusion): A Fusion client object. Defaults to the instance's _client.
                If instantiated from a Fusion object, then the client is set automatically.
            catalog (str, optional): A catalog identifier. Defaults to None.
            return_resp_obj (bool, optional): If True then return the response object. Defaults to False.

        Returns:
            requests.Response | None: The response object from the API call if return_resp_obj is True, otherwise None.

        Examples:

            >>> from fusion import Fusion
            >>> fusion = Fusion()
            >>> product = fusion.product("my_product").from_catalog(catalog="my_catalog")
            >>> product.title = "My Updated Product Title"
            >>> product.update(catalog="my_catalog")

        """
        client = self._use_client(client)
        catalog = client._use_catalog(catalog)

        release_date = self.release_date if self.release_date else pd.Timestamp("today").strftime("%Y-%m-%d")
        delivery_channel = self.delivery_channel if self.delivery_channel else ["API"]

        self.release_date = release_date
        self.delivery_channel = delivery_channel

        data = self.to_dict()

        url = f"{client.root_url}catalogs/{catalog}/products/{self.identifier}"
        resp: requests.Response = client.session.put(url, json=data)
        requests_raise_for_status(resp)
        return resp if return_resp_obj else None

    def delete(
        self,
        catalog: str | None = None,
        client: Fusion | None = None,
        return_resp_obj: bool = False,
    ) -> requests.Response | None:
        """Delete a product from a Fusion catalog.

        Args:
            client (Fusion): A Fusion client object. Defaults to the instance's _client.
                If instantiated from a Fusion object, then the client is set automatically.
            catalog (str, optional): A catalog identifier. Defaults to None.
            return_resp_obj (bool, optional): If True then return the response object. Defaults to False.

        Returns:
            requests.Response | None: The response object from the API call if return_resp_obj is True, otherwise None.

         Examples:

            >>> from fusion import Fusion
            >>> fusion = Fusion()
            >>> fusion.product("my_product").delete(catalog="my_catalog")

        """
        client = self._use_client(client)
        catalog = client._use_catalog(catalog)

        url = f"{client.root_url}catalogs/{catalog}/products/{self.identifier}"
        resp: requests.Response = client.session.delete(url)
        requests_raise_for_status(resp)
        return resp if return_resp_obj else None

    def copy(
        self,
        catalog_to: str,
        catalog_from: str | None = None,
        client: Fusion | None = None,
        client_to: Fusion | None = None,
        return_resp_obj: bool = False,
    ) -> requests.Response | None:
        """Copy product from one Fusion catalog and/or environment to another by copy.

        Args:
            catalog_to (str): Catalog identifier to which to copy product.
            catalog_from (str, optional): A catalog identifier from which to copy product. Defaults to "common".
            client (Fusion): A Fusion client object. Defaults to the instance's _client.
                If instantiated from a Fusion object, then the client is set automatically.
            client_to (Fusion | None, optional): Fusion client object. Defaults to current instance.
            return_resp_obj (bool, optional): If True then return the response object. Defaults to False.

        Returns:
            requests.Response | None: The response object from the API call if return_resp_obj is True, otherwise None.

        Examples:

            >>> from fusion import Fusion
            >>> fusion = Fusion()
            >>> fusion.product("my_product").copy(catalog_from="my_catalog", catalog_to="my_new_catalog")

        """
        client = self._use_client(client)
        catalog_from = client._use_catalog(catalog_from)
        if client_to is None:
            client_to = client
        product_obj = self.from_catalog(catalog=catalog_from, client=client)
        product_obj.client = client_to
        resp = product_obj.create(catalog=catalog_to, return_resp_obj=True)
        return resp if return_resp_obj else None
