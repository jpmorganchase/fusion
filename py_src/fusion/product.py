"""Fusion Product class and functions."""

from __future__ import annotations

import json as js
from dataclasses import asdict, dataclass, field, fields
from typing import TYPE_CHECKING, Any

import pandas as pd

from fusion.utils import _is_json, convert_date_format, make_bool, make_list, tidy_string

if TYPE_CHECKING:
    import requests

    from fusion import Fusion


@dataclass
class Product:
    "Product class."

    identifier: str
    title: str = ""
    category: str | list[str] | None = None
    shortAbstract: str = ""
    description: str = ""
    isActive: bool = True
    isRestricted: bool | None = None
    maintainer: str | list[str] | None = None
    region: str | list[str] | None = None
    publisher: str | None = None
    subCategory: str | list[str] | None = None
    tag: str | list[str] | None = None
    deliveryChannel: str | list[str] = field(default_factory=lambda: ["API"])
    theme: str | None = None
    releaseDate: str | None = None
    language: str = "English"
    status: str = "Available"
    image: str = ""
    logo: str = ""
    dataset: str | list[str] | None = None

    _client: Any = field(init=False, repr=False, compare=False, default=None)

    def __repr__(self: Product) -> str:
        """Return a string representation of the Product object."""
        return (
            f"Product(\n"
            f"  title={self.title!r},\n"
            f"  identifier={self.identifier!r},\n"
            f"  category={self.category!r},\n"
            f"  shortAbstract={self.shortAbstract!r},\n"
            f"  description={self.description!r},\n"
            f"  isActive={self.isActive!r},\n"
            f"  isRestricted={self.isRestricted!r},\n"
            f"  maintainer={self.maintainer!r},\n"
            f"  region={self.region!r},\n"
            f"  publisher={self.publisher!r},\n"
            f"  subCategory={self.subCategory!r},\n"
            f"  tag={self.tag!r},\n"
            f"  deliveryChannel={self.deliveryChannel!r},\n"
            f"  theme={self.theme!r},\n"
            f"  releaseDate={self.releaseDate!r},\n"
            f"  language={self.language!r},\n"
            f"  status={self.status!r},\n"
            f"  image={self.image!r},\n"
            f"  logo={self.logo!r},\n"
            f"  dataset={self.dataset!r}\n"
            f")"
        )

    def __post_init__(self: Product) -> None:
        """Format Product metadata fields after object instantiation."""
        self.title = tidy_string(self.title)
        self.identifier = tidy_string(self.identifier).upper().replace(" ", "_")
        self.shortAbstract = tidy_string(self.shortAbstract)
        self.description = tidy_string(self.description)
        self.category = (
            self.category if isinstance(self.category, list) or self.category is None else make_list(self.category)
        )
        self.tag = self.tag if isinstance(self.tag, list) or self.tag is None else make_list(self.tag)
        self.dataset = (
            self.dataset if isinstance(self.dataset, list) or self.dataset is None else make_list(self.dataset)
        )
        self.subCategory = (
            self.subCategory
            if isinstance(self.subCategory, list) or self.subCategory is None
            else make_list(self.subCategory)
        )
        self.isActive = self.isActive if isinstance(self.isActive, bool) else make_bool(self.isActive)
        self.isRestricted = (
            self.isRestricted
            if isinstance(self.isRestricted, bool) or self.isRestricted is None
            else make_bool(self.isRestricted)
        )
        self.maintainer = (
            self.maintainer
            if isinstance(self.maintainer, list) or self.maintainer is None
            else make_list(self.maintainer)
        )
        self.region = self.region if isinstance(self.region, list) or self.region is None else make_list(self.region)
        self.deliveryChannel = (
            self.deliveryChannel if isinstance(self.deliveryChannel, list) else make_list(self.deliveryChannel)
        )
        self.releaseDate = convert_date_format(self.releaseDate) if self.releaseDate else None

    def set_client(self, client: Any) -> None:
        """Set the client for the Product."""
        self._client = client

    @classmethod
    def from_series(cls: type[Product], series: pd.Series[Any]) -> Product:
        """Create a Product object from a pandas Series."""
        series = series.rename(lambda x: x.replace(" ", "").replace("_", "").lower())
        series = series.rename({"tag": "tags", "dataset": "datasets"})
        short_abstract = series.get("abstract", "")
        if short_abstract is None:
            short_abstract = series.get("shortabstract", "")

        return cls(
            title=series.get("title", ""),
            identifier=series.get("identifier", ""),
            category=series.get("category", None),
            shortAbstract=short_abstract,
            description=series.get("description", ""),
            theme=series.get("theme", None),
            releaseDate=series.get("releasedate", None),
            isActive=series.get("isactive", True),
            isRestricted=series.get("isrestricted", None),
            maintainer=series.get("maintainer", None),
            region=series.get("region", None),
            publisher=series.get("publisher", None),
            subCategory=series.get("subcategory", None),
            tag=series.get("tags", None),
            deliveryChannel=series.get("deliverychannel", "API"),
            language=series.get("language", "English"),
            status=series.get("status", "Available"),
            dataset=series.get("datasets", None),
        )

    @classmethod
    def from_dict(cls: type[Product], data: dict[str, Any]) -> Product:
        """Create a Product object from a dictionary."""
        keys = [f.name for f in fields(cls)]
        data = {k: v for k, v in data.items() if k in keys}
        return cls(**data)

    @classmethod
    def from_csv(cls: type[Product], file_path: str, identifier: str | None = None) -> Product:
        """Create a list of Product objects from a CSV file."""
        data = pd.read_csv(file_path)

        return (
            Product.from_series(data[data["identifier"] == identifier].reset_index(drop=True).iloc[0])
            if identifier
            else Product.from_series(data.reset_index(drop=True).iloc[0])
        )

    @classmethod
    def from_object(cls: type[Product], product_source: Any) -> Product:
        """Create a Product object from a dictionary."""
        if isinstance(product_source, Product):
            return product_source
        if isinstance(product_source, dict):
            return Product.from_dict(product_source)
        if isinstance(product_source, str):
            if _is_json(product_source):
                return Product.from_dict(js.loads(product_source))
            return Product.from_csv(product_source)
        if isinstance(product_source, pd.Series):
            return Product.from_series(product_source)

        raise TypeError(f"Could not resolve the object provided: {product_source}")

    def from_catalog(self, catalog: str, client: Fusion | None = None) -> Product:
        """Create a Product object from a catalog."""
        if client is None:
            client = self._client

        catalog = client._use_catalog(catalog)

        list_products = client.session.get(f"{client.root_url}catalogs/{catalog}/products").json()["resources"]
        dict_ = [dict_ for dict_ in list_products if dict_["identifier"] == self.identifier][0]
        product_obj = Product.from_dict(dict_)
        product_obj.set_client(client)

        return product_obj

    def to_dict(self: Product) -> dict[str, Any]:
        """Convert the Product object to a dictionary."""
        product_dict = asdict(self)
        product_dict.pop("_client")
        return product_dict

    def create(
        self,
        catalog: str | None = None,
        client: Fusion | None = None,
    ) -> requests.Response:
        """Create a new product in the catalog.

        Args:
            client (Fusion, optional): A Fusion client object. Defaults to the instance's _client.
            catalog (str, optional): A catalog identifier. Defaults to None.

        Returns:
            requests.Response: The response object from the API call.
        """
        client = self._client if client is None else client
        catalog = client._use_catalog(catalog)

        releaseDate = self.releaseDate if self.releaseDate else pd.Timestamp("today").strftime("%Y-%m-%d")
        deliveryChannel = self.deliveryChannel if self.deliveryChannel else ["API"]

        self.releaseDate = releaseDate
        self.deliveryChannel = deliveryChannel

        data = self.to_dict()

        url = f"{client.root_url}catalogs/{catalog}/products/{self.identifier}"
        resp: requests.Response = client.session.post(url, json=data)
        resp.raise_for_status()
        return resp

    def update(
        self,
        catalog: str | None = None,
        client: Fusion | None = None,
    ) -> requests.Response:
        """Update an existing product in the catalog.

        Args:
            client (Fusion): A Fusion client object.
            catalog (str, optional): A catalog identifier. Defaults to None.

        Returns:
            requests.Response: The response object from the API call.
        """
        client = self._client if client is None else client
        catalog = client._use_catalog(catalog)

        releaseDate = self.releaseDate if self.releaseDate else pd.Timestamp("today").strftime("%Y-%m-%d")
        deliveryChannel = self.deliveryChannel if self.deliveryChannel else ["API"]

        self.releaseDate = releaseDate
        self.deliveryChannel = deliveryChannel

        data = self.to_dict()

        url = f"{client.root_url}catalogs/{catalog}/products/{self.identifier}"
        resp: requests.Response = client.session.put(url, json=data)
        resp.raise_for_status()
        return resp

    def delete(
        self,
        catalog: str | None = None,
        client: Fusion | None = None,
    ) -> requests.Response:
        """Delete a product from the catalog.

        Args:
            client (Fusion): A Fusion client object.
            catalog (str, optional): A catalog identifier. Defaults to None.

        Returns:
            requests.Response: The response object from the API call.
        """
        client = self._client if client is None else client
        catalog = client._use_catalog(catalog)

        url = f"{client.root_url}catalogs/{catalog}/products/{self.identifier}"
        resp: requests.Response = client.session.delete(url)
        resp.raise_for_status()
        return resp

    def copy(
        self,
        catalog_to: str,
        catalog_from: str | None = None,
        client: Fusion | None = None,
        client_to: Fusion | None = None,
    ) -> requests.Response:
        """Copy product from one catalog and/or environment to another by copy.

        Args:
            product (str): Product  identifier.
            catalog_to (str): Catalog  identifier to wich to copy product.
            catalog_from (str, optional): A catalog identifier from which to copy product. Defaults to "common".
            client_to (Fusion | None, optional): Fusion client object. Defaults to current instance.

        Returns:
            requests.Response: The response object from the API call.
        """
        client = self._client if client is None else client
        catalog_from = client._use_catalog(catalog_from)
        if client_to is None:
            client_to = client
        product_obj = self.from_catalog(catalog=catalog_from, client=client)
        product_obj.set_client(client_to)
        return product_obj.create(catalog=catalog_to)
