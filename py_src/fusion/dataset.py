"""Fusion Dataset class and functions."""

from __future__ import annotations

import json as js
from dataclasses import asdict, dataclass, field, fields
from typing import TYPE_CHECKING, Any

import pandas as pd

from fusion.utils import _is_json, convert_date_format, make_bool, make_list, requests_raise_for_status, tidy_string

if TYPE_CHECKING:
    import requests

    from fusion import Fusion


@dataclass
class Dataset:
    """Fusion Dataset class for managing dataset metadata in a Fusion catalog.

    Attributes:
        identifier (str): A unique identifier for the dataset.
        title (str, optional): A title for the dataset. If not provided, defaults to identifier.
        category (str | list[str] | None, optional): A category or list of categories for the dataset. Defaults to None.
        description (str, optional): A description of the dataset. If not provided, defaults to identifier.
        frequency (str, optional): The frequency of the dataset. Defaults to "Once".
        isInternalOnlyDataset (bool, optional): Flag for internal datasets. Defaults to False.
        isThirdPartyData (bool, optional): Flag for third party data. Defaults to True.
        isRestricted (bool | None, optional): Flag for restricted datasets. Defaults to None.
        isRawData (bool, optional): Flag for raw datasets. Defaults to True.
        maintainer (str | None, optional): Dataset maintainer. Defaults to "J.P. Morgan Fusion".
        source (str | list[str] | None, optional): Name of data vendor which provided the data. Defaults to None.
        region (str | list[str] | None, optional): Region. Defaults to None.
        publisher (str, optional): Name of vendor that publishes the data.. Defaults to "J.P. Morgan".
        product (str | list[str] | None, optional): Product to associate dataset with. Defaults to None.
        subCategory (str | list[str] | None, optional): Sub-category. Defaults to None.
        tags (str | list[str] | None, optional): Tags used for search purposes. Defaults to None.
        createdDate (str | None, optional): Created date. Defaults to None.
        modifiedDate (str | None, optional): Modified date. Defaults to None.
        deliveryChannel (str | list[str], optional): Delivery channel. Defaults to "API".
        language (str, optional): Language. Defaults to "English".
        status (str, optional): Status. Defaults to "Available".
        type_ (str | None, optional): Dataset type. Defaults to "Source".
        containerType (str | None, optional): Container type. Defaults to "Snapshot-Full".
        snowflake (str | None, optional): Snowflake account connection. Defaults to None.
        complexity (str | None, optional): Complecist. Defaults to None.
        isImmutable (bool | None, optional): Flag for immutable datasets. Defaults to None.
        isMnpi (bool | None, optional): isMnpi. Defaults to None.
        isPci (bool | None, optional): isPci. Defaults to None.
        isPii (bool | None, optional): isPii. Defaults to None.
        isClient (bool | None, optional): isClient. Defaults to None.
        isPublic (bool | None, optional): isPublic. Defaults to None.
        isInternal (bool | None, optional): IsInternal. Defaults to None.
        isConfidential (bool | None, optional): IsConfidential. Defaults to None.
        isHighlyConfidential (bool | None, optional): isHighlyConfidential. Defaults to None.
        isActive (bool | None, optional): isActive. Defaults to None.
        owners (list[str] | None, optional): The owners of the dataset. Defaults to None.
        applicationId (str | None, optional): The application ID of the dataset. Defaults to None.
        _client (Any, optional): A Fusion client object. Defaults to None.

    """

    identifier: str
    title: str = ""
    category: str | list[str] | None = None
    description: str = ""
    frequency: str = "Once"
    isInternalOnlyDataset: bool = False
    isThirdPartyData: bool = True
    isRestricted: bool | None = None
    isRawData: bool = True
    maintainer: str | None = "J.P. Morgan Fusion"
    source: str | list[str] | None = None
    region: str | list[str] | None = None
    publisher: str = "J.P. Morgan"
    product: str | list[str] | None = None
    subCategory: str | list[str] | None = None
    tags: str | list[str] | None = None
    createdDate: str | None = None
    modifiedDate: str | None = None
    deliveryChannel: str | list[str] = field(default_factory=lambda: ["API"])
    language: str = "English"
    status: str = "Available"
    type_: str | None = "Source"
    containerType: str | None = "Snapshot-Full"
    snowflake: str | None = None
    complexity: str | None = None
    isImmutable: bool | None = None
    isMnpi: bool | None = None
    isPci: bool | None = None
    isPii: bool | None = None
    isClient: bool | None = None
    isPublic: bool | None = None
    isInternal: bool | None = None
    isConfidential: bool | None = None
    isHighlyConfidential: bool | None = None
    isActive: bool | None = None
    owners: list[str] | None = None
    applicationId: str | None = None

    _client: Any = field(init=False, repr=False, compare=False, default=None)

    def __repr__(self: Dataset) -> str:
        """Return an object representation of the Dataset object.

        Returns:
            str: Object representaiton of the dataset.

        """
        attrs = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        return f"Dataset(\n" + ",\n ".join(f"{k}={v!r}" for k, v in attrs.items()) + "\n)"

    def __post_init__(self: Dataset) -> None:
        """Format Dataset metadata fields after object initialization."""
        self.identifier = tidy_string(self.identifier).upper().replace(" ", "_")
        self.title = tidy_string(self.title) if self.title != "" else self.identifier.replace("_", " ").title()
        self.description = tidy_string(self.description) if self.description != "" else self.title
        self.category = (
            self.category if isinstance(self.category, list) or self.category is None else make_list(self.category)
        )
        self.deliveryChannel = (
            self.deliveryChannel if isinstance(self.deliveryChannel, list) else make_list(self.deliveryChannel)
        )
        self.source = self.source if isinstance(self.source, list) or self.source is None else make_list(self.source)
        self.region = self.region if isinstance(self.region, list) or self.region is None else make_list(self.region)
        self.product = (
            self.product if isinstance(self.product, list) or self.product is None else make_list(self.product)
        )
        self.subCategory = (
            self.subCategory
            if isinstance(self.subCategory, list) or self.subCategory is None
            else make_list(self.subCategory)
        )
        self.tags = self.tags if isinstance(self.tags, list) or self.tags is None else make_list(self.tags)
        self.isInternalOnlyDataset = (
            self.isInternalOnlyDataset
            if isinstance(self.isInternalOnlyDataset, bool)
            else make_bool(self.isInternalOnlyDataset)
        )
        self.createdDate = convert_date_format(self.createdDate) if self.createdDate else None
        self.modifiedDate = convert_date_format(self.modifiedDate) if self.modifiedDate else None
        self.owners = self.owners if isinstance(self.owners, list) or self.owners is None else make_list(self.owners)

    def set_client(self, client: Any) -> None:
        """Set the client for the Dataset. Set automatically, if the Dataset is instantiated from a Fusion object.

        Args:
            client (Any): Fusion client object.

        Examples:
            >>> from fusion import Fusion
            >>> fusion = Fusion()
            >>> dataset = fusion.dataset("my_dataset")
            >>> dataset.set_client(fusion)

        """
        self._client = client

    @classmethod
    def _from_series(cls: type[Dataset], series: pd.Series[Any]) -> Dataset:
        """Instantiate a Dataset object from a pandas Series.

        Args:
            series (pd.Series[Any]): Dataset metadata as a pandas Series.

        Returns:
            Dataset: Dataset object.

        """
        series = series.rename(lambda x: x.replace(" ", "").replace("_", "").lower())
        series = series.rename({"tag": "tags"})
        series = series.rename({"type_": "type"})
        series = series.rename({"productId": "product"})

        isInternalOnlyDataset = series.get("isinternalonlydataset", None)
        isInternalOnlyDataset = (
            make_bool(isInternalOnlyDataset) if isInternalOnlyDataset is not None else isInternalOnlyDataset
        )
        isRestricted = series.get("isrestricted", None)
        isRestricted = make_bool(isRestricted) if isRestricted is not None else isRestricted
        isImmutable = series.get("isimmutable", None)
        isImmutable = make_bool(isImmutable) if isImmutable is not None else isImmutable
        isMnpi = series.get("ismnpi", None)
        isMnpi = make_bool(isMnpi) if isMnpi is not None else isMnpi
        isPci = series.get("ispci", None)
        isPci = make_bool(isPci) if isPci is not None else isPci
        isPii = series.get("ispii", None)
        isPii = make_bool(isPii) if isPii is not None else isPii
        isClient = series.get("isclient", None)
        isClient = make_bool(isClient) if isClient is not None else isClient
        isPublic = series.get("ispublic", None)
        isPublic = make_bool(isPublic) if isPublic is not None else isPublic
        isInternal = series.get("isinternal", None)
        isInternal = make_bool(isInternal) if isInternal is not None else isInternal
        isConfidential = series.get("isconfidential", None)
        isConfidential = make_bool(isConfidential) if isConfidential is not None else isConfidential
        isHighlyConfidential = series.get("ishighlyconfidential", None)
        isHighlyConfidential = (
            make_bool(isHighlyConfidential) if isHighlyConfidential is not None else isHighlyConfidential
        )
        isActive = series.get("isactive", None)
        isActive = make_bool(isActive) if isActive is not None else isActive

        dataset = cls(
            identifier=series.get("identifier", ""),
            category=series.get("category", None),
            deliveryChannel=series.get("deliverychannel", ["API"]),
            title=series.get("title", ""),
            description=series.get("description", ""),
            frequency=series.get("frequency", "Once"),
            isInternalOnlyDataset=isInternalOnlyDataset,  # type: ignore
            isThirdPartyData=series.get("isthirdpartydata", True),
            isRestricted=isRestricted,
            isRawData=series.get("israwdata", True),
            maintainer=series.get("maintainer", "J.P. Morgan Fusion"),
            source=series.get("source", None),
            region=series.get("region", None),
            publisher=series.get("publisher", "J.P. Morgan"),
            product=series.get("product", None),
            subCategory=series.get("subcategory", None),
            tags=series.get("tags", None),
            containerType=series.get("containertype", "Snapshot-Full"),
            language=series.get("language", "English"),
            status=series.get("status", "Available"),
            type_=series.get("type", "Source"),
            createdDate=series.get("createddate", None),
            modifiedDate=series.get("modifieddate", None),
            snowflake=series.get("snowflake", None),
            complexity=series.get("complexity", None),
            owners=series.get("owners", None),
            applicationId=series.get("applicationid", None),
            isImmutable=isImmutable,
            isMnpi=isMnpi,
            isPci=isPci,
            isPii=isPii,
            isClient=isClient,
            isPublic=isPublic,
            isInternal=isInternal,
            isConfidential=isConfidential,
            isHighlyConfidential=isHighlyConfidential,
            isActive=isActive,
        )
        return dataset

    @classmethod
    def _from_dict(cls: type[Dataset], data: dict[str, Any]) -> Dataset:
        """Instantiate a Dataset object from a dictionary.

        Args:
            data (dict[str, Any]): Dataset metadata as a dictionary.

        Returns:
            Dataset: Dataset object.

        """
        keys = [f.name for f in fields(cls)]
        keys = ["type" if key == "type_" else key for key in keys]
        data = {k: v for k, v in data.items() if k in keys}
        if "type" in data:
            data["type_"] = data.pop("type")
        return cls(**data)

    @classmethod
    def _from_csv(cls: type[Dataset], file_path: str, identifier: str | None = None) -> Dataset:
        """Instantiate a Dataset object from a CSV file.

        Args:
            file_path (str): Path to the CSV file.
            identifier (str | None, optional): Dataset identifer for filtering if multipler datasets are defined in csv.
                Defaults to None.

        Returns:
            Dataset: Dataset object.

        """
        data = pd.read_csv(file_path)

        return (
            Dataset._from_series(data[data["identifier"] == identifier].reset_index(drop=True).iloc[0])
            if identifier
            else Dataset._from_series(data.reset_index(drop=True).iloc[0])
        )

    def from_object(
        self,
        dataset_source: Dataset | dict[str, Any] | str | pd.Series[Any],
    ) -> Dataset:
        """Instantiate a Dataset object from a Dataset object, dictionary, JSON string, path to CSV, or pandas Series.

        Args:
            dataset_source (Dataset | dict[str, Any] | str | pd.Series[Any]): Dataset metadata source.

        Raises:
            TypeError: If the object provided is not a Dataset, dictionary, JSON string, path to CSV file,
                or pandas Series.

        Returns:
            Dataset: Dataset object.

        Examples:
            Instantiate a Dataset object from a dictionary:

            >>> from fusion import Fusion
            >>> fusion = Fusion()
            >>> dataset_dict = {
            ...     "identifier": "my_dataset",
            ...     "title": "My Dataset",
            ...     "description": "My dataset description",
            ...     "category": "Finance",
            ...     "frequency": "Daily",
            ...     "isRestricted": False,
            ...     "isRawData": True,
            ...     "maintainer": "J.P. Morgan Fusion",
            ...     "source": "J.P. Morgan",
            ...     }
            >>> dataset = fusion.dataset("my_dataset").from_object(dataset_dict)

            Instantiate a Dataset object from a JSON string:

            >>> from fusion import Fusion
            >>> fusion = Fusion()
            >>> dataset_json = '{
            ...     "identifier": "my_dataset",
            ...     "title": "My Dataset",
            ...     "description": "My dataset description",
            ...     "category": "Finance",
            ...     "frequency": "Daily",
            ...     "isRestricted": False,
            ...     "isRawData": True,
            ...     "maintainer": "J.P. Morgan Fusion",
            ...     "source": "J.P. Morgan"
            ...     }'
            >>> dataset = fusion.dataset("my_dataset").from_object(dataset_json)

            Instantiate a Dataset object from a CSV file:

            >>> from fusion import Fusion
            >>> fusion = Fusion()
            >>> dataset = fusion.dataset("my_dataset").from_object("path/to/dataset.csv")

            Instantiate a Dataset object from a pandas Series:

            >>> from fusion import Fusion
            >>> fusion = Fusion()
            >>> dataset_series = pd.Series({
            ...     "identifier": "my_dataset",
            ...     "title": "My Dataset",
            ...     "description": "My dataset description",
            ...     "category": "Finance",
            ...     "frequency": "Daily",
            ...     "isRestricted": False,
            ...     "isRawData": True,
            ...     "maintainer": "J.P. Morgan Fusion",
            ...     "source": "J.P. Morgan"
            ...     })
            >>> dataset = fusion.dataset("my_dataset").from_object(dataset_series)

        """
        if isinstance(dataset_source, Dataset):
            dataset = dataset_source
        elif isinstance(dataset_source, dict):
            dataset = Dataset._from_dict(dataset_source)
        elif isinstance(dataset_source, str):
            if _is_json(dataset_source):
                dataset = Dataset._from_dict(js.loads(dataset_source))
            else:
                dataset = Dataset._from_csv(dataset_source)
        elif isinstance(dataset_source, pd.Series):
            dataset = Dataset._from_series(dataset_source)
        else:
            raise TypeError(f"Could not resolve the object provided: {dataset_source}")

        dataset.set_client(self._client)

        return dataset

    def from_catalog(self, catalog: str | None = None, client: Fusion | None = None) -> Dataset:
        """Instantiate a Dataset object from a Fusion catalog.

        Args:
            catalog (str | None, optional): Catalog identifer. Defaults to None.
            client (Fusion | None, optional): Fusion session. Defaults to None.
                If instantiated from a Fusion object, then the client is set automatically.

        Returns:
            Dataset: Dataset object.

        Examples:

            >>> from fusion import Fusion
            >>> fusion = Fusion()
            >>> dataset = fusion.dataset("my_dataset").from_catalog(catalog="my_catalog")

        """
        if client is None:
            client = self._client
        catalog = client._use_catalog(catalog)
        dataset = self.identifier
        resp = client.session.get(f"{client.root_url}catalogs/{catalog}/datasets")
        requests_raise_for_status(resp)
        list_datasets = resp.json()["resources"]
        dict_ = [dict_ for dict_ in list_datasets if dict_["identifier"] == dataset][0]
        dataset_obj = Dataset._from_dict(dict_)
        dataset_obj.set_client(client)

        prod_df = client.list_product_dataset_mapping(catalog=catalog)

        if dataset.lower() in list(prod_df.dataset.str.lower()):
            product = [prod_df[prod_df["dataset"].str.lower() == dataset.lower()]["product"].iloc[0]]
            dataset_obj.product = product

        return dataset_obj

    def to_dict(self) -> dict[str, Any]:
        """Convert the Dataset instance to a dictionary.

        Returns:
            dict[str, Any]: Dataset metadata as a dictionary.

        Examples:
            >>> from fusion import Fusion
            >>> fusion = Fusion()
            >>> dataset = fusion.dataset("my_dataset")
            >>> dataset_dict = dataset.to_dict()

        """
        dataset_dict = asdict(self)
        dataset_dict["type"] = dataset_dict.pop("type_")
        dataset_dict.pop("_client")
        return dataset_dict

    def create(
        self,
        catalog: str | None = None,
        client: Fusion | None = None,
        return_resp_obj: bool = False,
    ) -> requests.Response | None:
        """Upload a new dataset to a Fusion catalog.

        Args:
            catalog (str | None, optional): A catalog identifier. Defaults to "common".
            client (Fusion, optional): A Fusion client object. Defaults to the instance's _client.
                If instantiated from a Fusion object, then the client is set automatically.
            return_resp_obj (bool, optional): If True then return the response object. Defaults to False.

        Returns:
            requests.Response | None: The response object from the API call if return_resp_obj is True, otherwise None.

        Examples:

            From scratch:

            >>> from fusion import Fusion
            >>> fusion = Fusion()
            >>> dataset = fusion.dataset(
            ...     identifier= "my_dataset",
            ...     title= "My Dataset",
            ...     description= "My dataset description",
            ...     category= "Finance",
            ...     frequency= "Daily",
            ...     isRestricted= False
            ...     )
            >>> dataset.create(catalog="my_catalog")

            From a dictionary:

            >>> from fusion import Fusion
            >>> fusion = Fusion()
            >>> dataset_dict = {
            ...     "identifier": "my_dataset",
            ...     "title": "My Dataset",
            ...     "description": "My dataset description",
            ...     "category": "Finance",
            ...     "frequency": "Daily",
            ...     "isRestricted": False
            ...     }
            >>> dataset = fusion.dataset("my_dataset").from_object(dataset_dict)
            >>> dataset.create(catalog="my_catalog")

            From a JSON string:

            >>> from fusion import Fusion
            >>> fusion = Fusion()
            >>> dataset_json = '{
            ...     "identifier": "my_dataset",
            ...     "title": "My Dataset",
            ...     "description": "My dataset description",
            ...     "category": "Finance",
            ...     "frequency": "Daily",
            ...     "isRestricted": False
            ...     }'
            >>> dataset = fusion.dataset("my_dataset").from_object(dataset_json)
            >>> dataset.create(catalog="my_catalog")

            From a CSV file:

            >>> from fusion import Fusion
            >>> fusion = Fusion()
            >>> dataset = fusion.dataset("my_dataset").from_object("path/to/dataset.csv")
            >>> dataset.create(catalog="my_catalog")

            From a pandas Series:

            >>> from fusion import Fusion
            >>> fusion = Fusion()
            >>> dataset_series = pd.Series({
            ...     "identifier": "my_dataset",
            ...     "title": "My Dataset",
            ...     "description": "My dataset description",
            ...     "category": "Finance",
            ...     "frequency": "Daily",
            ...     "isRestricted": False
            ...     })
            >>> dataset = fusion.dataset("my_dataset").from_object(dataset_series)

            From existing dataset in a catalog:

            >>> from fusion import Fusion
            >>> fusion = Fusion()
            >>> dataset = fusion.dataset("my_dataset").from_catalog(catalog="my_catalog")
            >>> dataset.identifier = "my_new_dataset"
            >>> dataset.create(catalog="my_catalog")

        """
        client = self._client if client is None else client
        catalog = client._use_catalog(catalog)

        self.createdDate = self.createdDate if self.createdDate else pd.Timestamp("today").strftime("%Y-%m-%d")

        self.modifiedDate = self.modifiedDate if self.modifiedDate else pd.Timestamp("today").strftime("%Y-%m-%d")

        data = self.to_dict()

        url = f"{client.root_url}catalogs/{catalog}/datasets/{self.identifier}"
        resp: requests.Response = client.session.post(url, json=data)
        requests_raise_for_status(resp)

        return resp if return_resp_obj else None

    def update(
        self,
        catalog: str | None = None,
        client: Fusion | None = None,
        return_resp_obj: bool = False,
    ) -> requests.Response | None:
        """Updates a dataset via API from dataset object.

        Args:
            catalog (str | None, optional): A catalog identifier. Defaults to "common".
            client (Fusion, optional): A Fusion client object. Defaults to the instance's _client.
                If instantiated from a Fusion object, then the client is set automatically.
            return_resp_obj (bool, optional): If True then return the response object. Defaults to False.

        Returns:
            requests.Response | None: The response object from the API call if return_resp_obj is True, otherwise None.

        Examples:

            >>> from fusion import Fusion
            >>> fusion = Fusion()
            >>> dataset = fusion.dataset("my_dataset").from_catalog(catalog="my_catalog")
            >>> dataset.title = "My Updated Dataset"
            >>> dataset.update(catalog="my_catalog")

        """
        client = self._client if client is None else client
        catalog = client._use_catalog(catalog)

        self.createdDate = self.createdDate if self.createdDate else pd.Timestamp("today").strftime("%Y-%m-%d")
        self.modifiedDate = self.modifiedDate if self.modifiedDate else pd.Timestamp("today").strftime("%Y-%m-%d")

        data = self.to_dict()

        url = f"{client.root_url}catalogs/{catalog}/datasets/{self.identifier}"
        resp: requests.Response = client.session.put(url, json=data)
        requests_raise_for_status(resp)
        return resp if return_resp_obj else None

    def delete(
        self,
        catalog: str | None = None,
        client: Fusion | None = None,
        return_resp_obj: bool = False,
    ) -> requests.Response | None:
        """Delete a dataset via API from its dataset identifier.

        Args:
            catalog (str | None, optional): A catalog identifier. Defaults to "common".
            client (Fusion, optional): A Fusion client object. Defaults to the instance's _client.
                If instantiated from a Fusion object, then the client is set automatically.
            return_resp_obj (bool, optional): If True then return the response object. Defaults to False.

        Returns:
            requests.Response | None: The response object from the API call if return_resp_obj is True, otherwise None.

        Examples:

            >>> from fusion import Fusion
            >>> fusion = Fusion()
            >>> fusion.dataset("my_dataset").delete(catalog="my_catalog")

        """
        client = self._client if client is None else client
        catalog = client._use_catalog(catalog)

        url = f"{client.root_url}catalogs/{catalog}/datasets/{self.identifier}"
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
        """Copy dataset from one catalog and/or environment to another by copy.

        Args:
            catalog_to (str): A catalog identifier to which to copy dataset.
            catalog_from (str, optional): A catalog identifier from which to copy dataset. Defaults to "common".
            client (Fusion, optional): A Fusion client object. Defaults to the instance's _client.
                If instantiated from a Fusion object, then the client is set automatically.
            client_to (Fusion | None, optional): Fusion client object. Defaults to current instance.
            return_resp_obj (bool, optional): If True then return the response object. Defaults to False.

        Returns:
            requests.Response | None: The response object from the API call if return_resp_obj is True, otherwise None.

        Examples:

            >>> from fusion import Fusion
            >>> fusion = Fusion()
            >>> dataset = fusion.dataset("my_dataset").copy(catalog_from="my_catalog", catalog_to="my_new_catalog")

        """
        client = self._client if client is None else client
        catalog_from = client._use_catalog(catalog_from)

        if client_to is None:
            client_to = client
        dataset_obj = self.from_catalog(catalog=catalog_from, client=client)
        dataset_obj.set_client(client_to)
        resp = dataset_obj.create(client=client_to, catalog=catalog_to, return_resp_obj=True)
        return resp if return_resp_obj else None
