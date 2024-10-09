"""Fusion Dataset class and functions."""

from __future__ import annotations

import json as js
from dataclasses import asdict, dataclass, field, fields
from typing import TYPE_CHECKING, Any

import pandas as pd

from fusion.utils import _is_json, convert_date_format, make_bool, make_list, tidy_string

if TYPE_CHECKING:
    from fusion import Fusion


@dataclass
class Dataset:
    """Dataset class."""

    title: str
    identifier: str
    category: str | list[str] | None = None
    description: str = ""
    frequency: str = "Once"
    isInternalOnlyDataset: bool = False
    isThirdPartyData: bool = True
    isRestricted: bool | None = None
    isRawData: bool = False
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

    def __repr__(self: Dataset) -> str:
        """Format object representation."""
        return (
            f"Dataset(\n"
            f"  title={self.title!r},\n"
            f"  identifier={self.identifier!r},\n"
            f"  category={self.category!r},\n"
            f"  description={self.description!r},\n"
            f"  frequency={self.frequency!r},\n"
            f"  isInternalOnlyDataset={self.isInternalOnlyDataset!r},\n"
            f"  isThirdPartyData={self.isThirdPartyData!r},\n"
            f"  isRestricted={self.isRestricted!r},\n"
            f"  isRawData={self.isRawData!r},\n"
            f"  maintainer={self.maintainer!r},\n"
            f"  source={self.source!r},\n"
            f"  region={self.region!r},\n"
            f"  publisher={self.publisher!r},\n"
            f"  product={self.product!r},\n"
            f"  subCategory={self.subCategory!r},\n"
            f"  tags={self.tags!r},\n"
            f"  createdDate={self.createdDate!r},\n"
            f"  modifiedDate={self.modifiedDate!r},\n"
            f"  deliveryChannel={self.deliveryChannel!r},\n"
            f"  language={self.language!r},\n"
            f"  status={self.status!r},\n"
            f"  type_={self.type_!r},\n"
            f"  containerType={self.containerType!r},\n"
            f"  snowflake={self.snowflake!r},\n"
            f"  complexity={self.complexity!r},\n"
            f"  isImmutable={self.isImmutable!r},\n"
            f"  isMnpi={self.isMnpi!r},\n"
            f"  isPci={self.isPci!r},\n"
            f"  isPii={self.isPii!r},\n"
            f"  isClient={self.isClient!r},\n"
            f"  isPublic={self.isPublic!r},\n"
            f"  isInternal={self.isInternal!r},\n"
            f"  isConfidential={self.isConfidential!r},\n"
            f"  isHighlyConfidential={self.isHighlyConfidential!r},\n"
            f"  isActive={self.isActive!r}\n"
        )

    def __post_init__(self: Dataset) -> None:
        """Format Dataset metadata fields after object initialization."""
        self.identifier = tidy_string(self.identifier).upper().replace(" ", "_")
        self.title = tidy_string(self.title)
        self.description = tidy_string(self.description)
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

    @classmethod
    def from_series(cls: type[Dataset], series: pd.Series) -> Dataset:
        """Create a Dataset object from a pandas Series."""
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
            identifier=series.get("identifier", None),
            category=series.get("category", None),
            deliveryChannel=series.get("deliverychannel", ["API"]),
            title=series.get("title", None),
            description=series.get("description", ""),
            frequency=series.get("frequency", "Once"),
            isInternalOnlyDataset=isInternalOnlyDataset,
            isThirdPartyData=series.get("isthirdpartydata", True),
            isRestricted=isRestricted,
            isRawData=series.get("israwdata", False),
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
    def from_dict(cls: type[Dataset], data: dict[str, Any]) -> Dataset:
        """Create a Dataset object from a dictionary."""
        keys = [f.name for f in fields(cls)]
        keys = ["type" if key == "type_" else key for key in keys]
        data = {k: v for k, v in data.items() if k in keys}
        if "type" in data:
            data["type_"] = data.pop("type")
        return cls(**data)

    @classmethod
    def from_csv(cls: type[Dataset], file_path: str, identifier: str | None = None) -> Dataset:
        """Create a list of Dataset objects from a CSV file."""
        data = pd.read_csv(file_path)

        return (
            Dataset.from_series(data[data["identifier"] == identifier].reset_index(drop=True).iloc[0])
            if identifier
            else Dataset.from_series(data.reset_index(drop=True).iloc[0])
        )

    @classmethod
    def from_object(
        cls: type[Dataset],
        dataset_source: Dataset | dict[str, Any] | str | pd.Series,
    ) -> Dataset:
        """Create a Dataset object from a dictionary."""
        if isinstance(dataset_source, Dataset):
            return dataset_source
        if isinstance(dataset_source, dict):
            return Dataset.from_dict(dataset_source)
        if isinstance(dataset_source, str):
            if _is_json(dataset_source):
                return Dataset.from_dict(js.loads(dataset_source))
            return Dataset.from_csv(dataset_source)
        if isinstance(dataset_source, pd.Series):
            return Dataset.from_series(dataset_source)

        raise TypeError(f"Could not resolve the object provided: {dataset_source}")

    @classmethod
    def from_catalog(cls: type[Dataset], client: Fusion, dataset: str, catalog: str) -> Dataset:
        """Create a Dataset object from a catalog."""
        list_datasets = client.session.get(f"{client.root_url}catalogs/{catalog}/datasets").json()["resources"]
        dict_ = [dict_ for dict_ in list_datasets if dict_["identifier"] == dataset][0]
        dataset_obj = Dataset.from_dict(dict_)

        prod_df = client.list_product_dataset_mapping(catalog=catalog)

        if dataset.lower() in list(prod_df.dataset.str.lower()):
            product = [prod_df[prod_df["dataset"].str.lower() == dataset.lower()]["product"].iloc[0]]
            dataset_obj.product = product

        return dataset_obj

    def to_dict(self) -> dict[str, Any]:
        """Convert the Dataset object to a dictionary."""
        dataset_dict = asdict(self)
        dataset_dict["type"] = dataset_dict.pop("type_")
        return dataset_dict
