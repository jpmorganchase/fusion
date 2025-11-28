from __future__ import (
    annotations,  # Enables postponed evaluation of type annotations (forward references) in this file.
)

import logging  # Standard logging library for warnings/info/errors.
from dataclasses import dataclass, field, fields  # Dataclass utilities for declarative classes and field introspection.
from typing import TYPE_CHECKING, Any  # Typing helpers; TYPE_CHECKING avoids runtime imports, Any is a wide type.

import numpy as np  # NumPy used for NaN/inf cleanup when building from DataFrames.
import pandas as pd  # Pandas used for Series/DataFrame conversions and CSV reading.

from .utils import (  # Project utilities imported from local package.
    CamelCaseMeta,  # Metaclass that lets you access snake_case fields via camelCase names.
    camel_to_snake,  # Converts camelCase to snake_case strings.
    requests_raise_for_status,  # Wraps response.raise_for_status() with uniform error handling.
    snake_to_camel,  # Converts snake_case to camelCase strings.
    tidy_string,  # Normalizes strings (trim, collapse whitespace, etc.).
)

if TYPE_CHECKING:  # Only import these types for type-checkers, not at runtime (avoids circular deps / costs).
    import requests  # Type for HTTP responses (used in annotations only).

    from fusion import Fusion  # The client/session type we attach to objects.

logger = logging.getLogger(__name__)  # Module-level logger for warnings and debug messages.


@dataclass  # Turn the class into a dataclass (auto __init__, __repr__, etc.).
class Dataflow(metaclass=CamelCaseMeta):  # Use CamelCaseMeta so camelCase attribute access maps to snake_case.
    """Represents a single Dataflow object with CRUD operations and Dataset-style loaders."""  # Class docstring.

    # Required  # Informal section header to clarify which fields are required by API.
    providerNode: dict[str, str]  # Provider node metadata as a dict, e.g., {"name": "...", "dataNodeType": "..."}.
    consumerNode: dict[str, str]  # Consumer node metadata as a dict, similar structure as providerNode.

    # Optional fields  # Informal header for clarity.
    description: str | None = None  # Optional description; normalized in __post_init__.
    id: str | None = None  # Dataflow ID assigned by server (required for update/delete/patch).
    alternativeId: dict[str, Any] | None = None  # Optional alternate identifier block.
    transportType: str | None = None  # Transport mechanism, e.g., "Batch" / "Streaming".
    frequency: str | None = None  # Frequency info, e.g., "Daily".
    startTime: str | None = None  # Optional start time as string/ISO.
    endTime: str | None = None  # Optional end time as string/ISO.
    boundarySets: list[dict[str, Any]] = field(default_factory=list)  # Optional list of boundary-set dicts.
    dataAssets: list[dict[str, Any]] = field(default_factory=list)  # Optional list of data-asset dicts.

    _client: Fusion | None = field(
        init=False, repr=False, compare=False, default=None
    )  # Attached Fusion client (hidden).

    def __post_init__(self) -> None:  # Runs automatically after dataclass __init__.
        """Normalize description immediately after initialization."""  # Docstring describing post-init normalization.
        self.description = tidy_string(self.description or "")  # Clean description; empty becomes "" (not None).

    def __getattr__(self, name: str) -> Any:  # Fallback when attribute not found in normal dict.
        """Allow camelCase access for snake_case attributes."""  # Docstring: supports camelCase accessors.
        snake_name = camel_to_snake(name)  # Convert requested attr to our snake_case storage key.
        return self.__dict__.get(snake_name, None)  # Return value if present; otherwise None (soft access).

    def __setattr__(self, name: str, value: Any) -> None:  # Intercept all assignments.
        """Allow camelCase assignment to snake_case attributes."""  # Docstring: supports camelCase setters.
        if name == "client":  # Special-case: writing 'client' should use property (avoid recursion).
            object.__setattr__(self, name, value)  # Bypass our mapping for the client property.
        else:
            snake_name = camel_to_snake(name)  # Convert attribute name to snake_case.
            self.__dict__[snake_name] = value  # Store directly in the instance dict.

    @property  # Define a getter for the client property.
    def client(self) -> Fusion | None:  # Returns either Fusion or None.
        """Fusion client associated with this Dataflow."""  # Docstring: what 'client' represents.
        return self._client  # Return the private _client field.

    @client.setter  # Define a setter for the client property.
    def client(self, client: Fusion | None) -> None:  # Assigns a Fusion client to the object.
        self._client = client  # Store the client object (or None) internally.

    def _use_client(self, client: Fusion | None) -> Fusion:  # Utility to resolve the active client.
        """Resolve the Fusion client (either provided or bound to the object)."""  # Docstring: explains resolution.
        res = self._client if client is None else client  # Prefer passed client; otherwise use attached client.
        if res is None:  # If neither is available:
            raise ValueError("A Fusion client object is required.")  # Tell the caller we need a client.
        return res  # Return the resolved client.

    @classmethod  # Class constructor helper for dicts.
    def from_dict(cls: type[Dataflow], data: dict[str, Any]) -> Dataflow:  # Convert a raw dict into a Dataflow.
        """Create a Dataflow object from a dictionary."""  # Docstring.

        def normalize_value(val: Any) -> Any:  # Inner helper to normalize scalar values.
            if isinstance(val, str) and val.strip() == "":  # If a string is empty/whitespace:
                return None  # Treat it as None (for clean payloads).
            return val  # Otherwise return as-is.

        def convert_keys(d: dict[str, Any]) -> dict[str, Any]:  # Inner helper to recursively snake_case keys.
            converted = {}  # New dict to populate.
            for k, v in d.items():  # Iterate source dictionary items.
                key = camel_to_snake(k)  # Convert field name to snake_case.
                if isinstance(v, dict):  # For nested dicts:
                    converted[key] = convert_keys(v)  # Recurse to convert nested keys.
                elif isinstance(v, list):  # For lists:
                    converted[key] = [
                        convert_keys(i) if isinstance(i, dict) else i for i in v
                    ]  # Convert dict items in lists.
                else:
                    converted[key] = normalize_value(v)  # Normalize scalars (e.g., "" -> None).
            return converted  # Return the converted mapping.

        converted_data = convert_keys(data)  # Apply key conversion and normalization to entire dict.
        valid_fields = {f.name for f in fields(cls)}  # All dataclass field names for filtering.
        filtered_data = {
            k: v for k, v in converted_data.items() if k in valid_fields
        }  # Keep only fields known to the class.

        obj = cls.__new__(cls)  # Allocate instance without calling default __init__ (we'll set attrs manually).
        for field_obj in fields(cls):  # For each declared dataclass field:
            setattr(obj, field_obj.name, filtered_data.get(field_obj.name, None))  # Set value or None if missing.

        obj.__post_init__()  # Manually run post-init to normalize description.
        return obj  # Return the built instance.

    @classmethod  # Class constructor helper for pandas Series.
    def _from_series(cls: type[Dataflow], series: pd.Series) -> Dataflow:  # Build from one DataFrame row.
        """Instantiate a single Dataflow from a pandas Series."""  # Docstring.
        return cls.from_dict(series.to_dict())  # Convert Series to dict and delegate to from_dict.

    @classmethod  # Class constructor helper for CSV.
    def _from_csv(
        cls: type[Dataflow], file_path: str, row: int | None = None
    ) -> Dataflow:  # Build from CSV file (one row).
        """Instantiate a Dataflow object from a CSV file.

        Args:
            file_path (str): Path to the CSV file.
            row (int | None, optional): Row index to pick (defaults to 0).

        Returns:
            Dataflow: A single Dataflow instance.
        """  # Detailed docstring for CSV ingestion.
        import json  # Local import to avoid polluting module unless needed.

        df = pd.read_csv(file_path)  # Load CSV into a DataFrame.

        idx = 0 if row is None else row  # Choose row index (default first row).
        if not (0 <= idx < len(df)):  # Validate the index is in range.
            raise IndexError(f"Row index {idx} out of range for CSV with {len(df)} rows")  # Helpful error if not.

        series = df.iloc[idx].copy()  # Extract the chosen row as a Series.

        # Parse JSON string columns into dicts  # Comment explaining the loop purpose.
        for col in ["providerNode", "consumerNode"]:  # Only these two columns need JSON parsing if strings.
            if col in series and isinstance(series[col], str):  # If the column exists and is a string:
                try:
                    series[col] = json.loads(series[col])  # Try to parse the JSON string into a dict.
                except Exception:  # If parsing fails:
                    raise ValueError(f"Column {col} must contain a JSON object string")  # Raise a clear message.

        return cls._from_series(series)  # Build the Dataflow from the parsed Series.

    @classmethod  # Alias constructor (kept for symmetry with Dataset).
    def _from_dict(cls: type[Dataflow], data: dict[str, Any]) -> Dataflow:  # Private-ish dict constructor.
        """Instantiate a single Dataflow from a dict ."""  # Short docstring.
        return cls.from_dict(data)  # Delegate to the public from_dict for consistency.

    @classmethod  # Bulk builder from a DataFrame.
    def from_dataframe(
        cls, frame: pd.DataFrame, client: Fusion | None = None
    ) -> list[Dataflow]:  # Build many from DataFrame.
        """Instantiate multiple Dataflow objects from a DataFrame (row-wise)."""  # Docstring.
        frame = frame.replace([np.nan, np.inf, -np.inf], None).where(frame.notna(), None)  # Clean up NaN/inf to None.
        results: list[Dataflow] = []  # Accumulator for successfully built and validated Dataflows.
        for _, row in frame.iterrows():  # Iterate over rows as Series.
            try:
                obj = cls._from_series(row)  # Build a Dataflow from this row.
                obj.client = client  # Attach the provided client (if any).
                obj.validate()  # Ensure required fields exist.
                results.append(obj)  # Keep the valid object.
            except ValueError as e:  # noqa: PERF203  # Skip invalid rows with a warning (Perf rule silenced intentionally).
                logger.warning("Skipping invalid row: %s", e)  # Log why the row was skipped.
        return results  # Return only good Dataflows.

    def from_object(self, dataflow_source: Dataflow | dict[str, Any] | str | pd.Series) -> Dataflow:  # type: ignore[type-arg]  # Flexible single-object builder.
        """Instantiate a single Dataflow from a Dataflow, dict, JSON-object string, CSV path, or pandas Series.

        - Dataflow: returned as-is (client re-bound).
        - dict: converted via _from_dict.
        - pandas Series: converted via _from_series.
        - str:
            * If it looks like a JSON object (starts with '{'), parse and build via _from_dict.
            * If it ends with '.csv', read the CSV and take the first row via _from_csv.
            (Use _from_csv(file_path, row=0) semantics.)
        """  # Rich docstring explaining behavior.
        import json  # Local import for JSON parsing.

        if isinstance(dataflow_source, Dataflow):  # If already a Dataflow:
            obj = dataflow_source  # Reuse it directly.
        elif isinstance(dataflow_source, dict):  # If dict:
            obj = self._from_dict(dataflow_source)  # Convert via dict path.
        elif isinstance(dataflow_source, pd.Series):  # If Series:
            obj = self._from_series(dataflow_source)  # Convert via series path.
        elif isinstance(dataflow_source, str):  # If string:
            s = dataflow_source.strip()  # Trim whitespace.
            if s.startswith("{"):  # JSON object string:
                obj = self._from_dict(json.loads(s))  # Parse JSON and build.
            elif s.lower().endswith(".csv"):  # CSV file path:
                obj = self._from_csv(s)  # Build from the first row of the CSV.
            else:
                raise ValueError(
                    "Unsupported string input — must be JSON object string or a .csv file path"
                )  # Guardrail error.
        else:
            raise TypeError(
                f"Could not resolve the object provided: {type(dataflow_source).__name__}"
            )  # Type error if unrecognized.

        obj.client = self._client  # Re-bind this object's client to the new Dataflow.
        return obj  # Return the constructed Dataflow.

    def validate(self) -> None:  # Simple required-field validator.
        """Validate that required fields exist."""  # Docstring.
        required_fields = ["provider_node", "consumer_node"]  # Fields that must be present and non-empty.
        missing = [f for f in required_fields if getattr(self, f, None) in [None, ""]]  # Collect missing/blank ones.
        if missing:  # If any required fields are missing:
            raise ValueError(f"Missing required fields in Dataflow: {', '.join(missing)}")  # Raise a clear error.

    def to_dict(  # Serializer to dict with camelCase keys for API payloads.
        self,
        *,
        drop_none: bool = True,  # If True, omit None/blank string values.
        exclude: set[str] | None = None,  # Optional set of snake_case fields to exclude.
    ) -> dict[str, Any]:  # Returns a dict suitable for JSON encoding.
        """Convert Dataflow object into a JSON-serializable dictionary."""  # Docstring.
        out: dict[str, Any] = {}  # Accumulator for result.
        for k, v in self.__dict__.items():  # Iterate instance fields.
            if k.startswith("_"):  # Skip private/internal fields (like _client).
                continue  # Do not include internal keys.
            if exclude and k in exclude:  # If caller asked to exclude a field:
                continue  # Skip it.
            # Treat empty strings like None when dropping Nones  # Clarifies the next conditional.
            if drop_none and (  # If dropping null-like values:
                v is None or (isinstance(v, str) and v.strip() == "")  # Consider empty strings as null-like.
            ):
                continue  # Omit this key/value.
            out[snake_to_camel(k)] = v  # Use camelCase field name in output dict.
        return out  # Return the serialized mapping.

    def create(  # Create a server-side dataflow.
        self,
        client: Fusion | None = None,  # Optional client override (else use attached).
        return_resp_obj: bool = False,  # If True, return the raw HTTP response.
    ) -> requests.Response | None:  # Return type depends on flag.
        """Create the dataflow via API."""  # Docstring.
        client = self._use_client(client)  # Resolve which client/session to use.

        payload = self.to_dict(drop_none=True, exclude={"id"})  # Build request body without 'id'.
        url = f"{client._get_new_root_url()}/api/corelineage-service/v1/lineage/dataflows"  # Create endpoint URL.
        resp: requests.Response = client.session.post(url, json=payload)  # Send POST to create the resource.
        requests_raise_for_status(resp)  # Raise if non-2xx, with friendly formatting.

        return resp if return_resp_obj else None  # Optionally return the HTTP response.

    def update(  # Full update (PUT) for an existing dataflow.
        self,
        client: Fusion | None = None,  # Optional client override.
        return_resp_obj: bool = False,  # If True, return raw response.
    ) -> requests.Response | None:  # Return type based on flag.
        """Full replace (PUT) using self.id, excluding provider/consumer nodes from the payload."""  # Docstring.
        client = self._use_client(client)  # Resolve client.
        if not self.id:  # Ensure we know which resource to update.
            raise ValueError("Dataflow ID is required on the object (set self.id before update()).")  # Helpful error.

        payload = self.to_dict(  # Build the request body for PUT.
            drop_none=True,  # Drop None/empty values to avoid overwriting with nulls.
            exclude={"id", "provider_node", "consumer_node"},  # Provider/consumer immutable via PUT here.
        )

        url = f"{client._get_new_root_url()}/api/corelineage-service/v1/lineage/dataflows/{self.id}"  # PUT endpoint for this ID.
        resp: requests.Response = client.session.put(url, json=payload)  # Perform the PUT.
        requests_raise_for_status(resp)  # Raise on failure.
        return resp if return_resp_obj else None  # Optionally return raw response.

    def update_fields(  # Partial update (PATCH) for specific fields.
        self,
        changes: dict[str, Any],  # Fields to change; keys can be snake_case or camelCase.
        client: Fusion | None = None,  # Optional client override.
        return_resp_obj: bool = False,  # If True, return raw response.
    ) -> requests.Response | None:  # Return type based on flag.
        """Partial update (PATCH) using self.id. Provider/consumer nodes are not allowed."""  # Docstring.
        client = self._use_client(client)  # Resolve client.
        if not self.id:  # We must know which resource to patch.
            raise ValueError("Dataflow ID is required on the object (set self.id before update_fields()).")  # Error.

        forbidden = {"provider_node", "consumer_node"}  # Fields disallowed in PATCH (immutable for this op).
        normalized = {camel_to_snake(k): v for k, v in changes.items()}  # Normalize keys to snake_case internally.
        used = forbidden.intersection(normalized.keys())  # Check for forbidden keys.
        if used:  # If any forbidden keys were provided:
            raise ValueError(  # Raise a clear error explaining the issue.
                f"Cannot update {sorted(used)} via PATCH; provider/consumer nodes are immutable for updates."
            )

        patch_body = {snake_to_camel(k): v for k, v in normalized.items()}  # Build camelCase JSON body.

        url = f"{client._get_new_root_url()}/api/corelineage-service/v1/lineage/dataflows/{self.id}"  # PATCH endpoint.
        resp: requests.Response = client.session.patch(url, json=patch_body)  # Perform PATCH call.
        requests_raise_for_status(resp)  # Raise on HTTP error codes.
        return resp if return_resp_obj else None  # Optionally return response.

    def delete(  # Delete this dataflow by ID.
        self,
        client: Fusion | None = None,  # Optional client override.
        return_resp_obj: bool = False,  # If True, return raw response.
    ) -> requests.Response | None:  # Return type based on flag.
        """Delete this dataflow using self.id."""  # Docstring.
        client = self._use_client(client)  # Resolve client/session.
        if not self.id:  # Ensure we know which resource to delete.
            raise ValueError("Dataflow ID is required on the object (set self.id before delete()).")  # Helpful error.

        url = f"{client._get_new_root_url()}/api/corelineage-service/v1/lineage/dataflows/{self.id}"  # Delete endpoint.
        resp: requests.Response = client.session.delete(url)  # Perform DELETE.
        requests_raise_for_status(resp)  # Raise on error codes.
        return resp if return_resp_obj else None  # Optionally return raw response.
