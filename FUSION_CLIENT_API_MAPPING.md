# Fusion Client - Complete API Method Mapping

This document provides a comprehensive list of all methods exposed through the Fusion client and the underlying API endpoints they use, including runtime-constructed URLs.

---

## Quick Reference

| Category | Methods Count | Key Endpoints |
|----------|---------------|---------------|
| **Catalog Operations** | 2 | `GET /catalogs/`, `GET /catalogs/{catalog}` |
| **Product Operations** | 4 | `GET/POST/PUT/DELETE /catalogs/{catalog}/products/...` |
| **Dataset Operations** | 15 | `GET/POST/PUT/DELETE /catalogs/{catalog}/datasets/...` |
| **Attribute Operations** | 6 | `GET/POST/PUT/DELETE /catalogs/{catalog}/attributes/...` |
| **Download Operations** | 4 | `GET .../distributions/{format}/files/operationType/download` |
| **Upload Operations** | 2 | `PUT/POST .../distributions/{format}` (single/multipart) |
| **Report Operations** | 7 | `GET/POST/PUT/PATCH/DELETE /api/corelineage-service/v1/reports/...` |
| **Report Attribute Ops** | 6 | `GET/POST/PUT/PATCH/DELETE /api/corelineage-service/v1/reports/{id}/attributes/...` |
| **Dataflow Operations** | 5 | `GET/POST/PUT/PATCH/DELETE /api/corelineage-service/v1/lineage/dataflows/...` |
| **Data Dependency Ops** | 2 | `POST/DELETE /api/corelineage-service/v1/data-dependencies/...` |
| **Data Mapping Ops** | 3 | `POST/DELETE/PATCH /api/corelineage-service/v1/data-mapping/...` |
| **Lineage Operations** | 4 | Dataset & attribute lineage endpoints |
| **Events/Notifications** | 2 | `GET /catalogs/{catalog}/notifications/subscribe` (SSE) |
| **Embeddings/Vector** | 3 | `GET /dataspaces/{catalog}/datasets/{kb}/indexes/` |
| **Filesystem Operations** | Multiple | FusionHTTPFileSystem runtime endpoints |

**Total Public Methods**: 68+  
**Total Unique API Endpoints**: 50+  
**HTTP Methods**: GET, HEAD, POST, PUT, PATCH, DELETE  
**Special Protocols**: Server-Sent Events (SSE), Multipart Upload  
**Range Request Patterns**: Query parameters (?downloadRange=, &downloadRange=), Standard HTTP Range header

---

## Table of Contents
1. [Method to API Endpoint Quick Reference Table](#method-to-api-endpoint-quick-reference-table)
2. [Fusion Client Public Methods](#fusion-client-public-methods)
3. [Dataset Object Methods](#dataset-object-methods)
4. [Product Object Methods](#product-object-methods)
5. [Attribute/Attributes Object Methods](#attributeattributes-object-methods)
6. [Report Object Methods](#report-object-methods)
7. [ReportAttribute/ReportAttributes Object Methods](#reportattributereportattributes-object-methods)
8. [Dataflow Object Methods](#dataflow-object-methods)
9. [Data Dependency & Mapping Methods](#data-dependency--mapping-methods)

---

## Performance Metrics - Method Reference Table

This table shows expected performance metrics for each method across different load scenarios.

**⚠️ IMPORTANT - Performance Metrics Disclaimer:**

These metrics are **ESTIMATED TARGETS** that need validation in your environment. To get actual measurements:

**How to Obtain Real Measurements:**
```bash
# Install dependencies
pip install -e ".[test]"
pip install pytest pytest-benchmark pytest-mock

# Run performance tests with your credentials
export FUSION_E2E_CLIENT_ID="your_client_id"
export FUSION_E2E_CLIENT_SECRET="your_client_secret"

# Run all performance tests
pytest py_tests/test_performance.py -v --benchmark-only

# Run specific method tests
pytest py_tests/test_performance.py::test_list_catalogs_light -v --benchmark-only
pytest py_tests/test_performance.py::test_download_heavy -v --benchmark-only

# Generate performance report
pytest py_tests/test_performance.py --benchmark-json=output.json
```

**Performance Varies Based On:**
- 🌐 Network latency and bandwidth
- 📦 Data size (KB vs GB)
- 🏋️ Concurrent load
- 🖥️ Client/server resources
- 📍 Geographic location

**These Values Represent:**
- Expected performance under typical conditions
- Baseline for performance regression testing
- SLA planning guidelines

**Metrics Explained:**
- **Latency (P95)**: 95th percentile response time in seconds - Light/Heavy scenarios
- **Throughput**: Operations per second or MB/sec for data transfer operations
- **Success Rate**: Expected percentage of successful operations (Light/Heavy)
- **Scalability**: Performance behavior under increasing load (Linear/Sub-linear/Constant)

| # | Method Name | Latency P95 (sec)<br>Light / Heavy | Throughput<br>(ops/sec or MB/s) | Success Rate (%)<br>Light / Heavy | Scalability<br>Pattern |
|---|-------------|-------------------------------------|----------------------------------|-----------------------------------|------------------------|
| **FUSION CLIENT METHODS** |
| 1 | `list_catalogs()` | 2.5 / 8.0 | 10 ops/sec | 100 / 95 | Constant |
| 2 | `catalog_resources()` | 2.0 / 6.0 | 12 ops/sec | 100 / 95 | Constant |
| 3 | `list_products()` | 2.5 / 7.0 | 10 ops/sec | 99 / 95 | Linear |
| 4 | `list_datasets()` | 2.5 / 10.0 | 8 ops/sec | 99 / 95 | Linear |
| 5 | `list_reports()` | 2.0 / 6.0 | 10 ops/sec | 99 / 95 | Constant |
| 6 | `list_report_attributes()` | 1.8 / 5.0 | 12 ops/sec | 99 / 98 | Linear |
| 7 | `dataset_resources()` | 0.8 / 3.0 | 15 ops/sec | 100 / 98 | Constant |
| 8 | `list_dataset_attributes()` | 0.9 / 4.0 | 12 ops/sec | 100 / 98 | Linear |
| 9 | `list_datasetmembers()` | 1.5 / 6.0 | 10 ops/sec | 99 / 95 | Linear |
| 10 | `datasetmember_resources()` | 1.0 / 4.0 | 12 ops/sec | 99 / 97 | Constant |
| 11 | `list_distributions()` | 1.2 / 4.5 | 10 ops/sec | 99 / 97 | Constant |
| 12 | `list_distribution_files()` | 1.5 / 5.0 | 8 ops/sec | 99 / 95 | Linear |
| 13 | `download()` | 8.0 / 110.0 | 5-15 MB/s | 100 / 95 | Sub-linear |
| 14 | `to_df()` | 7.0 / 100.0 | 5-12 MB/s | 100 / 95 | Sub-linear |
| 15 | `to_bytes()` | 5.0 / 80.0 | 8-15 MB/s | 100 / 97 | Sub-linear |
| 16 | `to_table()` | 7.5 / 105.0 | 5-12 MB/s | 100 / 95 | Sub-linear |
| 17 | `upload()` | 12.0 / 160.0 | 3-10 MB/s | 99 / 95 | Sub-linear |
| 18 | `from_bytes()` | 8.0 / 120.0 | 4-8 MB/s | 99 / 95 | Sub-linear |
| 19 | `list_dataset_lineage()` | 2.5 / 8.0 | 8 ops/sec | 99 / 95 | Linear |
| 20 | `create_dataset_lineage()` | 2.0 / 6.0 | 10 ops/sec | 99 / 97 | Linear |
| 21 | `list_product_dataset_mapping()` | 2.0 / 5.0 | 10 ops/sec | 99 / 95 | Linear |
| 22 | `delete_all_datasetmembers()` | 3.0 / 15.0 | 5 ops/sec | 99 / 95 | Linear |
| 23 | `delete_datasetmembers()` | 2.5 / 10.0 | 8 ops/sec | 99 / 95 | Linear |
| 24 | `list_datasetmembers_distributions()` | 2.0 / 8.0 | 8 ops/sec | 99 / 95 | Linear |
| 25 | `list_registered_attributes()` | 2.5 / 9.0 | 8 ops/sec | 99 / 95 | Linear |
| 26 | `list_attribute_lineage()` | 1.8 / 6.0 | 10 ops/sec | 99 / 97 | Linear |
| 27 | `list_business_terms_for_attribute()` | 1.5 / 5.0 | 12 ops/sec | 99 / 97 | Constant |
| 28 | `list_indexes()` | 2.0 / 6.0 | 10 ops/sec | 99 / 97 | Constant |
| 29 | `list_dataflows()` | 1.8 / 5.0 | 10 ops/sec | 99 / 97 | Constant |
| 30 | `listen_to_events()` | N/A (streaming) | N/A | 99 / 95 | N/A |
| 31 | `get_events()` | N/A (streaming) | N/A | 99 / 95 | N/A |
| 32 | `link_attributes_to_terms()` | 2.5 / 8.0 | 8 ops/sec | 99 / 95 | Linear |
| 33 | `get_fusion_filesystem()` | <0.05 (instant) | N/A | 100 / 100 | Constant |
| 34 | `get_fusion_vector_store_client()` | <0.05 (instant) | N/A | 100 / 100 | Constant |
| 35 | `get_async_fusion_vector_store_client()` | <0.05 (instant) | N/A | 100 / 100 | Constant |
| **DATASET METHODS** |
| 36 | `Dataset.create()` | 3.0 / 10.0 | 8 ops/sec | 99 / 95 | Linear |
| 37 | `Dataset.update()` | 2.5 / 8.0 | 10 ops/sec | 99 / 97 | Linear |
| 38 | `Dataset.delete()` | 2.0 / 7.0 | 10 ops/sec | 99 / 97 | Linear |
| 39 | `Dataset.add_to_product()` | 2.0 / 6.0 | 10 ops/sec | 99 / 97 | Constant |
| 40 | `Dataset.remove_from_product()` | 2.0 / 6.0 | 10 ops/sec | 99 / 97 | Constant |
| **PRODUCT METHODS** |
| 41 | `Product.create()` | 3.0 / 9.0 | 8 ops/sec | 99 / 95 | Linear |
| 42 | `Product.update()` | 2.5 / 7.0 | 10 ops/sec | 99 / 97 | Linear |
| 43 | `Product.delete()` | 2.0 / 6.0 | 10 ops/sec | 99 / 97 | Linear |
| **ATTRIBUTE METHODS** |
| 44 | `Attribute.update()` | 2.0 / 6.0 | 10 ops/sec | 99 / 97 | Linear |
| 45 | `Attribute.delete()` | 1.8 / 5.5 | 12 ops/sec | 99 / 97 | Linear |
| 46 | `Attribute.create_lineage()` | 2.0 / 6.0 | 10 ops/sec | 99 / 97 | Linear |
| 47 | `Attributes.create()` | 3.0 / 12.0 | 6 ops/sec | 99 / 95 | Linear |
| 48 | `Attributes.delete()` | 2.5 / 9.0 | 8 ops/sec | 99 / 95 | Linear |
| **REPORT METHODS** |
| 49 | `Report.create()` | 2.5 / 8.0 | 10 ops/sec | 99 / 97 | Linear |
| 50 | `Report.update()` | 2.0 / 6.5 | 10 ops/sec | 99 / 97 | Linear |
| 51 | `Report.update_fields()` | 1.8 / 5.5 | 12 ops/sec | 99 / 98 | Constant |
| 52 | `Report.delete()` | 2.0 / 6.0 | 10 ops/sec | 99 / 97 | Linear |
| 53 | `Report.link_attributes_to_terms()` | 2.5 / 8.0 | 8 ops/sec | 99 / 95 | Linear |
| **REPORT ATTRIBUTE METHODS** |
| 54 | `ReportAttribute.create()` | 2.0 / 6.0 | 10 ops/sec | 99 / 97 | Constant |
| 55 | `ReportAttribute.update()` | 1.8 / 5.5 | 12 ops/sec | 99 / 98 | Constant |
| 56 | `ReportAttribute.update_fields()` | 1.5 / 5.0 | 12 ops/sec | 99 / 98 | Constant |
| 57 | `ReportAttribute.delete()` | 2.0 / 6.0 | 10 ops/sec | 99 / 97 | Constant |
| 58 | `ReportAttributes.create()` | 3.0 / 10.0 | 6 ops/sec | 99 / 95 | Linear |
| 59 | `ReportAttributes.delete()` | 2.5 / 8.0 | 8 ops/sec | 99 / 95 | Linear |
| **DATAFLOW METHODS** |
| 60 | `Dataflow.create()` | 2.5 / 7.5 | 8 ops/sec | 99 / 97 | Linear |
| 61 | `Dataflow.update()` | 2.0 / 6.5 | 10 ops/sec | 99 / 97 | Linear |
| 62 | `Dataflow.update_fields()` | 1.8 / 5.5 | 12 ops/sec | 99 / 98 | Constant |
| 63 | `Dataflow.delete()` | 2.0 / 6.0 | 10 ops/sec | 99 / 97 | Linear |
| **DATA DEPENDENCY & MAPPING METHODS** |
| 64 | `DataDependency.link_attributes()` | 2.5 / 8.0 | 8 ops/sec | 99 / 95 | Linear |
| 65 | `DataDependency.unlink_attributes()` | 2.0 / 6.5 | 10 ops/sec | 99 / 97 | Linear |
| 66 | `DataMapping.link_attribute_to_term()` | 2.5 / 8.0 | 8 ops/sec | 99 / 95 | Linear |
| 67 | `DataMapping.unlink_attribute_from_term()` | 2.0 / 6.0 | 10 ops/sec | 99 / 97 | Linear |
| 68 | `DataMapping.update_attribute_to_term_kde_status()` | 2.0 / 6.5 | 10 ops/sec | 99 / 97 | Constant |

**Scalability Patterns Explained:**
- **Constant**: Performance remains stable regardless of load (e.g., single resource lookups)
- **Linear**: Performance degrades proportionally with load (e.g., pagination, batch operations)
- **Sub-linear**: Performance degrades less than proportionally (e.g., cached data, parallel processing)

**How to Interpret:**
- **Latency**: Lower is better. Light = small data/low concurrency, Heavy = large data/high concurrency
- **Throughput**: Higher is better. For data operations (download/upload), measured in MB/s; for metadata operations, in ops/sec
- **Success Rate**: Higher is better. Should be 95%+ even under heavy load
- **Scalability**: Indicates how well the method handles increasing load

**Testing These Metrics:**
Use the performance testing framework in `py_tests/test_performance.py` to validate these targets:
```bash
# Test specific method
pytest py_tests/test_performance.py -k "test_download" -v

# Test all methods
pytest py_tests/test_performance.py -v -m performance
```

---

## Method to API Endpoint Quick Reference Table

**Note on Root URLs:**
- `{root_url}` - Base Fusion API endpoint (e.g., `https://fusion.jpmorgan.com/api/v1/`)
- `{root_url_new}` - New lineage/metadata service endpoint (e.g., `https://fusion.jpmorgan.com/`)

The SDK automatically manages these root URLs based on the API being called.

| # | Method Name | API Endpoints Used | HTTP Method(s) |
|---|-------------|-------------------|----------------|
| **FUSION CLIENT METHODS** |
| 1 | `list_catalogs()` | `GET {root_url}catalogs/` | GET |
| 2 | `catalog_resources()` | `GET {root_url}catalogs/{catalog}` | GET |
| 3 | `list_products()` | `GET {root_url}catalogs/{catalog}/products` | GET |
| 4 | `list_datasets()` | `GET {root_url}catalogs/{catalog}/datasets/{dataset}` (exact match)<br>`GET {root_url}catalogs/{catalog}/datasets`<br>`GET {root_url}catalogs/{catalog}/productDatasets` (if product filter) | GET |
| 5 | `list_reports()` | `GET {root_url_new}api/corelineage-service/v1/reports/{id}` (single)<br>`POST {root_url_new}api/corelineage-service/v1/reports/list` (all) | GET, POST |
| 6 | `list_report_attributes()` | `GET {root_url_new}api/corelineage-service/v1/reports/{id}/attributes` | GET |
| 7 | `dataset_resources()` | `GET {root_url}catalogs/{catalog}/datasets/{dataset}` | GET |
| 8 | `list_dataset_attributes()` | `GET {root_url}catalogs/{catalog}/datasets/{dataset}/attributes` | GET |
| 9 | `list_datasetmembers()` | `GET {root_url}catalogs/{catalog}/datasets/{dataset}/datasetseries` | GET |
| 10 | `datasetmember_resources()` | `GET {root_url}catalogs/{catalog}/datasets/{dataset}/datasetseries/{series}` | GET |
| 11 | `list_distributions()` | `GET {root_url}catalogs/{catalog}/datasets/{dataset}/datasetseries/{series}/distributions` | GET |
| 12 | `list_distribution_files()` | `GET {root_url}catalogs/{catalog}/datasets/{dataset}/datasetseries/{series}/distributions/{format}/files` | GET |
| 13 | `download()` | `GET {root_url}catalogs/{catalog}/datasets/{dataset}` (access check)<br>`GET {root_url}catalogs/{catalog}/datasets/{dataset}/datasetseries`<br>`GET {root_url}catalogs/{catalog}/datasets/changes?datasets={dataset}`<br>`GET {root_url}catalogs/{catalog}/datasets/{dataset}/datasetseries/{series}/distributions/{format}/files`<br>`HEAD {root_url}.../operationType/download` (get file size)<br>`HEAD {rpath}` (get headers for parallel download planning)<br>`GET {root_url}.../files/operationType/download?file={name}` (per file)<br>`GET {root_url}...&downloadRange=bytes={start}-{end}` (parallel chunks)<br>`GET {root_url}.../operationType/download?downloadRange=bytes={start}-{end}` (FusionFile reads)<br>`GET {url}` with `Range: bytes={start}-{end}` header (cat/pagination) | GET, HEAD |
| 14 | `to_df()` | Same as `download()` + local file reading | GET, HEAD |
| 15 | `to_bytes()` | `GET {root_url}catalogs/{catalog}/datasets/{dataset}/datasetseries/{series}/distributions/{format}/files`<br>`GET {root_url}.../files/operationType/download?file={name}` | GET |
| 16 | `to_table()` | Same as `download()` + local file reading | GET, HEAD |
| 17 | `upload()` | **Single-part:**<br>`PUT {root_url}catalogs/{catalog}/datasets/{dataset}/datasetseries/{series}/distributions/{format}`<br>**Multipart:**<br>`POST {root_url}.../operationType/upload` (init)<br>`POST {root_url}.../operations/upload?operationId={id}&partNumber={n}` (chunks)<br>`POST {root_url}.../operations/upload?operationId={id}` (finalize)<br>**Validation:**<br>`GET {root_url}catalogs/{catalog}/datasets/{dataset}` (raw check) | PUT, POST, GET |
| 18 | `from_bytes()` | Same as `upload()` | PUT, POST, GET |
| 19 | `list_dataset_lineage()` | `GET {root_url}catalogs/{catalog}/datasets/{dataset}/lineage` | GET |
| 20 | `create_dataset_lineage()` | `POST {root_url}catalogs/{catalog}/datasets/{dataset}/lineage` | POST |
| 21 | `list_product_dataset_mapping()` | `GET {root_url}catalogs/{catalog}/productDatasets` | GET |
| 22 | `delete_all_datasetmembers()` | `DELETE {root_url}catalogs/{catalog}/datasets/{dataset}/datasetseries` | DELETE |
| 23 | `delete_datasetmembers()` | `DELETE {root_url}catalogs/{catalog}/datasets/{dataset}/datasetseries/{series}` (per member) | DELETE |
| 24 | `list_datasetmembers_distributions()` | `GET {root_url}catalogs/{catalog}/datasets/changes?datasets={dataset}` | GET |
| 25 | `list_registered_attributes()` | `GET {root_url}catalogs/{catalog}/attributes` | GET |
| 26 | `list_attribute_lineage()` | `POST {root_url_new}api/corelineage-service/v1/data-dependencies/source-attributes/query` | POST |
| 27 | `list_business_terms_for_attribute()` | `POST {root_url_new}api/corelineage-service/v1/data-mapping/term/query` | POST |
| 28 | `list_indexes()` | `GET {root_url}dataspaces/{catalog}/datasets/{kb}/indexes/` | GET |
| 29 | `list_dataflows()` | `GET {root_url_new}api/corelineage-service/v1/lineage/dataflows/{id}` | GET |
| 30 | `listen_to_events()` | `GET {root_url}catalogs/{catalog}/notifications/subscribe` (SSE) | GET (SSE) |
| 31 | `get_events()` | `GET {root_url}catalogs/{catalog}/notifications/subscribe` (SSE) | GET (SSE) |
| 32 | `link_attributes_to_terms()` | `POST {root_url_new}api/corelineage-service/v1/data-mapping/attributes/terms` | POST |
| 33 | `get_fusion_filesystem()` | No API calls (returns client object) | N/A |
| 34 | `get_fusion_vector_store_client()` | No API calls (returns client object) | N/A |
| 35 | `get_async_fusion_vector_store_client()` | No API calls (returns client object) | N/A |
| **DATASET METHODS** |
| 36 | `Dataset.create()` | `POST {root_url}catalogs/{catalog}/datasets/{identifier}` | POST |
| 37 | `Dataset.update()` | `PUT {root_url}catalogs/{catalog}/datasets/{identifier}` | PUT |
| 38 | `Dataset.delete()` | `DELETE {root_url}catalogs/{catalog}/datasets/{identifier}` | DELETE |
| 39 | `Dataset.add_to_product()` | `PUT {root_url}catalogs/{catalog}/productDatasets` | PUT |
| 40 | `Dataset.remove_from_product()` | `DELETE {root_url}catalogs/{catalog}/productDatasets/{product}/{dataset}` | DELETE |
| **PRODUCT METHODS** |
| 41 | `Product.create()` | `POST {root_url}catalogs/{catalog}/products/{identifier}` | POST |
| 42 | `Product.update()` | `PUT {root_url}catalogs/{catalog}/products/{identifier}` | PUT |
| 43 | `Product.delete()` | `DELETE {root_url}catalogs/{catalog}/products/{identifier}` | DELETE |
| **ATTRIBUTE METHODS** |
| 44 | `Attribute.update()` | `PUT {root_url}catalogs/{catalog}/datasets/{dataset}/attributes/{identifier}` | PUT |
| 45 | `Attribute.delete()` | `DELETE {root_url}catalogs/{catalog}/datasets/{dataset}/attributes/{identifier}` | DELETE |
| 46 | `Attribute.create_lineage()` | `POST {root_url}catalogs/{catalog}/attributes/lineage` | POST |
| 47 | `Attributes.create()` | **For dataset attrs:**<br>`PUT {root_url}catalogs/{catalog}/datasets/{dataset}/attributes`<br>**For registered attrs:**<br>`POST {root_url}catalogs/{catalog}/attributes` | PUT, POST |
| 48 | `Attributes.delete()` | `DELETE {root_url}catalogs/{catalog}/datasets/{dataset}/attributes/{identifier}` (per attr) | DELETE |
| **REPORT METHODS** |
| 49 | `Report.create()` | `POST {root_url_new}api/corelineage-service/v1/reports` | POST |
| 50 | `Report.update()` | `PUT {root_url_new}api/corelineage-service/v1/reports/{id}` | PUT |
| 51 | `Report.update_fields()` | `PATCH {root_url_new}api/corelineage-service/v1/reports/{id}` | PATCH |
| 52 | `Report.delete()` | `DELETE {root_url_new}api/corelineage-service/v1/reports/{id}` | DELETE |
| 53 | `Report.link_attributes_to_terms()` | `POST {root_url_new}api/corelineage-service/v1/data-mapping/attributes/terms` | POST |
| **REPORT ATTRIBUTE METHODS** |
| 54 | `ReportAttribute.create()` | `POST {root_url_new}api/corelineage-service/v1/reports/{id}/attributes` | POST |
| 55 | `ReportAttribute.update()` | `PUT {root_url_new}api/corelineage-service/v1/reports/{id}/attributes/{attr_id}` | PUT |
| 56 | `ReportAttribute.update_fields()` | `PATCH {root_url_new}api/corelineage-service/v1/reports/{id}/attributes/{attr_id}` | PATCH |
| 57 | `ReportAttribute.delete()` | `DELETE {root_url_new}api/corelineage-service/v1/reports/{id}/attributes` | DELETE |
| 58 | `ReportAttributes.create()` | `POST {root_url_new}api/corelineage-service/v1/reports/{id}/attributes` (bulk) | POST |
| 59 | `ReportAttributes.delete()` | `DELETE {root_url_new}api/corelineage-service/v1/reports/{id}/attributes` (bulk) | DELETE |
| **DATAFLOW METHODS** |
| 60 | `Dataflow.create()` | `POST {root_url_new}api/corelineage-service/v1/lineage/dataflows` | POST |
| 61 | `Dataflow.update()` | `PUT {root_url_new}api/corelineage-service/v1/lineage/dataflows/{id}` | PUT |
| 62 | `Dataflow.update_fields()` | `PATCH {root_url_new}api/corelineage-service/v1/lineage/dataflows/{id}` | PATCH |
| 63 | `Dataflow.delete()` | `DELETE {root_url_new}api/corelineage-service/v1/lineage/dataflows/{id}` | DELETE |
| **DATA DEPENDENCY & MAPPING METHODS** |
| 64 | `DataDependency.link_attributes()` | `POST {root_url_new}api/corelineage-service/v1/data-dependencies/attributes` | POST |
| 65 | `DataDependency.unlink_attributes()` | `DELETE {root_url_new}api/corelineage-service/v1/data-dependencies/attributes` | DELETE |
| 66 | `DataMapping.link_attribute_to_term()` | `POST {root_url_new}api/corelineage-service/v1/data-mapping/attributes/terms` | POST |
| 67 | `DataMapping.unlink_attribute_from_term()` | `DELETE {root_url_new}api/corelineage-service/v1/data-mapping/attributes/terms` | DELETE |
| 68 | `DataMapping.update_attribute_to_term_kde_status()` | `PATCH {root_url_new}api/corelineage-service/v1/data-mapping/attributes/terms` | PATCH |

**Legend:**
- All endpoints support pagination via `x-jpmc-next-token` header where applicable
- `{catalog}` defaults to "common" if not specified
- `{root_url}` = Base API URL (default: `https://fusion.jpmorgan.com/api/v1/`)
- `{root_url_new}` = Modified URL without `/api/v1/` suffix
- SSE = Server-Sent Events (long-lived connection)

---

## Fusion Client Public Methods

### 1. `list_catalogs(output=False)`
**Purpose**: Lists all available catalogs  
**API Endpoints Used**:
- `GET {root_url}catalogs/`
  - Via: `_call_for_dataframe()` → `handle_paginated_request()` → `session.get()`

---

### 2. `catalog_resources(catalog=None, output=False)`
**Purpose**: List resources within a catalog  
**API Endpoints Used**:
- `GET {root_url}catalogs/{catalog}`
  - Via: `_call_for_dataframe()` → `handle_paginated_request()` → `session.get()`

---

### 3. `list_products(contains=None, id_contains=False, catalog=None, output=False, max_results=-1, display_all_columns=False)`
**Purpose**: Get products in a catalog  
**API Endpoints Used**:
- `GET {root_url}catalogs/{catalog}/products`
  - Via: `_call_for_dataframe()` → `handle_paginated_request()` → `session.get()`

---

### 4. `list_datasets(contains=None, id_contains=False, product=None, catalog=None, output=False, max_results=-1, display_all_columns=False, status=None, dataset_type=None)`
**Purpose**: Get datasets in a catalog  
**API Endpoints Used**:
- `GET {root_url}catalogs/{catalog}/datasets/{dataset}` (for exact match)
  - Via: `session.get()`
- `GET {root_url}catalogs/{catalog}/datasets`
  - Via: `_call_for_dataframe()` → `handle_paginated_request()` → `session.get()`
- `GET {root_url}catalogs/{catalog}/productDatasets` (if filtering by product)
  - Via: `_call_for_dataframe()` → `handle_paginated_request()` → `session.get()`

---

### 5. `list_reports(report_id=None, output=False, display_all_columns=False)`
**Purpose**: Retrieve reports from the system  
**API Endpoints Used**:
- `GET {root_url_new}/api/corelineage-service/v1/reports/{report_id}` (if report_id provided)
  - Via: `session.get()`
- `POST {root_url_new}/api/corelineage-service/v1/reports/list` (for all reports)
  - Via: `session.post()`

---

### 6. `list_report_attributes(report_id, output=False, display_all_columns=False)`
**Purpose**: Retrieve attributes of a specific report  
**API Endpoints Used**:
- `GET {root_url_new}/api/corelineage-service/v1/reports/{report_id}/attributes`
  - Via: `session.get()`

---

### 7. `dataset_resources(dataset, catalog=None, output=False)`
**Purpose**: List resources available for a dataset  
**API Endpoints Used**:
- `GET {root_url}catalogs/{catalog}/datasets/{dataset}`
  - Via: `_call_for_dataframe()` → `handle_paginated_request()` → `session.get()`

---

### 8. `list_dataset_attributes(dataset, catalog=None, output=False, display_all_columns=False)`
**Purpose**: Returns attributes in a dataset  
**API Endpoints Used**:
- `GET {root_url}catalogs/{catalog}/datasets/{dataset}/attributes`
  - Via: `_call_for_dataframe()` → `handle_paginated_request()` → `session.get()`

---

### 9. `list_datasetmembers(dataset, catalog=None, output=False, max_results=-1)`
**Purpose**: List available members in the dataset series  
**API Endpoints Used**:
- `GET {root_url}catalogs/{catalog}/datasets/{dataset}/datasetseries`
  - Via: `_call_for_dataframe()` → `handle_paginated_request()` → `session.get()`

---

### 10. `datasetmember_resources(dataset, series, catalog=None, output=False)`
**Purpose**: List resources for a datasetseries member  
**API Endpoints Used**:
- `GET {root_url}catalogs/{catalog}/datasets/{dataset}/datasetseries/{series}`
  - Via: `_call_for_dataframe()` → `handle_paginated_request()` → `session.get()`

---

### 11. `list_distributions(dataset, series, catalog=None, output=False)`
**Purpose**: List available distributions  
**API Endpoints Used**:
- `GET {root_url}catalogs/{catalog}/datasets/{dataset}/datasetseries/{series}/distributions`
  - Via: `_call_for_dataframe()` → `handle_paginated_request()` → `session.get()`

---

### 12. `list_distribution_files(dataset, series, file_format='parquet', catalog=None, output=False, max_results=-1)`
**Purpose**: List available files for a specific dataset distribution  
**API Endpoints Used**:
- `GET {root_url}catalogs/{catalog}/datasets/{dataset}/datasetseries/{series}/distributions/{file_format}/files`
  - Via: `_call_for_dataframe()` → `handle_paginated_request()` → `session.get()`

---

### 13. `download(dataset, dt_str='latest', dataset_format='parquet', catalog=None, n_par=None, show_progress=True, force_download=False, download_folder=None, return_paths=False, partitioning=None, preserve_original_name=False, file_name=None)`
**Purpose**: Downloads dataset distributions to disk  
**API Endpoints Used**:
- `GET {root_url}catalogs/{catalog}/datasets/{dataset}` (access check)
  - Via: `session.get()`
- `GET {root_url}catalogs/{catalog}/datasets/{dataset}/datasetseries` (via `list_datasetmembers()`)
  - Via: `_call_for_dataframe()` → `handle_paginated_request()` → `session.get()`
- `GET {root_url}catalogs/{catalog}/datasets/changes?datasets={dataset}` (via `list_datasetmembers_distributions()`)
  - Via: `handle_paginated_request()` → `session.get()`
- **Runtime-constructed download URLs** (via `distribution_to_url()` and `FusionHTTPFileSystem`):
  - For regular series: `GET {root_url}catalogs/{catalog}/datasets/{dataset}/datasetseries/{series}/distributions/{format}/files/operationType/download?file={file_name}`
  - For sample data: `GET {root_url}catalogs/{catalog}/datasets/{dataset}/sample/distributions/{format}`
  - Executed through `FusionHTTPFileSystem._get()` with chunked/parallel downloads
  - Optional range downloads (two patterns):
    - Pattern 1: `GET {url}&downloadRange=bytes={start}-{end-1}` (when URL has existing query params)
    - Pattern 2: `GET {url}/operationType/download?downloadRange=bytes={start}-{end-1}` (for FusionFile reads)

---

### 14. `to_df(dataset, dt_str='latest', dataset_format='parquet', catalog=None, n_par=None, show_progress=True, columns=None, filters=None, force_download=False, download_folder=None, dataframe_type='pandas', file_name=None, **kwargs)`
**Purpose**: Downloads and returns data as a dataframe  
**API Endpoints Used**:
- All APIs from `download()` method (see above)
- Additional file reading operations (no direct API calls)

---

### 15. `to_bytes(dataset, series_member, dataset_format='parquet', catalog=None, file_name=None)`
**Purpose**: Returns dataset distribution as bytes object  
**API Endpoints Used**:
- `GET {root_url}catalogs/{catalog}/datasets/{dataset}/datasetseries/{series}/distributions/{format}/files` (if no file_name)
  - Via: `list_distribution_files()` → `_call_for_dataframe()` → `handle_paginated_request()` → `session.get()`
- **Runtime-constructed download URL** (for each file):
  - `GET {root_url}catalogs/{catalog}/datasets/{dataset}/datasetseries/{series}/distributions/{format}/files/operationType/download?file={file_name}`
  - Via: `_call_for_bytes_object()` → `session.get()`
  - Returns response content as `BytesIO`

---

### 16. `to_table(dataset, dt_str='latest', dataset_format='parquet', catalog=None, n_par=None, show_progress=True, columns=None, filters=None, force_download=False, download_folder=None, file_name=None, **kwargs)`
**Purpose**: Downloads and returns data as PyArrow table  
**API Endpoints Used**:
- All APIs from `download()` method (see above)
- Additional file reading operations (no direct API calls)

---

### 17. `upload(path, dataset=None, dt_str='latest', catalog=None, n_par=None, show_progress=True, return_paths=False, multipart=True, chunk_size=5*2**20, from_date=None, to_date=None, preserve_original_name=False, additional_headers=None)`
**Purpose**: Uploads files to Fusion  
**API Endpoints Used**:
- **Runtime-constructed upload URLs** (via `distribution_to_url()` and `FusionHTTPFileSystem`):
  - For single-part upload: 
    - `PUT {root_url}catalogs/{catalog}/datasets/{dataset}/datasetseries/{series}/distributions/{format}`
    - Via: `FusionHTTPFileSystem.put()` → `_put_file()` → `session.put()`
    - Headers: `Content-Type`, `x-jpmc-distribution-created-date`, `x-jpmc-distribution-from-date`, `x-jpmc-distribution-to-date`, `Digest` (SHA-256), `File-Name` (optional)
  - For multipart upload:
    - Initialize: `POST {root_url}catalogs/{catalog}/datasets/{dataset}/datasetseries/{series}/distributions/{format}/operationType/upload`
      - Returns `operationId`
    - Upload chunks: `POST {root_url}catalogs/{catalog}/datasets/{dataset}/datasetseries/{series}/distributions/{format}/operations/upload?operationId={id}&partNumber={n}` (for each chunk)
      - Headers per chunk: `Content-Type: application/octet-stream`, `Digest` (SHA-256 of chunk)
    - Finalize: `POST {root_url}catalogs/{catalog}/datasets/{dataset}/datasetseries/{series}/distributions/{format}/operations/upload?operationId={id}`
      - Body: `{"parts": [response_objects]}`
  - For raw dataset validation: `GET {catalog}/datasets/{dataset}` (to check `isRawData` flag)
    - Via: `FusionHTTPFileSystem.cat()` → async/sync HTTP GET

---

### 18. `from_bytes(data, dataset, series_member='latest', catalog=None, distribution='parquet', show_progress=True, multipart=True, return_paths=False, chunk_size=5*2**20, from_date=None, to_date=None, file_name=None, **kwargs)`
**Purpose**: Uploads data from memory  
**API Endpoints Used**:
- `GET {catalog}/datasets/{dataset}` (to check if raw data)
  - Via: `FusionHTTPFileSystem.cat()` → async/sync HTTP GET
- **Runtime-constructed upload URLs** (same as `upload()` method above):
  - For single-part: `PUT {root_url}catalogs/{catalog}/datasets/{dataset}/datasetseries/{series}/distributions/{format}`
  - For multipart: Same three-step process as in `upload()` (initialize → upload chunks → finalize)

---

### 19. `list_dataset_lineage(dataset_id, catalog=None, output=False, max_results=-1)`
**Purpose**: List upstream and downstream lineage of dataset  
**API Endpoints Used**:
- `GET {root_url}catalogs/{catalog}/datasets/{dataset_id}/lineage`
  - Via: `handle_paginated_request()` → `session.get()`

---

### 20. `create_dataset_lineage(base_dataset, source_dataset_catalog_mapping, catalog=None, return_resp_obj=False)`
**Purpose**: Upload lineage to a dataset  
**API Endpoints Used**:
- `POST {root_url}catalogs/{catalog}/datasets/{base_dataset}/lineage`
  - Via: `session.post()`

---

### 21. `list_product_dataset_mapping(dataset=None, product=None, catalog=None)`
**Purpose**: Get product to dataset linking  
**API Endpoints Used**:
- `GET {root_url}catalogs/{catalog}/productDatasets`
  - Via: `_call_for_dataframe()` → `handle_paginated_request()` → `session.get()`

---

### 22. `delete_all_datasetmembers(dataset, catalog=None, return_resp_obj=False)`
**Purpose**: Delete all dataset members within a dataset  
**API Endpoints Used**:
- `DELETE {root_url}catalogs/{catalog}/datasets/{dataset}/datasetseries`
  - Via: `session.delete()`

---

### 23. `delete_datasetmembers(dataset, series_members, catalog=None, return_resp_obj=False)`
**Purpose**: Delete specific dataset members  
**API Endpoints Used**:
- `DELETE {root_url}catalogs/{catalog}/datasets/{dataset}/datasetseries/{series_member}` (for each member)
  - Via: `session.delete()`

---

### 24. `list_datasetmembers_distributions(dataset, catalog=None)`
**Purpose**: List distributions of dataset members  
**API Endpoints Used**:
- `GET {root_url}catalogs/{catalog}/datasets/changes?datasets={dataset}`
  - Via: `handle_paginated_request()` → `session.get()`

---

### 25. `list_registered_attributes(catalog=None, output=False, display_all_columns=False)`
**Purpose**: Returns list of attributes in a catalog  
**API Endpoints Used**:
- `GET {root_url}catalogs/{catalog}/attributes`
  - Via: `_call_for_dataframe()` → `handle_paginated_request()` → `session.get()`

---

### 26. `list_attribute_lineage(entity_type, entity_identifier, attribute_identifier, data_space=None, output=False)`
**Purpose**: List source attributes linked to a target attribute  
**API Endpoints Used**:
- `POST {root_url_new}/api/corelineage-service/v1/data-dependencies/source-attributes/query`
  - Via: `session.post()`

---

### 27. `list_business_terms_for_attribute(entity_type, entity_identifier, attribute_identifier, data_space=None, output=False)`
**Purpose**: List business terms linked to an attribute  
**API Endpoints Used**:
- `POST {root_url_new}/api/corelineage-service/v1/data-mapping/term/query`
  - Via: `session.post()`

---

### 28. `list_indexes(knowledge_base, catalog=None, show_details=False)`
**Purpose**: List indexes in a knowledge base  
**API Endpoints Used**:
- `GET {root_url}dataspaces/{catalog}/datasets/{knowledge_base}/indexes/`
  - Via: `session.get()`

---

### 29. `list_dataflows(id_contains, output=False)`
**Purpose**: Retrieve a single dataflow from the system  
**API Endpoints Used**:
- `GET {root_url_new}/api/corelineage-service/v1/lineage/dataflows/{id_contains}`
  - Via: `session.get()`

---

### 30. `listen_to_events(last_event_id=None, catalog=None, url=None)`
**Purpose**: Run SSE listener in the background  
**API Endpoints Used**:
- `GET {url}catalogs/{catalog}/notifications/subscribe` (Server-Sent Events)
  - Via: Async SSE client connection

---

### 31. `get_events(last_event_id=None, catalog=None, in_background=True, url=None)`
**Purpose**: Run SSE listener and print events  
**API Endpoints Used**:
- `GET {url}catalogs/{catalog}/notifications/subscribe` (Server-Sent Events)
  - Via: SSEClient or async connection

---

### 32. `link_attributes_to_terms(mappings, return_resp_obj=False)`
**Purpose**: Link attributes to business terms  
**API Endpoints Used**:
- Delegates to `Report.link_attributes_to_terms()` which uses:
- `POST {root_url_new}/api/corelineage-service/v1/data-mapping/attributes/terms`
  - Via: `session.post()`

---

### 33. `get_fusion_filesystem(**kwargs)`
**Purpose**: Retrieve Fusion file system instance  
**API Endpoints Used**:
- No direct API calls (returns filesystem object)

---

### 34. `get_fusion_vector_store_client(knowledge_base, catalog=None)`
**Purpose**: Returns Fusion Embeddings Search client  
**API Endpoints Used**:
- No direct API calls (returns OpenSearch client)

---

### 35. `get_async_fusion_vector_store_client(knowledge_base, catalog=None)`
**Purpose**: Returns async Fusion Embeddings Search client  
**API Endpoints Used**:
- No direct API calls (returns AsyncOpenSearch client)

---

## Dataset Object Methods

### 36. `Dataset.create(catalog=None, dataset=None, client=None, return_resp_obj=False)`
**Purpose**: Create a new dataset  
**API Endpoints Used**:
- `GET {root_url}catalogs/{catalog}/datasets` (via `from_catalog()`)
  - Via: `handle_paginated_request()` → `session.get()`
- `POST {root_url}catalogs/{catalog}/datasets/{dataset_identifier}`
  - Via: `session.post()`

---

### 37. `Dataset.update(catalog=None, dataset=None, client=None, return_resp_obj=False)`
**Purpose**: Update an existing dataset  
**API Endpoints Used**:
- `PUT {root_url}catalogs/{catalog}/datasets/{dataset_identifier}`
  - Via: `session.put()`

---

### 38. `Dataset.delete(catalog=None, client=None, return_resp_obj=False)`
**Purpose**: Delete a dataset  
**API Endpoints Used**:
- `DELETE {root_url}catalogs/{catalog}/datasets/{dataset_identifier}`
  - Via: `session.delete()`

---

### 39. `Dataset.add_to_product(product, catalog=None, client=None, return_resp_obj=False)`
**Purpose**: Link dataset to a product  
**API Endpoints Used**:
- `PUT {root_url}catalogs/{catalog}/productDatasets`
  - Via: `session.put()`

---

### 40. `Dataset.remove_from_product(product, catalog=None, client=None, return_resp_obj=False)`
**Purpose**: Unlink dataset from a product  
**API Endpoints Used**:
- `DELETE {root_url}catalogs/{catalog}/productDatasets/{product}/{dataset}`
  - Via: `session.delete()`

---

## Product Object Methods

### 41. `Product.create(catalog=None, client=None, return_resp_obj=False)`
**Purpose**: Create a new product  
**API Endpoints Used**:
- `GET {root_url}catalogs/{catalog}/products` (via `from_catalog()`)
  - Via: `handle_paginated_request()` → `session.get()`
- `POST {root_url}catalogs/{catalog}/products/{product_identifier}`
  - Via: `session.post()`

---

### 42. `Product.update(catalog=None, client=None, return_resp_obj=False)`
**Purpose**: Update an existing product  
**API Endpoints Used**:
- `PUT {root_url}catalogs/{catalog}/products/{product_identifier}`
  - Via: `session.put()`

---

### 43. `Product.delete(catalog=None, client=None, return_resp_obj=False)`
**Purpose**: Delete a product  
**API Endpoints Used**:
- `DELETE {root_url}catalogs/{catalog}/products/{product_identifier}`
  - Via: `session.delete()`

---

## Attribute/Attributes Object Methods

### 44. `Attribute.update(dataset, catalog=None, client=None, return_resp_obj=False)`
**Purpose**: Update a single attribute  
**API Endpoints Used**:
- `PUT {root_url}catalogs/{catalog}/datasets/{dataset}/attributes/{attribute_identifier}`
  - Via: `session.put()`

---

### 45. `Attribute.delete(dataset, catalog=None, client=None, return_resp_obj=False)`
**Purpose**: Delete a single attribute  
**API Endpoints Used**:
- `DELETE {root_url}catalogs/{catalog}/datasets/{dataset}/attributes/{attribute_identifier}`
  - Via: `session.delete()`

---

### 46. `Attribute.create_lineage(source_dataset, source_attribute, target_dataset, target_attribute, catalog=None, client=None, return_resp_obj=False)`
**Purpose**: Create attribute lineage  
**API Endpoints Used**:
- `POST {root_url}catalogs/{catalog}/attributes/lineage`
  - Via: `session.post()`

---

### 47. `Attributes.create(dataset, catalog=None, dataset_only=False, client=None)`
**Purpose**: Create multiple attributes  
**API Endpoints Used**:
- `GET {root_url}catalogs/{catalog}/datasets/{dataset}/attributes` (via `from_dataset()`)
  - Via: `handle_paginated_request()` → `session.get()`
- For dataset attributes:
  - `PUT {root_url}catalogs/{catalog}/datasets/{dataset}/attributes`
    - Via: `session.put()`
- For registered attributes:
  - `POST {root_url}catalogs/{catalog}/attributes` (for each attribute)
    - Via: `session.post()`

---

### 48. `Attributes.delete(dataset, catalog=None, client=None)`
**Purpose**: Delete multiple attributes  
**API Endpoints Used**:
- `DELETE {root_url}catalogs/{catalog}/datasets/{dataset}/attributes/{attribute_identifier}` (for each)
  - Via: `session.delete()`

---

## Report Object Methods

### 49. `Report.create(client=None, return_resp_obj=False)`
**Purpose**: Create a new report  
**API Endpoints Used**:
- `POST {root_url_new}/api/corelineage-service/v1/reports`
  - Via: `session.post()`

---

### 50. `Report.update(client=None, return_resp_obj=False)`
**Purpose**: Update an existing report  
**API Endpoints Used**:
- `PUT {root_url_new}/api/corelineage-service/v1/reports/{report_id}`
  - Via: `session.put()`

---

### 51. `Report.update_fields(client=None, return_resp_obj=False, **fields)`
**Purpose**: Partially update report fields  
**API Endpoints Used**:
- `PATCH {root_url_new}/api/corelineage-service/v1/reports/{report_id}`
  - Via: `session.patch()`

---

### 52. `Report.delete(client=None, return_resp_obj=False)`
**Purpose**: Delete a report  
**API Endpoints Used**:
- `DELETE {root_url_new}/api/corelineage-service/v1/reports/{report_id}`
  - Via: `session.delete()`

---

### 53. `Report.link_attributes_to_terms(mappings, client=None, return_resp_obj=False)` (static)
**Purpose**: Link attributes to business terms  
**API Endpoints Used**:
- `POST {root_url_new}/api/corelineage-service/v1/data-mapping/attributes/terms`
  - Via: `session.post()`

---

## ReportAttribute/ReportAttributes Object Methods

### 54. `ReportAttribute.create(report_id, client=None, return_resp_obj=False)`
**Purpose**: Create a single report attribute  
**API Endpoints Used**:
- `POST {root_url_new}/api/corelineage-service/v1/reports/{report_id}/attributes`
  - Via: `session.post()`

---

### 55. `ReportAttribute.update(report_id, client=None, return_resp_obj=False)`
**Purpose**: Update a single report attribute  
**API Endpoints Used**:
- `PUT {root_url_new}/api/corelineage-service/v1/reports/{report_id}/attributes/{attribute_id}`
  - Via: `session.put()`

---

### 56. `ReportAttribute.update_fields(report_id, client=None, return_resp_obj=False, **fields)`
**Purpose**: Partially update report attribute fields  
**API Endpoints Used**:
- `PATCH {root_url_new}/api/corelineage-service/v1/reports/{report_id}/attributes/{attribute_id}`
  - Via: `session.patch()`

---

### 57. `ReportAttribute.delete(report_id, client=None, return_resp_obj=False)`
**Purpose**: Delete a single report attribute  
**API Endpoints Used**:
- `DELETE {root_url_new}/api/corelineage-service/v1/reports/{report_id}/attributes`
  - Via: `session.delete()` with JSON payload containing attribute ID

---

### 58. `ReportAttributes.create(report_id, client=None)`
**Purpose**: Create multiple report attributes  
**API Endpoints Used**:
- `POST {root_url_new}/api/corelineage-service/v1/reports/{report_id}/attributes` (bulk)
  - Via: `session.post()`

---

### 59. `ReportAttributes.delete(report_id, client=None)`
**Purpose**: Delete multiple report attributes  
**API Endpoints Used**:
- `DELETE {root_url_new}/api/corelineage-service/v1/reports/{report_id}/attributes` (bulk)
  - Via: `session.delete()` with JSON payload

---

## Dataflow Object Methods

### 60. `Dataflow.create(client=None, return_resp_obj=False)`
**Purpose**: Create a new dataflow  
**API Endpoints Used**:
- `POST {root_url_new}/api/corelineage-service/v1/lineage/dataflows`
  - Via: `session.post()`

---

### 61. `Dataflow.update(client=None, return_resp_obj=False)`
**Purpose**: Update an existing dataflow  
**API Endpoints Used**:
- `PUT {root_url_new}/api/corelineage-service/v1/lineage/dataflows/{dataflow_id}`
  - Via: `session.put()`

---

### 62. `Dataflow.update_fields(client=None, return_resp_obj=False, **fields)`
**Purpose**: Partially update dataflow fields  
**API Endpoints Used**:
- `PATCH {root_url_new}/api/corelineage-service/v1/lineage/dataflows/{dataflow_id}`
  - Via: `session.patch()`

---

### 63. `Dataflow.delete(client=None, return_resp_obj=False)`
**Purpose**: Delete a dataflow  
**API Endpoints Used**:
- `DELETE {root_url_new}/api/corelineage-service/v1/lineage/dataflows/{dataflow_id}`
  - Via: `session.delete()`

---

## Data Dependency & Mapping Methods

### 64. `DataDependency.link_attributes(mappings, client=None, return_resp_obj=False)`
**Purpose**: Link source attributes to target attribute (create attribute dependencies)  
**API Endpoints Used**:
- `POST {root_url_new}/api/corelineage-service/v1/data-dependencies/attributes`
  - Via: `session.post()`

---

### 65. `DataDependency.unlink_attributes(mappings, client=None, return_resp_obj=False)`
**Purpose**: Unlink source attributes from target attribute (delete attribute dependencies)  
**API Endpoints Used**:
- `DELETE {root_url_new}/api/corelineage-service/v1/data-dependencies/attributes`
  - Via: `session.delete()`

---

### 66. `DataMapping.link_attribute_to_term(mappings, client=None, return_resp_obj=False)`
**Purpose**: Link attributes to business terms  
**API Endpoints Used**:
- `POST {root_url_new}/api/corelineage-service/v1/data-mapping/attributes/terms`
  - Via: `session.post()`

---

### 67. `DataMapping.unlink_attribute_from_term(mappings, client=None, return_resp_obj=False)`
**Purpose**: Unlink attributes from business terms  
**API Endpoints Used**:
- `DELETE {root_url_new}/api/corelineage-service/v1/data-mapping/attributes/terms`
  - Via: `session.delete()`

---

### 68. `DataMapping.update_attribute_to_term_kde_status(mappings, client=None, return_resp_obj=False)`
**Purpose**: Update KDE (Key Data Element) status for attribute-to-term mappings  
**API Endpoints Used**:
- `PATCH {root_url_new}/api/corelineage-service/v1/data-mapping/attributes/terms`
  - Via: `session.patch()`

---

## Helper Methods (Not Directly Exposed but Used Internally)

### Internal Helper: `handle_paginated_request(session, url, headers=None)`
**Purpose**: Handles paginated API responses  
**API Endpoints Used**:
- `GET {url}` (with x-jpmc-next-token header for pagination)
  - Via: `session.get()`

---

### Internal Helper: `_call_for_dataframe(url, session)`
**Purpose**: Calls API and returns DataFrame  
**API Endpoints Used**:
- Uses `handle_paginated_request()` which calls `session.get()`

---

### Internal Helper: `_call_for_bytes_object(url, session)`
**Purpose**: Calls API and returns bytes  
**API Endpoints Used**:
- Direct `session.get()`

---

## Summary Statistics

**Total Public Methods Exposed**: 68+  
**Total Unique API Endpoints**: 50+

### API Endpoint Categories:
1. **Catalog APIs**: 3 endpoints
2. **Product APIs**: 4 endpoints  
3. **Dataset APIs**: 12 endpoints
4. **Attribute APIs**: 6 endpoints
5. **Distribution/Download APIs**: 5 endpoints
6. **Upload APIs**: Multiple (via FusionHTTPFileSystem)
7. **Lineage APIs**: 3 endpoints (dataset level)
8. **Report APIs**: 4 endpoints (corelineage-service)
9. **Report Attribute APIs**: 4 endpoints (corelineage-service)
10. **Dataflow APIs**: 4 endpoints (corelineage-service)
11. **Data Dependency APIs**: 2 endpoints (corelineage-service)
12. **Data Mapping APIs**: 4 endpoints (corelineage-service)
13. **Events/Notifications APIs**: 1 endpoint (SSE)
14. **Embeddings/Vector Store APIs**: 1 endpoint

### HTTP Methods Used:
- **GET**: Used for listing, retrieving, downloading
- **HEAD**: Used for checking file metadata and size before download
- **POST**: Used for creating resources and queries
- **PUT**: Used for updating entire resources
- **PATCH**: Used for partial updates
- **DELETE**: Used for removing resources

### Root URL Patterns:
1. `{root_url}` - Standard API v1 endpoints
2. `{root_url_new}/api/corelineage-service/v1/` - Core lineage service endpoints
3. SSE endpoints for real-time notifications
4. Download/upload URLs with special formatting

---

---

## Runtime-Constructed API Endpoints

The following endpoints are dynamically constructed at runtime based on method parameters:

### Download Endpoints (via `distribution_to_url()`)

**Standard Distribution Download**:
```
GET {root_url}catalogs/{catalog}/datasets/{dataset}/datasetseries/{series}/distributions/{format}/files/operationType/download?file={file_name}
```
- Used by: `download()`, `to_df()`, `to_table()`, `to_bytes()`
- Constructed when: `is_download=True` and specific file_name provided
- Query parameters: `file` (specific filename to download)

**Sample Data Download**:
```
GET {root_url}catalogs/{catalog}/datasets/{dataset}/sample/distributions/{format}
```
- Used by: `download()` when `dt_str="sample"`
- No authentication required for public samples

**Range Download** (two patterns depending on context):

Pattern 1 - Query parameter append (for parallel chunk downloads):
```
GET {url}&downloadRange=bytes={start}-{end-1}
```
- Used by: `FusionHTTPFileSystem._fetch_range()` in `_download_single_file_async()`
- Context: URL already has query parameters (e.g., `?file=xyz`), so uses `&` to append
- Enables parallel chunked downloads for large files
- Each thread downloads a specific byte range

Pattern 2 - Direct range query (for FusionFile reads):
```
GET {url}/operationType/download?downloadRange=bytes={start}-{end-1}
```
- Used by: `FusionFile.async_fetch_range()` for buffered file reads
- Context: Adds both operation type and range parameter together
- Used when reading file-like objects with specific byte ranges
- Supports partial content responses (HTTP 206)

Pattern 3 - Standard HTTP Range header (for cat/pagination):
```
GET {url}
Headers: Range: bytes={start}-{end-1}
```
- Used by: `FusionFile._fetch_range_with_headers()` and `_async_fetch_range_with_headers()`
- Context: Used by `cat()` method for fetching file contents with pagination support
- Standard HTTP Range header per RFC 7233
- Also used for metadata retrieval with pagination tokens

**HEAD Request for File Metadata**:
```
HEAD {url}/operationType/download
```
- Used by: `FusionHTTPFileSystem._ls_real()` when checking distribution file sizes
- Returns: `Content-Length` header with file size
- Context: Used to determine file size before download operations

```
HEAD {rpath}
```
- Used by: `FusionHTTPFileSystem.download()` via `get_headers()`
- Returns: Response headers including `Content-Length` for download planning
- Enables parallel download decision-making based on file size

---

### Upload Endpoints (via `FusionHTTPFileSystem.put()`)

**Single-Part Upload**:
```
PUT {root_url}catalogs/{catalog}/datasets/{dataset}/datasetseries/{series}/distributions/{format}
```
- Used when: File size ≤ chunk_size OR multipart=False
- Headers required:
  - `Content-Type: application/octet-stream`
  - `x-jpmc-distribution-created-date: YYYY-MM-DD`
  - `x-jpmc-distribution-from-date: YYYY-MM-DD`
  - `x-jpmc-distribution-to-date: YYYY-MM-DD`
  - `Digest: SHA-256={base64_hash}`
  - `File-Name: {name}` (optional)

**Multipart Upload** (3-step process):

Step 1 - Initialize:
```
POST {root_url}catalogs/{catalog}/datasets/{dataset}/datasetseries/{series}/distributions/{format}/operationType/upload
```
- Headers: Same as single-part but `Content-Type: application/json`
- Returns: `{"operationId": "uuid"}`

Step 2 - Upload Chunks (repeated for each chunk):
```
POST {root_url}catalogs/{catalog}/datasets/{dataset}/datasetseries/{series}/distributions/{format}/operations/upload?operationId={uuid}&partNumber={n}
```
- Headers per chunk:
  - `Content-Type: application/octet-stream`
  - `Digest: SHA-256={chunk_hash}`
- Body: Raw chunk bytes
- Returns: Chunk response object

Step 3 - Finalize:
```
POST {root_url}catalogs/{catalog}/datasets/{dataset}/datasetseries/{series}/distributions/{format}/operations/upload?operationId={uuid}
```
- Headers: Same as Step 1 but with combined `Digest: SHA-256={tree_hash}`
- Body: `{"parts": [chunk_response_1, chunk_response_2, ...]}`

**Raw Dataset Check** (before upload):
```
GET {root_url}catalogs/{catalog}/datasets/{dataset}
```
- Used to determine if dataset is raw (`isRawData` flag)
- Via: `FusionHTTPFileSystem.cat()`

---

### Filesystem Catalog Endpoints (via `FusionHTTPFileSystem`)

**List Resources with Pagination**:
```
GET {root_url}catalogs/{catalog}/datasets/{dataset}/datasetseries
```
- Headers: `x-jpmc-next-token: {token}` (for subsequent pages)
- Returns: `{"resources": [...], ...}`
- Automatically handles pagination in `_ls_real()` and `_changes()`

**Changes API** (for dataset distributions):
```
GET {root_url}catalogs/{catalog}/datasets/changes?datasets={dataset}
```
- Headers: `x-jpmc-next-token: {token}` (for pagination)
- Used by: `list_datasetmembers_distributions()`
- Returns: Full change history with distribution metadata

---

### URL Construction Helper Functions

**`distribution_to_url(root_url, dataset, datasetseries, file_format, catalog, is_download, file_name)`**:
- Returns properly formatted distribution URLs
- Handles special cases: sample data, download operations, file-specific URLs

**`distribution_to_filename(download_folder, dataset, datasetseries, file_format, catalog, partitioning, file_name)`**:
- Constructs local file paths for downloads
- Supports Hive-style partitioning: `{folder}/{catalog}/{dataset}/{series}/`
- Default naming: `{dataset}__{catalog}__{series}.{format}`
- Preserves original filename when `preserve_original_name=True`

**`file_name_to_url(file_name, dataset, catalog, is_download)`**:
- Constructs URLs from flattened filenames (e.g., `folder__subfolder__file.ext`)
- Uses filename components as series member identifier

**`path_to_url(path, is_raw, is_download)`**:
- Parses filename to extract catalog, dataset, date, format
- Auto-detects raw datasets and adjusts format accordingly

---

### Authentication & Session Management

All API calls use:
- **Bearer token authentication**: `Authorization: Bearer {token}`
- **Token refresh**: Automatic via `FusionOAuthAdapter`
- **Retry logic**: Configurable retries with backoff (default: 5 retries)
- **Proxy support**: Via `FusionCredentials.proxies`

Headers automatically injected:
- `Authorization: Bearer {access_token}`
- Token refreshed before expiry via OAuth2 flow

---

### Pagination Implementation

**Automatic Pagination Handler** (`handle_paginated_request()`):
```python
# First request
GET {url}
Response Headers: x-jpmc-next-token: {next_token}

# Subsequent requests (automatically repeated until no token)
GET {url}
Request Headers: x-jpmc-next-token: {previous_token}
Response Headers: x-jpmc-next-token: {next_token}  # if more pages exist
```

**Merged Response Structure**:
- All list fields (e.g., `resources`, `datasets`, `distributions`) are automatically concatenated
- Non-list fields taken from first response
- Implemented in `_merge_responses()` helper

**Methods Using Pagination**:
- All `list_*` methods that use `_call_for_dataframe()`
- `FusionHTTPFileSystem._changes()`
- `FusionHTTPFileSystem._ls_real()`
- Dataset/product/catalog listing endpoints

---

### Server-Sent Events (SSE) Endpoints

**Subscribe to Notifications**:
```
GET {url}catalogs/{catalog}/notifications/subscribe
```
- Protocol: Server-Sent Events (SSE)
- Used by: `listen_to_events()`, `get_events()`
- Headers:
  - `Last-Event-ID: {id}` (resume from specific event)
  - `Authorization: Bearer {token}`
- Connection: Long-lived, event stream
- Event types: Dataset updates, distribution changes, HeartBeatNotification

**Event Stream Format**:
```
event: message
data: {"id": "...", "type": "...", "timestamp": "...", "metaData": {...}}
```

---

### Embeddings/Vector Store Endpoints

**List Indexes**:
```
GET {root_url}dataspaces/{catalog}/datasets/{knowledge_base}/indexes/
```
- Returns: OpenSearch index configurations
- Used by: `list_indexes()`

**OpenSearch Client Operations**:
- Credentials passed to `OpenSearch`/`AsyncOpenSearch` clients
- Connection class: `FusionEmbeddingsConnection` / `FusionAsyncHttpConnection`
- Endpoints vary based on OpenSearch operations (search, index, bulk, etc.)

---

### Additional Runtime Patterns

**Catalog Resource Discovery**:
```
GET {root_url}catalogs/{catalog}
```
- Returns: Links to products, datasets, attributes in catalog
- Auto-discovered resources enable dynamic API navigation

**File Format Detection**:
- Recognized formats: csv, parquet, psv, json, pdf, txt, doc, docx, htm, html, xls, xlsx, xlsm, dot, dotx, docm, dotm, rtf, odt, xltx, xlsb, jpg, jpeg, bmp, png, tif, gif, mp3, wav, mp4, mov, mkv, gz, xml
- Raw format: Used when file extension not in recognized formats
- Detection happens at runtime in upload/download flows

**Conditional Endpoint Construction**:
- URLs adapt based on:
  - `is_download` flag (adds `/operationType/download`)
  - `is_raw` flag (uses 'raw' format instead of file extension)
  - `partitioning` style (affects local path, not API URL)
  - Presence of `file_name` (adds `?file={name}` query param)

---

---

## Complete API Call Flow Examples

### Example 1: Download Flow (`fusion.download("MY_DATASET", "20250101", "csv")`)

```
1. Check Access:
   GET https://fusion.jpmorgan.com/api/v1/catalogs/common/datasets/MY_DATASET
   → Returns: {"status": "Subscribed", ...}

2. List Dataset Members:
   GET https://fusion.jpmorgan.com/api/v1/catalogs/common/datasets/MY_DATASET/datasetseries
   → Returns: {"resources": [{"identifier": "20250101", ...}, ...]}
   → Handles pagination with x-jpmc-next-token

3. Get Distribution Details:
   GET https://fusion.jpmorgan.com/api/v1/catalogs/common/datasets/changes?datasets=MY_DATASET
   → Returns: {"datasets": [{"distributions": [...]}]}

4. List Files in Distribution:
   GET https://fusion.jpmorgan.com/api/v1/catalogs/common/datasets/MY_DATASET/datasetseries/20250101/distributions/csv/files
   → Returns: {"resources": [{"@id": "file1.csv", ...}]}

5. Download Each File:
   GET https://fusion.jpmorgan.com/api/v1/catalogs/common/datasets/MY_DATASET/datasetseries/20250101/distributions/csv/files/operationType/download?file=file1.csv
   → Optional parallel range requests (two patterns):
     - Pattern 1: ...&downloadRange=bytes=0-5242879 (appends to existing query params)
     - Pattern 2: .../operationType/download?downloadRange=bytes=0-5242879 (for FusionFile reads)
   → Returns: File content as bytes
```

---

### Example 2: Upload Flow (`fusion.upload("data.csv", "MY_DATASET", "20250101", "common")`)

**Small File (Single-Part):**
```
1. Construct URL from filename or parameters
   → URL: /catalogs/common/datasets/MY_DATASET/datasetseries/20250101/distributions/csv

2. Compute SHA-256 digest of entire file

3. Upload:
   PUT https://fusion.jpmorgan.com/api/v1/catalogs/common/datasets/MY_DATASET/datasetseries/20250101/distributions/csv
   Headers:
     - Content-Type: application/octet-stream
     - Authorization: Bearer {token}
     - Digest: SHA-256={base64_hash}
     - x-jpmc-distribution-created-date: 2025-12-03
     - x-jpmc-distribution-from-date: 2025-01-01
     - x-jpmc-distribution-to-date: 2199-12-31
   Body: <file_bytes>
```

**Large File (Multipart):**
```
1. Initialize Upload:
   POST https://fusion.jpmorgan.com/api/v1/catalogs/common/datasets/MY_DATASET/datasetseries/20250101/distributions/csv/operationType/upload
   Headers:
     - Content-Type: application/json
     - Authorization: Bearer {token}
     - x-jpmc-distribution-created-date: 2025-12-03
     - x-jpmc-distribution-from-date: 2025-01-01
     - x-jpmc-distribution-to-date: 2199-12-31
   → Returns: {"operationId": "abc-123-def"}

2. Upload Chunk 1:
   POST https://fusion.jpmorgan.com/api/v1/catalogs/common/datasets/MY_DATASET/datasetseries/20250101/distributions/csv/operations/upload?operationId=abc-123-def&partNumber=1
   Headers:
     - Content-Type: application/octet-stream
     - Digest: SHA-256={chunk1_hash}
   Body: <chunk1_bytes>
   → Returns: {chunk_response_1}

3. Upload Chunk 2:
   POST ...&partNumber=2
   → Returns: {chunk_response_2}

4. Finalize Upload:
   POST https://fusion.jpmorgan.com/api/v1/catalogs/common/datasets/MY_DATASET/datasetseries/20250101/distributions/csv/operations/upload?operationId=abc-123-def
   Headers:
     - Content-Type: application/json
     - Digest: SHA-256={tree_hash}  # Hash of chunk hashes
   Body: {"parts": [chunk_response_1, chunk_response_2]}
```

---

### Example 3: Metadata Creation Flow (`dataset.create()`)

```
1. Validate Dataset Name Not in Use:
   GET https://fusion.jpmorgan.com/api/v1/catalogs/common/datasets
   → Checks if identifier already exists in returned resources

2. Create Dataset:
   POST https://fusion.jpmorgan.com/api/v1/catalogs/common/datasets/MY_DATASET
   Headers:
     - Content-Type: application/json
     - Authorization: Bearer {token}
   Body: {
     "identifier": "MY_DATASET",
     "title": "My Dataset",
     "category": ["Category1"],
     "isRawData": true,
     ...
   }
```

---

### Example 4: Report with Attributes Flow

```
1. Create Report:
   POST https://fusion.jpmorgan.com/api/corelineage-service/v1/reports
   Body: {
     "title": "Monthly Report",
     "description": "...",
     "frequency": "Monthly",
     ...
   }
   → Returns: {"id": "report-123", ...}

2. Create Report Attributes (Bulk):
   POST https://fusion.jpmorgan.com/api/corelineage-service/v1/reports/report-123/attributes
   Body: [
     {"title": "Field1", "sourceIdentifier": "f1", ...},
     {"title": "Field2", "sourceIdentifier": "f2", ...}
   ]

3. Link Attributes to Business Terms:
   POST https://fusion.jpmorgan.com/api/corelineage-service/v1/data-mapping/attributes/terms
   Body: [
     {
       "attribute": {
         "entityType": "Report",
         "entityIdentifier": "report-123",
         "attributeIdentifier": "f1"
       },
       "term": {"id": "term-xyz"},
       "isKde": true
     }
   ]
```

---

### Example 5: Lineage Flow

```
1. Create Dataset Lineage:
   POST https://fusion.jpmorgan.com/api/v1/catalogs/common/datasets/TARGET_DS/lineage
   Body: {
     "source": [
       {"dataset": "SOURCE_DS_1", "catalog": "common"},
       {"dataset": "SOURCE_DS_2", "catalog": "common"}
     ]
   }

2. Create Attribute Dependencies:
   POST https://fusion.jpmorgan.com/api/corelineage-service/v1/data-dependencies/attributes
   Body: [
     {
       "sourceAttributes": [
         {
           "entityType": "Dataset",
           "entityIdentifier": "SOURCE_DS_1",
           "attributeIdentifier": "col_a",
           "dataSpace": "common"
         }
       ],
       "targetAttribute": {
         "entityType": "Dataset",
         "entityIdentifier": "TARGET_DS",
         "attributeIdentifier": "col_x",
         "dataSpace": "common"
       }
     }
   ]

3. Query Attribute Lineage:
   POST https://fusion.jpmorgan.com/api/corelineage-service/v1/data-dependencies/source-attributes/query
   Body: {
     "entityType": "Dataset",
     "entityIdentifier": "TARGET_DS",
     "attributeIdentifier": "col_x",
     "dataSpace": "common"
   }
```

---

## Notes:
1. All API calls require authentication via FusionCredentials
2. Pagination is handled automatically via `x-jpmc-next-token` header
3. Most methods support catalog parameter (defaults to "common")
4. Error handling uses `requests_raise_for_status()` wrapper
5. FusionHTTPFileSystem handles file upload/download with multipart support
6. Multipart upload is automatically used when file size > chunk_size (default: 5MB)
7. SHA-256 digests are computed for all uploads (both single-part and per-chunk for multipart)
8. Parallel downloads use byte-range requests for improved performance
9. All dates are formatted as YYYY-MM-DD for API calls
10. File downloads support both synchronous and asynchronous operations
11. Bearer tokens are automatically refreshed before expiration
12. All URLs are properly URL-encoded, especially file names with special characters
13. Retry logic (5 attempts with backoff) is applied to all HTTP requests
14. Proxy settings from credentials are automatically applied to all requests


