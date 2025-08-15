# Installation

## Stable release

To install PyFusion, run this command in your
terminal:

``` console
$ pip install pyfusion
```

This is the preferred method to install PyFusion, as it will always install the most recent stable release. Documentation of changes made during each release can be found in the [Changelog](changelog.md).

If you don't have [pip](https://pip.pypa.io/en/stable/installation/) installed, this [Python installation guide](https://docs.python-guide.org/starting/installation/)
can guide you through the process.

## From source

The source for PyFusion can be downloaded from
the [Github repo](https://github.com/jpmorganchase/fusion).

You can either clone the public repository:

``` console
$ git clone git://github.com/jpmorganchase/fusion
```

Or download the [tarball](https://www.makeuseof.com/extract-tar-gz/):

``` console
$ curl -OJL https://github.com/jpmorganchase/fusion/tarball/master
```

Once you have a copy of the source, you can install it with:

``` console
$ pip install .
```

# Storing Credentials

In order to connect to the Fusion API via SDK, we recommend storing your credentials in a JSON file. This file can be generated using Fusion’s application registration page.

Your credentials file should be located in a directory accessible from the location you are using the SDK.
By default, the SDK will look for the credentials file at the path ``'config/client_credentials.json'``.

## Formatting Credentials File

Your credentials file should be formatted as follows.

```python
{
    "client_id": "YOUR_CLIENT_ID",
    "client_secret": "YOUR_CLIENT_SECRET",
    "resource": "",
    "auth_url": "",
    "proxies": {}
}
```

- **client_id**: Generated using Fusion's application registration page.
- **client_secret**: Generated using Fusion's application registration page.
- **resource**: Can be found on Fusion's calling the API page.
- **auth_url**: Can be found on Fusion's calling the API page.
- **proxies**: HTTP and HTTPS proxy values. Optional.

!!! warning "Populating Proxies"
    If your application is running behind a proxy, for example a corporate firewall, then the ``proxies`` value will also need to be defined. For example:
        ``"proxies" : {"http": "http://proxy.myfirm.com:8080", "https": "https://proxy.myfirm.com:8080"}``
    
    If you do not require proxies, either remove this argument or leave it as an empty dictionary ``{}``.


# Usage 

## Import Fusion

To begin using pyfusion, simply execute the import below.

```python
from fusion import Fusion
```
## Fusion Object

Connection to the Fusion platform can be easily established by instantiating a ``Fusion()`` object.

This object will act as a Fusion client, managing your credentials and connectivity to the API. This client also provides an extensive list of methods for browsing, retrieving, and creating metadata and data.

If your credenitals are stored in ``'config/client_credentials.json'``, you can instantiate your client as follow:

```python
fusion = Fusion()
```

If your credentials are stored in an alternative location, you can provide the appropriate path as the ``credentials`` argument:

```python
fusion = Fusion(credentials="path/to/my/credentials.json")
```

Alternatively, if you wish to provide your credentials directly to the client, you can utilize the ``FusionCredentials`` object:

```python
from fusion import FusionCredentials

credentials = FusionCredentials(
    client_id="<CLIENT_ID>",
    client_secret="<CLIENT_SECRET>",
    resource="<RESOURCE>"
)

fusion = Fusion(credentials=credentials)

```

## View Available Methods

Once you have instantiated the client ``fusion``, running the following cell will display its available methods:

```python
fusion
```
The output will be a table containing the available methods along with a short description:
```
Fusion object 
Available methods:
+--------------------------------------+--------------------------------------------------------------------------------------------------------------+
| attribute                            | Instantiate an Attribute object with this client for metadata creation.                                      |
| attributes                           | Instantiate an Attributes object with this client for metadata creation.                                     |
| catalog_resources                    | List the resources contained within the catalog, for example products and datasets.                          |
| create_dataset_lineage               | Upload lineage to a dataset.                                                                                 |
| dataset                              | Instantiate a Dataset object with this client for metadata creation.                                         |
| dataset_resources                    | List the resources available for a dataset, currently this will always be a datasetseries.                   |
| datasetmember_resources              | List the available resources for a datasetseries member.                                                     |
| delete_all_datasetmembers            | Delete all dataset members within a dataset.                                                                 |
| delete_datasetmembers                | Delete dataset members.                                                                                      |
| download                             | Downloads the requested distributions of a dataset to disk.                                                  |
| from_bytes                           | Uploads data from an object in memory.                                                                       |
| get_async_fusion_vector_store_client | Returns Fusion Embeddings Search client.                                                                     |
| get_events                           | Run server sent event listener and print out the new events. Keyboard terminate to stop.                     |
| get_fusion_filesystem                | Retrieve Fusion file system instance.                                                                        |
| get_fusion_vector_store_client       | Returns Fusion Embeddings Search client.                                                                     |
| input_dataflow                       | Instantiate an Input Dataflow object with this client for metadata creation.                                 |
| list_catalogs                        | Lists the catalogs available to the API account.                                                             |
| list_dataset_attributes              | Returns the list of attributes that are in the dataset.                                                      |
| list_dataset_lineage                 | List the upstream and downstream lineage of the dataset.                                                     |
| list_datasetmembers                  | List the available members in the dataset series.                                                            |
| list_datasetmembers_distributions    | List the distributions of dataset members.                                                                   |
| list_datasets                        | Get the datasets contained in a catalog.                                                                     |
| list_distributions                   | List the available distributions (downloadable instances of the dataset with a format type).                 |
| list_indexes                         | List the indexes in a knowledge base.                                                                        |
| list_product_dataset_mapping         | get the product to dataset linking contained in  a catalog. A product is a grouping of datasets.             |
| list_products                        | Get the products contained in a catalog. A product is a grouping of datasets.                                |
| list_registered_attributes           | Returns the list of attributes in a catalog.                                                                 |
| listen_to_events                     | Run server sent event listener in the background. Retrieve results by running get_events.                    |
| output_dataflow                      | Instantiate an Output Dataflow object with this client for metadata creation.                                |
| product                              | Instantiate a Product object with this client for metadata creation.                                         |
| report                               | Instantiate Report object with this client for metadata creation for managing regulatory reporting metadata. |
| to_bytes                             | Returns an instance of dataset (the distribution) as a bytes object.                                         |
| to_df                                | Gets distributions for a specified date or date range and returns the data as a dataframe.                   |
| to_table                             | Gets distributions for a specified date or date range and returns the data as an arrow table.                |
| upload                               | Uploads the requested files/files to Fusion.                                                                 |
| default_catalog                      | Returns the default catalog.                                                                                 |
+--------------------------------------+--------------------------------------------------------------------------------------------------------------+
```

## Executing Available Methods

While detailed documentation for each function is located on the [Modules](api.md) page, below are a few examples to get you started.

### Retrieve your available catalogs

```python
fusion.list_catalogs()
```

### Browse products available within a catalog
```python
fusion.list_products(catalog="<CATALOG_ID>")
```

### Browse datasets available within a catalog
```python
fusion.list_datasets(catalog="<CATALOG_ID>")
```

### Browse datasets associated with a product
```python
fusion.list_datasets(catalog="<CATALOG_ID>", product="<PRODUCT_ID>")
```

### Display all metadata for datasets available within a catalog
```python
fusion.list_datasets(catalog="<CATALOG_ID>", display_all_columns=True)
```

### Retrieve attributes for a dataset
```python
fusion.list_dataset_attributes(dataset="<DATASET_ID>", catalog="<CATALOG_ID>")
```

# Additional Resources

Now that you are set up with the SDK, you can start exploring our pages on [downloading](download.md) and [uploading](upload.md).