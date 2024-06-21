# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.3] - 2024-06-20

* fix fsync upload

## [1.1.2] - 2024-06-16

* default root_url change

## [1.1.1] - 2024-06-07

* project folder rename
* minor fix in the upload internals

## [1.1.0] - 2024-06-06

* Fusion e2e monitoring


## [1.1.0-dev3] - 2024-05-22

* Internal build updates


## [1.1.0-dev1] - 2024-05-14

* Internal build updates


## [1.0.23] - 2024-04-12

* bug fix to unlock the upload functionality

## [1.0.22] - 2024-04-06

* limit number of threads for download to default to 10

## [1.0.21] - 2024-03-27

* fix to_df for parquet files

## [1.0.20] - 2024-03-27

* multipart download for a single file download

## [1.0.19] - 2024-03-12

* eliminate dependency on async-retrying

## [1.0.18] - 2024-03-11

* upload from s3 to API fix

## [1.0.17] - 2024-02-26

* allow datetime dataseries members
* to_bytes method
* encode dataset name in the changes endpoint

## [1.0.16] - 2024-02-19

* fix multi-dataset fsync

## [1.0.15] - 2024-02-08

* fix get_events
* support for bytes-range requests in fusion filesystem
* support for per column downloads via pyarrow parquet dataset

## [1.0.14] - 2023-12-13

* progress bar fix
* upload error propagation

## [1.0.13] - 2023-12-13

* polars integration
* file size in fs.info function
* progress bar improvement to capture exceptions
* sample dataset download
* server events functionality

## [1.0.12] - 2023-06-12

* minor bug fixes

## [1.0.11] - 2023-05-10

* support bearer token authentication
* fix proxy support to aiohttp
* fix filtering support for csv and json

## [1.0.10] - 2023-03-23

* md5 to sha256 convention change
* fsync continuous updates bug fix
* to_table function addition
* saving files in a hive friendly folder structure
* new bearer token add for download/upload operations
* raw data upload functionality fix

## [1.0.9] - 2023-01-23

* operational enhancements

## [1.0.8] - 2023-01-19

* cloud storage compatibility

## [1.0.7] - 2023-01-12

* Multi-part upload
* fsync

## [1.0.6] - 2022-11-21

* Support setting of the default catalog
* Fusion filesystem module
* Upload functionality
* Folder traversing for credentials
* Filters for parquet and csv file opening

## [1.0.5] - 2022-06-22

* Add support for internal auth methods

## [1.0.4] - 2022-05-19

* Support proxy servers in auth post requests
* Add back support for '2020-01-01' and '20200101' date formats
* Various bug fixes
* Streamline credentials creation

## [1.0.3] - 2022-05-12

* Add support for 'latest' datasets

## [1.0.2] - 2022-05-12

* Integrate build with docs

## [1.0.1] - 2022-05-12

* First live release on JPMC gitub
