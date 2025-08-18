# Wecome to PyFusion #

The PyFusion SDK eases the consumption of the Fusion API in Python, taking care of credential and session management, efficient parallel data downloads and uploads, and metadata browsing and creation.

## What is Fusion?

Fusion by J.P. Morgan is a cloud-native data platform for institutional investors, providing end-to-end data management, analytics, and reporting solutions across the investment lifecycle. The platform allows clients to seamlessly integrate and combine data from multiple sources into a single data model that delivers the benefits and scale and reduces costs, along with the ability to more easily unlock timely analysis and insights. Fusion's open data architecture supports flexible distribution, including partnerships with cloud and data providers, all managed by J.P. Morgan data experts. 

For more information, please visit [fusion.jpmorgan.com](https://fusion.jpmorgan.com)

!!! info "How to Gain Access to the Fusion Data Management Platform"
    You must have the appropriate access to use the SDK. Please refer to the [Fusion page](https://fusion.jpmorgan.com) and reach out to the concerned department for additonal information.

## Overview

The PyFusion SDK is designed to provide users with a seamless way to programmatically interact with the Fusion Data Management Platform. With PyFusion, you can browse and query available data and metadata, enabling efficient data discovery and integration with workflows. Additionally, the SDK allows you to create products, define datasets and schemas, and upload your data directly to the platform.

## Key Features

- **Data Download and Upload**: Seamlessly download and upload data.
- **Metadata Discovery**: Browse available products, datasets and attributes within your catalogs.
- **Metadata Creation**: Define metadata for products, datasets and attributes and upload to your catalog.

## Key Terms
- **Catalog**: A catalog is an inventory of data products and datasets. It maintains metadata that describes each product or dataset, allowing data to be classified and effectively managed.
- **Data Product**: A grouping of related datasets with its own metadata that may reflect a logical way to group datasets.
- **Dataset**: A grouping of related data, for example the data held in a database table or data relating to a specific entity.
- **Dataset Series Member**: A specific instance of a dataset. For structured data, a series member typically represents an instance of a time series range. For unstructured data, this contain a variety of forms, for example, a PDF within a corpus of documents.
- **Distribution**: Downloadable instances of a dataset, containing a file type, for example CSV or Parquet.

## Getting Started

To help you get started, we have provided a comprehensive [Getting Started Guide](quickstart.md) that covers setting up your environment, installing the library, and starting your fusion session.
