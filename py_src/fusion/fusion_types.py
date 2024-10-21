"""Fusion types definitions."""

from enum import Enum

import polars as pl
import pyarrow as pa
import pyspark.sql.types

class Types(Enum):
    """Fusion types.

    Args:
        Enum (class: `enum.Enum`): Enum inheritance.
    """

    String = 1
    Boolean = 2
    Decimal = 3
    Float = 4
    Double = 5
    Timestamp = 6
    Date = 8
    Binary = 9
    Long = 11
    Integer = 12
    Short = 13
    Byte = 14
    Datetime = 6


spark_types = {
    1: pyspark.sql.types.StringType(),
    2: pyspark.sql.types.BooleanType(),
    3: pyspark.sql.types.DecimalType(),
    4: pyspark.sql.types.FloatType(),
    5: pyspark.sql.types.DoubleType(),
    6: pyspark.sql.types.TimestampType(),
    8: pyspark.sql.types.DateType(),
    9: pyspark.sql.types.BinaryType(),
    11: pyspark.sql.types.LongType(),
    12: pyspark.sql.types.IntegerType(),
    13: pyspark.sql.types.ShortType(),
    14: pyspark.sql.types.ByteType(),
}


inv_spark_types = {v: k for k, v in spark_types.items()}

arrow_types = {
    1: pa.utf8(),
    2: pa.bool_(),
    3: pa.Decimal128Type,
    4: pa.float32(),
    5: pa.float64(),
    6: pa.TimestampType,
    8: pa.date32(),
    9: pa.binary(),
    11: pa.int64(),
    12: pa.int32(),
    13: pa.int16(),
    14: pa.int8(),
}

inv_arrow_types = {v: k for k, v in arrow_types.items()}

polars_types = {
    1: pl.Utf8,
    2: pl.Boolean,
    3: pl.Decimal,
    4: pl.Float32,
    5: pl.Float64,
    6: pl.Datetime,
    8: pl.Date,
    9: pl.Binary,
    11: pl.Int64,
    12: pl.Int32,
    13: pl.Int16,
    14: pl.Int8,
}

inv_polars_types = {v: k for k, v in polars_types.items()}

int_types = [
    pyspark.sql.types.LongType(),
    pyspark.sql.types.IntegerType(),
    pyspark.sql.types.ShortType(),
    pyspark.sql.types.ByteType(),
]

float_types = [
    pyspark.sql.types.FloatType(),
    pyspark.sql.types.DoubleType(),
    pyspark.sql.types.DecimalType(),
]

numeric_types = int_types + float_types

int_types_arrow = [pa.int64(), pa.int32(), pa.int16(), pa.int8()]

float_types_arrow = [pa.float32(), pa.float64(), pa.decimal128(2, 0)]

numeric_types_arrow = int_types_arrow + float_types_arrow

int_types_polars = [pl.Int64, pl.Int32, pl.Int16, pl.Int8]

float_types_polars = [pl.Float32, pl.Float64, pl.Decimal]

numeric_types_polars = int_types_polars + float_types_polars
