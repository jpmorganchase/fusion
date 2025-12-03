# Fusion SDK Performance Testing Guide

## Overview

This document describes the comprehensive performance testing framework for the Fusion SDK, covering latency, throughput, success rate, and scalability testing across all major operations.

## Performance Metrics

### 1. **Latency**
- **Min/Max**: Fastest and slowest operation times
- **Mean/Median**: Average and median response times
- **P95/P99**: 95th and 99th percentile response times
- **Measured in**: Milliseconds (ms)

### 2. **Throughput**
- Operations per second (ops/sec)
- Data transfer rate (MB/s for uploads/downloads)
- Measured across different concurrency levels

### 3. **Success Rate**
- Percentage of successful operations
- Error types and frequencies
- Retry behavior analysis

### 4. **Scalability**
- Performance under varying load (1x, 5x, 10x, 50x)
- Concurrent user simulation
- Resource utilization patterns

---

## Test Scenarios

Each method is tested across 4-5 different scenarios:

### Scenario Categories

#### **Light Scenarios**
- Single operation execution
- Small data volumes (<10MB)
- Minimal concurrency (1-5 threads)
- Baseline performance measurement

#### **Medium Scenarios**
- Multiple sequential operations
- Medium data volumes (10MB-100MB)
- Moderate concurrency (5-10 threads)
- Typical production workload

#### **Heavy Scenarios**
- Large data volumes (>100MB, up to 1GB+)
- High concurrency (10-50 threads)
- Bulk operations
- Maximum throughput testing

#### **Scalability Scenarios**
- Increasing load patterns
- Burst traffic simulation
- Sustained load over time
- Resource constraint testing

#### **Edge Case Scenarios**
- Network instability simulation
- Timeout handling
- Rate limiting behavior
- Error recovery testing

---

## Test Coverage by Operation Type

### 1. Catalog Operations

#### `list_catalogs()`

| Scenario | Description | Parameters | Expected Metrics |
|----------|-------------|------------|------------------|
| Light - Single Call | 5 sequential calls | Default | P95 < 5s, Success 100% |
| Light - Cached | Repeat calls | Same params | P95 < 1s |
| Heavy - Concurrent | 50 parallel calls | 10 workers | Success > 90% |
| Scalability - Increasing Load | 10→50→100 calls | Progressive | Throughput > 10 ops/sec |
| Edge - Rate Limiting | 1000 rapid calls | No delay | Monitor throttling |

#### `catalog_resources()`

| Scenario | Description | Parameters | Expected Metrics |
|----------|-------------|------------|------------------|
| Light - Single Catalog | 10 calls for common | catalog="common" | P95 < 3s |
| Medium - Multiple Catalogs | Query 5 catalogs | Different catalogs | Mean < 2s |
| Heavy - Concurrent Access | 30 parallel queries | 15 workers | Throughput > 8 ops/sec |
| Scalability - All Catalogs | Query all available | Loop all | Success 100% |

---

### 2. Dataset Operations

#### `list_datasets()`

| Scenario | Description | Parameters | Expected Metrics |
|----------|-------------|------------|------------------|
| Light - Limited Results | 10 results | max_results=10 | P95 < 3s |
| Medium - Filtered Search | Contains filter | contains="TEST" | Mean < 5s |
| Heavy - Full Pagination | All datasets | max_results=-1 | Success 95% |
| Scalability - Increasing Pages | 10→100→1000→all | Progressive | Linear scaling |
| Edge - Large Result Set | 10,000+ datasets | No limits | Memory < 500MB |

#### `dataset_resources()`

| Scenario | Description | Parameters | Expected Metrics |
|----------|-------------|------------|------------------|
| Light - Single Dataset | 20 sequential | Fixed dataset | P95 < 1s |
| Heavy - Concurrent | 100 parallel | 20 workers | Throughput > 15 ops/sec |
| Scalability - Many Datasets | 50 different datasets | Loop | Mean < 2s |
| Edge - Non-existent | Query invalid ID | Error handling | Graceful failure |

#### `list_dataset_attributes()`

| Scenario | Description | Parameters | Expected Metrics |
|----------|-------------|------------|------------------|
| Light - Few Attributes | Dataset with <10 attrs | Simple dataset | P95 < 1s |
| Medium - Many Attributes | Dataset with 100+ attrs | Complex dataset | Mean < 3s |
| Heavy - Concurrent | 50 parallel queries | 10 workers | Success 100% |
| Scalability - Wide Tables | 1000+ attributes | Wide schema | Memory efficient |

#### `list_datasetmembers()`

| Scenario | Description | Parameters | Expected Metrics |
|----------|-------------|------------|------------------|
| Light - Recent Members | max_results=10 | Latest 10 | P95 < 2s |
| Medium - Monthly Data | 30-50 members | Typical series | Mean < 5s |
| Heavy - Historical | 1000+ members | max_results=-1 | Pagination works |
| Scalability - Time Series | Daily for 5 years | 1800+ members | P95 < 30s |
| Edge - Empty Dataset | No members | Handle gracefully | No error |

---

### 3. Download Operations

#### `download()`

| Scenario | Description | File Size | Parallelization | Expected Metrics |
|----------|-------------|-----------|-----------------|------------------|
| Light - Small CSV | Single 2MB file | n_par=1 | P95 < 5s, Success 100% |
| Medium - Parquet | 50MB parquet | n_par=4 | 10-20 MB/s throughput |
| Heavy - Large Dataset | 1GB+ file | n_par=16 | P95 < 120s |
| Scalability - Parallel | Same file, vary n_par | 1,2,4,8,16 | Show speedup |
| Scalability - Concurrent | 20 users download | 5 workers | Success > 95% |
| Edge - Network Retry | Simulate failures | Auto-retry | Success after retries |

#### `to_df()`

| Scenario | Description | Data Size | Filters | Expected Metrics |
|----------|-------------|-----------|---------|------------------|
| Light - Small Table | 1000 rows | No filters | P95 < 3s |
| Medium - Filtered | 100K rows | Column subset | Mean < 10s |
| Medium - Column Select | 1M rows | 5 of 100 cols | Filter speedup |
| Heavy - Large Memory | 10M rows | All columns | Memory < 2GB |
| Scalability - Increasing Size | 1K→10K→100K→1M→10M | Progressive | Linear time |

#### `to_bytes()`

| Scenario | Description | File Size | Expected Metrics |
|----------|-------------|-----------|------------------|
| Light - Small Binary | <1MB | P95 < 2s |
| Medium - Multiple Files | 5 files, 10MB each | Mean < 15s total |
| Heavy - Large Binary | 500MB | P95 < 60s |
| Scalability - Concurrent | 30 parallel | Success 100% |

#### `to_table()`

| Scenario | Description | Expected Metrics |
|----------|-------------|------------------|
| Light - PyArrow Small | P95 < 3s |
| Medium - Partitioned | Handle partitions |
| Heavy - Large Arrow | Memory efficient |

---

### 4. Upload Operations

#### `upload()`

| Scenario | Description | File Size | Mode | Expected Metrics |
|----------|-------------|-----------|------|------------------|
| Light - Single-part | 2MB | multipart=False | P95 < 5s, Success 100% |
| Medium - Small Multipart | 10MB | chunk_size=5MB | Mean < 15s |
| Heavy - Large Multipart | 500MB | chunk_size=20MB | 5-10 MB/s upload |
| Scalability - Chunk Sizes | 50MB file | 5,10,20,50MB | Optimal chunk size |
| Scalability - Parallel | vary n_par | 1,2,4,8 | Show speedup |
| Edge - Interrupted Upload | Simulate failure | Resume/retry | Success after retry |

#### `from_bytes()`

| Scenario | Description | Data Size | Expected Metrics |
|----------|-------------|-----------|------------------|
| Light - Small Buffer | 1MB in memory | P95 < 3s |
| Medium - Generated Data | 50MB DataFrame | Mean < 20s |
| Heavy - Large Memory | 500MB buffer | Success 100% |
| Scalability - Sequential | 10 uploads | Throughput > 0.5 ops/sec |

---

### 5. Metadata Operations

#### `list_products()`

| Scenario | Description | Expected Metrics |
|----------|-------------|------------------|
| Light - Basic List | P95 < 2s |
| Heavy - Concurrent | 75 parallel, Success > 95% |
| Scalability - All Products | Pagination works |

#### `list_registered_attributes()`

| Scenario | Description | Expected Metrics |
|----------|-------------|------------------|
| Light - Single Catalog | P95 < 3s |
| Medium - Large Catalog | 1000+ attributes, Mean < 10s |
| Scalability - Pagination | Handle efficiently |

#### Product/Dataset/Attribute CRUD Operations

| Operation | Light | Medium | Heavy | Expected |
|-----------|-------|--------|-------|----------|
| Create | Single | Batch 10 | Batch 100 | P95 < 5s |
| Update | Single | Batch 10 | Batch 50 | P95 < 10s |
| Delete | Single | Batch 10 | Batch 50 | Success 100% |

---

### 6. Lineage Operations

#### `list_dataset_lineage()`

| Scenario | Description | Expected Metrics |
|----------|-------------|------------------|
| Light - Simple Lineage | 1-2 levels | P95 < 3s |
| Medium - Complex Graph | 5+ levels | Mean < 10s |
| Heavy - Wide Lineage | 100+ connections | P95 < 30s |
| Scalability - Many Queries | 50 datasets | Throughput > 3 ops/sec |

#### `list_attribute_lineage()`

| Scenario | Description | Expected Metrics |
|----------|-------------|------------------|
| Light - Single Attribute | P95 < 2s |
| Medium - 20 Attributes | Sequential, Mean < 5s |
| Heavy - Complex Dependencies | Deep graph, P95 < 15s |

#### `create_dataset_lineage()`

| Scenario | Description | Expected Metrics |
|----------|-------------|------------------|
| Light - Simple Link | 1 source | P95 < 3s |
| Medium - Multiple Sources | 5 sources | Mean < 5s |
| Heavy - Batch Creation | 50 lineage links | Success 100% |

---

### 7. Report Operations

#### CRUD Operations

| Scenario | Description | Expected Metrics |
|----------|-------------|------------------|
| Light - Create Report | Single report | P95 < 3s |
| Medium - With Attributes | Report + 50 attrs | Mean < 10s |
| Heavy - Bulk Operations | 20 reports | P95 < 60s |
| Scalability - Large Report | 500+ attributes | Success 100% |

---

### 8. End-to-End Workflows

#### Complete Data Pipeline

| Scenario | Description | Expected Metrics |
|----------|-------------|------------------|
| Light - Simple Workflow | Create→Upload→Download | Total < 30s |
| Medium - Full Cycle | Metadata + Data + Lineage | Total < 2min |
| Heavy - Production Sim | Multiple datasets + deps | Success 100% |

---

## Running Performance Tests

### Prerequisites

```bash
# Install required packages
pip install pytest pytest-benchmark pandas numpy pyarrow

# Set up test environment
export FUSION_CREDENTIALS_PATH=/path/to/credentials.json
export PERF_TEST_DATASET_ID=your_test_dataset
export PERF_TEST_LARGE_DATASET_ID=your_large_dataset
```

### Run All Tests

```bash
# Run all performance tests
pytest py_tests/test_performance.py -v -m performance

# Run specific category
pytest py_tests/test_performance.py::TestDownloadPerformance -v

# Run only light scenarios
pytest py_tests/test_performance.py -k "light" -v

# Run only heavy scenarios
pytest py_tests/test_performance.py -k "heavy" -v

# Run stress tests
pytest py_tests/test_performance.py -m stress -v
```

### Generate Performance Report

```python
from py_tests.test_performance import generate_performance_report

# Collect metrics from test runs
metrics_list = [
    # ... collected metrics ...
]

generate_performance_report(metrics_list, "performance_report.html")
```

---

## Performance Benchmarks

### Target SLAs

| Operation Category | P95 Latency | Throughput | Success Rate |
|-------------------|-------------|------------|--------------|
| Metadata Queries | < 3s | > 10 ops/sec | 99%+ |
| Small Downloads (<10MB) | < 10s | > 5 MB/s | 99%+ |
| Large Downloads (>1GB) | < 120s | > 10 MB/s | 95%+ |
| Small Uploads (<10MB) | < 15s | > 3 MB/s | 99%+ |
| Large Uploads (>100MB) | < 180s | > 5 MB/s | 95%+ |
| Lineage Queries | < 5s | > 5 ops/sec | 99%+ |
| CRUD Operations | < 5s | > 10 ops/sec | 99%+ |

### Known Bottlenecks

1. **Network Latency**: Dominant factor for small operations
2. **Pagination**: Can be slow for very large result sets
3. **Multipart Upload**: Overhead for small files
4. **Concurrent Downloads**: May hit rate limits

---

## Monitoring & Profiling

### Built-in Metrics

```python
from py_tests.test_performance import PerformanceMetrics, timed_operation

# Track custom operations
metrics = PerformanceMetrics("my_operation", "scenario_name")
metrics.start_timer()

for i in range(100):
    success, latency, result = timed_operation(my_function, arg1, arg2)
    if success:
        metrics.record_success(latency)
    else:
        metrics.record_failure(latency, result)

metrics.stop_timer()
metrics.print_summary()
```

### Memory Profiling

```bash
# Use memory_profiler for detailed analysis
pip install memory_profiler
python -m memory_profiler py_tests/test_performance.py
```

### Network Monitoring

```python
# Enable request logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## Continuous Performance Testing

### Integration with CI/CD

```yaml
# .github/workflows/performance.yml
name: Performance Tests

on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM
  workflow_dispatch:

jobs:
  performance:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run performance tests
        run: pytest py_tests/test_performance.py -v -m performance
      - name: Upload report
        uses: actions/upload-artifact@v2
        with:
          name: performance-report
          path: performance_report.html
```

---

## Performance Optimization Tips

### For Users

1. **Use parallelization**: Set `n_par` for large downloads/uploads
2. **Filter early**: Use `columns` and `filters` in `to_df()`
3. **Batch operations**: Group multiple operations when possible
4. **Cache metadata**: Store frequently accessed metadata locally
5. **Choose optimal chunk size**: 10-20MB for multipart uploads

### For Developers

1. **Minimize API calls**: Batch requests where possible
2. **Implement caching**: Cache catalog/dataset metadata
3. **Optimize pagination**: Use efficient page sizes
4. **Connection pooling**: Reuse HTTP connections
5. **Async operations**: Use async methods for I/O-bound tasks

---

## Troubleshooting Performance Issues

### Common Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| Slow downloads | P95 > 60s | Increase n_par, check network |
| High memory usage | OOM errors | Use streaming, reduce batch size |
| Timeout errors | Requests fail | Increase timeout, check connectivity |
| Low throughput | < 1 MB/s | Check network, use larger chunks |
| Rate limiting | 429 errors | Implement backoff, reduce concurrency |

### Debug Mode

```python
import fusion
fusion.set_debug_mode(True)  # Enable verbose logging
```

---

## Future Enhancements

- [ ] Automated performance regression detection
- [ ] Real-time performance dashboard
- [ ] Cross-region performance comparison
- [ ] Load testing tool with configurable scenarios
- [ ] Performance profiling API
- [ ] Automated bottleneck identification

---

## Contact & Support

For performance-related issues or questions:
- GitHub Issues: [jpmorganchase/fusion](https://github.com/jpmorganchase/fusion/issues)
- Documentation: [Fusion SDK Docs](https://fusion.jpmorgan.com/docs)
