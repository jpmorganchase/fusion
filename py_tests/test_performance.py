"""
Performance Testing Suite for Fusion SDK
=========================================

Tests latency, throughput, success rate, and scalability across different scenarios.

Usage:
    pytest py_tests/test_performance.py -v
    pytest py_tests/test_performance.py::TestCatalogPerformance -v
    pytest py_tests/test_performance.py -k "light" -v
"""

import time
import pytest
import pandas as pd
import numpy as np
from io import BytesIO
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Callable
import json
from datetime import datetime
import statistics


class PerformanceMetrics:
    """Track and calculate performance metrics"""
    
    def __init__(self, operation_name: str, scenario: str):
        self.operation_name = operation_name
        self.scenario = scenario
        self.latencies: List[float] = []
        self.successes = 0
        self.failures = 0
        self.errors: List[str] = []
        self.start_time = None
        self.end_time = None
    
    def record_success(self, latency: float):
        """Record a successful operation"""
        self.latencies.append(latency)
        self.successes += 1
    
    def record_failure(self, latency: float, error: str):
        """Record a failed operation"""
        self.latencies.append(latency)
        self.failures += 1
        self.errors.append(error)
    
    def start_timer(self):
        """Start overall timing"""
        self.start_time = time.time()
    
    def stop_timer(self):
        """Stop overall timing"""
        self.end_time = time.time()
    
    def get_metrics(self) -> Dict:
        """Calculate and return all metrics"""
        total_ops = self.successes + self.failures
        success_rate = (self.successes / total_ops * 100) if total_ops > 0 else 0
        
        metrics = {
            "operation": self.operation_name,
            "scenario": self.scenario,
            "total_operations": total_ops,
            "successes": self.successes,
            "failures": self.failures,
            "success_rate_pct": round(success_rate, 2),
        }
        
        if self.latencies:
            metrics.update({
                "latency_min_ms": round(min(self.latencies) * 1000, 2),
                "latency_max_ms": round(max(self.latencies) * 1000, 2),
                "latency_mean_ms": round(statistics.mean(self.latencies) * 1000, 2),
                "latency_median_ms": round(statistics.median(self.latencies) * 1000, 2),
                "latency_p95_ms": round(np.percentile(self.latencies, 95) * 1000, 2),
                "latency_p99_ms": round(np.percentile(self.latencies, 99) * 1000, 2),
            })
        
        if self.start_time and self.end_time:
            duration = self.end_time - self.start_time
            metrics["total_duration_sec"] = round(duration, 2)
            metrics["throughput_ops_per_sec"] = round(total_ops / duration, 2) if duration > 0 else 0
        
        if self.errors:
            metrics["error_samples"] = self.errors[:5]  # First 5 errors
        
        return metrics
    
    def print_summary(self):
        """Print formatted metrics summary"""
        metrics = self.get_metrics()
        print(f"\n{'='*80}")
        print(f"Performance Report: {metrics['operation']} - {metrics['scenario']}")
        print(f"{'='*80}")
        print(f"Total Operations:    {metrics['total_operations']}")
        print(f"Successes:           {metrics['successes']}")
        print(f"Failures:            {metrics['failures']}")
        print(f"Success Rate:        {metrics['success_rate_pct']}%")
        
        if 'latency_mean_ms' in metrics:
            print(f"\nLatency Statistics (ms):")
            print(f"  Min:      {metrics['latency_min_ms']}")
            print(f"  Max:      {metrics['latency_max_ms']}")
            print(f"  Mean:     {metrics['latency_mean_ms']}")
            print(f"  Median:   {metrics['latency_median_ms']}")
            print(f"  P95:      {metrics['latency_p95_ms']}")
            print(f"  P99:      {metrics['latency_p99_ms']}")
        
        if 'throughput_ops_per_sec' in metrics:
            print(f"\nThroughput:          {metrics['throughput_ops_per_sec']} ops/sec")
            print(f"Total Duration:      {metrics['total_duration_sec']} seconds")
        
        if metrics.get('error_samples'):
            print(f"\nSample Errors:")
            for i, error in enumerate(metrics['error_samples'], 1):
                print(f"  {i}. {error[:100]}")
        
        print(f"{'='*80}\n")


def timed_operation(func: Callable, *args, **kwargs) -> Tuple[bool, float, any]:
    """
    Execute an operation and measure its performance
    
    Returns:
        (success, latency, result_or_error)
    """
    start = time.time()
    try:
        result = func(*args, **kwargs)
        latency = time.time() - start
        return True, latency, result
    except Exception as e:
        latency = time.time() - start
        return False, latency, str(e)


# ============================================================================
# Catalog Operations Performance Tests
# ============================================================================

@pytest.mark.performance
class TestCatalogPerformance:
    """Performance tests for catalog operations"""
    
    @pytest.mark.skip(reason="Requires live Fusion connection")
    def test_list_catalogs_light(self, fusion_client):
        """Light: Single catalog listing"""
        metrics = PerformanceMetrics("list_catalogs", "light_single_call")
        metrics.start_timer()
        
        for i in range(5):
            success, latency, result = timed_operation(
                fusion_client.list_catalogs
            )
            if success:
                metrics.record_success(latency)
            else:
                metrics.record_failure(latency, result)
        
        metrics.stop_timer()
        metrics.print_summary()
        
        assert metrics.success_rate_pct >= 95, "Success rate should be >= 95%"
        assert metrics.get_metrics()['latency_p95_ms'] < 5000, "P95 latency should be < 5s"
    
    @pytest.mark.skip(reason="Requires live Fusion connection")
    def test_list_catalogs_concurrent(self, fusion_client):
        """Heavy: Concurrent catalog listings"""
        metrics = PerformanceMetrics("list_catalogs", "heavy_concurrent")
        metrics.start_timer()
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for i in range(50):
                future = executor.submit(timed_operation, fusion_client.list_catalogs)
                futures.append(future)
            
            for future in as_completed(futures):
                success, latency, result = future.result()
                if success:
                    metrics.record_success(latency)
                else:
                    metrics.record_failure(latency, result)
        
        metrics.stop_timer()
        metrics.print_summary()
        
        assert metrics.success_rate_pct >= 90, "Success rate should be >= 90% under load"


# ============================================================================
# Dataset Operations Performance Tests
# ============================================================================

@pytest.mark.performance
class TestDatasetPerformance:
    """Performance tests for dataset operations"""
    
    @pytest.mark.skip(reason="Requires live Fusion connection")
    def test_list_datasets_light(self, fusion_client):
        """Light: List datasets from single catalog"""
        metrics = PerformanceMetrics("list_datasets", "light_single_catalog")
        metrics.start_timer()
        
        for i in range(10):
            success, latency, result = timed_operation(
                fusion_client.list_datasets,
                catalog="common",
                max_results=10
            )
            if success:
                metrics.record_success(latency)
            else:
                metrics.record_failure(latency, result)
        
        metrics.stop_timer()
        metrics.print_summary()
        
        assert metrics.get_metrics()['latency_mean_ms'] < 3000
    
    @pytest.mark.skip(reason="Requires live Fusion connection")
    def test_list_datasets_heavy_pagination(self, fusion_client):
        """Heavy: List all datasets with pagination"""
        metrics = PerformanceMetrics("list_datasets", "heavy_full_pagination")
        metrics.start_timer()
        
        for i in range(3):
            success, latency, result = timed_operation(
                fusion_client.list_datasets,
                catalog="common",
                max_results=-1  # All results
            )
            if success:
                metrics.record_success(latency)
            else:
                metrics.record_failure(latency, result)
        
        metrics.stop_timer()
        metrics.print_summary()
        
        assert metrics.success_rate_pct >= 95
    
    @pytest.mark.skip(reason="Requires live Fusion connection")
    def test_list_datasets_scalability(self, fusion_client):
        """Scalability: Increasing result sizes"""
        result_sizes = [10, 50, 100, 500, 1000, -1]
        
        for size in result_sizes:
            metrics = PerformanceMetrics("list_datasets", f"scalability_max_{size}")
            metrics.start_timer()
            
            for i in range(3):
                success, latency, result = timed_operation(
                    fusion_client.list_datasets,
                    catalog="common",
                    max_results=size
                )
                if success:
                    metrics.record_success(latency)
                else:
                    metrics.record_failure(latency, result)
            
            metrics.stop_timer()
            metrics.print_summary()
    
    @pytest.mark.skip(reason="Requires live Fusion connection")
    def test_dataset_resources_concurrent(self, fusion_client, test_dataset_id):
        """Heavy: Concurrent dataset resource queries"""
        metrics = PerformanceMetrics("dataset_resources", "heavy_concurrent")
        metrics.start_timer()
        
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = []
            for i in range(100):
                future = executor.submit(
                    timed_operation,
                    fusion_client.dataset_resources,
                    dataset=test_dataset_id
                )
                futures.append(future)
            
            for future in as_completed(futures):
                success, latency, result = future.result()
                if success:
                    metrics.record_success(latency)
                else:
                    metrics.record_failure(latency, result)
        
        metrics.stop_timer()
        metrics.print_summary()
        
        assert metrics.get_metrics()['throughput_ops_per_sec'] > 5


# ============================================================================
# Download Performance Tests
# ============================================================================

@pytest.mark.performance
class TestDownloadPerformance:
    """Performance tests for download operations"""
    
    @pytest.mark.skip(reason="Requires live Fusion connection")
    def test_download_small_file_light(self, fusion_client, test_dataset_id, tmp_path):
        """Light: Download single small file (<10MB)"""
        metrics = PerformanceMetrics("download", "light_small_file")
        metrics.start_timer()
        
        for i in range(5):
            success, latency, result = timed_operation(
                fusion_client.download,
                dataset=test_dataset_id,
                dt_str="latest",
                dataset_format="csv",
                download_folder=str(tmp_path / f"run_{i}"),
                show_progress=False
            )
            if success:
                metrics.record_success(latency)
            else:
                metrics.record_failure(latency, result)
        
        metrics.stop_timer()
        metrics.print_summary()
        
        assert metrics.success_rate_pct == 100
        assert metrics.get_metrics()['latency_p95_ms'] < 10000
    
    @pytest.mark.skip(reason="Requires live Fusion connection")
    def test_download_large_file_heavy(self, fusion_client, large_dataset_id, tmp_path):
        """Heavy: Download large file (>1GB) with parallelization"""
        metrics = PerformanceMetrics("download", "heavy_large_file")
        metrics.start_timer()
        
        parallel_configs = [1, 2, 4, 8, 16]
        
        for n_par in parallel_configs:
            success, latency, result = timed_operation(
                fusion_client.download,
                dataset=large_dataset_id,
                dt_str="latest",
                dataset_format="parquet",
                download_folder=str(tmp_path / f"parallel_{n_par}"),
                show_progress=False,
                n_par=n_par
            )
            if success:
                metrics.record_success(latency)
                print(f"  n_par={n_par}: {latency:.2f}s")
            else:
                metrics.record_failure(latency, result)
        
        metrics.stop_timer()
        metrics.print_summary()
    
    @pytest.mark.skip(reason="Requires live Fusion connection")
    def test_download_concurrent_users(self, fusion_client, test_dataset_id, tmp_path):
        """Scalability: Multiple concurrent downloads"""
        metrics = PerformanceMetrics("download", "scalability_concurrent_users")
        metrics.start_timer()
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for i in range(20):
                future = executor.submit(
                    timed_operation,
                    fusion_client.download,
                    dataset=test_dataset_id,
                    dt_str="latest",
                    download_folder=str(tmp_path / f"user_{i}"),
                    show_progress=False
                )
                futures.append(future)
            
            for future in as_completed(futures):
                success, latency, result = future.result()
                if success:
                    metrics.record_success(latency)
                else:
                    metrics.record_failure(latency, result)
        
        metrics.stop_timer()
        metrics.print_summary()
    
    @pytest.mark.skip(reason="Requires live Fusion connection")
    def test_to_df_light(self, fusion_client, test_dataset_id, tmp_path):
        """Light: to_df single file"""
        metrics = PerformanceMetrics("to_df", "light_single_file")
        metrics.start_timer()
        
        for i in range(5):
            success, latency, result = timed_operation(
                fusion_client.to_df,
                dataset=test_dataset_id,
                dt_str="latest",
                dataset_format="parquet",
                download_folder=str(tmp_path),
                show_progress=False
            )
            if success:
                metrics.record_success(latency)
                # Also track dataframe size
                if isinstance(result, pd.DataFrame):
                    print(f"  DataFrame shape: {result.shape}")
            else:
                metrics.record_failure(latency, result)
        
        metrics.stop_timer()
        metrics.print_summary()
    
    @pytest.mark.skip(reason="Requires live Fusion connection")
    def test_to_df_with_filters(self, fusion_client, test_dataset_id, tmp_path):
        """Medium: to_df with column selection and filters"""
        metrics = PerformanceMetrics("to_df", "medium_with_filters")
        metrics.start_timer()
        
        for i in range(5):
            success, latency, result = timed_operation(
                fusion_client.to_df,
                dataset=test_dataset_id,
                dt_str="latest",
                columns=["col1", "col2", "col3"],
                filters=[("col1", ">", 100)],
                download_folder=str(tmp_path),
                show_progress=False
            )
            if success:
                metrics.record_success(latency)
            else:
                metrics.record_failure(latency, result)
        
        metrics.stop_timer()
        metrics.print_summary()


# ============================================================================
# Upload Performance Tests
# ============================================================================

@pytest.mark.performance
class TestUploadPerformance:
    """Performance tests for upload operations"""
    
    def create_test_file(self, path: Path, size_mb: int):
        """Create a test file of specified size"""
        data = np.random.rand(size_mb * 1024 * 128, 8)  # ~size_mb MB
        df = pd.DataFrame(data, columns=[f"col_{i}" for i in range(8)])
        df.to_parquet(path, compression="snappy")
        return path
    
    @pytest.mark.skip(reason="Requires live Fusion connection")
    def test_upload_small_file_light(self, fusion_client, test_dataset_id, tmp_path):
        """Light: Upload single small file (<5MB)"""
        metrics = PerformanceMetrics("upload", "light_small_file")
        metrics.start_timer()
        
        for i in range(5):
            test_file = self.create_test_file(tmp_path / f"small_{i}.parquet", 2)
            success, latency, result = timed_operation(
                fusion_client.upload,
                path=str(test_file),
                dataset=test_dataset_id,
                dt_str=f"2025010{i}",
                show_progress=False,
                multipart=False
            )
            if success:
                metrics.record_success(latency)
            else:
                metrics.record_failure(latency, result)
        
        metrics.stop_timer()
        metrics.print_summary()
        
        assert metrics.success_rate_pct == 100
    
    @pytest.mark.skip(reason="Requires live Fusion connection")
    def test_upload_large_file_multipart(self, fusion_client, test_dataset_id, tmp_path):
        """Heavy: Upload large file with multipart"""
        metrics = PerformanceMetrics("upload", "heavy_large_multipart")
        metrics.start_timer()
        
        chunk_sizes = [5 * 2**20, 10 * 2**20, 20 * 2**20]  # 5MB, 10MB, 20MB
        
        for i, chunk_size in enumerate(chunk_sizes):
            test_file = self.create_test_file(tmp_path / f"large_{i}.parquet", 50)
            success, latency, result = timed_operation(
                fusion_client.upload,
                path=str(test_file),
                dataset=test_dataset_id,
                dt_str=f"2025020{i}",
                show_progress=False,
                multipart=True,
                chunk_size=chunk_size
            )
            if success:
                metrics.record_success(latency)
                print(f"  chunk_size={chunk_size/(2**20):.0f}MB: {latency:.2f}s")
            else:
                metrics.record_failure(latency, result)
        
        metrics.stop_timer()
        metrics.print_summary()
    
    @pytest.mark.skip(reason="Requires live Fusion connection")
    def test_upload_parallel_threads(self, fusion_client, test_dataset_id, tmp_path):
        """Heavy: Upload with different parallelization levels"""
        metrics = PerformanceMetrics("upload", "heavy_parallel_upload")
        metrics.start_timer()
        
        n_par_configs = [1, 2, 4, 8]
        
        for n_par in n_par_configs:
            test_file = self.create_test_file(tmp_path / f"parallel_{n_par}.parquet", 30)
            success, latency, result = timed_operation(
                fusion_client.upload,
                path=str(test_file),
                dataset=test_dataset_id,
                dt_str=f"20250301",
                show_progress=False,
                n_par=n_par
            )
            if success:
                metrics.record_success(latency)
                print(f"  n_par={n_par}: {latency:.2f}s")
            else:
                metrics.record_failure(latency, result)
        
        metrics.stop_timer()
        metrics.print_summary()
    
    @pytest.mark.skip(reason="Requires live Fusion connection")
    def test_from_bytes_light(self, fusion_client, test_dataset_id):
        """Light: Upload from memory"""
        metrics = PerformanceMetrics("from_bytes", "light_memory_upload")
        metrics.start_timer()
        
        for i in range(10):
            # Create small DataFrame in memory
            data = np.random.rand(1000, 10)
            df = pd.DataFrame(data, columns=[f"col_{i}" for i in range(10)])
            buffer = BytesIO()
            df.to_parquet(buffer)
            buffer.seek(0)
            
            success, latency, result = timed_operation(
                fusion_client.from_bytes,
                data=buffer,
                dataset=test_dataset_id,
                series_member=f"2025040{i}",
                show_progress=False
            )
            if success:
                metrics.record_success(latency)
            else:
                metrics.record_failure(latency, result)
        
        metrics.stop_timer()
        metrics.print_summary()


# ============================================================================
# Metadata Operations Performance Tests
# ============================================================================

@pytest.mark.performance
class TestMetadataPerformance:
    """Performance tests for metadata operations"""
    
    @pytest.mark.skip(reason="Requires live Fusion connection")
    def test_list_attributes_light(self, fusion_client, test_dataset_id):
        """Light: List dataset attributes"""
        metrics = PerformanceMetrics("list_dataset_attributes", "light")
        metrics.start_timer()
        
        for i in range(20):
            success, latency, result = timed_operation(
                fusion_client.list_dataset_attributes,
                dataset=test_dataset_id
            )
            if success:
                metrics.record_success(latency)
            else:
                metrics.record_failure(latency, result)
        
        metrics.stop_timer()
        metrics.print_summary()
        
        assert metrics.get_metrics()['latency_mean_ms'] < 1000
    
    @pytest.mark.skip(reason="Requires live Fusion connection")
    def test_list_datasetmembers_scalability(self, fusion_client, test_dataset_id):
        """Scalability: Dataset with many members"""
        metrics = PerformanceMetrics("list_datasetmembers", "scalability_many_members")
        metrics.start_timer()
        
        for i in range(10):
            success, latency, result = timed_operation(
                fusion_client.list_datasetmembers,
                dataset=test_dataset_id,
                max_results=-1
            )
            if success:
                metrics.record_success(latency)
            else:
                metrics.record_failure(latency, result)
        
        metrics.stop_timer()
        metrics.print_summary()
    
    @pytest.mark.skip(reason="Requires live Fusion connection")
    def test_list_products_concurrent(self, fusion_client):
        """Heavy: Concurrent product queries"""
        metrics = PerformanceMetrics("list_products", "heavy_concurrent")
        metrics.start_timer()
        
        with ThreadPoolExecutor(max_workers=15) as executor:
            futures = []
            for i in range(75):
                future = executor.submit(
                    timed_operation,
                    fusion_client.list_products,
                    catalog="common"
                )
                futures.append(future)
            
            for future in as_completed(futures):
                success, latency, result = future.result()
                if success:
                    metrics.record_success(latency)
                else:
                    metrics.record_failure(latency, result)
        
        metrics.stop_timer()
        metrics.print_summary()


# ============================================================================
# Lineage Operations Performance Tests
# ============================================================================

@pytest.mark.performance
class TestLineagePerformance:
    """Performance tests for lineage operations"""
    
    @pytest.mark.skip(reason="Requires live Fusion connection")
    def test_list_dataset_lineage_light(self, fusion_client, test_dataset_id):
        """Light: Query dataset lineage"""
        metrics = PerformanceMetrics("list_dataset_lineage", "light")
        metrics.start_timer()
        
        for i in range(10):
            success, latency, result = timed_operation(
                fusion_client.list_dataset_lineage,
                dataset_id=test_dataset_id
            )
            if success:
                metrics.record_success(latency)
            else:
                metrics.record_failure(latency, result)
        
        metrics.stop_timer()
        metrics.print_summary()
    
    @pytest.mark.skip(reason="Requires live Fusion connection")
    def test_list_attribute_lineage_heavy(self, fusion_client, test_dataset_id):
        """Heavy: Query attribute lineage for multiple attributes"""
        metrics = PerformanceMetrics("list_attribute_lineage", "heavy_multiple_attrs")
        metrics.start_timer()
        
        attributes = [f"attr_{i}" for i in range(20)]
        
        for attr in attributes:
            success, latency, result = timed_operation(
                fusion_client.list_attribute_lineage,
                entity_type="Dataset",
                entity_identifier=test_dataset_id,
                attribute_identifier=attr
            )
            if success:
                metrics.record_success(latency)
            else:
                metrics.record_failure(latency, result)
        
        metrics.stop_timer()
        metrics.print_summary()


# ============================================================================
# End-to-End Workflow Performance Tests
# ============================================================================

@pytest.mark.performance
class TestWorkflowPerformance:
    """End-to-end workflow performance tests"""
    
    @pytest.mark.skip(reason="Requires live Fusion connection")
    def test_full_upload_download_cycle(self, fusion_client, test_dataset_id, tmp_path):
        """Complete workflow: Create data → Upload → Download → Verify"""
        metrics = PerformanceMetrics("full_workflow", "upload_download_cycle")
        metrics.start_timer()
        
        for i in range(3):
            # Create test data
            data = np.random.rand(10000, 20)
            df = pd.DataFrame(data, columns=[f"col_{i}" for i in range(20)])
            upload_file = tmp_path / f"upload_{i}.parquet"
            df.to_parquet(upload_file)
            
            workflow_start = time.time()
            
            # Upload
            try:
                fusion_client.upload(
                    path=str(upload_file),
                    dataset=test_dataset_id,
                    dt_str=f"2025050{i}",
                    show_progress=False
                )
                
                # Download
                download_folder = tmp_path / f"download_{i}"
                fusion_client.download(
                    dataset=test_dataset_id,
                    dt_str=f"2025050{i}",
                    download_folder=str(download_folder),
                    show_progress=False
                )
                
                # Verify
                downloaded_df = pd.read_parquet(download_folder)
                assert downloaded_df.shape == df.shape
                
                workflow_latency = time.time() - workflow_start
                metrics.record_success(workflow_latency)
                print(f"  Workflow {i+1} completed in {workflow_latency:.2f}s")
                
            except Exception as e:
                workflow_latency = time.time() - workflow_start
                metrics.record_failure(workflow_latency, str(e))
        
        metrics.stop_timer()
        metrics.print_summary()
        
        assert metrics.success_rate_pct == 100


# ============================================================================
# Stress Tests
# ============================================================================

@pytest.mark.performance
@pytest.mark.stress
class TestStressScenarios:
    """Stress testing scenarios"""
    
    @pytest.mark.skip(reason="Requires live Fusion connection")
    def test_sustained_load(self, fusion_client):
        """Sustained load: Continuous operations over 5 minutes"""
        metrics = PerformanceMetrics("sustained_load", "5_minute_test")
        metrics.start_timer()
        
        end_time = time.time() + 300  # 5 minutes
        operation_count = 0
        
        while time.time() < end_time:
            success, latency, result = timed_operation(
                fusion_client.list_catalogs
            )
            if success:
                metrics.record_success(latency)
            else:
                metrics.record_failure(latency, result)
            
            operation_count += 1
            time.sleep(0.5)  # Small delay between operations
        
        metrics.stop_timer()
        metrics.print_summary()
        
        print(f"Total operations in 5 minutes: {operation_count}")
    
    @pytest.mark.skip(reason="Requires live Fusion connection")
    def test_burst_traffic(self, fusion_client):
        """Burst traffic: Sudden spike in concurrent requests"""
        metrics = PerformanceMetrics("burst_traffic", "100_concurrent")
        metrics.start_timer()
        
        with ThreadPoolExecutor(max_workers=100) as executor:
            futures = []
            for i in range(100):
                future = executor.submit(
                    timed_operation,
                    fusion_client.list_catalogs
                )
                futures.append(future)
            
            for future in as_completed(futures):
                success, latency, result = future.result()
                if success:
                    metrics.record_success(latency)
                else:
                    metrics.record_failure(latency, result)
        
        metrics.stop_timer()
        metrics.print_summary()


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def fusion_client():
    """Mock or real Fusion client based on environment"""
    # Replace with actual client initialization
    # from fusion import Fusion
    # return Fusion()
    pytest.skip("Requires Fusion client configuration")


@pytest.fixture
def test_dataset_id():
    """Test dataset ID for performance tests"""
    return "PERF_TEST_DATASET"


@pytest.fixture
def large_dataset_id():
    """Large dataset ID for heavy load tests"""
    return "PERF_TEST_LARGE_DATASET"


# ============================================================================
# Performance Report Generator
# ============================================================================

def generate_performance_report(metrics_list: List[Dict], output_file: str):
    """Generate HTML performance report from collected metrics"""
    
    df = pd.DataFrame(metrics_list)
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Fusion SDK Performance Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #4CAF50; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .metric {{ font-weight: bold; }}
            .good {{ color: green; }}
            .warning {{ color: orange; }}
            .bad {{ color: red; }}
        </style>
    </head>
    <body>
        <h1>Fusion SDK Performance Test Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>Summary Statistics</h2>
        {df.to_html(index=False, classes='dataframe')}
        
        <h2>Key Metrics</h2>
        <ul>
            <li>Total Test Scenarios: {len(df)}</li>
            <li>Average Success Rate: {df['success_rate_pct'].mean():.2f}%</li>
            <li>Average Throughput: {df['throughput_ops_per_sec'].mean():.2f} ops/sec</li>
            <li>Average P95 Latency: {df['latency_p95_ms'].mean():.2f} ms</li>
        </ul>
    </body>
    </html>
    """
    
    with open(output_file, 'w') as f:
        f.write(html)
    
    print(f"Performance report generated: {output_file}")


if __name__ == "__main__":
    print("Run with: pytest py_tests/test_performance.py -v")
