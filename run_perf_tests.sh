#!/bin/bash
# Quick Performance Test Script
# ==============================
# Run common performance test scenarios quickly

set -e

echo "================================================"
echo "  Fusion SDK Performance Testing Quick Start"
echo "================================================"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check Python availability
if ! command -v python &> /dev/null; then
    echo -e "${RED}Error: Python not found${NC}"
    exit 1
fi

# Check if pytest is installed
if ! python -c "import pytest" 2>/dev/null; then
    echo -e "${YELLOW}Installing pytest...${NC}"
    pip install pytest pytest-benchmark
fi

# Function to run tests
run_test() {
    local test_name=$1
    local test_cmd=$2
    
    echo -e "\n${GREEN}Running: ${test_name}${NC}"
    echo "Command: ${test_cmd}"
    echo "----------------------------------------"
    eval $test_cmd
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ ${test_name} completed${NC}"
    else
        echo -e "${RED}✗ ${test_name} failed${NC}"
    fi
}

# Menu
echo "Select test scenario:"
echo "1. Quick Smoke Test (5 min)"
echo "2. Light Scenarios Only (15 min)"
echo "3. Heavy Scenarios Only (30 min)"
echo "4. Full Performance Suite (1-2 hours)"
echo "5. Download Performance Tests"
echo "6. Upload Performance Tests"
echo "7. Stress Test (Custom duration)"
echo "8. Custom (specify test pattern)"
echo "9. Exit"
echo ""
read -p "Enter choice [1-9]: " choice

case $choice in
    1)
        echo -e "\n${YELLOW}Running Quick Smoke Test...${NC}"
        run_test "Catalog Light Test" "pytest py_tests/test_performance.py::TestCatalogPerformance::test_list_catalogs_light -v"
        run_test "Dataset Light Test" "pytest py_tests/test_performance.py::TestDatasetPerformance::test_list_datasets_light -v"
        run_test "Download Light Test" "pytest py_tests/test_performance.py::TestDownloadPerformance::test_download_small_file_light -v"
        ;;
        
    2)
        echo -e "\n${YELLOW}Running Light Scenarios...${NC}"
        run_test "All Light Tests" "pytest py_tests/test_performance.py -k 'light' -v"
        ;;
        
    3)
        echo -e "\n${YELLOW}Running Heavy Scenarios...${NC}"
        run_test "All Heavy Tests" "pytest py_tests/test_performance.py -k 'heavy' -v"
        ;;
        
    4)
        echo -e "\n${YELLOW}Running Full Performance Suite...${NC}"
        run_test "All Performance Tests" "pytest py_tests/test_performance.py -v -m performance"
        ;;
        
    5)
        echo -e "\n${YELLOW}Running Download Performance Tests...${NC}"
        run_test "Download Tests" "pytest py_tests/test_performance.py::TestDownloadPerformance -v"
        ;;
        
    6)
        echo -e "\n${YELLOW}Running Upload Performance Tests...${NC}"
        run_test "Upload Tests" "pytest py_tests/test_performance.py::TestUploadPerformance -v"
        ;;
        
    7)
        read -p "Enter duration in seconds (default 300): " duration
        duration=${duration:-300}
        echo -e "\n${YELLOW}Running Stress Test for ${duration}s...${NC}"
        run_test "Stress Test" "pytest py_tests/test_performance.py::TestStressScenarios -v"
        ;;
        
    8)
        read -p "Enter test pattern (e.g., 'download' or 'test_list'): " pattern
        echo -e "\n${YELLOW}Running tests matching: ${pattern}${NC}"
        run_test "Custom Tests" "pytest py_tests/test_performance.py -k '${pattern}' -v"
        ;;
        
    9)
        echo "Exiting..."
        exit 0
        ;;
        
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

# Generate summary
echo ""
echo "================================================"
echo "  Test Execution Complete"
echo "================================================"
echo ""
echo "Next steps:"
echo "  1. Review test output above"
echo "  2. Check generated reports in ./perf_reports/"
echo "  3. Compare with baseline: pytest py_tests/test_performance.py --compare-baseline"
echo ""
echo "For more options, see PERFORMANCE_TESTING.md"
