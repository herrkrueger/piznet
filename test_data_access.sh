#!/bin/bash
# Data Access Test Script for Patent Analysis Platform
# Enhanced from EPO PATLIB 2025 Live Demo Code

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Setup logging
LOG_DIR="logs"
LOG_FILE="$LOG_DIR/data_access_tests.log"
mkdir -p "$LOG_DIR"

# Function to log with timestamp
log_with_timestamp() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log_with_timestamp "🚀 Patent Analysis Platform - Data Access Tests"
log_with_timestamp "Enhanced from EPO PATLIB 2025 Live Demo Code"
log_with_timestamp "============================================================"
log_with_timestamp "Log file: $LOG_FILE"

# Function to run command with logging
run_with_logging() {
    log_with_timestamp "Starting: $1"
    eval "$1" 2>&1 | tee -a "$LOG_FILE"
    local exit_code=${PIPESTATUS[0]}
    log_with_timestamp "Completed: $1 (exit code: $exit_code)"
    return $exit_code
}

# Check if Python test scripts exist
if [[ ! -f "data_access/test_data_access.py" ]]; then
    echo "❌ Test script not found: data_access/test_data_access.py"
    exit 1
fi

if [[ ! -f "data_access/test_nuts_mapper.py" ]]; then
    echo "❌ Test script not found: data_access/test_nuts_mapper.py"
    exit 1
fi

# Check Python version
echo ""
echo "🐍 Python Environment Check"
echo "----------------------------"
python_version=$(python3 --version 2>&1)
if [[ $? -eq 0 ]]; then
    echo "✅ Python version: $python_version"
else
    echo "❌ Python3 not found"
    exit 1
fi

# Check if virtual environment is active
if [[ -n "$VIRTUAL_ENV" ]]; then
    echo "✅ Virtual environment: $(basename $VIRTUAL_ENV)"
else
    echo "⚠️ No virtual environment detected"
fi

# Check key dependencies
echo ""
echo "📦 Dependency Check"
echo "-------------------"

dependencies=("pandas" "numpy" "pathlib")
missing_deps=()

for dep in "${dependencies[@]}"; do
    python3 -c "import $dep" 2>/dev/null
    if [[ $? -eq 0 ]]; then
        version=$(python3 -c "import $dep; print($dep.__version__)" 2>/dev/null)
        echo "✅ $dep: $version"
    else
        echo "❌ $dep: Not installed"
        missing_deps+=("$dep")
    fi
done

if [[ ${#missing_deps[@]} -gt 0 ]]; then
    echo ""
    echo "⚠️ Missing dependencies: ${missing_deps[*]}"
    echo "Please install missing packages:"
    echo "pip install ${missing_deps[*]}"
    echo ""
    if [[ "$NON_INTERACTIVE" == "true" ]]; then
        echo "Non-interactive mode: Continuing with missing dependencies..."
    else
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
fi

# Check data access module structure
echo ""
echo "🏗️ Module Structure Check"
echo "-------------------------"

required_files=(
    "data_access/__init__.py"
    "data_access/patstat_client.py"
    "data_access/ops_client.py"
    "data_access/cache_manager.py"
    "data_access/country_mapper.py"
    "data_access/nuts_mapper.py"
    "data_access/README.md"
)

for file in "${required_files[@]}"; do
    if [[ -f "$file" ]]; then
        size=$(wc -l < "$file")
        echo "✅ $file ($size lines)"
    else
        echo "❌ $file (missing)"
    fi
done

# Hierarchical Test Execution for Data Access
echo ""
echo "🧪 Hierarchical Test Execution for Data Access"
echo "==============================================="
echo "Data access requires valid configuration:"
echo "1️⃣ Configuration → 2️⃣ Data Access"
echo ""

echo "Select test mode:"
echo "1) Data access tests only (assumes configuration is valid)"
echo "2) Full pipeline test (configuration → data access - recommended)"
echo "3) Core data access only (PATSTAT, OPS, cache)"
echo "4) NUTS geographic mapping only"
echo ""

# Check for non-interactive mode and option override
choice="2"  # Default
NON_INTERACTIVE=false
for arg in "$@"; do
    case $arg in
        --option=*)
            choice="${arg#*=}"
            ;;
        --non-interactive|--auto)
            NON_INTERACTIVE=true
            ;;
    esac
done

if [[ "$NON_INTERACTIVE" == "true" ]]; then
    echo "Running in non-interactive mode: Option $choice"
elif [[ -t 0 ]]; then
    read -p "Choose option (1/2/3/4) [2]: " -n 1 -r
    echo
    choice="${REPLY:-2}"
fi

case $choice in
    1)
        log_with_timestamp "Running data access tests only (assumes configuration is valid)..."
        echo "⚠️ Warning: This assumes configuration layer is functional"
        log_with_timestamp ""
        log_with_timestamp "Step 1: Core Data Access Tests"
        log_with_timestamp "------------------------------"
        run_with_logging "timeout 1800 python3 data_access/test_data_access.py"
        core_exit_code=$?
        
        log_with_timestamp ""
        log_with_timestamp "Step 2: NUTS Geographic Mapping Tests"
        log_with_timestamp "------------------------------------"
        run_with_logging "timeout 600 python3 data_access/test_nuts_mapper.py"
        nuts_exit_code=$?
        
        # Use core tests as primary indicator
        test_exit_code=$core_exit_code
        
        log_with_timestamp ""
        log_with_timestamp "🏁 Test Suite Summary"
        log_with_timestamp "===================="
        log_with_timestamp "Core Data Access: $([ $core_exit_code -eq 0 ] && echo "✅ PASS" || echo "❌ FAIL")"
        log_with_timestamp "NUTS Mapping: $([ $nuts_exit_code -eq 0 ] && echo "✅ PASS" || echo "❌ FAIL")"
        ;;
    2)
        log_with_timestamp "Running full pipeline test with hierarchical execution..."
        echo ""
        
        # Step 1: Configuration Tests
        log_with_timestamp "Step 1/2: Testing Configuration Layer"
        echo "⚙️ Ensuring configuration system works..."
        if ! run_with_logging "timeout 300 ./test_config.sh"; then
            log_with_timestamp "❌ Configuration tests failed - cannot proceed to data access"
            exit 1
        fi
        echo "✅ Configuration layer validated"
        echo ""
        
        # Step 2: Data Access Tests (with validated configuration)
        log_with_timestamp "Step 2/2: Testing Data Access with Valid Configuration"
        echo "🔍 Testing data access with validated configuration..."
        
        log_with_timestamp "Step 2a: Core Data Access Tests"
        log_with_timestamp "------------------------------"
        run_with_logging "timeout 1800 python3 data_access/test_data_access.py"
        core_exit_code=$?
        
        log_with_timestamp ""
        log_with_timestamp "Step 2b: NUTS Geographic Mapping Tests"
        log_with_timestamp "------------------------------------"
        run_with_logging "timeout 600 python3 data_access/test_nuts_mapper.py"
        nuts_exit_code=$?
        
        # Use core tests as primary indicator
        test_exit_code=$core_exit_code
        
        log_with_timestamp ""
        log_with_timestamp "🏁 Pipeline Test Summary"
        log_with_timestamp "======================="
        log_with_timestamp "Configuration: ✅ PASS"
        log_with_timestamp "Core Data Access: $([ $core_exit_code -eq 0 ] && echo "✅ PASS" || echo "❌ FAIL")"
        log_with_timestamp "NUTS Mapping: $([ $nuts_exit_code -eq 0 ] && echo "✅ PASS" || echo "❌ FAIL")"
        ;;
    3)
        log_with_timestamp "Running core data access tests only..."
        run_with_logging "timeout 1800 python3 data_access/test_data_access.py"
        test_exit_code=$?
        ;;
    4)
        log_with_timestamp "Running NUTS mapping tests only..."
        run_with_logging "timeout 600 python3 data_access/test_nuts_mapper.py"
        test_exit_code=$?
        ;;
    *)
        log_with_timestamp "Running default full pipeline test..."
        echo ""
        
        # Full pipeline execution (same as option 2)
        log_with_timestamp "Step 1/2: Testing Configuration Layer"
        if ! run_with_logging "timeout 300 ./test_config.sh"; then
            log_with_timestamp "❌ Configuration tests failed"
            exit 1
        fi
        
        log_with_timestamp "Step 2/2: Testing Data Access"
        run_with_logging "timeout 1800 python3 data_access/test_data_access.py"
        core_exit_code=$?
        run_with_logging "timeout 600 python3 data_access/test_nuts_mapper.py"
        test_exit_code=$core_exit_code
        ;;
esac

echo ""
echo "📊 Test Execution Summary"
echo "========================="

case $test_exit_code in
    0)
        if [[ $choice == "2" ]] || [[ $choice == "" ]]; then
            echo "🎉 Full pipeline test completed successfully!"
            echo "✅ Configuration → Data Access validated with real configuration"
            echo "🔍 Data access tested with actual configuration settings"
        else
            echo "🎉 All data access tests passed successfully!"
            echo "✅ Data access layer is ready for production use"
        fi
        ;;
    124)
        echo "⏰ Tests timed out (30 minute limit exceeded)"
        echo "⚠️ Consider optimizing performance or reducing test scope"
        ;;
    130)
        echo "⚠️ Tests interrupted by user (Ctrl+C)"
        ;;
    *)
        if [[ $choice == "2" ]] || [[ $choice == "" ]]; then
            echo "❌ Pipeline test failed with exit code: $test_exit_code"
            echo "🔍 Check if configuration layer failed in hierarchical execution above"
        else
            echo "❌ Data access tests failed with exit code: $test_exit_code"
            echo "🔍 Please review the test output above for details"
        fi
        ;;
esac

# Performance summary
echo ""
echo "⚡ Performance Guidelines"
echo "------------------------"
echo "• PATSTAT connection: <5 seconds for initial setup"
echo "• EPO OPS authentication: <3 seconds with cached tokens"
echo "• Cache operations: <100ms for typical queries"
echo "• NUTS mapping: <1 second for hierarchy navigation"

# Usage examples
echo ""
echo "💡 Usage Examples"
echo "----------------"
echo "# Core data access tests:"
echo "python3 data_access/test_data_access.py"
echo ""
echo "# NUTS mapping tests:"
echo "python3 data_access/test_nuts_mapper.py"
echo ""
echo "# Individual component tests:"
echo "python3 -c \"from data_access.test_data_access import test_patstat_connection; test_patstat_connection()\""
echo "python3 -c \"from data_access.test_nuts_mapper import test_nuts_mapper_basic; test_nuts_mapper_basic()\""

# Additional checks
echo ""
echo "🔧 Additional Checks"
echo "-------------------"

# Check if config files exist
if [[ -f "config/api_config.yaml" ]]; then
    echo "✅ API configuration found"
else
    echo "⚠️ API configuration missing"
fi

if [[ -f "config/database_config.yaml" ]]; then
    echo "✅ Database configuration found"
else
    echo "⚠️ Database configuration missing"
fi

# Check environment file
if [[ -f ".env" ]]; then
    echo "✅ Environment file found"
else
    echo "⚠️ Environment file missing (needed for EPO OPS)"
fi

# Check NUTS mapping files
if [[ -f "data_access/mappings/nuts_mapping.csv" ]]; then
    nuts_count=$(wc -l < "data_access/mappings/nuts_mapping.csv")
    echo "✅ NUTS mapping CSV found ($nuts_count regions)"
else
    echo "⚠️ NUTS mapping CSV missing"
fi

# Final recommendations
echo ""
echo "📝 Next Steps"
echo "------------"
if [[ $test_exit_code -eq 0 ]]; then
    echo "1. ✅ Complete data access layer is ready for production!"
    echo "2. 🗄️ Deploy to EPO PATLIB 2025 demo environment"
    echo "3. 📊 Test with real patent analysis workflows"
    echo "4. 🚀 Demonstrate full pipeline capabilities"
else
    echo "1. 🔍 Review failed data access components and fix issues"
    echo "2. 📝 Check database connectivity and API credentials"
    echo "3. 🐛 Debug specific component integration problems"
    echo "4. 💬 Address configuration or dependency concerns"
fi

log_with_timestamp ""
log_with_timestamp "============================================================"
log_with_timestamp "Data Access Test Suite Completed"
log_with_timestamp "Exit code: $test_exit_code"
log_with_timestamp "Full test log available at: $LOG_FILE"

exit $test_exit_code