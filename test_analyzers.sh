#!/bin/bash
# Analyzers Testing Script for Patent Analysis Platform
# Tests technology, regional, and trends analysis modules

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Setup logging
LOG_DIR="logs"
LOG_FILE="$LOG_DIR/analyzers_tests.log"
mkdir -p "$LOG_DIR"

# Function to log with timestamp
log_with_timestamp() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log_with_timestamp "🚀 Patent Analysis Platform - Analyzers Test Suite"
log_with_timestamp "Enhanced from EPO PATLIB 2025 Live Demo Code"
log_with_timestamp "=================================================="
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
if [[ ! -f "analyzers/test_analyzers.py" ]]; then
    echo "❌ Test script not found: analyzers/test_analyzers.py"
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

dependencies=("pandas" "numpy" "networkx")
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
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check analyzer module structure
echo ""
echo "🏗️ Analyzer Module Structure Check"
echo "----------------------------------"

required_files=(
    "analyzers/__init__.py"
    "analyzers/technology.py"
    "analyzers/regional.py"
    "analyzers/trends.py"
    "analyzers/test_analyzers.py"
)

for file in "${required_files[@]}"; do
    if [[ -f "$file" ]]; then
        size=$(wc -l < "$file")
        echo "✅ $file ($size lines)"
    else
        echo "❌ $file (missing)"
    fi
done

# Check for CPC/IPC integration
echo ""
echo "🔬 Classification Integration Check"
echo "----------------------------------"

if [[ -f "data_access/cpc_database_client.py" ]]; then
    echo "✅ CPC database client available"
else
    echo "⚠️ CPC database client missing"
fi

if [[ -f "data_access/ipc_database_client.py" ]]; then
    echo "✅ IPC database client available"
else
    echo "⚠️ IPC database client missing"
fi

if [[ -f "data_access/mappings/cpc_database.sqlite" ]]; then
    echo "✅ CPC database available"
else
    echo "⚠️ CPC database missing"
fi

if [[ -f "data_access/mappings/ipc_database.sqlite" ]]; then
    echo "✅ IPC database available"
else
    echo "⚠️ IPC database missing"
fi

# Hierarchical Test Execution for Analyzers
echo ""
echo "🧪 Hierarchical Test Execution for Analyzers"
echo "============================================="
echo "Analyzers require processed data from the full pipeline:"
echo "1️⃣ Data Access → 2️⃣ Processors → 3️⃣ Analyzers"
echo ""

echo "Select test mode:"
echo "1) Quick analyzer tests only (assumes processed data exists)"
echo "2) Full pipeline test (data access → processors → analyzers - recommended)"  
echo "3) Analyzers only with performance testing"
echo ""

# Default to full pipeline if no interaction
if [[ -t 0 ]]; then
    read -p "Choose option (1/2/3) [2]: " -n 1 -r
    echo
    choice="${REPLY:-2}"
else
    choice="2"
fi

case $choice in
    1)
        log_with_timestamp "Running analyzer tests only (assumes processed data exists)..."
        echo "⚠️ Warning: This assumes data access and processors are functional"
        run_with_logging "timeout 600 python3 analyzers/test_analyzers.py"
        ;;
    2)
        log_with_timestamp "Running full pipeline test with hierarchical execution..."
        echo ""
        
        # Step 1: Data Access Tests
        log_with_timestamp "Step 1/3: Testing Data Access Layer"
        echo "🔍 Ensuring data access functionality works..."
        if ! run_with_logging "timeout 300 ./test_data_access.sh"; then
            log_with_timestamp "❌ Data access tests failed - cannot proceed to analyzers"
            exit 1
        fi
        echo "✅ Data access layer validated"
        echo ""
        
        # Step 2: Processor Tests
        log_with_timestamp "Step 2/3: Testing Processors"
        echo "⚙️ Ensuring data processing functionality works..."
        if ! run_with_logging "timeout 600 ./test_processors.sh"; then
            log_with_timestamp "❌ Processor tests failed - cannot proceed to analyzers"
            exit 1
        fi
        echo "✅ Processors validated"
        echo ""
        
        # Step 3: Analyzer Tests (with real processed data)
        log_with_timestamp "Step 3/3: Testing Analyzers with Real Processed Data"
        echo "📊 Testing analyzers with processed data from pipeline..."
        run_with_logging "timeout 600 python3 analyzers/test_analyzers.py"
        ;;
    3)
        log_with_timestamp "Running performance tests with extended timeout..."
        echo "⚠️ Warning: This assumes data access and processors are functional"
        run_with_logging "timeout 1800 python3 analyzers/test_analyzers.py"
        ;;
    *)
        log_with_timestamp "Running default full pipeline test..."
        echo ""
        
        # Full pipeline execution (same as option 2)
        log_with_timestamp "Step 1/3: Testing Data Access Layer"
        if ! run_with_logging "timeout 300 ./test_data_access.sh"; then
            log_with_timestamp "❌ Data access tests failed"
            exit 1
        fi
        
        log_with_timestamp "Step 2/3: Testing Processors"  
        if ! run_with_logging "timeout 600 ./test_processors.sh"; then
            log_with_timestamp "❌ Processor tests failed"
            exit 1
        fi
        
        log_with_timestamp "Step 3/3: Testing Analyzers"
        run_with_logging "timeout 600 python3 analyzers/test_analyzers.py"
        ;;
esac

test_exit_code=$?

echo ""
echo "📊 Test Execution Summary"
echo "========================="

case $test_exit_code in
    0)
        if [[ $choice == "2" ]] || [[ $choice == "" ]]; then
            echo "🎉 Full pipeline test completed successfully!"
            echo "✅ Data Access → Processors → Analyzers validated with real data flow"
            echo "📊 Analyzers tested with actual processed data from pipeline"
        else
            echo "🎉 All analyzer tests passed!"
            echo "✅ Technology, regional, and trends analyzers are ready for production"
        fi
        ;;
    124)
        echo "⏰ Tests timed out (10 minute limit exceeded)"
        echo "⚠️ Consider optimizing analyzer performance or reducing test scope"
        ;;
    130)
        echo "⚠️ Tests interrupted by user (Ctrl+C)"
        ;;
    *)
        if [[ $choice == "2" ]] || [[ $choice == "" ]]; then
            echo "❌ Pipeline test failed with exit code: $test_exit_code"
            echo "🔍 Check which layer failed in the hierarchical execution above"
        else
            echo "❌ Analyzer tests failed with exit code: $test_exit_code"
            echo "🔍 Please review the test output above for details"
        fi
        ;;
esac

# Performance summary
echo ""
echo "⚡ Analyzer Performance Guidelines"
echo "--------------------------------"
echo "• Technology analysis: Process 1000+ patents in <5 seconds"
echo "• Regional analysis: Handle 50+ jurisdictions efficiently"
echo "• Trends analysis: Analyze multi-year time series data"
echo "• Network building: Support graphs with 10,000+ nodes"

# Usage examples
echo ""
echo "💡 Usage Examples"
echo "----------------"
echo "# Run analyzer tests:"
echo "python3 analyzers/test_analyzers.py"
echo ""
echo "# Test specific analyzer:"
echo "python3 -c \"from analyzers import TechnologyAnalyzer; print('Technology analyzer available')\""
echo ""
echo "# Import analyzers in your code:"
echo "from analyzers import TechnologyAnalyzer, RegionalAnalyzer, TrendsAnalyzer"

# Additional checks
echo ""
echo "🔧 Additional Checks"
echo "-------------------"

# Check if analyzers can be imported
python3 -c "from analyzers import TechnologyAnalyzer, RegionalAnalyzer, TrendsAnalyzer" 2>/dev/null
if [[ $? -eq 0 ]]; then
    echo "✅ All analyzer classes import successfully"
else
    echo "❌ Analyzer import issues detected"
fi

# Check classification integration
python3 -c "from data_access.classification_config import get_classification_config" 2>/dev/null
if [[ $? -eq 0 ]]; then
    echo "✅ Classification system integration available"
else
    echo "⚠️ Classification system integration issues"
fi

# Final recommendations
echo ""
echo "📝 Next Steps"
echo "------------"
if [[ $test_exit_code -eq 0 ]]; then
    echo "1. ✅ All analyzers tested and verified!"
    echo "2. 🔬 Run with real patent data for validation"
    echo "3. 📊 Test integration with visualization components"
    echo "4. 🚀 Deploy to production patent analysis pipeline"
else
    echo "1. 🔍 Review failed analyzer tests and fix issues"
    echo "2. 📝 Check analyzer integration with classification systems"
    echo "3. 🔧 Verify network building and complex analysis functionality"
    echo "4. 💬 Test with different data sizes and complexity levels"
fi

log_with_timestamp ""
log_with_timestamp "=================================================="
log_with_timestamp "Analyzers Test Suite Completed"
log_with_timestamp "Exit code: $test_exit_code"
log_with_timestamp "Full test log available at: $LOG_FILE"

exit $test_exit_code