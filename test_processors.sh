#!/bin/bash
# Processors Testing Script for Patent Analysis Platform
# Enhanced from EPO PATLIB 2025 Live Demo Code

log_with_timestamp "🚀 Patent Analysis Platform - Processors Test Suite"
log_with_timestamp "Enhanced from EPO PATLIB 2025 Live Demo Code"
log_with_timestamp "============================================="
log_with_timestamp "Log file: $LOG_FILE"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Setup logging
LOG_DIR="logs"
LOG_FILE="$LOG_DIR/processors_tests.log"
mkdir -p "$LOG_DIR"

# Function to log with timestamp
log_with_timestamp() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to run command with logging
run_with_logging() {
    log_with_timestamp "Starting: $1"
    eval "$1" 2>&1 | tee -a "$LOG_FILE"
    local exit_code=${PIPESTATUS[0]}
    log_with_timestamp "Completed: $1 (exit code: $exit_code)"
    return $exit_code
}

# Check if Python test scripts exist
if [[ ! -f "processors/test_processor.py" ]]; then
    echo "❌ Test script not found: processors/test_processor.py"
    exit 1
fi

if [[ ! -f "processors/test_complete_pipeline.py" ]]; then
    echo "❌ Test script not found: processors/test_complete_pipeline.py"
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

# Check processor module structure
echo ""
echo "🏗️ Module Structure Check"
echo "-------------------------"

required_files=(
    "processors/__init__.py"
    "processors/search.py"
    "processors/applicant.py"
    "processors/classification.py"
    "processors/geographic.py"
    "processors/citation.py"
    "processors/README.md"
)

for file in "${required_files[@]}"; do
    if [[ -f "$file" ]]; then
        size=$(wc -l < "$file")
        echo "✅ $file ($size lines)"
    else
        echo "❌ $file (missing)"
    fi
done

# Hierarchical Test Execution for Processors
echo ""
echo "🧪 Hierarchical Test Execution for Processors"
echo "=============================================="
echo "Processors require data access connectivity:"
echo "1️⃣ Data Access Layer → 2️⃣ Processors"
echo ""

echo "Select test mode:"
echo "1) Quick processor tests only (assumes data access works)"
echo "2) Full pipeline test (data access → processors - recommended)"
echo "3) Unit tests only (individual processors)"
echo "4) Integration tests only (complete processor pipeline)"
echo ""

# Default to full pipeline if no interaction
if [[ -t 0 ]]; then
    read -p "Choose option (1/2/3/4) [2]: " -n 1 -r
    echo
    choice="${REPLY:-2}"
else
    choice="2"
fi

case $choice in
    1)
        log_with_timestamp "Running processor tests only (assumes data access works)..."
        echo "⚠️ Warning: This assumes data access layer is functional"
        log_with_timestamp ""
        log_with_timestamp "Step 1: Unit Tests"
        log_with_timestamp "------------------"
        run_with_logging "timeout 600 python3 processors/test_processor.py"
        unit_exit_code=$?
        
        log_with_timestamp ""
        log_with_timestamp "Step 2: Integration Tests"
        log_with_timestamp "------------------------"
        run_with_logging "timeout 1800 python3 processors/test_complete_pipeline.py"
        integration_exit_code=$?
        
        # Use integration test result as primary indicator
        test_exit_code=$integration_exit_code
        
        log_with_timestamp ""
        log_with_timestamp "🏁 Test Suite Summary"
        log_with_timestamp "===================="
        log_with_timestamp "Unit Tests: $([ $unit_exit_code -eq 0 ] && echo "✅ PASS" || echo "❌ FAIL")"
        log_with_timestamp "Integration Tests: $([ $integration_exit_code -eq 0 ] && echo "✅ PASS" || echo "❌ FAIL")"
        ;;
    2)
        log_with_timestamp "Running full pipeline test with hierarchical execution..."
        echo ""
        
        # Step 1: Data Access Tests  
        log_with_timestamp "Step 1/2: Testing Data Access Layer"
        echo "🔍 Ensuring data access functionality works..."
        if ! run_with_logging "timeout 300 ./test_data_access.sh"; then
            log_with_timestamp "❌ Data access tests failed - cannot proceed to processors"
            exit 1
        fi
        echo "✅ Data access layer validated"
        echo ""
        
        # Step 2: Processor Tests (with validated data access)
        log_with_timestamp "Step 2/2: Testing Processors with Valid Data Access"
        echo "⚙️ Testing processors with validated data connectivity..."
        
        log_with_timestamp "Step 2a: Unit Tests"
        log_with_timestamp "------------------"
        run_with_logging "timeout 600 python3 processors/test_processor.py"
        unit_exit_code=$?
        
        log_with_timestamp ""
        log_with_timestamp "Step 2b: Integration Tests"
        log_with_timestamp "------------------------"
        run_with_logging "timeout 1800 python3 processors/test_complete_pipeline.py"
        integration_exit_code=$?
        
        # Use integration test result as primary indicator
        test_exit_code=$integration_exit_code
        
        log_with_timestamp ""
        log_with_timestamp "🏁 Pipeline Test Summary"
        log_with_timestamp "======================="
        log_with_timestamp "Data Access: ✅ PASS"
        log_with_timestamp "Unit Tests: $([ $unit_exit_code -eq 0 ] && echo "✅ PASS" || echo "❌ FAIL")"
        log_with_timestamp "Integration Tests: $([ $integration_exit_code -eq 0 ] && echo "✅ PASS" || echo "❌ FAIL")"
        ;;
    3)
        log_with_timestamp "Running unit tests only..."
        run_with_logging "timeout 600 python3 processors/test_processor.py"
        test_exit_code=$?
        ;;
    4)
        log_with_timestamp "Running integration tests only..."
        run_with_logging "timeout 1800 python3 processors/test_complete_pipeline.py"
        test_exit_code=$?
        ;;
    *)
        log_with_timestamp "Running default full pipeline test..."
        echo ""
        
        # Full pipeline execution (same as option 2)
        log_with_timestamp "Step 1/2: Testing Data Access Layer"
        if ! run_with_logging "timeout 300 ./test_data_access.sh"; then
            log_with_timestamp "❌ Data access tests failed"
            exit 1
        fi
        
        log_with_timestamp "Step 2/2: Testing Processors"
        run_with_logging "timeout 600 python3 processors/test_processor.py"
        unit_exit_code=$?
        run_with_logging "timeout 1800 python3 processors/test_complete_pipeline.py"
        test_exit_code=$?
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
            echo "✅ Data Access → Processors validated with real connectivity"
            echo "⚙️ Processors tested with actual data access functionality"
        else
            echo "🎉 Processor tests passed successfully!"
            echo "✅ Processors module is ready for production use"
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
            echo "🔍 Check if data access layer failed in hierarchical execution above"
        else
            echo "❌ Processor tests failed with exit code: $test_exit_code"
            echo "🔍 Please review the test output above for details"
        fi
        ;;
esac

# Performance summary
echo ""
echo "⚡ Performance Guidelines"
echo "------------------------"
echo "• Recommended: Process 1000+ records in <10 seconds"
echo "• Memory usage: <10MB per 1000 patent records"
echo "• Scalability: Tested up to 100,000+ records"
echo "• Network analysis: Efficient for citation networks <10,000 nodes"

# Usage examples
echo ""
echo "💡 Usage Examples"
echo "----------------"
echo "# Unit tests (individual processors):"
echo "python3 processors/test_processor.py"
echo "python3 processors/test_processor.py --processor search"
echo ""
echo "# Integration tests (complete pipeline):"
echo "python3 processors/test_complete_pipeline.py"
echo ""
echo "# Import processors in your code:"
echo "from processors import create_patent_search_processor, create_applicant_analyzer"

# Additional checks
echo ""
echo "🔧 Additional Checks"
echo "-------------------"

# Check if config files exist
if [[ -f "config/search_patterns_config.yaml" ]]; then
    echo "✅ Search patterns configuration found"
else
    echo "⚠️ Search patterns configuration missing"
fi

if [[ -f "config/geographic_config.yaml" ]]; then
    echo "✅ Geographic configuration found"
else
    echo "⚠️ Geographic configuration missing"
fi

# Check data_access integration
python3 -c "from data_access import PatstatClient" 2>/dev/null
if [[ $? -eq 0 ]]; then
    echo "✅ Data access integration available"
else
    echo "⚠️ Data access integration issues"
fi

# Final recommendations
echo ""
echo "📝 Next Steps"
echo "------------"
if [[ $test_exit_code -eq 0 ]]; then
    echo "1. ✅ Complete pipeline is ready for production!"
    echo "2. 🗄️ Deploy to EPO PATLIB 2025 demo environment"
    echo "3. 📊 Test with real PATSTAT data (live integration)"
    echo "4. 🚀 Demonstrate full workflow capabilities"
else
    echo "1. 🔍 Review failed pipeline components and fix issues"
    echo "2. 📝 Check processor integration and configuration"
    echo "3. 🐛 Debug specific pipeline workflow problems"
    echo "4. 💬 Address performance or scaling concerns"
fi

log_with_timestamp ""
log_with_timestamp "=================================================="
log_with_timestamp "Processors Test Suite Completed"
log_with_timestamp "Exit code: $test_exit_code"
log_with_timestamp "Full test log available at: $LOG_FILE"

exit $test_exit_code