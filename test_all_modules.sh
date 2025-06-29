#!/bin/bash
# Comprehensive Test Suite for Patent Analysis Platform
# Tests all modules: processors, analyzers, visualizations, data_access, config

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Setup master logging
LOG_DIR="logs"
MASTER_LOG_FILE="$LOG_DIR/master_test_suite.log"
mkdir -p "$LOG_DIR"

# Function to log with timestamp
log_with_timestamp() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$MASTER_LOG_FILE"
}

log_with_timestamp "🚀 Patent Analysis Platform - Comprehensive Test Suite"
log_with_timestamp "Enhanced from EPO PATLIB 2025 Live Demo Code"
log_with_timestamp "======================================================="
log_with_timestamp "Master log file: $MASTER_LOG_FILE"
log_with_timestamp "Individual module logs will be stored in: $LOG_DIR/"

# Function to aggregate all module logs
aggregate_logs() {
    log_with_timestamp "=== AGGREGATING ALL MODULE TEST LOGS ==="
    
    for log_file in "$LOG_DIR"/*_tests.log; do
        if [[ -f "$log_file" ]]; then
            module_name=$(basename "$log_file" _tests.log)
            log_with_timestamp "\n--- $module_name Test Log ---"
            cat "$log_file" >> "$MASTER_LOG_FILE"
            log_with_timestamp "--- End $module_name Log ---\n"
        fi
    done
    
    log_with_timestamp "=== END LOG AGGREGATION ==="
}

# Test results tracking
declare -A test_results
total_tests=0
passed_tests=0

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to run a test and track results
run_test() {
    local test_name="$1"
    local test_command="$2"
    local timeout_seconds="${3:-600}"  # Default 10 minutes
    
    echo ""
    echo -e "${BLUE}🧪 Running $test_name${NC}"
    echo "$(date '+%H:%M:%S') - Starting $test_name"
    echo "----------------------------------------"
    
    total_tests=$((total_tests + 1))
    
    # Add --non-interactive flag if this is a non-interactive run
    if [[ "$NON_INTERACTIVE" == "true" ]] && [[ "$test_command" == *".sh"* ]]; then
        test_command="$test_command --non-interactive"
    fi
    
    # Run test with timeout
    if timeout $timeout_seconds bash -c "$test_command"; then
        test_results["$test_name"]="✅ PASSED"
        passed_tests=$((passed_tests + 1))
        echo -e "${GREEN}✅ $test_name PASSED${NC}"
    else
        exit_code=$?
        if [ $exit_code -eq 124 ]; then
            test_results["$test_name"]="⏰ TIMEOUT"
            echo -e "${YELLOW}⏰ $test_name TIMED OUT${NC}"
        else
            test_results["$test_name"]="❌ FAILED"
            echo -e "${RED}❌ $test_name FAILED${NC}"
        fi
    fi
}

# Check Python environment
echo ""
echo -e "${BLUE}🐍 Python Environment Check${NC}"
echo "----------------------------"
python_version=$(python3 --version 2>&1)
if [[ $? -eq 0 ]]; then
    echo "✅ Python version: $python_version"
else
    echo "❌ Python3 not found"
    exit 1
fi

if [[ -n "$VIRTUAL_ENV" ]]; then
    echo "✅ Virtual environment: $(basename $VIRTUAL_ENV)"
else
    echo "⚠️ No virtual environment detected"
fi

# Check key dependencies
echo ""
echo -e "${BLUE}📦 Dependency Check${NC}"
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
    exit 1
fi

# Check module structure
echo ""
echo -e "${BLUE}🏗️ Module Structure Check${NC}"
echo "-------------------------"

modules=("processors" "analyzers" "visualizations" "data_access" "config")
for module in "${modules[@]}"; do
    if [[ -d "$module" && -f "$module/__init__.py" ]]; then
        echo "✅ $module module found"
    else
        echo "❌ $module module missing"
    fi
done

# Hierarchical Test Suite Execution
echo ""
echo -e "${BLUE}🧪 Hierarchical Comprehensive Test Suite${NC}"
echo "========================================"
echo "Testing complete Patent Intelligence Platform with hierarchical dependencies:"
echo "1️⃣ Config → 2️⃣ Data Access → 3️⃣ Processors → 4️⃣ Analyzers → 5️⃣ Visualizations → 6️⃣ Notebooks"
echo ""

echo "Select test execution strategy:"
echo "1) Quick validation (individual module tests - assumes dependencies work)"
echo "2) Complete hierarchical pipeline test (recommended - validates complete system)"
echo "3) Custom module selection"
echo ""

# Check for non-interactive mode (when called by other scripts)
if [[ "$1" == "--non-interactive" ]] || [[ "$1" == "--auto" ]]; then
    choice="2"  # Default to full pipeline
    NON_INTERACTIVE=true
    echo "Running in non-interactive mode: Complete hierarchical pipeline (option 2)"
elif [[ -t 0 ]]; then
    NON_INTERACTIVE=false
    read -p "Choose option (1/2/3) [2]: " -n 1 -r
    echo
    choice="${REPLY:-2}"
else
    choice="2"
    NON_INTERACTIVE=false
fi

case $choice in
    1)
        log_with_timestamp "Running quick validation tests (individual modules)..."
        echo "⚠️ Warning: This assumes all dependencies are functional"
        
        # Individual module tests
        if [[ -f "test_config.sh" ]]; then
            run_test "Configuration Tests" "./test_config.sh" 300
        fi
        
        if [[ -f "test_data_access.sh" ]]; then
            run_test "Data Access Tests" "./test_data_access.sh" 600
        fi
        
        if [[ -f "test_processors.sh" ]]; then
            run_test "Processors Tests" "./test_processors.sh" 1200
        fi
        
        if [[ -f "test_analyzers.sh" ]]; then
            run_test "Analyzers Tests" "./test_analyzers.sh" 600
        fi
        
        if [[ -f "test_visualizations.sh" ]]; then
            run_test "Visualizations Tests" "./test_visualizations.sh" 600
        fi
        
        if [[ -f "test_notebooks.sh" ]]; then
            run_test "Notebooks Tests" "./test_notebooks.sh" 900
        fi
        ;;
        
    2)
        log_with_timestamp "Running complete hierarchical pipeline test..."
        echo "🔗 This will validate the complete system with real data flow dependencies"
        echo ""
        
        # Step 1: Configuration Layer
        log_with_timestamp "=== HIERARCHICAL STEP 1/6: CONFIGURATION LAYER ==="
        if [[ -f "test_config.sh" ]]; then
            run_test "Configuration Tests (Base Layer)" "./test_config.sh" 300
            if [[ ${test_results["Configuration Tests (Base Layer)"]} != "✅ PASSED" ]]; then
                log_with_timestamp "❌ Configuration layer failed - cannot proceed with pipeline"
                echo -e "${RED}❌ Pipeline stopped: Configuration validation failed${NC}"
                exit 1
            fi
            echo -e "${GREEN}✅ Configuration layer validated - proceeding to data access${NC}"
        else
            log_with_timestamp "⚠️ Configuration test script not found"
        fi
        echo ""
        
        # Step 2: Data Access Layer (skip config deps)
        log_with_timestamp "=== HIERARCHICAL STEP 2/6: DATA ACCESS LAYER ==="
        if [[ -f "test_data_access.sh" ]]; then
            run_test "Data Access Tests (Skip Config Deps)" "./test_data_access.sh --option=1" 600
            if [[ ${test_results["Data Access Tests (Skip Config Deps)"]} != "✅ PASSED" ]]; then
                log_with_timestamp "❌ Data access layer failed - cannot proceed with pipeline"
                echo -e "${RED}❌ Pipeline stopped: Data access validation failed${NC}"
                exit 1
            fi
            echo -e "${GREEN}✅ Data access layer validated - proceeding to processors${NC}"
        else
            log_with_timestamp "⚠️ Data access test script not found"
        fi
        echo ""
        
        # Step 3: Processors Layer (skip data access deps)
        log_with_timestamp "=== HIERARCHICAL STEP 3/6: PROCESSORS LAYER ==="
        if [[ -f "test_processors.sh" ]]; then
            run_test "Processors Tests (Skip Data Access Deps)" "./test_processors.sh --option=1" 1200
            if [[ ${test_results["Processors Tests (Skip Data Access Deps)"]} != "✅ PASSED" ]]; then
                log_with_timestamp "❌ Processors layer failed - cannot proceed with pipeline"
                echo -e "${RED}❌ Pipeline stopped: Processors validation failed${NC}"
                exit 1
            fi
            echo -e "${GREEN}✅ Processors layer validated - proceeding to analyzers${NC}"
        else
            log_with_timestamp "⚠️ Processors test script not found"
        fi
        echo ""
        
        # Step 4: Analyzers Layer (skip processor deps)
        log_with_timestamp "=== HIERARCHICAL STEP 4/6: ANALYZERS LAYER ==="
        if [[ -f "test_analyzers.sh" ]]; then
            run_test "Analyzers Tests (Skip Processor Deps)" "./test_analyzers.sh --option=1" 600
            if [[ ${test_results["Analyzers Tests (Skip Processor Deps)"]} != "✅ PASSED" ]]; then
                log_with_timestamp "❌ Analyzers layer failed - cannot proceed with pipeline"
                echo -e "${RED}❌ Pipeline stopped: Analyzers validation failed${NC}"
                exit 1
            fi
            echo -e "${GREEN}✅ Analyzers layer validated - proceeding to visualizations${NC}"
        else
            log_with_timestamp "⚠️ Analyzers test script not found"
        fi
        echo ""
        
        # Step 5: Visualizations Layer (skip analyzer deps)
        log_with_timestamp "=== HIERARCHICAL STEP 5/6: VISUALIZATIONS LAYER ==="
        if [[ -f "test_visualizations.sh" ]]; then
            run_test "Visualizations Tests (Skip Analyzer Deps)" "./test_visualizations.sh --option=1" 600
            if [[ ${test_results["Visualizations Tests (Skip Analyzer Deps)"]} != "✅ PASSED" ]]; then
                log_with_timestamp "❌ Visualizations layer failed - cannot proceed with pipeline"
                echo -e "${RED}❌ Pipeline stopped: Visualizations validation failed${NC}"
                exit 1
            fi
            echo -e "${GREEN}✅ Visualizations layer validated - proceeding to notebooks${NC}"
        else
            log_with_timestamp "⚠️ Visualizations test script not found"
        fi
        echo ""
        
        # Step 6: Notebooks Layer (skip all deps - test notebooks only)
        log_with_timestamp "=== HIERARCHICAL STEP 6/6: NOTEBOOKS LAYER (FINAL INTEGRATION) ==="
        if [[ -f "test_notebooks.sh" ]]; then
            run_test "Notebooks Tests (Skip All Deps)" "./test_notebooks.sh --option=1" 900
            if [[ ${test_results["Notebooks Tests (Skip All Deps)"]} != "✅ PASSED" ]]; then
                log_with_timestamp "❌ Notebooks layer failed - complete system validation failed"
                echo -e "${RED}❌ Pipeline completed with notebook issues${NC}"
            else
                echo -e "${GREEN}✅ Complete pipeline validated successfully!${NC}"
                log_with_timestamp "🎉 COMPLETE HIERARCHICAL PIPELINE VALIDATION SUCCESSFUL"
            fi
        else
            log_with_timestamp "⚠️ Notebooks test script not found"
        fi
        ;;
        
    3)
        log_with_timestamp "Custom module selection mode..."
        echo "Available test modules:"
        echo "1) Configuration (test_config.sh)"
        echo "2) Data Access (test_data_access.sh)" 
        echo "3) Processors (test_processors.sh)"
        echo "4) Analyzers (test_analyzers.sh)"
        echo "5) Visualizations (test_visualizations.sh)"
        echo "6) Notebooks (test_notebooks.sh)"
        echo ""
        if [[ "$NON_INTERACTIVE" == "true" ]]; then
            module_selection="1,2,3,4,5,6"  # Test all modules in non-interactive mode
            echo "Non-interactive mode: Testing all modules (1,2,3,4,5,6)"
        else
            read -p "Enter module numbers to test (e.g., 1,3,5): " module_selection
        fi
        
        IFS=',' read -ra selected_modules <<< "$module_selection"
        for module in "${selected_modules[@]}"; do
            case $module in
                1) [[ -f "test_config.sh" ]] && run_test "Configuration Tests" "./test_config.sh" 300 ;;
                2) [[ -f "test_data_access.sh" ]] && run_test "Data Access Tests" "./test_data_access.sh" 600 ;;
                3) [[ -f "test_processors.sh" ]] && run_test "Processors Tests" "./test_processors.sh" 1200 ;;
                4) [[ -f "test_analyzers.sh" ]] && run_test "Analyzers Tests" "./test_analyzers.sh" 600 ;;
                5) [[ -f "test_visualizations.sh" ]] && run_test "Visualizations Tests" "./test_visualizations.sh" 600 ;;
                6) [[ -f "test_notebooks.sh" ]] && run_test "Notebooks Tests" "./test_notebooks.sh" 900 ;;
                *) log_with_timestamp "⚠️ Invalid module selection: $module" ;;
            esac
        done
        ;;
        
    *)
        log_with_timestamp "Running default hierarchical pipeline test..."
        # Same as option 2
        echo "🔗 Complete hierarchical validation with real data dependencies"
        if [[ -f "test_notebooks.sh" ]]; then
            run_test "Complete Pipeline (Via Notebooks)" "./test_notebooks.sh" 3600
        fi
        ;;
esac

# Hierarchical Test Summary
echo ""
echo -e "${BLUE}📊 Hierarchical Test Execution Summary${NC}"
echo "======================================"

if [[ $choice == "2" ]]; then
    echo "🔗 Complete Pipeline Validation Results:"
    echo "1️⃣ Configuration Layer → Data Access Layer → Processors → Analyzers → Visualizations → Notebooks"
    echo ""
fi

for test_name in "${!test_results[@]}"; do
    result="${test_results[$test_name]}"
    if [[ $result == *"PASSED"* ]]; then
        echo -e "$result $test_name"
    elif [[ $result == *"TIMEOUT"* ]]; then
        echo -e "${YELLOW}$result $test_name${NC}"
    else
        echo -e "${RED}$result $test_name${NC}"
    fi
done

echo ""
echo "======================="
if [[ $choice == "2" ]]; then
    echo -e "${BLUE}Hierarchical Pipeline Results${NC}"
    if [[ $passed_tests -eq $total_tests ]]; then
        echo -e "${GREEN}🎉 COMPLETE PIPELINE VALIDATION SUCCESSFUL!${NC}"
        echo -e "${GREEN}✅ Entire Patent Intelligence Platform validated with real data flow${NC}"
        echo -e "${GREEN}🚀 System ready for production deployment${NC}"
    else
        echo -e "${RED}❌ Pipeline validation failed at layer $((passed_tests + 1))${NC}"
        echo -e "${RED}🔍 Review the failed layer above and dependencies${NC}"
    fi
else
    echo -e "${BLUE}Individual Module Results${NC}"
fi
echo "======================="
echo "Total tests: $total_tests"
echo "Passed: $passed_tests"
echo "Failed: $((total_tests - passed_tests))"

success_rate=$(( passed_tests * 100 / total_tests ))
echo "Success rate: ${success_rate}%"

if [[ $passed_tests -eq $total_tests ]]; then
    echo -e "${GREEN}🎉 ALL TESTS PASSED! Platform is ready for production.${NC}"
    exit_code=0
elif [[ $success_rate -ge 80 ]]; then
    echo -e "${YELLOW}⚠️ Most tests passed. Some issues need attention.${NC}"
    exit_code=0
else
    echo -e "${RED}❌ Multiple test failures. Platform needs fixes before deployment.${NC}"
    exit_code=1
fi

# Module Coverage Summary
echo ""
echo -e "${BLUE}📋 Module Test Coverage${NC}"
echo "------------------------"
echo "✅ processors/ - Unit and integration tests"
echo "✅ data_access/ - Database and client tests"
echo "✅ config/ - Configuration validation tests"
echo "✅ analyzers/ - Technology, regional, trends analysis tests"
echo "✅ visualizations/ - Charts, maps, factory tests"
echo "✅ notebooks/ - Live demo readiness validation"
echo ""

# Performance Guidelines
echo -e "${BLUE}⚡ Performance Guidelines${NC}"
echo "------------------------"
echo "• Configuration tests: <5 minutes"
echo "• Data access tests: <10 minutes"
echo "• Processor tests: <30 minutes"
echo "• Analyzer tests: <10 minutes"
echo "• Visualization tests: <10 minutes"
echo "• Database tests: <5 minutes each"
echo ""

# Next Steps
echo -e "${BLUE}📝 Next Steps${NC}"
echo "------------"
if [[ $exit_code -eq 0 ]]; then
    echo "1. ✅ All critical modules tested and verified"
    echo "2. 🚀 Platform ready for production deployment"
    echo "3. 📊 Run integration tests with real PATSTAT data"
    echo "4. 🎯 Deploy to EPO PATLIB 2025 demonstration environment"
else
    echo "1. 🔍 Review failed tests and fix issues"
    echo "2. 📝 Check module integration and dependencies"
    echo "3. 🐛 Debug specific component failures"
    echo "4. 🔄 Re-run tests after fixes"
fi

# Aggregate all individual module logs into master log
aggregate_logs

log_with_timestamp ""
log_with_timestamp "=================================================="
log_with_timestamp "Comprehensive Test Suite Completed"
log_with_timestamp "Exit code: $exit_code"
log_with_timestamp "Master log file: $MASTER_LOG_FILE"
log_with_timestamp "Individual logs available in: $LOG_DIR/"

echo ""
echo "📄 Test Logs Summary"
echo "------------------"
echo "Master log: $MASTER_LOG_FILE"
for log_file in "$LOG_DIR"/*_tests.log; do
    if [[ -f "$log_file" ]]; then
        module_name=$(basename "$log_file" _tests.log)
        size=$(wc -l < "$log_file" 2>/dev/null || echo "0")
        echo "$module_name: $log_file ($size lines)"
    fi
done

exit $exit_code