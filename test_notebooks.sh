#!/bin/bash
# Notebooks Testing Script for Patent Analysis Platform
# Tests demo notebook readiness and validation

log_with_timestamp "🚀 Patent Analysis Platform - Notebooks Test Suite"
log_with_timestamp "Enhanced from EPO PATLIB 2025 Live Demo Code"
log_with_timestamp "=================================================="
log_with_timestamp "Log file: $LOG_FILE"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Setup logging
LOG_DIR="logs"
LOG_FILE="$LOG_DIR/notebooks_tests.log"
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
if [[ ! -f "notebooks/test_notebook_cells.py" ]]; then
    echo "❌ Test script not found: notebooks/test_notebook_cells.py"
    exit 1
fi

if [[ ! -f "notebooks/Patent_Intelligence_Platform_Demo.ipynb" ]]; then
    echo "❌ Demo notebook not found: notebooks/Patent_Intelligence_Platform_Demo.ipynb"
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

# Check key dependencies for notebook execution
echo ""
echo "📦 Dependency Check for Notebook Execution"
echo "-------------------------------------------"

dependencies=("pandas" "numpy" "jupyter")
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

# Check notebook-specific dependencies
notebook_deps=("json" "pathlib")
for dep in "${notebook_deps[@]}"; do
    python3 -c "import $dep" 2>/dev/null
    if [[ $? -eq 0 ]]; then
        echo "✅ $dep: Available"
    else
        echo "❌ $dep: Not available"
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

# Check notebook testing infrastructure
echo ""
echo "📓 Notebook Testing Infrastructure Check"
echo "---------------------------------------"

notebook_files=(
    "notebooks/Patent_Intelligence_Platform_Demo.ipynb"
    "notebooks/test_notebook_cells.py"
    "notebooks/test_notebook_validation.py"
    "notebooks/notebook_fixed_cells.py"
    "notebooks/README.md"
)

for file in "${notebook_files[@]}"; do
    if [[ -f "$file" ]]; then
        if [[ "$file" == *.ipynb ]]; then
            cell_count=$(grep -c '"cell_type"' "$file" 2>/dev/null || echo "?")
            echo "✅ $file ($cell_count cells)"
        else
            size=$(wc -l < "$file")
            echo "✅ $file ($size lines)"
        fi
    else
        echo "❌ $file (missing)"
    fi
done

# Check platform integration
echo ""
echo "🔧 Platform Integration Check"
echo "-----------------------------"

integration_modules=("config" "data_access" "processors" "visualizations")
for module in "${integration_modules[@]}"; do
    python3 -c "import $module" 2>/dev/null
    if [[ $? -eq 0 ]]; then
        echo "✅ $module module available for notebook"
    else
        echo "❌ $module module import issues"
    fi
done

# Hierarchical Test Execution for Notebooks
echo ""
echo "🧪 Hierarchical Test Execution for Notebooks"
echo "============================================="
echo "Notebooks require the complete functional platform:"
echo "1️⃣ Config → 2️⃣ Data Access → 3️⃣ Processors → 4️⃣ Analyzers → 5️⃣ Visualizations → 6️⃣ Notebooks"
echo ""

echo "Select test mode:"
echo "1) Notebook tests only (assumes full platform is functional)"
echo "2) Complete pipeline test (config → data → processors → analyzers → visualizations → notebooks - recommended)"
echo "3) Quick notebook validation (basic cell syntax)"
echo "4) Demo readiness check (EPO PATLIB 2025 preparation)"
echo "5) Fix mode (show and apply fixes)"
echo ""

# Default to complete pipeline if no interaction
if [[ -t 0 ]]; then
    read -p "Choose option (1/2/3/4/5) [2]: " -n 1 -r
    echo
    choice="${REPLY:-2}"
else
    choice="2"
fi

case $choice in
    1)
        log_with_timestamp "Running notebook tests only (assumes platform is functional)..."
        echo "⚠️ Warning: This assumes the complete platform stack is functional"
        log_with_timestamp "Testing notebook execution simulation..."
        run_with_logging "timeout 600 python3 notebooks/test_notebook_validation.py"
        
        validation_exit_code=$?
        log_with_timestamp ""
        log_with_timestamp "Testing comprehensive cell validation..."
        run_with_logging "timeout 600 python3 notebooks/test_notebook_cells.py"
        
        cells_exit_code=$?
        
        # Use the worse of the two exit codes
        test_exit_code=$((validation_exit_code > cells_exit_code ? validation_exit_code : cells_exit_code))
        ;;
    2)
        log_with_timestamp "Running complete pipeline test with hierarchical execution..."
        echo ""
        
        # Step 1: Configuration Tests
        log_with_timestamp "Step 1/6: Testing Configuration Layer"
        echo "⚙️ Ensuring configuration system works..."
        if ! run_with_logging "timeout 300 ./test_config.sh"; then
            log_with_timestamp "❌ Configuration tests failed - cannot proceed to notebooks"
            exit 1
        fi
        echo "✅ Configuration layer validated"
        echo ""
        
        # Step 2: Data Access Tests
        log_with_timestamp "Step 2/6: Testing Data Access Layer"
        echo "🔍 Ensuring data access functionality works..."
        if ! run_with_logging "timeout 300 ./test_data_access.sh"; then
            log_with_timestamp "❌ Data access tests failed - cannot proceed to notebooks"
            exit 1
        fi
        echo "✅ Data access layer validated"
        echo ""
        
        # Step 3: Processor Tests
        log_with_timestamp "Step 3/6: Testing Processors"
        echo "⚙️ Ensuring data processing functionality works..."
        if ! run_with_logging "timeout 600 ./test_processors.sh"; then
            log_with_timestamp "❌ Processor tests failed - cannot proceed to notebooks"
            exit 1
        fi
        echo "✅ Processors validated"
        echo ""
        
        # Step 4: Analyzer Tests
        log_with_timestamp "Step 4/6: Testing Analyzers"
        echo "📊 Ensuring analysis functionality works..."
        if ! run_with_logging "timeout 600 ./test_analyzers.sh"; then
            log_with_timestamp "❌ Analyzer tests failed - cannot proceed to notebooks"
            exit 1
        fi
        echo "✅ Analyzers validated"
        echo ""
        
        # Step 5: Visualization Tests
        log_with_timestamp "Step 5/6: Testing Visualizations"
        echo "🎨 Ensuring visualization functionality works..."
        if ! run_with_logging "timeout 600 ./test_visualizations.sh"; then
            log_with_timestamp "❌ Visualization tests failed - cannot proceed to notebooks"
            exit 1
        fi
        echo "✅ Visualizations validated"
        echo ""
        
        # Step 6: Notebook Tests (with complete functional platform)
        log_with_timestamp "Step 6/6: Testing Notebooks with Complete Functional Platform"
        echo "📓 Testing notebooks with fully validated platform..."
        
        log_with_timestamp "Step 6a: Notebook Execution Simulation"
        log_with_timestamp "------------------------------------"
        run_with_logging "timeout 600 python3 notebooks/test_notebook_validation.py"
        validation_exit_code=$?
        
        log_with_timestamp ""
        log_with_timestamp "Step 6b: Comprehensive Cell Validation"
        log_with_timestamp "-------------------------------------"
        run_with_logging "timeout 600 python3 notebooks/test_notebook_cells.py"
        cells_exit_code=$?
        
        # Use the worse of the two exit codes
        test_exit_code=$((validation_exit_code > cells_exit_code ? validation_exit_code : cells_exit_code))
        
        log_with_timestamp ""
        log_with_timestamp "🏁 Complete Pipeline Test Summary"
        log_with_timestamp "================================"
        log_with_timestamp "Configuration: ✅ PASS"
        log_with_timestamp "Data Access: ✅ PASS"
        log_with_timestamp "Processors: ✅ PASS"
        log_with_timestamp "Analyzers: ✅ PASS"
        log_with_timestamp "Visualizations: ✅ PASS"
        log_with_timestamp "Notebook Validation: $([ $validation_exit_code -eq 0 ] && echo "✅ PASS" || echo "❌ FAIL")"
        log_with_timestamp "Notebook Cells: $([ $cells_exit_code -eq 0 ] && echo "✅ PASS" || echo "❌ FAIL")"
        ;;
    3)
        log_with_timestamp "Running quick notebook validation (basic cell syntax)..."
        run_with_logging "timeout 300 python3 notebooks/test_notebook_cells.py"
        test_exit_code=$?
        ;;
    4)
        log_with_timestamp "Running demo readiness check (EPO PATLIB 2025 preparation)..."
        log_with_timestamp "Testing notebook execution simulation..."
        run_with_logging "timeout 600 python3 notebooks/test_notebook_validation.py"
        
        validation_exit_code=$?
        log_with_timestamp ""
        log_with_timestamp "Testing comprehensive cell validation..."
        run_with_logging "timeout 600 python3 notebooks/test_notebook_cells.py"
        
        cells_exit_code=$?
        
        # Use the worse of the two exit codes
        test_exit_code=$((validation_exit_code > cells_exit_code ? validation_exit_code : cells_exit_code))
        ;;
    5)
        log_with_timestamp "Running fix mode - showing available fixes..."
        run_with_logging "python3 notebooks/test_notebook_cells.py --fix"
        log_with_timestamp ""
        log_with_timestamp "Showing fixed cell documentation..."
        run_with_logging "python3 notebooks/notebook_fixed_cells.py"
        test_exit_code=$?
        ;;
    *)
        log_with_timestamp "Running default demo readiness check..."
        run_with_logging "timeout 600 python3 notebooks/test_notebook_validation.py"
        test_exit_code=$?
        ;;
esac

# If we didn't set test_exit_code above, get it from the last command
if [[ -z "$test_exit_code" ]]; then
    test_exit_code=$?
fi

echo ""
echo "📊 Test Execution Summary"
echo "========================="

case $test_exit_code in
    0)
        if [[ $choice == "2" ]] || [[ $choice == "" ]]; then
            echo "🎉 Complete pipeline test completed successfully!"
            echo "✅ Config → Data → Processors → Analyzers → Visualizations → Notebooks all validated"
            echo "📓 Notebooks tested with complete functional platform"
            echo "🚀 Demo notebook is ready for EPO PATLIB 2025 presentation!"
        else
            echo "🎉 All notebook tests passed!"
            echo "✅ Demo notebook is ready for EPO PATLIB 2025 presentation!"
        fi
        ;;
    124)
        echo "⏰ Tests timed out (15 minute limit exceeded)"
        echo "⚠️ Consider optimizing notebook cell execution or reducing complexity"
        ;;
    130)
        echo "⚠️ Tests interrupted by user (Ctrl+C)"
        ;;
    *)
        if [[ $choice == "2" ]] || [[ $choice == "" ]]; then
            echo "❌ Complete pipeline test failed with exit code: $test_exit_code"
            echo "🔍 Check which layer failed in the hierarchical execution above"
            echo "💡 Run individual layer tests to isolate the issue"
        else
            echo "❌ Notebook tests failed with exit code: $test_exit_code"
            echo "🔍 Please review the test output above for details"
            echo "💡 Run with option 5 (fix mode) to see available fixes"
        fi
        ;;
esac

# Demo readiness guidelines
echo ""
echo "⚡ Demo Readiness Guidelines"
echo "---------------------------"
echo "• Each cell should execute in <90 seconds for live demo"
echo "• No undefined variables or missing imports"
echo "• Graceful error handling for network/database issues"
echo "• Clear, professional output for audience"
echo "• Business intelligence focus for non-technical stakeholders"

# Usage examples
echo ""
echo "💡 Usage Examples"
echo "----------------"
echo "# Test notebook cells:"
echo "python3 notebooks/test_notebook_cells.py"
echo ""
echo "# Validate specific notebook:"
echo "python3 notebooks/test_notebook_cells.py --notebook path/to/notebook.ipynb"
echo ""
echo "# Quick cell validation:"
echo "python3 notebooks/test_notebook_validation.py"
echo ""
echo "# Show fixes:"
echo "python3 notebooks/notebook_fixed_cells.py"

# Additional checks
echo ""
echo "🔧 Additional Checks"
echo "-------------------"

# Check if notebook can be loaded
python3 -c "import json; json.load(open('notebooks/Patent_Intelligence_Platform_Demo.ipynb'))" 2>/dev/null
if [[ $? -eq 0 ]]; then
    echo "✅ Demo notebook JSON structure is valid"
else
    echo "❌ Demo notebook JSON structure issues"
fi

# Check for common notebook issues
if grep -q "NameError" notebooks/Patent_Intelligence_Platform_Demo.ipynb 2>/dev/null; then
    echo "⚠️ Potential NameError found in notebook output"
else
    echo "✅ No obvious NameError issues in notebook"
fi

# Check for output data
if [[ -f "notebooks/test_report.json" ]]; then
    echo "✅ Test report generated successfully"
else
    echo "⚠️ No test report generated (expected after running tests)"
fi

# Final recommendations
echo ""
echo "📝 Next Steps"
echo "------------"
if [[ $test_exit_code -eq 0 ]]; then
    echo "1. ✅ Notebook is demo-ready!"
    echo "2. 🎬 Practice 90-second timing for each cell"
    echo "3. 📊 Test with fresh kernel restart"
    echo "4. 🌐 Verify database connectivity for live demo"
    echo "5. 🎯 Prepare backup slides in case of connectivity issues"
else
    echo "1. 🔍 Review failed notebook tests and fix issues"
    echo "2. 📝 Check for undefined variables and missing imports"
    echo "3. 🔧 Use fix mode to see suggested corrections"
    echo "4. 💬 Test individual cells in Jupyter environment"
    echo "5. 🔄 Re-run tests after fixes"
fi

# Demo-specific recommendations
echo ""
echo "🎬 Live Demo Recommendations"
echo "---------------------------"
echo "• Test database connectivity before presentation"
echo "• Have backup data ready in case of network issues"
echo "• Practice cell execution timing (90 seconds each)"
echo "• Prepare audience-appropriate explanations"
echo "• Test all visualizations render correctly"
echo "• Verify export functionality works"

log_with_timestamp ""
log_with_timestamp "=================================================="
log_with_timestamp "Notebooks Test Suite Completed"
log_with_timestamp "Exit code: $test_exit_code"
log_with_timestamp "Full test log available at: $LOG_FILE"

exit $test_exit_code