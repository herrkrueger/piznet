#!/bin/bash
# Visualizations Testing Script for Patent Analysis Platform
# Tests charts, maps, dashboards, and factory components

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Setup logging
LOG_DIR="logs"
LOG_FILE="$LOG_DIR/visualizations_tests.log"
mkdir -p "$LOG_DIR"

# Function to log with timestamp
log_with_timestamp() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log_with_timestamp "🚀 Patent Analysis Platform - Visualizations Test Suite"
log_with_timestamp "Enhanced from EPO PATLIB 2025 Live Demo Code"
log_with_timestamp "======================================================="
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
if [[ ! -f "visualizations/test_visualizations.py" ]]; then
    echo "❌ Test script not found: visualizations/test_visualizations.py"
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

dependencies=("pandas" "numpy")
optional_deps=("plotly" "matplotlib" "folium")
missing_deps=()
missing_optional=()

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

for dep in "${optional_deps[@]}"; do
    python3 -c "import $dep" 2>/dev/null
    if [[ $? -eq 0 ]]; then
        version=$(python3 -c "import $dep; print($dep.__version__)" 2>/dev/null)
        echo "✅ $dep: $version (optional)"
    else
        echo "⚠️ $dep: Not installed (optional - visualizations may use fallback)"
        missing_optional+=("$dep")
    fi
done

if [[ ${#missing_deps[@]} -gt 0 ]]; then
    echo ""
    echo "❌ Missing required dependencies: ${missing_deps[*]}"
    echo "Please install missing packages:"
    echo "pip install ${missing_deps[*]}"
    exit 1
fi

if [[ ${#missing_optional[@]} -gt 0 ]]; then
    echo ""
    echo "⚠️ Missing optional dependencies: ${missing_optional[*]}"
    echo "Install for full visualization features:"
    echo "pip install ${missing_optional[*]}"
    echo ""
    if [[ "$NON_INTERACTIVE" == "true" ]]; then
        echo "Non-interactive mode: Continuing with fallback mode..."
    else
        read -p "Continue with fallback mode? (Y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Nn]$ ]]; then
            exit 1
        fi
    fi
fi

# Check visualization module structure
echo ""
echo "🏗️ Visualization Module Structure Check"
echo "---------------------------------------"

required_files=(
    "visualizations/__init__.py"
    "visualizations/charts.py"
    "visualizations/maps.py"
    "visualizations/dashboards.py"
    "visualizations/factory.py"
    "visualizations/test_visualizations.py"
)

for file in "${required_files[@]}"; do
    if [[ -f "$file" ]]; then
        size=$(wc -l < "$file")
        echo "✅ $file ($size lines)"
    else
        echo "❌ $file (missing)"
    fi
done

# Check configuration integration
echo ""
echo "🎨 Visualization Configuration Check"
echo "-----------------------------------"

if [[ -f "config/visualization_config.yaml" ]]; then
    echo "✅ Visualization configuration available"
    
    # Check for themes and settings
    theme_count=$(grep -c "theme" config/visualization_config.yaml 2>/dev/null || echo "0")
    echo "   Themes configured: $theme_count"
else
    echo "⚠️ Visualization configuration missing (will use defaults)"
fi

# Check output directories
echo ""
echo "📁 Output Directory Check"
echo "-------------------------"

output_dirs=("exports" "tmp" "cache")
for dir in "${output_dirs[@]}"; do
    if [[ -d "$dir" ]]; then
        echo "✅ $dir/ directory available"
    else
        echo "⚠️ $dir/ directory missing (will create if needed)"
    fi
done

# Hierarchical Test Execution
echo ""
echo "🧪 Hierarchical Test Execution for Visualizations"
echo "================================================="
echo "Visualizations require processed data from the complete pipeline:"
echo "1️⃣ Data Access Layer → 2️⃣ Processors → 3️⃣ Analyzers → 4️⃣ Visualizations"
echo ""

echo "Select test mode:"
echo "1) Quick validation (core functionality only)"  
echo "2) Full pipeline test (data access → processors → analyzers → visualizations)"
echo "3) Visualizations only (assumes processed data exists)"
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
    read -p "Choose option (1/2/3) [2]: " -n 1 -r
    echo
    choice="${REPLY:-2}"
fi

case $choice in
    1)
        log_with_timestamp "Running quick validation tests (visualizations only)..."
        run_with_logging "timeout 300 python3 visualizations/test_visualizations.py"
        ;;
    2)
        log_with_timestamp "Running full pipeline test with hierarchical execution..."
        echo ""
        
        # Step 1: Data Access Tests (full pipeline - tests config too)
        log_with_timestamp "Step 1/4: Testing Configuration → Data Access"
        echo "🔍 Ensuring data access functionality works..."
        if ! run_with_logging "timeout 300 ./test_data_access.sh --non-interactive --option=2"; then
            log_with_timestamp "❌ Data access tests failed - cannot proceed to visualizations"
            exit 1
        fi
        echo "✅ Data access layer validated (config validated too)"
        echo ""
        
        # Step 2: Processor Tests (assumes data access works)
        log_with_timestamp "Step 2/4: Testing Processors (skip data access deps)"
        echo "⚙️ Ensuring data processing functionality works..."
        if ! run_with_logging "timeout 600 ./test_processors.sh --non-interactive --option=1"; then
            log_with_timestamp "❌ Processor tests failed - cannot proceed to visualizations"
            exit 1
        fi
        echo "✅ Processors validated"
        echo ""
        
        # Step 3: Analyzer Tests (assumes processors work)
        log_with_timestamp "Step 3/4: Testing Analyzers (skip processor deps)" 
        echo "📊 Ensuring analysis functionality works..."
        if ! run_with_logging "timeout 600 ./test_analyzers.sh --non-interactive --option=1"; then
            log_with_timestamp "❌ Analyzer tests failed - cannot proceed to visualizations"
            exit 1
        fi
        echo "✅ Analyzers validated"
        echo ""
        
        # Step 4: Visualization Tests (with real processed data)
        log_with_timestamp "Step 4/4: Testing Visualizations with Real Data"
        echo "🎨 Testing visualizations with processed data from pipeline..."
        run_with_logging "timeout 600 python3 visualizations/test_visualizations.py"
        ;;
    3)
        log_with_timestamp "Running visualization tests only (assumes processed data exists)..."
        echo "⚠️ Warning: This assumes processed data is available from previous pipeline runs"
        run_with_logging "timeout 600 python3 visualizations/test_visualizations.py"
        ;;
    *)
        log_with_timestamp "Running default full pipeline test..."
        echo ""
        
        # Full pipeline execution (same as option 2)
        log_with_timestamp "Step 1/4: Testing Configuration → Data Access"
        if ! run_with_logging "timeout 300 ./test_data_access.sh --non-interactive --option=2"; then
            log_with_timestamp "❌ Data access tests failed"
            exit 1
        fi
        
        log_with_timestamp "Step 2/4: Testing Processors (skip data access deps)"
        if ! run_with_logging "timeout 600 ./test_processors.sh --non-interactive --option=1"; then
            log_with_timestamp "❌ Processor tests failed"
            exit 1
        fi
        
        log_with_timestamp "Step 3/4: Testing Analyzers (skip processor deps)"
        if ! run_with_logging "timeout 600 ./test_analyzers.sh --non-interactive --option=1"; then
            log_with_timestamp "❌ Analyzer tests failed"
            exit 1
        fi
        
        log_with_timestamp "Step 4/4: Testing Visualizations"
        run_with_logging "timeout 600 python3 visualizations/test_visualizations.py"
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
            echo "✅ Data Access → Processors → Analyzers → Visualizations all validated"
            echo "🎨 Visualizations tested with real processed data"
        else
            echo "🎉 Visualization tests passed!"
            echo "✅ Charts, maps, and dashboards are ready for production"
        fi
        ;;
    124)
        echo "⏰ Tests timed out (10 minute limit exceeded)"
        echo "⚠️ Consider optimizing visualization rendering or reducing test scope"
        ;;
    130)
        echo "⚠️ Tests interrupted by user (Ctrl+C)"
        ;;
    *)
        if [[ $choice == "2" ]] || [[ $choice == "" ]]; then
            echo "❌ Pipeline test failed with exit code: $test_exit_code"
            echo "🔍 Check which layer failed in the hierarchical execution above"
        else
            echo "❌ Visualization tests failed with exit code: $test_exit_code"
            echo "🔍 Please review the test output above for details"
        fi
        ;;
esac

# Performance summary
echo ""
echo "⚡ Visualization Performance Guidelines"
echo "-------------------------------------"
echo "• Chart generation: <2 seconds for 1000+ data points"
echo "• Map rendering: <5 seconds for geographic datasets"
echo "• Dashboard creation: <10 seconds for multi-component dashboards"
echo "• Export operations: <3 seconds for standard formats"

# Usage examples
echo ""
echo "💡 Usage Examples"
echo "----------------"
echo "# Run visualization tests:"
echo "python3 visualizations/test_visualizations.py"
echo ""
echo "# Test specific visualization type:"
echo "python3 -c \"from visualizations import create_patent_visualizations; print('Visualizations available')\""
echo ""
echo "# Import visualization classes:"
echo "from visualizations import ProductionChartCreator, ProductionMapsCreator"

# Additional checks
echo ""
echo "🔧 Additional Checks"
echo "-------------------"

# Check if visualizations can be imported
python3 -c "from visualizations import ProductionChartCreator, ProductionMapsCreator, PatentVisualizationFactory" 2>/dev/null
if [[ $? -eq 0 ]]; then
    echo "✅ All visualization classes import successfully"
else
    echo "❌ Visualization import issues detected"
fi

# Check factory pattern
python3 -c "from visualizations import create_patent_visualizations" 2>/dev/null
if [[ $? -eq 0 ]]; then
    echo "✅ Visualization factory pattern available"
else
    echo "❌ Visualization factory pattern issues"
fi

# Check configuration integration
python3 -c "from config import ConfigurationManager; cm = ConfigurationManager(); print('Config integration OK' if cm.get('visualization') else 'Config missing')" 2>/dev/null
if [[ $? -eq 0 ]]; then
    echo "✅ Configuration integration working"
else
    echo "⚠️ Configuration integration issues"
fi

# Final recommendations
echo ""
echo "📝 Next Steps"
echo "------------"
if [[ $test_exit_code -eq 0 ]]; then
    echo "1. ✅ All visualizations tested and verified!"
    echo "2. 🎨 Test with real patent analysis data"
    echo "3. 📊 Validate export formats and quality"
    echo "4. 🌐 Test in different browser environments"
    echo "5. 🚀 Deploy to production dashboard systems"
else
    echo "1. 🔍 Review failed visualization tests and fix issues"
    echo "2. 📝 Check visualization library dependencies"
    echo "3. 🎨 Verify chart and map generation functionality"
    echo "4. 💬 Test with different data sizes and formats"
fi

# Hierarchical Testing Explanation
echo ""
echo "🏗️ Hierarchical Testing Architecture"
echo "------------------------------------"
echo "This test suite implements a hierarchical testing approach:"
echo ""
echo "1️⃣ Data Access Layer: Validates PATSTAT/EPO OPS connectivity"
echo "2️⃣ Processors: Tests search, classification, citation processing"  
echo "3️⃣ Analyzers: Tests technology, regional, and trends analysis"
echo "4️⃣ Visualizations: Tests charts/maps with real processed data"
echo ""
echo "Benefits:"
echo "• ✅ True integration testing (no mocks)"
echo "• 🔗 Validates complete data pipeline"
echo "• 🎯 Catches real-world integration issues"
echo "• 📊 Tests visualizations with actual processor output"
echo ""
echo "Usage:"
echo "• Option 1: Quick visualization validation only"
echo "• Option 2: Full pipeline test (recommended)"
echo "• Option 3: Visualization-only (requires existing data)"

# Visualization-specific tips
echo ""
echo "🎨 Visualization Tips"
echo "-------------------"
echo "• Test with empty datasets to verify error handling"
echo "• Validate color schemes for accessibility"
echo "• Check responsive design for different screen sizes"
echo "• Test export functionality for all supported formats"
echo "• Verify interactive features work across browsers"

log_with_timestamp ""
log_with_timestamp "=================================================="
log_with_timestamp "Visualizations Test Suite Completed"
log_with_timestamp "Exit code: $test_exit_code"
log_with_timestamp "Full test log available at: $LOG_FILE"

exit $test_exit_code