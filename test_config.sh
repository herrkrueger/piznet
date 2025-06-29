#!/bin/bash
# Configuration Test Script for REE Patent Analysis Platform
# Enhanced from EPO PATLIB 2025 Live Demo Code

# Setup logging
LOG_DIR="logs"
LOG_FILE="$LOG_DIR/config_tests.log"
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

log_with_timestamp "🚀 Patent Analysis Platform - Configuration Tests"
log_with_timestamp "============================================================"
log_with_timestamp "Log file: $LOG_FILE"

# Change to the correct directory
cd "$(dirname "$0")"

# Run the configuration tests
log_with_timestamp "Running comprehensive configuration test suite..."
log_with_timestamp ""

run_with_logging "python config/test_config.py"

# Capture exit code
exit_code=$?

log_with_timestamp ""
if [ $exit_code -eq 0 ]; then
    log_with_timestamp "🎉 All configuration tests completed successfully!"
else
    log_with_timestamp "⚠️ Some configuration tests failed. Review the output above."
fi

log_with_timestamp "============================================================"
log_with_timestamp "Full test log available at: $LOG_FILE"
echo "To run individual test components:"
echo "  python -c 'from config.test_config import test_yaml_file_syntax; test_yaml_file_syntax()'"
echo "  python -c 'from config.test_config import test_centralized_search_patterns; test_centralized_search_patterns()'"
echo "  python -c 'from config.test_config import test_data_access_integration; test_data_access_integration()'"
echo ""
echo "Configuration files located in: ./config/"
echo "  - api_config.yaml"
echo "  - database_config.yaml" 
echo "  - visualization_config.yaml"
echo "  - search_patterns_config.yaml"

exit $exit_code