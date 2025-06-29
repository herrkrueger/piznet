# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **Patent Intelligence Platform v2.0** featuring Clean Architecture principles. The platform provides comprehensive patent analysis capabilities with standardized interfaces, dependency injection, and production-ready reliability.

### Core Architecture

The platform follows Clean Architecture with clear separation of concerns:

- **Processors** (`src/processors/`): Data processing layer (search, retrieval, filtering)
- **Analyzers** (`src/analyzers/`): Intelligence analysis layer (regional, technology, trends)
- **Data Providers** (`src/data_providers/`): External data source connectors (PATSTAT, EPO OPS, etc.)
- **Config Management** (`src/config/`): Configuration and dependency injection
- **Validation** (`src/validation/`): Quality assurance and data validation

All components implement standardized interfaces with consistent result objects (`ProcessorResult`, `AnalyzerResult`).

## Development Commands

### Running the Application
```bash
# Run demonstration analysis
python patent_intelligence.py --demo

# Run data provider demonstrations
python demo/demo_data_providers.py

# Show version information
python patent_intelligence.py --version

# Run with custom parameters (modify main() function)
python patent_intelligence.py
```

### Testing
```bash
# Run comprehensive test suite
python -m pytest tests/ -v

# Test real data provider connections
python tests/test_real_connections.py

# Run specific test files
python -m pytest tests/test_specific_file.py -v
```

### Package Management
```bash
# Install dependencies
pip install -r requirements.txt

# For development with optional packages, uncomment in requirements.txt:
# - plotly, matplotlib (visualizations)
# - sqlalchemy, psycopg2-binary (database)
# - jupyter, ipython (development)
```

## Key Components

### Main Orchestration
- `patent_intelligence.py`: Main platform orchestrator with complete analysis workflow
- Demonstrates clean data flow: Search → Process → Analyze → Validate → Results

### Base Classes
- `src/processors/base.py`: `BaseProcessor` and `ProcessorResult` interfaces
- `src/analyzers/base.py`: `BaseAnalyzer` and `AnalyzerResult` interfaces
- All processors/analyzers inherit from these standardized base classes

### Configuration
- `config/analysis.yaml`: Analysis configuration presets (comprehensive, default, quick)
- `config/data_providers.yaml`: Data provider configurations
- `config/processing.yaml`: Processing pipeline settings
- `config/visualization.yaml`: Visualization configurations

### Data Providers
- `src/data_providers/patstat/`: PATSTAT database provider
- `src/data_providers/epo_ops/`: EPO OPS API provider
- `src/data_providers/classification/`: Patent classification providers (CPC, IPC)
- `src/data_providers/geographic/`: Geographic data providers (NUTS)
- Each provider implements standardized query interface

## Development Patterns

### Adding New Processors
1. Inherit from `BaseProcessor` in `src/processors/base.py`
2. Implement `process(data, **kwargs) -> ProcessorResult` method
3. Use `_validate_input()` and `_create_metadata()` helper methods
4. Register in pipeline via `ProcessorPipeline.add_processor()`

### Adding New Analyzers  
1. Inherit from `BaseAnalyzer` in `src/analyzers/base.py`
2. Implement `analyze(data, analysis_params) -> AnalyzerResult` method
3. Use `_extract_data_from_processors()` to get data from processor results
4. Generate executive summary via `_generate_executive_summary()`

### Configuration Management
- Use YAML files in `config/` directory for external configuration
- Pass config dictionaries to component constructors
- Configuration is injected via dependency injection pattern

## Data Flow Architecture

The platform implements a standardized data flow:

1. **Search/Processing**: `PatentSearchProcessor` retrieves patent data
2. **Analysis**: Multiple analyzers process data in parallel (Regional, Technology, etc.)
3. **Validation**: Results validated for quality and completeness
4. **Compilation**: Executive summary and insights generated

Each step produces standardized result objects with metadata, error handling, and performance tracking.

## Environment Setup

### Required Environment Variables (for data providers)
```bash
# EPO OPS API credentials (optional, for EPO OPS provider)
OPS_KEY=your_epo_ops_api_key
OPS_SECRET=your_epo_ops_secret

# PATSTAT environment (defaults to PROD)
PATSTAT_ENVIRONMENT=PROD
```

### Performance Configuration
The platform includes comprehensive performance monitoring:
- Processing times tracked per component
- Success rates and error handling
- Memory usage monitoring
- Executive performance summaries

Use `PatentIntelligencePlatform.get_platform_summary()` to view performance metrics.

## Architecture Benefits

This clean architecture provides:
- **Testability**: All components mockable via dependency injection
- **Maintainability**: Clear separation of concerns
- **Extensibility**: Easy to add new processors/analyzers
- **Production Ready**: Comprehensive error handling and validation
- **Performance**: Built-in monitoring and optimization