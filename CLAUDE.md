# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Structure & Branch Strategy

This repository contains **two distinct architectures** for the Patent Intelligence Platform:

### **Main Branch**: Production-Ready Platform (EPO PATLIB 2025)
- **Purpose**: Production-ready patent analysis platform for live demonstrations
- **Architecture**: 5-layer modular design with YAML configuration
- **Status**: ✅ **Complete & Demo Ready** with 100% test coverage
- **Key Features**: Real PATSTAT connectivity, comprehensive testing, business intelligence visualizations

### **v2-clean-architecture Branch**: Clean Architecture Refactor
- **Purpose**: Complete architectural refactor demonstrating Clean Architecture principles
- **Architecture**: Clean separation with dependency injection and standardized interfaces
- **Status**: 🚧 **In Development** - Architectural foundation with demonstration capabilities
- **Key Features**: SOLID principles, enterprise-grade design patterns, 100% testable components

## Current Branch: v2-clean-architecture

This is the **Patent Intelligence Platform v2.0** featuring Clean Architecture principles. The platform provides comprehensive patent analysis capabilities with standardized interfaces, dependency injection, and production-ready reliability.

### Core Architecture

The platform follows Clean Architecture with clear separation of concerns:

- **Processors** (`src/processors/`): Data processing layer (search, retrieval, filtering)
- **Analyzers** (`src/analyzers/`): Intelligence analysis layer (regional, technology, trends)
- **Data Providers** (`src/data_providers/`): External data source connectors (PATSTAT, EPO OPS, etc.)
- **Config Management** (`src/config/`): Configuration and dependency injection
- **Validation** (`src/validation/`): Quality assurance and data validation

All components implement standardized interfaces with consistent result objects (`ProcessorResult`, `AnalyzerResult`).

## Branch-Specific Development Commands

### v2-clean-architecture Branch (Current)

#### Running the Application
```bash
# Run demonstration analysis
python patent_intelligence.py --demo

# Run data provider demonstrations  
python demo/demo_data_providers.py

# Show version information
python patent_intelligence.py --version
```

#### Testing
```bash
# Run comprehensive test suite
python -m pytest tests/ -v

# Test real data provider connections
python tests/test_real_connections.py

# Run specific test files
python -m pytest tests/test_specific_file.py -v
```

### Main Branch Commands (Production Platform)

#### Running the Application
```bash
# Run complete platform test
python scripts/test_complete_fix.py

# Open production-ready demo notebook
jupyter notebook notebooks/Patent_Intelligence_Platform_Demo.ipynb

# Run automated setup
python setup.py
```

#### Testing Scripts
```bash
# Complete test suite with all modules
./test_config.sh              # Configuration validation (8/8 tests)
./test_data_access.sh          # Data access layer (9/9 tests) 
./test_processors.sh           # Processor pipeline (5/5 tests)
./test_analyzers.sh            # Intelligence analyzers
./test_visualizations.sh       # Charts/maps/dashboards (19/19 tests)
./test_notebooks.sh            # Demo notebook validation

# Run all tests
./test_all_modules.sh
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

## Architecture Comparison

### v2-clean-architecture Branch (Current)

#### Key Components
- **Main Orchestration**: `patent_intelligence.py` - Clean Architecture demonstration
- **Base Classes**: `src/processors/base.py`, `src/analyzers/base.py` - Standardized interfaces
- **Data Providers**: `src/data_providers/` - External data source connectors
- **Configuration**: YAML-based configuration in `config/` directory
- **Validation**: `src/validation/` - Quality assurance framework

#### Clean Architecture Layers
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Search API    │───▶│   Processors    │───▶│   Analyzers     │
│  (Data Input)   │    │ (Data Process)  │    │ (Intelligence)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Validation    │◀───│  Visualizations │◀───│   Results       │
│  (Quality)      │    │   (Charts)      │    │ (Intelligence)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Main Branch (Production Platform)

#### Key Components
- **Configuration**: `config/` - YAML-driven modular configuration
- **Data Access**: `data_access/` - PATSTAT client, EPO OPS API, geographic intelligence
- **Processors**: `processors/` - Search, applicant, classification, geographic, citation analysis
- **Analyzers**: `analyzers/` - Technology, regional, trends intelligence
- **Visualizations**: `visualizations/` - Charts, maps, dashboards
- **Notebooks**: `notebooks/` - Live demo and interactive analysis

#### Production Architecture
```
Configuration Layer → Data Access Layer → Processing Layer → Analysis Layer → Visualization Layer
       ↓                      ↓                   ↓                ↓                   ↓
 YAML configs        PATSTAT + EPO OPS      5 Processors     6 Analyzers        Charts + Maps
```

## Branch-Specific Development Patterns

### v2-clean-architecture Branch Development

#### Adding New Processors
1. Inherit from `BaseProcessor` in `src/processors/base.py`
2. Implement `process(data, **kwargs) -> ProcessorResult` method
3. Use `_validate_input()` and `_create_metadata()` helper methods
4. Register in pipeline via `ProcessorPipeline.add_processor()`

#### Adding New Analyzers
1. Inherit from `BaseAnalyzer` in `src/analyzers/base.py`
2. Implement `analyze(data, analysis_params) -> AnalyzerResult` method
3. Use `_extract_data_from_processors()` to get data from processor results
4. Generate executive summary via `_generate_executive_summary()`

#### Configuration Management
- Use YAML files in `config/` directory for external configuration
- Pass config dictionaries to component constructors
- Configuration is injected via dependency injection pattern

### Main Branch Development

#### Adding New Components
1. **Processors**: Inherit from processor base classes in `processors/base.py`
2. **Analyzers**: Follow patterns in `analyzers/` with technology focus
3. **Visualizations**: Use factory pattern in `visualizations/factory.py`
4. **Configuration**: Add YAML configuration files in `config/`

#### Testing Requirements
- Write comprehensive test functions for each module
- Follow naming convention: `test_[module_name].py`
- Include performance benchmarks for production components
- Test with real PATSTAT data connections

## Data Flow Architecture

The platform implements a standardized data flow:

1. **Search/Processing**: `PatentSearchProcessor` retrieves patent data
2. **Analysis**: Multiple analyzers process data in parallel (Regional, Technology, etc.)
3. **Validation**: Results validated for quality and completeness
4. **Compilation**: Executive summary and insights generated

Each step produces standardized result objects with metadata, error handling, and performance tracking.

## Environment Setup

### Both Branches
```bash
# EPO OPS API credentials (optional, for EPO OPS provider)
OPS_KEY=your_epo_ops_api_key
OPS_SECRET=your_epo_ops_secret

# PATSTAT environment (defaults to PROD)
PATSTAT_ENVIRONMENT=PROD
```

### Main Branch Additional Requirements
```bash
# PATSTAT Database credentials (if external)
PATSTAT_USER=your_patstat_username
PATSTAT_PASSWORD=your_patstat_password

# Google Cloud BigQuery (if external)
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service_account.json
```

## Branch-Specific Features

### v2-clean-architecture Branch Benefits

**Clean Architecture Advantages**:
- **Testability**: All components mockable via dependency injection
- **Maintainability**: Clear separation of concerns with SOLID principles
- **Extensibility**: Easy to add new processors/analyzers through standardized interfaces
- **Enterprise Ready**: Comprehensive validation and error handling
- **Performance**: Built-in monitoring and optimization

**Performance Monitoring**:
Use `PatentIntelligencePlatform.get_platform_summary()` to view performance metrics.

### Main Branch Benefits

**Production Advantages**:
- **Real Data Connectivity**: Live PATSTAT PROD and EPO OPS integration
- **100% Test Coverage**: Comprehensive test suite with 19/19 visualization tests passing
- **Business Intelligence**: Executive dashboards and professional visualizations
- **Performance Optimization**: 8-10x faster citation analysis for large datasets
- **Demo Ready**: Live demonstration capabilities for EPO PATLIB 2025

**Proven Capacity**:
- 5,000 patents processed in ~30 seconds
- 1,000+ families/second citation analysis
- Zero garbage collection issues with intelligent sampling

## Branch Selection Guide

### Choose **main branch** when:
- Building production patent analysis applications
- Need real database connectivity (PATSTAT PROD)
- Require comprehensive testing and validation
- Working with business intelligence and visualizations
- Preparing live demonstrations or client presentations

### Choose **v2-clean-architecture branch** when:
- Learning Clean Architecture principles
- Building enterprise software with SOLID design
- Need maximum testability and dependency injection
- Developing components with standardized interfaces
- Studying architectural patterns and best practices