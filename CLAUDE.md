# Claude Code Project Configuration

This file helps Claude Code remember project-specific preferences and settings.

## Package Management Preference
**Preferred Environment**: Both venv and Pipenv (development flexibility)
- Use venv for CI/CD and production deployments
- Use pipenv for local development and dependency management
- Setup script supports both automatically

## Repository Configuration
**Repository Type**: Public GitHub Repository
**Target Users**: Patent intelligence professionals, researchers, developers

## Testing Strategy
**Test Scripts Available**:
- `test_config.sh` - Configuration validation tests
- `test_data_access.sh` - Data access layer tests  
- `test_processors.sh` - Processor pipeline tests (updated: test_processor.py)
- `test_analyzers.sh` - Technology/regional/trends analysis tests
- `test_visualizations.sh` - Charts/maps/dashboard tests
- `test_notebooks.sh` - Demo notebook validation tests

**Centralized Logging**: All test scripts write to timestamped log files in `/logs/`
**CI/CD Status**: Not currently enabled (GitHub Actions recommended)
**Test Coverage**: Comprehensive unit and integration tests available (100% processor tests passing)

## Development Workflow
1. **Environment Setup**: Run `python setup.py` for interactive setup
2. **Dependencies**: Both requirements.txt and Pipfile maintained
3. **Configuration**: YAML-based config with .env for credentials
4. **Testing**: Run individual test scripts or use pytest for full suite

## Documentation Standards
- **Installation**: Detailed INSTALL.md for GitHub users
- **README**: Executive-level overview with quick start
- **Code Documentation**: Docstrings for all public APIs
- **Demo Ready**: Jupyter notebook for live demonstrations

## Security Notes
- All credentials externalized to .env file
- .env.template provided for GitHub users
- Sensitive files properly git-ignored
- No hardcoded secrets in codebase

## System Access and Capabilities

### PATSTAT Database Access
- **Full PATSTAT Production Access**: Complete database access via TIP's pre-installed environment
- **Connection Method**: `PatstatClient(environment='PROD')` using `epo.tipdata.patstat`
- **Tables Available**: All PATSTAT tables (TLS201_APPLN, TLS209_APPLN_IPC, TLS224_APPLN_CPC, etc.)
- **Query Capabilities**: Direct SQL queries, ORM operations, and bulk data processing
- **Performance**: Production-grade performance for large-scale patent analysis

### EPO OPS API Integration
- **Full API Access**: Complete EPO Open Patent Services functionality
- **Authentication**: Real credentials configured for production use
- **Services Available**: Patent search, bibliographic data, full-text retrieval, images
- **Rate Limits**: Production-level access with appropriate throttling
- **Data Coverage**: Global patent data from EPO, USPTO, JPO, and other offices

### Environment Configuration
- **Production Environment**: Ready for live patent intelligence operations
- **Development Flexibility**: Both venv and Pipenv support for team collaboration
- **CI/CD Ready**: GitHub Actions compatible with comprehensive test coverage
- **Scalability**: Architecture supports high-volume patent processing workflows

## Architecture Preferences
- **5-Layer Design**: Config → Data Access → Processors → Analyzers → Visualizations
- **Clean Separation**: Processors (data processing) vs Analyzers (intelligence analysis)
- **Factory Patterns**: For component creation and management
- **YAML Configuration**: Technology-agnostic, environment-specific settings
- **Production Ready**: Comprehensive error handling and monitoring

## Recent Architecture Updates (2025-06-29)
### Critical Terminology & Structure Fixes
- **Renamed**: `ClassificationAnalyzer` → `ClassificationProcessor` (correct module scope)
- **Renamed**: `test_unit.py` → `test_processor.py` (clarity)
- **Fixed**: Architecture boundaries between `/processors/` and `/analyzers/`
- **Added**: Centralized logging system with timestamped audit trails
- **Updated**: All test scripts, shell scripts, and demo notebook
- **Result**: Clean architectural separation enabling scalable development

### Module Boundaries Clarified
- **`/processors/`**: Data processing, enrichment, preparation (ClassificationProcessor)
- **`/analyzers/`**: Intelligence analysis, insights, trends (TechnologyAnalyzer) 
- **Data Flow**: Search → Process → Analyze → Visualize
- **Benefits**: Clear ownership, testable components, maintainable codebase

Last Updated: 2025-06-29