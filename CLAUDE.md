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
- `test_processors.sh` - Processor pipeline tests

**CI/CD Status**: Not currently enabled (GitHub Actions recommended)
**Test Coverage**: Comprehensive unit and integration tests available

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

## Architecture Preferences
- **4-Layer Design**: Config → Data Access → Processors → Visualizations
- **Factory Patterns**: For component creation and management
- **YAML Configuration**: Technology-agnostic, environment-specific settings
- **Production Ready**: Comprehensive error handling and monitoring

Last Updated: 2025-06-28