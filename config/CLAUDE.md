# Configuration Module - CLAUDE.md

**Developer Documentation for AI-Assisted Development**

## Module Overview

The configuration module provides centralized, YAML-driven configuration management for the Patent Intelligence Platform. This documentation details all classes, functions, and their interfaces for AI-assisted development.

## Core Classes

### ConfigurationManager

**Location**: `config/__init__.py`

**Purpose**: Central orchestrator for all platform configuration

#### Constructor

```python
ConfigurationManager(config_dir: str = None, environment: str = 'development')
```

**Parameters**:
- `config_dir` (str, optional): Path to configuration directory. Defaults to `./config/`
- `environment` (str, optional): Environment name for configuration variants. Defaults to `'development'`

**Initialization Process**:
1. Sets up configuration directory path
2. Loads environment variables from `.env` file
3. Loads all YAML configuration files
4. Validates configuration integrity
5. Sets up environment variable resolution

#### Primary Methods

##### `get(section: str, key: str = None) -> Any`

**Purpose**: Retrieve configuration values with dot notation support

**Parameters**:
- `section` (str): Top-level configuration section (e.g., 'api', 'database')
- `key` (str, optional): Nested key using dot notation (e.g., 'epo_ops.authentication.consumer_key')

**Returns**: 
- Configuration value (str, dict, list, or primitive type)
- `None` if key not found

**Examples**:
```python
# Get entire section
api_config = config.get('api')

# Get nested value
consumer_key = config.get('api', 'epo_ops.authentication.consumer_key')

# Get search patterns
patterns = config.get('search_patterns', 'technology_areas.rare_earth_elements')
```

##### `get_configuration_summary() -> Dict[str, Any]`

**Purpose**: Provide metadata about loaded configuration

**Returns**: Dictionary containing:
- `environment` (str): Current environment name
- `config_directory` (str): Path to configuration directory
- `loaded_configs` (List[str]): List of successfully loaded config files
- `total_sections` (int): Number of configuration sections
- `has_env_file` (bool): Whether .env file was found

**Example**:
```python
summary = config.get_configuration_summary()
print(f"Environment: {summary['environment']}")
print(f"Configs: {summary['loaded_configs']}")
```

#### Internal Methods

##### `_load_config_file(filename: str) -> Dict[str, Any]`

**Purpose**: Load and parse individual YAML configuration file

**Parameters**:
- `filename` (str): Name of YAML file (without path)

**Returns**: 
- Parsed YAML content as dictionary
- Empty dict if file not found or invalid

**Process**:
1. Constructs full file path
2. Loads YAML content with safe loader
3. Resolves environment variables in values
4. Validates basic structure
5. Returns parsed configuration

##### `_resolve_env_variables(value: Any) -> Any`

**Purpose**: Recursively resolve environment variable references in configuration

**Parameters**:
- `value` (Any): Configuration value (string, dict, list, or primitive)

**Returns**: Value with environment variables resolved

**Environment Variable Syntax**:
- `${ENV:VARIABLE_NAME}` - Required variable (fails if missing)
- `${ENV:VARIABLE_NAME:default_value}` - Optional with default

**Examples**:
```yaml
# In YAML file
api_key: "${ENV:OPS_KEY}"
optional_setting: "${ENV:OPTIONAL_VAR:default_value}"
```

##### `_validate_configuration() -> bool`

**Purpose**: Validate configuration completeness and integrity

**Returns**: `True` if configuration is valid, `False` otherwise

**Validation Checks**:
1. Required configuration sections present
2. Critical environment variables resolved
3. File structure integrity
4. Data type validation for known fields

## Configuration File Structure

### api_config.yaml

**Purpose**: API settings and authentication

**Structure**:
```python
{
    'epo_ops': {
        'authentication': {
            'consumer_key': str,      # From ENV:OPS_KEY
            'consumer_secret': str    # From ENV:OPS_SECRET
        },
        'endpoints': {
            'base_url': str,
            'search_endpoint': str,
            'biblio_endpoint': str
        },
        'rate_limiting': {
            'requests_per_minute': int,
            'burst_limit': int,
            'retry_delay': float
        },
        'request_config': {
            'timeout': int,
            'max_retries': int,
            'backoff_factor': float
        }
    },
    'patstat': {
        'connection': dict,
        'query_config': dict,
        'tables': dict
    }
}
```

### database_config.yaml

**Purpose**: Database connections and query configurations

**Structure**:
```python
{
    'patstat': {
        'environments': {
            'PROD': {
                'description': str,
                'database_type': str,
                'connection_config': dict
            }
        },
        'query_optimization': {
            'batch_size': int,
            'timeout_seconds': int,
            'max_results': int
        },
        'table_mappings': {
            'applications': str,    # TLS201_APPLN
            'citations': str,       # TLS212_CITATION
            'classifications': str   # TLS209_APPLN_IPC
        }
    }
}
```

### search_patterns_config.yaml

**Purpose**: Technology-specific search patterns and business logic

**Structure**:
```python
{
    'global_settings': {
        'default_quality_threshold': float,
        'max_results_per_search': int,
        'search_timeout_seconds': int
    },
    'technology_areas': {
        'rare_earth_elements': {
            'description': str,
            'keywords': List[str],
            'cpc_codes': List[str],
            'ipc_codes': List[str]
        }
    },
    'search_strategies': {
        'focused_mode': {
            'description': str,
            'max_results': int,
            'quality_threshold': float
        },
        'comprehensive_mode': {
            'description': str,
            'max_results': int,
            'quality_threshold': float
        }
    },
    'demo_parameters': {
        'test_families': List[int],
        'test_applications': List[int]
    }
}
```

### visualization_config.yaml

**Purpose**: Visualization themes, chart settings, and export configurations

**Structure**:
```python
{
    'general': {
        'themes': {
            'default_theme': str,
            'available_themes': List[str]
        },
        'output': {
            'default_format': str,
            'supported_formats': List[str],
            'image_resolution': int
        },
        'branding': {
            'title_prefix': str,
            'subtitle_format': str,
            'watermark': str
        }
    },
    'charts': {
        'layout': dict,           # Default dimensions and styling
        'color_schemes': dict,    # Color palettes for different chart types
        'scatter_plots': dict,    # Bubble scatter configurations
        'bar_charts': dict,       # Bar chart settings
        'pie_charts': dict,       # Pie chart configurations
        'time_series': dict,      # Time series chart settings
        'heatmaps': dict          # Heatmap configurations
    },
    'maps': {
        'choropleth': dict,       # Geographic map settings
        'strategic_maps': dict,   # Strategic positioning maps
        'country_coordinates': dict  # Country center coordinates
    },
    'dashboards': {
        'layouts': dict,          # Dashboard layout templates
        'panels': dict            # Panel configurations
    },
    'export': {
        'output_directory': str,  # Export directory path
        'file_naming': dict,      # File naming conventions
        'html_export': dict,      # HTML export settings
        'png_export': dict,       # PNG export settings
        'batch_export': dict      # Multi-format export settings
    }
}
```

### geographic_config.yaml

**Purpose**: Geographic mapping and regional configurations

**Structure**:
```python
{
    'country_mapping': {
        'patstat_to_iso': dict,   # PATSTAT country codes to ISO mapping
        'name_variations': dict,   # Country name variations
        'default_coordinates': dict  # Default country coordinates
    },
    'regional_groups': {
        'ip5_offices': List[str],
        'eu_countries': List[str],
        'oecd_countries': List[str]
    },
    'nuts_regions': {
        'levels': dict,           # NUTS level configurations
        'hierarchies': dict       # NUTS hierarchical relationships
    }
}
```

## Helper Functions

### `get_search_patterns_config(key: str = None) -> Any`

**Purpose**: Convenience function for accessing search patterns configuration

**Parameters**:
- `key` (str, optional): Specific configuration key using dot notation

**Returns**: Search patterns configuration or specific value

**Example**:
```python
from config import get_search_patterns_config

# Get all rare earth elements configuration
ree_config = get_search_patterns_config('technology_areas.rare_earth_elements')

# Get CPC codes
cpc_codes = get_search_patterns_config('technology_areas.rare_earth_elements.cpc_codes')
```

### `run_configuration_tests() -> bool`

**Purpose**: Execute comprehensive configuration test suite

**Returns**: `True` if all tests pass, `False` otherwise

**Test Coverage**:
1. YAML file syntax validation
2. Configuration manager functionality
3. Environment variable resolution
4. Configuration validation framework
5. Data access integration
6. Geographic configuration

## Environment Variables

### Required Variables

**OPS_KEY** - EPO OPS Consumer Key
- **Type**: String
- **Purpose**: Authentication for EPO OPS API
- **Example**: `"your_consumer_key_here"`

**OPS_SECRET** - EPO OPS Consumer Secret
- **Type**: String  
- **Purpose**: Authentication secret for EPO OPS API
- **Example**: `"your_consumer_secret_here"`

### Optional Variables

**PATSTAT_USER** - PATSTAT Database Username
- **Type**: String
- **Purpose**: Database authentication for external PATSTAT access
- **Default**: Uses TIP environment credentials

**PATSTAT_PASSWORD** - PATSTAT Database Password
- **Type**: String
- **Purpose**: Database password for external PATSTAT access
- **Default**: Uses TIP environment credentials

**GOOGLE_APPLICATION_CREDENTIALS** - Google Cloud Service Account
- **Type**: String (file path)
- **Purpose**: BigQuery access for external environments
- **Example**: `"/path/to/service_account.json"`

## Error Handling

### Common Exceptions

**ConfigurationError** - Raised when configuration is invalid or incomplete
**FileNotFoundError** - Raised when required configuration files are missing
**EnvironmentError** - Raised when required environment variables are missing
**yaml.YAMLError** - Raised when YAML syntax is invalid

### Error Recovery

1. **Missing Configuration Files**: Uses default values and logs warnings
2. **Invalid YAML Syntax**: Skips file and continues with other configurations
3. **Missing Environment Variables**: Uses placeholder values in development mode
4. **Connection Issues**: Provides fallback configurations

## Performance Characteristics

- **Configuration Loading**: <100ms for all YAML files
- **Environment Variable Resolution**: <10ms per variable
- **Memory Usage**: <5MB for complete configuration
- **Validation Time**: <200ms for full validation

## Integration Patterns

### Processor Integration
```python
from config import ConfigurationManager
from processors import ApplicantAnalyzer

config = ConfigurationManager()
analyzer = ApplicantAnalyzer(config)  # Passes config to processor
```

### Data Access Integration
```python
from config import ConfigurationManager
from data_access import PatstatClient

config = ConfigurationManager()
patstat = PatstatClient(environment='PROD', config=config)
```

### Visualization Integration
```python
from config import ConfigurationManager
from visualizations import ProductionChartCreator

config = ConfigurationManager()
chart_creator = ProductionChartCreator(config)
```

## Testing Interface

### Test Functions

**test_yaml_file_syntax()** - Validates YAML syntax for all configuration files
**test_configuration_manager()** - Tests ConfigurationManager functionality
**test_configuration_validation()** - Validates configuration completeness
**test_centralized_search_patterns()** - Tests search pattern consolidation
**test_environment_handling()** - Tests environment variable resolution
**test_data_access_integration()** - Tests integration with data access module
**test_geographic_configuration()** - Tests geographic configuration loading

### Running Tests

```python
# Run all configuration tests
from config import run_configuration_tests
success = run_configuration_tests()

# Run specific test
from config.test_config import test_yaml_file_syntax
test_yaml_file_syntax()
```

---

**Last Updated**: 2025-06-29  
**Module Version**: 1.0  
**API Stability**: Stable