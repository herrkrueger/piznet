"""
Configuration Manager - Clean Architecture
Centralized YAML configuration with environment variable support
"""

import yaml
import os
from typing import Dict, Any, Optional, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ConfigurationManager:
    """
    Centralized configuration manager supporting YAML files and environment variables
    Maintains compatibility with existing config structure while adding clean architecture
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager
        
        Args:
            config_path: Path to configuration directory or file
        """
        self.config_path = Path(config_path) if config_path else self._find_config_path()
        self.config_data = {}
        self.env_overrides = {}
        
        # Load configurations
        self._load_configurations()
        self._load_environment_overrides()
    
    def _find_config_path(self) -> Path:
        """Find configuration directory automatically"""
        # Look for config directory
        possible_paths = [
            Path.cwd() / 'config',
            Path.cwd() / 'src' / 'config', 
            Path(__file__).parent.parent.parent / 'config',  # Original config dir
            Path(__file__).parent / 'yaml'  # New config location
        ]
        
        for path in possible_paths:
            if path.exists() and path.is_dir():
                logger.info(f"Found config directory: {path}")
                return path
        
        # Default to config directory next to this file
        config_dir = Path(__file__).parent / 'yaml'
        config_dir.mkdir(exist_ok=True)
        return config_dir
    
    def _load_configurations(self):
        """Load all YAML configuration files"""
        
        # Core configuration files
        config_files = [
            'data_providers.yaml',
            'analysis.yaml', 
            'visualization.yaml',
            'processing.yaml'
        ]
        
        for config_file in config_files:
            file_path = self.config_path / config_file
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        config_data = yaml.safe_load(f)
                        if config_data:
                            # Use filename without extension as top-level key
                            config_key = file_path.stem
                            self.config_data[config_key] = config_data
                            logger.debug(f"✅ Loaded config: {config_file}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load {config_file}: {e}")
            else:
                logger.debug(f"Config file not found: {config_file}")
                # Create default config
                self._create_default_config(file_path)
    
    def _load_environment_overrides(self):
        """Load environment variable overrides"""
        
        # Data provider overrides
        self.env_overrides['data_providers'] = {
            'patstat': {
                'environment': os.getenv('PATSTAT_ENVIRONMENT', 'PROD'),
                'enabled': os.getenv('ENABLE_PATSTAT', 'true').lower() == 'true'
            },
            'epo_ops': {
                'consumer_key': os.getenv('OPS_KEY'),
                'consumer_secret': os.getenv('OPS_SECRET'),
                'enabled': os.getenv('ENABLE_EPO_OPS', 'true').lower() == 'true',
                'rate_limit_per_minute': int(os.getenv('OPS_RATE_LIMIT_PER_MINUTE', '30')),
                'rate_limit_per_week': int(os.getenv('OPS_RATE_LIMIT_PER_WEEK', '4000'))
            },
            'wipo_ipc': {
                'enabled': os.getenv('ENABLE_WIPO_IPC', 'true').lower() == 'true'
            }
        }
        
        # Caching overrides
        self.env_overrides['caching'] = {
            'enabled': os.getenv('ENABLE_PROVIDER_CACHING', 'true').lower() == 'true',
            'ttl_hours': int(os.getenv('CACHE_TTL_HOURS', '24'))
        }
        
        # Logging overrides
        self.env_overrides['logging'] = {
            'level': os.getenv('LOG_LEVEL', 'INFO')
        }
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        
        Args:
            key_path: Dot-separated path (e.g., 'data_providers.patstat.environment')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        # Check environment overrides first
        env_value = self._get_from_dict(self.env_overrides, key_path)
        if env_value is not None:
            return env_value
        
        # Check config files
        config_value = self._get_from_dict(self.config_data, key_path)
        if config_value is not None:
            return config_value
        
        return default
    
    def _get_from_dict(self, data: Dict[str, Any], key_path: str) -> Any:
        """Get value from nested dictionary using dot notation"""
        keys = key_path.split('.')
        current = data
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        
        return current
    
    def get_data_provider_config(self, provider_name: str) -> Dict[str, Any]:
        """
        Get complete configuration for a data provider
        
        Args:
            provider_name: Name of the provider
            
        Returns:
            Provider configuration dictionary
        """
        # Get base config
        base_config = self.get(f'data_providers.{provider_name}', {})
        
        # Get environment overrides
        env_config = self.get(f'data_providers.{provider_name}', {})
        
        # Merge configurations (env overrides base)
        merged_config = {**base_config, **env_config}
        
        return merged_config
    
    def get_analysis_config(self, analysis_type: str = 'default') -> Dict[str, Any]:
        """Get analysis configuration"""
        return self.get(f'analysis.{analysis_type}', self.get('analysis.default', {}))
    
    def get_visualization_config(self, viz_type: str = 'default') -> Dict[str, Any]:
        """Get visualization configuration"""
        return self.get(f'visualization.{viz_type}', self.get('visualization.default', {}))
    
    def get_processing_config(self, processor_type: str = 'default') -> Dict[str, Any]:
        """Get processing configuration"""
        return self.get(f'processing.{processor_type}', self.get('processing.default', {}))
    
    def _create_default_config(self, file_path: Path):
        """Create default configuration file"""
        
        config_templates = {
            'data_providers.yaml': {
                'patstat': {
                    'environment': 'PROD',
                    'timeout': 30,
                    'max_results': 100000,
                    'enabled': True
                },
                'epo_ops': {
                    'base_url': 'https://ops.epo.org/3.2',
                    'rate_limit_per_minute': 30,
                    'rate_limit_per_week': 4000,
                    'timeout': 30,
                    'enabled': True
                },
                'wipo_ipc': {
                    'classification_version': '2023.01',
                    'language': 'en',
                    'cache_classifications': True,
                    'enabled': True
                }
            },
            'analysis.yaml': {
                'default': {
                    'include_regional': True,
                    'include_technology': True,
                    'include_trends': True,
                    'validation_enabled': True
                },
                'quick': {
                    'include_regional': True,
                    'include_technology': False,
                    'include_trends': False,
                    'validation_enabled': False
                },
                'comprehensive': {
                    'include_regional': True,
                    'include_technology': True,
                    'include_trends': True,
                    'include_citation': True,
                    'validation_enabled': True
                }
            },
            'visualization.yaml': {
                'default': {
                    'include_charts': True,
                    'include_maps': True,
                    'include_dashboards': True,
                    'export_formats': ['html', 'png', 'pdf']
                },
                'executive': {
                    'include_charts': True,
                    'include_maps': False,
                    'include_dashboards': True,
                    'export_formats': ['html', 'pdf']
                }
            },
            'processing.yaml': {
                'default': {
                    'max_records': 10000,
                    'chunk_size': 1000,
                    'parallel_processing': True,
                    'validation_enabled': True
                },
                'high_volume': {
                    'max_records': 100000,
                    'chunk_size': 5000,
                    'parallel_processing': True,
                    'validation_enabled': True
                }
            }
        }
        
        template_data = config_templates.get(file_path.name, {})
        if template_data:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    yaml.dump(template_data, f, default_flow_style=False, indent=2)
                logger.info(f"✅ Created default config: {file_path.name}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to create default config {file_path.name}: {e}")
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate configuration integrity"""
        
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'provider_status': {}
        }
        
        # Validate data providers
        for provider in ['patstat', 'epo_ops', 'wipo_ipc']:
            provider_config = self.get_data_provider_config(provider)
            
            if not provider_config.get('enabled', False):
                validation_results['warnings'].append(f"Provider {provider} is disabled")
                validation_results['provider_status'][provider] = 'disabled'
                continue
            
            # Provider-specific validation
            if provider == 'epo_ops':
                if not provider_config.get('consumer_key') or not provider_config.get('consumer_secret'):
                    validation_results['errors'].append(f"EPO OPS missing credentials")
                    validation_results['valid'] = False
                    validation_results['provider_status'][provider] = 'invalid'
                else:
                    validation_results['provider_status'][provider] = 'valid'
            else:
                validation_results['provider_status'][provider] = 'valid'
        
        return validation_results
    
    def get_summary(self) -> Dict[str, Any]:
        """Get configuration summary"""
        
        return {
            'config_path': str(self.config_path),
            'config_files_loaded': list(self.config_data.keys()),
            'environment_overrides': bool(self.env_overrides),
            'data_providers_configured': len(self.get('data_providers', {})),
            'validation_status': self.validate_configuration()
        }