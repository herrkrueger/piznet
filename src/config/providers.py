"""
Data Provider Configuration Helper
Specialized configuration for data providers
"""

from typing import Dict, Any
from .manager import ConfigurationManager


class DataProviderConfig:
    """
    Helper class for data provider configurations
    Bridges new clean architecture with existing YAML configs
    """
    
    def __init__(self, config_manager: ConfigurationManager = None):
        """
        Initialize data provider configuration
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager or ConfigurationManager()
    
    def get_provider_config(self, provider_name: str) -> Dict[str, Any]:
        """Get configuration for specific provider"""
        return self.config_manager.get_data_provider_config(provider_name)
    
    def get_all_provider_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get configurations for all providers"""
        
        provider_names = [
            'patstat', 'epo_ops', 'wipo_ipc', 'cpc', 
            'nuts_geo', 'usgs_market', 'depa_tech', 
            'lens_org', 'hochschulkompass'
        ]
        
        configs = {}
        for provider_name in provider_names:
            configs[f'{provider_name}_config'] = self.get_provider_config(provider_name)
        
        return configs
    
    def get_enabled_providers(self) -> Dict[str, bool]:
        """Get enabled status for all providers"""
        
        enabled_status = {}
        all_configs = self.get_all_provider_configs()
        
        for provider_key, config in all_configs.items():
            provider_name = provider_key.replace('_config', '')
            enabled_status[provider_name] = config.get('enabled', False)
        
        return enabled_status
    
    def validate_providers(self) -> Dict[str, Any]:
        """Validate all provider configurations"""
        return self.config_manager.validate_configuration()