"""
Classification System Configuration
Easy switching between IPC and CPC classification systems.

Reads configuration from YAML config file instead of environment variables.
Provides unified interface for patent classification analysis.
"""

import yaml
from typing import Union, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ClassificationConfig:
    """
    Configuration manager for patent classification systems.
    
    Reads configuration from YAML config file:
    1. YAML config file settings (primary)
    2. Direct configuration override
    3. Auto-detection fallback
    """
    
    # Supported classification systems
    SUPPORTED_SYSTEMS = ['ipc', 'cpc']
    DEFAULT_SYSTEM = 'cpc'  # CPC is more comprehensive
    
    def __init__(self, classification_system: Optional[str] = None, config_path: Optional[Path] = None):
        """
        Initialize classification configuration.
        
        Args:
            classification_system: 'ipc' or 'cpc', or None to read from config
            config_path: Path to YAML config file
        """
        self.config_path = config_path or Path(__file__).parent.parent / 'config' / 'search_patterns_config.yaml'
        self.config = self._load_config()
        self.system = self._determine_system(classification_system)
        self.client = None
        
        logger.debug(f"📋 Classification system: {self.system.upper()}")
    
    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    logger.debug(f"✅ Loaded config from {self.config_path}")
                    return config
            else:
                logger.warning(f"⚠️ Config file not found: {self.config_path}")
                return {}
        except Exception as e:
            logger.error(f"❌ Failed to load config: {e}")
            return {}
    
    def _determine_system(self, override: Optional[str] = None) -> str:
        """Determine which classification system to use."""
        
        # 1. Direct override
        if override:
            system = override.lower()
            if system in self.SUPPORTED_SYSTEMS:
                return system
            else:
                logger.warning(f"⚠️ Unsupported system '{override}', using config/default")
        
        # 2. YAML configuration
        if 'classification' in self.config:
            config_system = self.config['classification'].get('system', '').lower()
            if config_system in self.SUPPORTED_SYSTEMS:
                logger.debug(f"📄 Using system from config: {config_system}")
                return config_system
        
        # 3. Check database availability
        available_systems = self._check_available_systems()
        
        if self.DEFAULT_SYSTEM in available_systems:
            return self.DEFAULT_SYSTEM
        elif available_systems:
            # Use first available system
            system = available_systems[0]
            logger.info(f"💡 Using available system: {system}")
            return system
        else:
            logger.warning(f"⚠️ No databases found, defaulting to {self.DEFAULT_SYSTEM}")
            return self.DEFAULT_SYSTEM
    
    def _check_available_systems(self) -> list:
        """Check which classification databases are available."""
        available = []
        
        # Check for CPC database
        try:
            from .cpc_database_client import get_cpc_client
            cpc_client = get_cpc_client()
            if cpc_client.available:
                available.append('cpc')
                logger.debug("✅ CPC database available")
            else:
                logger.debug("❌ CPC database not available")
        except Exception as e:
            logger.debug(f"❌ CPC client error: {e}")
        
        # Check for IPC database
        try:
            from .ipc_database_client import get_ipc_client
            ipc_client = get_ipc_client()
            if ipc_client.available:
                available.append('ipc')
                logger.debug("✅ IPC database available")
            else:
                logger.debug("❌ IPC database not available")
        except Exception as e:
            logger.debug(f"❌ IPC client error: {e}")
        
        return available
    
    def get_client(self):
        """Get the appropriate classification database client."""
        if self.client is None:
            if self.system == 'cpc':
                from .cpc_database_client import get_cpc_client
                self.client = get_cpc_client()
            elif self.system == 'ipc':
                from .ipc_database_client import get_ipc_client
                self.client = get_ipc_client()
            else:
                raise ValueError(f"Unknown classification system: {self.system}")
        
        return self.client
    
    def get_description(self, code: str) -> str:
        """Get description for classification code."""
        client = self.get_client()
        
        if self.system == 'cpc':
            return client.get_cpc_description(code)
        elif self.system == 'ipc':
            return client.get_ipc_description(code)
        else:
            return f"Unknown system: {self.system}"
    
    def get_technology_domains(self, codes: list) -> 'pd.DataFrame':
        """Get technology domains breakdown."""
        client = self.get_client()
        return client.get_technology_domains(codes)
    
    def search_codes(self, search_term: str, limit: int = 20) -> 'pd.DataFrame':
        """Search classification codes."""
        client = self.get_client()
        
        if self.system == 'cpc':
            return client.search_cpc_codes(search_term, limit)
        elif self.system == 'ipc':
            return client.search_ipc_codes(search_term, limit)
        else:
            import pandas as pd
            return pd.DataFrame()
    
    def switch_system(self, new_system: str):
        """Switch to different classification system."""
        if new_system.lower() not in self.SUPPORTED_SYSTEMS:
            raise ValueError(f"Unsupported system: {new_system}")
        
        old_system = self.system
        self.system = new_system.lower()
        self.client = None  # Reset client
        
        logger.info(f"🔄 Switched from {old_system.upper()} to {self.system.upper()}")
    
    def get_system_info(self) -> dict:
        """Get information about current classification system."""
        client = self.get_client()
        
        info = {
            'system': self.system.upper(),
            'available': client.available if client else False,
            'description': self._get_system_description(),
        }
        
        if client and client.available:
            # Get basic stats
            try:
                if self.system == 'cpc':
                    summary = client.get_subclass_summary(limit=1)
                    info['subclasses'] = len(client.get_subclass_summary(limit=1000))
                elif self.system == 'ipc':
                    summary = client.get_subclass_summary(limit=1)
                    info['subclasses'] = len(client.get_subclass_summary(limit=1000))
                    info['illustrations'] = True  # IPC has illustrations
            except:
                pass
        
        return info
    
    def _get_system_description(self) -> str:
        """Get description of classification system."""
        descriptions = {
            'cpc': 'Cooperative Patent Classification (EPO+USPTO) - 680+ subclasses',
            'ipc': 'International Patent Classification (WIPO) - 654+ subclasses with illustrations'
        }
        return descriptions.get(self.system, 'Unknown classification system')

# Global configuration instance
_config = None

def get_classification_config(system: Optional[str] = None) -> ClassificationConfig:
    """
    Get global classification configuration.
    
    Args:
        system: Override classification system
        
    Returns:
        ClassificationConfig instance
    """
    global _config
    
    if _config is None or system:
        _config = ClassificationConfig(system)
    
    return _config

def set_classification_system(system: str):
    """
    Set global classification system.
    
    Args:
        system: 'ipc' or 'cpc'
    """
    config = get_classification_config()
    config.switch_system(system)

def get_classification_client():
    """Get the current classification database client."""
    config = get_classification_config()
    return config.get_client()

# Convenience functions for easy use
def describe_code(code: str) -> str:
    """Get description for classification code using current system."""
    config = get_classification_config()
    return config.get_description(code)

def analyze_technology_domains(codes: list) -> 'pd.DataFrame':
    """Analyze technology domains using current system."""
    config = get_classification_config()
    return config.get_technology_domains(codes)

def search_classifications(search_term: str, limit: int = 20) -> 'pd.DataFrame':
    """Search classifications using current system."""
    config = get_classification_config()
    return config.search_codes(search_term, limit)