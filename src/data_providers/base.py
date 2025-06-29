"""
Base Data Provider Classes - Clean Architecture Foundation
Standardized interfaces for all external data sources
"""

import pandas as pd
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import time
from enum import Enum

# Setup logging
logger = logging.getLogger(__name__)


class DataProviderType(Enum):
    """Enumeration of data provider types"""
    PATSTAT = "patstat"
    EPO_OPS = "epo_ops"
    WIPO_IPC = "wipo_ipc"
    CPC = "cpc"
    NUTS_GEO = "nuts_geo"
    USGS_MARKET = "usgs_market"
    DEPA_TECH = "depa_tech"
    LENS_ORG = "lens_org"
    HOCHSCHULKOMPASS = "hochschulkompass"


class DataProviderResult:
    """
    Standardized result container for all data providers
    """
    
    def __init__(self, data: Union[pd.DataFrame, Dict[str, Any]], metadata: Dict[str, Any], 
                 status: str = "success", errors: List[str] = None, warnings: List[str] = None):
        """
        Initialize data provider result
        
        Args:
            data: Retrieved data (DataFrame or structured dict)
            metadata: Provider metadata and query information
            status: Query status ('success', 'partial', 'failed', 'no_data')
            errors: List of any errors encountered
            warnings: List of any warnings encountered
        """
        self.data = data
        self.metadata = metadata or {}
        self.status = status
        self.errors = errors or []
        self.warnings = warnings or []
        self.timestamp = datetime.now().isoformat()
        
        # Add basic statistics
        if isinstance(data, pd.DataFrame):
            self.metadata.update({
                'record_count': len(data),
                'column_count': len(data.columns),
                'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024 / 1024,
                'data_type': 'dataframe'
            })
        elif isinstance(data, dict):
            self.metadata.update({
                'data_keys': list(data.keys()),
                'data_type': 'dictionary'
            })
        
        self.metadata['query_timestamp'] = self.timestamp
    
    @property
    def is_successful(self) -> bool:
        """Check if query was successful"""
        return self.status == "success" and not self.errors
    
    @property
    def has_data(self) -> bool:
        """Check if result contains data"""
        if isinstance(self.data, pd.DataFrame):
            return not self.data.empty
        elif isinstance(self.data, dict):
            return bool(self.data)
        return False
    
    def add_error(self, error_message: str):
        """Add an error to the result"""
        self.errors.append(error_message)
        if self.status == "success":
            self.status = "partial"
    
    def add_warning(self, warning_message: str):
        """Add a warning to the result"""
        self.warnings.append(warning_message)


class DataProvider(ABC):
    """
    Abstract base class for all data providers
    Ensures standardized interface for external data access
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize base data provider
        
        Args:
            config: Provider-specific configuration
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.provider_type = self._get_provider_type()
        
        # Connection tracking
        self.is_connected = False
        self.connection_metadata = {}
        
        # Query tracking
        self.query_history = []
        self.total_queries = 0
        self.successful_queries = 0
        self.average_query_time = 0.0
        
        # Rate limiting
        self.rate_limit_config = self.config.get('rate_limiting', {})
        self.last_query_time = 0
    
    @abstractmethod
    def _get_provider_type(self) -> DataProviderType:
        """Return the provider type enumeration"""
        pass
    
    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connection to data source
        
        Returns:
            True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    def disconnect(self):
        """Close connection to data source"""
        pass
    
    @abstractmethod
    def query(self, query_params: Dict[str, Any], **kwargs) -> DataProviderResult:
        """
        Execute query against data source
        
        Args:
            query_params: Query parameters specific to the provider
            **kwargs: Additional query options
            
        Returns:
            DataProviderResult with retrieved data and metadata
        """
        pass
    
    @abstractmethod
    def validate_query_params(self, query_params: Dict[str, Any]) -> bool:
        """
        Validate query parameters before execution
        
        Args:
            query_params: Query parameters to validate
            
        Returns:
            True if valid, False otherwise
        """
        pass
    
    def test_connection(self) -> Dict[str, Any]:
        """
        Test connection to data source
        
        Returns:
            Dictionary with connection test results
        """
        test_start = time.time()
        
        try:
            if not self.is_connected:
                connection_success = self.connect()
            else:
                connection_success = True
            
            if connection_success:
                # Run a minimal test query
                test_result = self._run_connection_test()
                test_time = time.time() - test_start
                
                return {
                    'connection_status': 'success',
                    'provider_type': self.provider_type.value,
                    'test_time': test_time,
                    'test_result': test_result,
                    'metadata': self.connection_metadata
                }
            else:
                return {
                    'connection_status': 'failed',
                    'provider_type': self.provider_type.value,
                    'test_time': time.time() - test_start,
                    'error': 'Failed to establish connection'
                }
                
        except Exception as e:
            return {
                'connection_status': 'error',
                'provider_type': self.provider_type.value,
                'test_time': time.time() - test_start,
                'error': str(e)
            }
    
    def _run_connection_test(self) -> Dict[str, Any]:
        """
        Run provider-specific connection test
        Should be overridden by specific providers
        """
        return {'test': 'basic_connection', 'status': 'passed'}
    
    def _enforce_rate_limits(self):
        """Enforce rate limiting if configured"""
        if not self.rate_limit_config:
            return
        
        min_interval = self.rate_limit_config.get('min_interval_seconds', 0)
        if min_interval > 0:
            time_since_last = time.time() - self.last_query_time
            if time_since_last < min_interval:
                sleep_time = min_interval - time_since_last
                self.logger.info(f"Rate limiting: sleeping {sleep_time:.2f}s")
                time.sleep(sleep_time)
        
        self.last_query_time = time.time()
    
    def _update_query_metrics(self, query_time: float, success: bool):
        """Update provider query metrics"""
        self.total_queries += 1
        if success:
            self.successful_queries += 1
        
        # Update average query time (moving average)
        if self.total_queries == 1:
            self.average_query_time = query_time
        else:
            self.average_query_time = (
                (self.average_query_time * (self.total_queries - 1) + query_time) / self.total_queries
            )
        
        # Store in history (keep last 100)
        self.query_history.append({
            'timestamp': datetime.now().isoformat(),
            'query_time': query_time,
            'success': success
        })
        
        if len(self.query_history) > 100:
            self.query_history = self.query_history[-100:]
    
    def get_provider_summary(self) -> Dict[str, Any]:
        """Get summary of provider status and performance"""
        success_rate = (
            self.successful_queries / self.total_queries 
            if self.total_queries > 0 else 0
        )
        
        return {
            'provider_type': self.provider_type.value,
            'connection_status': 'connected' if self.is_connected else 'disconnected',
            'total_queries': self.total_queries,
            'success_rate': f"{success_rate:.1%}",
            'average_query_time': f"{self.average_query_time:.3f}s",
            'configuration': {k: v for k, v in self.config.items() if 'password' not in k.lower() and 'key' not in k.lower()},
            'recent_queries': self.query_history[-5:] if self.query_history else []
        }
    
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()


class DataProviderFactory:
    """
    Factory for creating and managing data providers
    Handles auto-detection and configuration
    """
    
    _registered_providers = {}
    
    @classmethod
    def register_provider(cls, provider_type: DataProviderType, provider_class):
        """Register a data provider class"""
        cls._registered_providers[provider_type] = provider_class
    
    @classmethod
    def create_provider(cls, provider_type: Union[str, DataProviderType], 
                       config: Dict[str, Any] = None) -> DataProvider:
        """
        Create a data provider instance
        
        Args:
            provider_type: Type of provider to create
            config: Provider configuration
            
        Returns:
            Configured DataProvider instance
        """
        if isinstance(provider_type, str):
            provider_type = DataProviderType(provider_type.lower())
        
        if provider_type not in cls._registered_providers:
            raise ValueError(f"Unknown provider type: {provider_type}")
        
        provider_class = cls._registered_providers[provider_type]
        return provider_class(config)
    
    @classmethod
    def create_all_providers(cls, global_config: Dict[str, Any] = None) -> Dict[DataProviderType, DataProvider]:
        """
        Create all registered providers
        
        Args:
            global_config: Global configuration containing provider-specific configs
            
        Returns:
            Dictionary of provider instances by type
        """
        providers = {}
        global_config = global_config or {}
        
        for provider_type in cls._registered_providers:
            try:
                provider_config = global_config.get(f'{provider_type.value}_config', {})
                provider = cls.create_provider(provider_type, provider_config)
                providers[provider_type] = provider
                logger.info(f"✅ Created provider: {provider_type.value}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to create provider {provider_type.value}: {e}")
        
        return providers
    
    @classmethod
    def get_available_providers(cls) -> List[DataProviderType]:
        """Get list of available provider types"""
        return list(cls._registered_providers.keys())
    
    @classmethod
    def auto_detect_providers(cls, config: Dict[str, Any] = None) -> List[DataProviderType]:
        """
        Auto-detect available providers based on configuration and dependencies
        
        Args:
            config: Configuration to check for provider settings
            
        Returns:
            List of available provider types
        """
        available_providers = []
        config = config or {}
        
        # Check for each provider type
        for provider_type in cls._registered_providers:
            try:
                # Check if configuration exists
                provider_config = config.get(f'{provider_type.value}_config')
                if provider_config:
                    # Try to create and test the provider
                    provider = cls.create_provider(provider_type, provider_config)
                    test_result = provider.test_connection()
                    if test_result.get('connection_status') in ['success', 'partial']:
                        available_providers.append(provider_type)
                        logger.info(f"✅ Auto-detected: {provider_type.value}")
                    else:
                        logger.warning(f"⚠️ Provider unavailable: {provider_type.value}")
            except Exception as e:
                logger.debug(f"Provider {provider_type.value} not available: {e}")
        
        return available_providers


# Auto-registration decorator
def register_provider(provider_type: DataProviderType):
    """Decorator to auto-register provider classes"""
    def decorator(provider_class):
        DataProviderFactory.register_provider(provider_type, provider_class)
        return provider_class
    return decorator