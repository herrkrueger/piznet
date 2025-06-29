"""USGS Market Data Provider - Placeholder"""
import pandas as pd
from typing import Dict, Any
from ..base import DataProvider, DataProviderResult, DataProviderType, register_provider

@register_provider(DataProviderType.USGS_MARKET)
class UsgsMarketProvider(DataProvider):
    def _get_provider_type(self) -> DataProviderType:
        return DataProviderType.USGS_MARKET
    def connect(self) -> bool:
        return True
    def disconnect(self):
        pass
    def validate_query_params(self, query_params: Dict[str, Any]) -> bool:
        return True
    def query(self, query_params: Dict[str, Any], **kwargs) -> DataProviderResult:
        return DataProviderResult(data=pd.DataFrame(), metadata={'provider': 'USGS_MARKET'}, status='no_data')