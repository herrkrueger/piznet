"""depa.tech API Provider - Placeholder"""
import pandas as pd
from typing import Dict, Any
from ..base import DataProvider, DataProviderResult, DataProviderType, register_provider

@register_provider(DataProviderType.DEPA_TECH)
class DepaTechProvider(DataProvider):
    def _get_provider_type(self) -> DataProviderType:
        return DataProviderType.DEPA_TECH
    def connect(self) -> bool:
        return True
    def disconnect(self):
        pass
    def validate_query_params(self, query_params: Dict[str, Any]) -> bool:
        return True
    def query(self, query_params: Dict[str, Any], **kwargs) -> DataProviderResult:
        return DataProviderResult(data=pd.DataFrame(), metadata={'provider': 'DEPA_TECH'}, status='no_data')