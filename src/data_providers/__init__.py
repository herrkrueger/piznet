"""
Data Providers Package - Clean Architecture
Unified interface for all external data sources
"""

from .base import DataProvider, DataProviderResult, DataProviderFactory
from .patstat.provider import PatstatDataProvider
from .epo_ops.provider import EPOOpsDataProvider
from .classification.ipc_provider import WipoIpcProvider
from .classification.cpc_provider import CpcProvider
from .geographic.nuts_provider import NutsGeoProvider
from .market.usgs_provider import UsgsMarketProvider
from .research.depa_provider import DepaTechProvider
from .research.lens_provider import LensOrgProvider
from .research.hochschul_provider import HochschulkompassProvider

__all__ = [
    'DataProvider', 'DataProviderResult', 'DataProviderFactory',
    'PatstatDataProvider', 'EPOOpsDataProvider', 
    'WipoIpcProvider', 'CpcProvider', 'NutsGeoProvider',
    'UsgsMarketProvider', 'DepaTechProvider', 'LensOrgProvider', 
    'HochschulkompassProvider'
]