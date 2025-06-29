"""
Patent Search Processor - Clean Implementation
Demonstrates clean architecture with standardized interface
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import time

from .base import BaseProcessor, ProcessorResult


class PatentSearchProcessor(BaseProcessor):
    """
    Clean implementation of patent search processing
    Demonstrates standardized interface and dependency injection
    """
    
    def __init__(self, data_provider=None, config: Dict[str, Any] = None):
        """
        Initialize patent search processor
        
        Args:
            data_provider: Data access provider (injected dependency)
            config: Search configuration parameters
        """
        super().__init__(data_provider, config)
        
        # Default search configuration
        self.default_config = {
            'max_results': 10000,
            'include_families': True,
            'include_citations': True,
            'quality_filters': {
                'min_claims': 1,
                'exclude_withdrawn': True
            }
        }
        
        # Merge with provided config
        self.search_config = {**self.default_config, **self.config}
    
    def process(self, data: pd.DataFrame, search_params: Dict[str, Any] = None, **kwargs) -> ProcessorResult:
        """
        Process patent search with clean interface
        
        Args:
            data: Input data (can be empty for initial search)
            search_params: Search parameters (technology, years, countries, etc.)
            **kwargs: Additional search options
            
        Returns:
            ProcessorResult with search results and metadata
        """
        start_time = time.time()
        
        # Input validation
        if search_params is None:
            search_params = {}
        
        self.logger.info(f"🔍 Starting patent search with params: {search_params}")
        
        try:
            # In a real implementation, this would query PATSTAT/OPS
            # For demo, we'll generate realistic sample data
            search_results = self._execute_search(search_params)
            
            # Apply quality filters
            filtered_results = self._apply_quality_filters(search_results)
            
            processing_time = time.time() - start_time
            
            # Create metadata
            metadata = self._create_metadata(
                input_data=data,
                processing_time=processing_time,
                additional_metadata={
                    'search_params': search_params,
                    'raw_results': len(search_results),
                    'filtered_results': len(filtered_results),
                    'filter_ratio': len(filtered_results) / len(search_results) if len(search_results) > 0 else 0,
                    'search_config': self.search_config
                }
            )
            
            # Update performance metrics
            self._update_performance_metrics(processing_time, len(filtered_results))
            
            self.logger.info(f"✅ Search completed: {len(filtered_results)} results in {processing_time:.2f}s")
            
            return ProcessorResult(
                data=filtered_results,
                metadata=metadata,
                status="completed"
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Search processing failed: {str(e)}"
            self.logger.error(error_msg)
            
            return ProcessorResult(
                data=pd.DataFrame(),
                metadata=self._create_metadata(data, processing_time, {'error': str(e)}),
                status="failed",
                errors=[error_msg]
            )
    
    def _execute_search(self, search_params: Dict[str, Any]) -> pd.DataFrame:
        """
        Execute patent search (demo implementation)
        In production, this would query PATSTAT/OPS API
        """
        # Generate realistic demo data based on search parameters
        num_results = min(
            search_params.get('limit', 1000),
            self.search_config['max_results']
        )
        
        # Technology areas for demo
        technology_areas = search_params.get('technology_area', 'energy storage')
        if isinstance(technology_areas, str):
            technology_areas = [technology_areas]
        
        # Filing years
        filing_years = search_params.get('filing_years', [2020, 2021, 2022, 2023])
        
        # Countries
        countries = search_params.get('countries', ['US', 'DE', 'JP', 'CN', 'FR'])
        
        # Generate realistic patent data
        np.random.seed(42)  # For reproducible demo data
        
        data = {
            'docdb_family_id': range(1, num_results + 1),
            'appln_id': range(10001, 10001 + num_results),
            'person_id': np.random.randint(1, 200, num_results),
            'person_name': [f'Company_{i}' for i in np.random.randint(1, 50, num_results)],
            'person_ctry_code': np.random.choice(countries, num_results),
            'appln_filing_year': np.random.choice(filing_years, num_results),
            'appln_filing_date': pd.date_range('2020-01-01', '2023-12-31', periods=num_results),
            'ipc_class_symbol': self._generate_ipc_codes(technology_areas, num_results),
            'appln_title': [f'Patent Application {i}: {np.random.choice(technology_areas)}' for i in range(num_results)],
            'nb_citing_docdb_fam': np.random.poisson(5, num_results),
            'nb_cited_docdb_fam': np.random.poisson(3, num_results),
            'appln_kind': np.random.choice(['A', 'B', 'U'], num_results, p=[0.7, 0.2, 0.1])
        }
        
        return pd.DataFrame(data)
    
    def _generate_ipc_codes(self, technology_areas: List[str], count: int) -> List[str]:
        """Generate realistic IPC codes based on technology areas"""
        ipc_mapping = {
            'energy storage': ['H01M', 'H02J', 'H01G'],
            'computing': ['G06F', 'G06N', 'H04L'],
            'chemistry': ['C07D', 'C08F', 'A61K'],
            'medicine': ['A61K', 'A61P', 'C12N'],
            'telecommunications': ['H04L', 'H04W', 'H04B'],
            'automotive': ['B60L', 'F02D', 'B62D']
        }
        
        # Map technology areas to IPC codes
        relevant_ipcs = []
        for tech in technology_areas:
            tech_lower = tech.lower()
            for key, ipcs in ipc_mapping.items():
                if key in tech_lower or tech_lower in key:
                    relevant_ipcs.extend(ipcs)
        
        if not relevant_ipcs:
            relevant_ipcs = ['G06F', 'H01M', 'C07D', 'A61K']  # Default mix
        
        return np.random.choice(relevant_ipcs, count).tolist()
    
    def _apply_quality_filters(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply quality filters to search results"""
        filtered_data = data.copy()
        
        quality_config = self.search_config.get('quality_filters', {})
        
        # Filter by application kind if specified
        if quality_config.get('exclude_withdrawn', True):
            # In real implementation, would filter withdrawn applications
            pass
        
        # Apply minimum citations filter
        min_citations = quality_config.get('min_citations', 0)
        if min_citations > 0:
            filtered_data = filtered_data[
                filtered_data['nb_citing_docdb_fam'] >= min_citations
            ]
        
        # Remove duplicates by family ID
        filtered_data = filtered_data.drop_duplicates(subset=['docdb_family_id'])
        
        self.logger.info(f"Quality filters applied: {len(data)} → {len(filtered_data)} records")
        
        return filtered_data
    
    def configure_search(self, **config_updates):
        """Update search configuration"""
        self.search_config.update(config_updates)
        self.logger.info(f"Search configuration updated: {config_updates}")
    
    def get_search_summary(self) -> Dict[str, Any]:
        """Get summary of search performance"""
        return {
            'configuration': self.search_config,
            'total_searches': len(self.processing_history),
            'total_records_found': self.total_records_processed,
            'average_search_time': self.average_processing_time,
            'recent_searches': self.processing_history[-5:] if self.processing_history else []
        }