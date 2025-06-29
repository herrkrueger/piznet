"""
PATSTAT Data Provider - Real Implementation
Production-ready PATSTAT database access with clean architecture
"""

import pandas as pd
import logging
from typing import Dict, Any, List, Optional, Union
import time
from datetime import datetime

from ..base import DataProvider, DataProviderResult, DataProviderType, register_provider

# Setup logging
logger = logging.getLogger(__name__)


@register_provider(DataProviderType.PATSTAT)
class PatstatDataProvider(DataProvider):
    """
    Production PATSTAT data provider using epo.tipdata.patstat
    Provides access to complete EPO PATSTAT database with optimized queries
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize PATSTAT provider
        
        Args:
            config: PATSTAT configuration including environment and connection settings
        """
        super().__init__(config)
        
        # Default configuration
        self.default_config = {
            'environment': 'PROD',  # 'PROD' or 'TEST'
            'connection_timeout': 30,
            'query_timeout': 300,
            'chunk_size': 10000,
            'max_results': 100000,
            'use_indexes': True,
            'cache_connections': True
        }
        
        # Merge configurations
        self.patstat_config = {**self.default_config, **self.config}
        
        # PATSTAT client
        self.patstat_client = None
        self.connection_pool = None
        
        # Query optimization
        self.optimized_queries = self._load_optimized_queries()
        
        # Performance tracking
        self.query_performance = {
            'total_execution_time': 0,
            'total_records_retrieved': 0,
            'queries_by_type': {},
            'average_records_per_second': 0
        }
    
    def _get_provider_type(self) -> DataProviderType:
        """Return PATSTAT provider type"""
        return DataProviderType.PATSTAT
    
    def connect(self) -> bool:
        """
        Establish connection to PATSTAT database
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Import PATSTAT client
            from epo.tipdata.patstat import PatstatClient as EPOPatstatClient
            
            # Initialize client with environment (correct parameter name)
            environment = self.patstat_config['environment']
            self.patstat_client = EPOPatstatClient(env=environment)
            
            # Test connection with simple query first
            test_query = "SELECT 1 as test LIMIT 1"
            test_result = self.patstat_client.sql_query(test_query)
            
            # Handle both DataFrame and list results
            if test_result is not None:
                # Check if it's a DataFrame or list and has data
                has_data = False
                if hasattr(test_result, 'empty'):
                    has_data = not test_result.empty
                elif isinstance(test_result, list):
                    has_data = len(test_result) > 0
                else:
                    has_data = True  # Any other result type is considered valid
                
                if has_data:
                    self.is_connected = True
                    self.connection_metadata = {
                        'environment': environment,
                        'connection_time': datetime.now().isoformat(),
                        'client_type': 'EPOPatstatClient',
                        'database_version': self._get_database_version()
                    }
                    
                    self.logger.info(f"✅ PATSTAT connection established: {environment}")
                    return True
                else:
                    self.logger.error("❌ PATSTAT connection test failed")
                    return False
            else:
                self.logger.error("❌ PATSTAT connection test failed - no result")
                return False
                
        except ImportError as e:
            self.logger.error(f"❌ PATSTAT client not available: {e}")
            return False
        except Exception as e:
            self.logger.error(f"❌ PATSTAT connection failed: {e}")
            return False
    
    def disconnect(self):
        """Close PATSTAT connection"""
        if self.patstat_client:
            try:
                # Close any open connections
                if hasattr(self.patstat_client, 'close'):
                    self.patstat_client.close()
                self.is_connected = False
                self.logger.info("📡 PATSTAT connection closed")
            except Exception as e:
                self.logger.warning(f"⚠️ Error closing PATSTAT connection: {e}")
        
        self.patstat_client = None
    
    def validate_query_params(self, query_params: Dict[str, Any]) -> bool:
        """
        Validate PATSTAT query parameters
        
        Args:
            query_params: Query parameters to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(query_params, dict):
            self.logger.error("Query parameters must be a dictionary")
            return False
        
        # Check for required query type
        if 'query_type' not in query_params:
            self.logger.error("Missing required 'query_type' parameter")
            return False
        
        query_type = query_params['query_type']
        
        # Validate specific query types
        if query_type == 'patent_search':
            return self._validate_patent_search_params(query_params)
        elif query_type == 'applicant_search':
            return self._validate_applicant_search_params(query_params)
        elif query_type == 'classification_search':
            return self._validate_classification_search_params(query_params)
        elif query_type == 'citation_analysis':
            return self._validate_citation_analysis_params(query_params)
        elif query_type == 'custom_sql':
            return self._validate_custom_sql_params(query_params)
        else:
            self.logger.error(f"Unknown query type: {query_type}")
            return False
    
    def query(self, query_params: Dict[str, Any], **kwargs) -> DataProviderResult:
        """
        Execute PATSTAT query
        
        Args:
            query_params: PATSTAT query parameters
            **kwargs: Additional query options
            
        Returns:
            DataProviderResult with PATSTAT data
        """
        start_time = time.time()
        
        # Validate connection
        if not self.is_connected:
            if not self.connect():
                return DataProviderResult(
                    data=pd.DataFrame(),
                    metadata={'provider': 'PATSTAT', 'error': 'Connection failed'},
                    status='failed',
                    errors=['Failed to connect to PATSTAT']
                )
        
        # Validate parameters
        if not self.validate_query_params(query_params):
            return DataProviderResult(
                data=pd.DataFrame(),
                metadata={'provider': 'PATSTAT', 'query_params': query_params},
                status='failed',
                errors=['Invalid query parameters']
            )
        
        # Enforce rate limiting
        self._enforce_rate_limits()
        
        try:
            # Execute query based on type
            query_type = query_params['query_type']
            
            if query_type == 'patent_search':
                result = self._execute_patent_search(query_params, **kwargs)
            elif query_type == 'applicant_search':
                result = self._execute_applicant_search(query_params, **kwargs)
            elif query_type == 'classification_search':
                result = self._execute_classification_search(query_params, **kwargs)
            elif query_type == 'citation_analysis':
                result = self._execute_citation_analysis(query_params, **kwargs)
            elif query_type == 'custom_sql':
                result = self._execute_custom_sql(query_params, **kwargs)
            else:
                raise ValueError(f"Unsupported query type: {query_type}")
            
            # Update performance metrics
            query_time = time.time() - start_time
            self._update_query_metrics(query_time, result.is_successful)
            
            # Add performance metadata
            if result.is_successful:
                self._update_performance_stats(query_time, len(result.data) if isinstance(result.data, pd.DataFrame) else 0)
            
            return result
            
        except Exception as e:
            query_time = time.time() - start_time
            error_msg = f"PATSTAT query failed: {str(e)}"
            self.logger.error(error_msg)
            
            self._update_query_metrics(query_time, False)
            
            return DataProviderResult(
                data=pd.DataFrame(),
                metadata={
                    'provider': 'PATSTAT',
                    'query_params': query_params,
                    'query_time': query_time,
                    'error': str(e)
                },
                status='failed',
                errors=[error_msg]
            )
    
    def _execute_patent_search(self, params: Dict[str, Any], **kwargs) -> DataProviderResult:
        """Execute patent search query"""
        
        # Build optimized query
        sql_query = self._build_patent_search_query(params)
        
        # Execute query
        self.logger.info(f"🔍 Executing PATSTAT patent search...")
        data = self.patstat_client.sql_query(sql_query)
        
        # Process results
        if data is not None and not data.empty:
            # Apply post-processing
            processed_data = self._post_process_patent_data(data, params)
            
            metadata = {
                'provider': 'PATSTAT',
                'query_type': 'patent_search',
                'search_params': params,
                'sql_query': sql_query[:200] + '...' if len(sql_query) > 200 else sql_query,
                'raw_records': len(data),
                'processed_records': len(processed_data)
            }
            
            return DataProviderResult(
                data=processed_data,
                metadata=metadata,
                status='success'
            )
        else:
            return DataProviderResult(
                data=pd.DataFrame(),
                metadata={'provider': 'PATSTAT', 'query_type': 'patent_search'},
                status='no_data',
                warnings=['No patents found matching search criteria']
            )
    
    def _execute_applicant_search(self, params: Dict[str, Any], **kwargs) -> DataProviderResult:
        """Execute applicant search query"""
        
        # Build applicant search query
        sql_query = self._build_applicant_search_query(params)
        
        self.logger.info(f"🏢 Executing PATSTAT applicant search...")
        data = self.patstat_client.sql_query(sql_query)
        
        if data is not None and not data.empty:
            processed_data = self._post_process_applicant_data(data, params)
            
            metadata = {
                'provider': 'PATSTAT',
                'query_type': 'applicant_search',
                'search_params': params,
                'applicants_found': len(processed_data)
            }
            
            return DataProviderResult(
                data=processed_data,
                metadata=metadata,
                status='success'
            )
        else:
            return DataProviderResult(
                data=pd.DataFrame(),
                metadata={'provider': 'PATSTAT', 'query_type': 'applicant_search'},
                status='no_data'
            )
    
    def _execute_classification_search(self, params: Dict[str, Any], **kwargs) -> DataProviderResult:
        """Execute classification search query"""
        
        sql_query = self._build_classification_search_query(params)
        
        self.logger.info(f"🏷️ Executing PATSTAT classification search...")
        data = self.patstat_client.sql_query(sql_query)
        
        if data is not None and not data.empty:
            processed_data = self._post_process_classification_data(data, params)
            
            metadata = {
                'provider': 'PATSTAT',
                'query_type': 'classification_search',
                'search_params': params,
                'technologies_found': len(processed_data)
            }
            
            return DataProviderResult(
                data=processed_data,
                metadata=metadata,
                status='success'
            )
        else:
            return DataProviderResult(
                data=pd.DataFrame(),
                metadata={'provider': 'PATSTAT', 'query_type': 'classification_search'},
                status='no_data'
            )
    
    def _execute_citation_analysis(self, params: Dict[str, Any], **kwargs) -> DataProviderResult:
        """Execute citation analysis query"""
        
        sql_query = self._build_citation_analysis_query(params)
        
        self.logger.info(f"🔗 Executing PATSTAT citation analysis...")
        data = self.patstat_client.sql_query(sql_query)
        
        if data is not None and not data.empty:
            processed_data = self._post_process_citation_data(data, params)
            
            metadata = {
                'provider': 'PATSTAT',
                'query_type': 'citation_analysis',
                'search_params': params,
                'citation_records': len(processed_data)
            }
            
            return DataProviderResult(
                data=processed_data,
                metadata=metadata,
                status='success'
            )
        else:
            return DataProviderResult(
                data=pd.DataFrame(),
                metadata={'provider': 'PATSTAT', 'query_type': 'citation_analysis'},
                status='no_data'
            )
    
    def _execute_custom_sql(self, params: Dict[str, Any], **kwargs) -> DataProviderResult:
        """Execute custom SQL query"""
        
        sql_query = params.get('sql_query', '')
        if not sql_query:
            return DataProviderResult(
                data=pd.DataFrame(),
                metadata={'provider': 'PATSTAT'},
                status='failed',
                errors=['No SQL query provided']
            )
        
        # Security check: only allow SELECT statements
        if not sql_query.strip().upper().startswith('SELECT'):
            return DataProviderResult(
                data=pd.DataFrame(),
                metadata={'provider': 'PATSTAT'},
                status='failed',
                errors=['Only SELECT statements are allowed']
            )
        
        self.logger.info(f"🔧 Executing custom PATSTAT query...")
        data = self.patstat_client.sql_query(sql_query)
        
        if data is not None:
            metadata = {
                'provider': 'PATSTAT',
                'query_type': 'custom_sql',
                'sql_query': sql_query[:100] + '...' if len(sql_query) > 100 else sql_query
            }
            
            return DataProviderResult(
                data=data,
                metadata=metadata,
                status='success' if not data.empty else 'no_data'
            )
        else:
            return DataProviderResult(
                data=pd.DataFrame(),
                metadata={'provider': 'PATSTAT', 'query_type': 'custom_sql'},
                status='failed',
                errors=['Query execution failed']
            )
    
    def _build_patent_search_query(self, params: Dict[str, Any]) -> str:
        """Build optimized patent search SQL query"""
        
        # Base query with essential tables (correct uppercase table names from main branch)
        base_query = """
        SELECT DISTINCT
            a.appln_id,
            a.docdb_family_id,
            a.appln_filing_date,
            a.appln_filing_year,
            a.appln_kind,
            a.earliest_filing_date,
            a.earliest_filing_year,
            a.appln_auth,
            t.appln_title,
            p.person_id,
            p.person_name,
            p.person_ctry_code,
            ipc.ipc_class_symbol,
            cpc.cpc_class_symbol
        FROM TLS201_APPLN a
        LEFT JOIN TLS202_APPLN_TITLE t ON a.appln_id = t.appln_id AND t.appln_title_lg = 'en'
        LEFT JOIN TLS207_PERS_APPLN pa ON a.appln_id = pa.appln_id AND pa.applt_seq_nr > 0
        LEFT JOIN TLS206_PERSON p ON pa.person_id = p.person_id
        LEFT JOIN TLS209_APPLN_IPC ipc ON a.appln_id = ipc.appln_id AND ipc.ipc_position = 'F'
        LEFT JOIN TLS224_APPLN_CPC cpc ON a.appln_id = cpc.appln_id AND cpc.cpc_position = 'F'
        WHERE 1=1
        """
        
        conditions = []
        
        # Filing year filter
        if 'filing_years' in params:
            years = params['filing_years']
            if isinstance(years, list) and years:
                year_conditions = [str(year) for year in years if isinstance(year, int)]
                if year_conditions:
                    conditions.append(f"a.appln_filing_year IN ({','.join(year_conditions)})")
        
        # Technology/IPC filter
        if 'technology_area' in params or 'ipc_classes' in params:
            ipc_conditions = []
            if 'ipc_classes' in params:
                ipc_classes = params['ipc_classes']
                if isinstance(ipc_classes, list):
                    ipc_list = "','".join(ipc_classes)
                    ipc_conditions.append(f"ipc.ipc_class_symbol IN ('{ipc_list}')")
            
            if 'technology_area' in params:
                # Map technology areas to IPC classes
                tech_to_ipc = {
                    'energy storage': ['H01M', 'H02J', 'H01G'],
                    'computing': ['G06F', 'G06N', 'H04L'],
                    'chemistry': ['C07D', 'C08F', 'A61K'],
                    'telecommunications': ['H04L', 'H04W', 'H04B']
                }
                
                tech_area = params['technology_area'].lower()
                if tech_area in tech_to_ipc:
                    ipc_list = "','".join(tech_to_ipc[tech_area])
                    ipc_conditions.append(f"ipc.ipc_class_symbol IN ('{ipc_list}')")
            
            if ipc_conditions:
                conditions.append(f"({' OR '.join(ipc_conditions)})")
        
        # Country filter
        if 'countries' in params:
            countries = params['countries']
            if isinstance(countries, list) and countries:
                country_list = "','".join(countries)
                conditions.append(f"(a.appln_auth IN ('{country_list}') OR p.person_ctry_code IN ('{country_list}'))")
        
        # Applicant filter
        if 'applicant_names' in params:
            applicants = params['applicant_names']
            if isinstance(applicants, list) and applicants:
                applicant_conditions = [f"p.person_name LIKE '%{name}%'" for name in applicants]
                conditions.append(f"({' OR '.join(applicant_conditions)})")
        
        # Date range filter
        if 'date_from' in params:
            conditions.append(f"a.appln_filing_date >= '{params['date_from']}'")
        
        if 'date_to' in params:
            conditions.append(f"a.appln_filing_date <= '{params['date_to']}'")
        
        # Add conditions to query
        if conditions:
            base_query += " AND " + " AND ".join(conditions)
        
        # Add ordering and limit
        base_query += " ORDER BY a.appln_filing_date DESC"
        
        limit = min(params.get('limit', 10000), self.patstat_config['max_results'])
        base_query += f" LIMIT {limit}"
        
        return base_query
    
    def _build_applicant_search_query(self, params: Dict[str, Any]) -> str:
        """Build applicant search SQL query"""
        
        query = """
        SELECT 
            p.person_id,
            p.person_name,
            p.person_ctry_code,
            COUNT(DISTINCT a.docdb_family_id) as patent_families,
            COUNT(DISTINCT a.appln_id) as applications,
            MIN(a.appln_filing_year) as earliest_filing_year,
            MAX(a.appln_filing_year) as latest_filing_year
        FROM TLS206_PERSON p
        JOIN TLS207_PERS_APPLN pa ON p.person_id = pa.person_id AND pa.applt_seq_nr > 0
        JOIN TLS201_APPLN a ON pa.appln_id = a.appln_id
        WHERE 1=1
        """
        
        conditions = []
        
        if 'applicant_name' in params:
            conditions.append(f"p.person_name LIKE '%{params['applicant_name']}%'")
        
        if 'country_codes' in params:
            countries = params['country_codes']
            if isinstance(countries, list):
                country_list = "','".join(countries)
                conditions.append(f"p.person_ctry_code IN ('{country_list}')")
        
        if 'min_patents' in params:
            # This will be applied in HAVING clause
            pass
        
        if conditions:
            query += " AND " + " AND ".join(conditions)
        
        query += " GROUP BY p.person_id, p.person_name, p.person_ctry_code"
        
        if 'min_patents' in params:
            query += f" HAVING COUNT(DISTINCT a.docdb_family_id) >= {params['min_patents']}"
        
        query += " ORDER BY patent_families DESC"
        
        limit = params.get('limit', 1000)
        query += f" LIMIT {limit}"
        
        return query
    
    def _build_classification_search_query(self, params: Dict[str, Any]) -> str:
        """Build classification search SQL query"""
        
        query = """
        SELECT 
            ipc.ipc_class_symbol,
            COUNT(DISTINCT a.docdb_family_id) as patent_families,
            COUNT(DISTINCT a.appln_id) as applications,
            MIN(a.appln_filing_year) as earliest_year,
            MAX(a.appln_filing_year) as latest_year
        FROM TLS209_APPLN_IPC ipc
        JOIN TLS201_APPLN a ON ipc.appln_id = a.appln_id
        WHERE ipc.ipc_position = 'F'
        """
        
        conditions = []
        
        if 'ipc_classes' in params:
            ipc_list = "','".join(params['ipc_classes'])
            conditions.append(f"ipc.ipc_class_symbol IN ('{ipc_list}')")
        
        if 'filing_years' in params:
            years = params['filing_years']
            if isinstance(years, list):
                year_conditions = [str(year) for year in years]
                conditions.append(f"a.appln_filing_year IN ({','.join(year_conditions)})")
        
        if conditions:
            query += " AND " + " AND ".join(conditions)
        
        query += " GROUP BY ipc.ipc_class_symbol ORDER BY patent_families DESC"
        
        limit = params.get('limit', 500)
        query += f" LIMIT {limit}"
        
        return query
    
    def _build_citation_analysis_query(self, params: Dict[str, Any]) -> str:
        """Build citation analysis SQL query"""
        
        query = """
        SELECT 
            a.appln_id,
            a.docdb_family_id,
            a.appln_filing_year,
            COUNT(cit.cited_appln_id) as forward_citations,
            COUNT(citb.citing_appln_id) as backward_citations
        FROM TLS201_APPLN a
        LEFT JOIN TLS212_CITATION cit ON a.appln_id = cit.citing_appln_id
        LEFT JOIN TLS212_CITATION citb ON a.appln_id = citb.cited_appln_id
        WHERE 1=1
        """
        
        conditions = []
        
        if 'family_ids' in params:
            family_ids = params['family_ids']
            if isinstance(family_ids, list):
                id_list = ','.join([str(fid) for fid in family_ids])
                conditions.append(f"a.docdb_family_id IN ({id_list})")
        
        if 'min_citations' in params:
            # Applied in HAVING clause
            pass
        
        if conditions:
            query += " AND " + " AND ".join(conditions)
        
        query += " GROUP BY a.appln_id, a.docdb_family_id, a.appln_filing_year"
        
        if 'min_citations' in params:
            query += f" HAVING COUNT(cit.cited_appln_id) >= {params['min_citations']}"
        
        query += " ORDER BY forward_citations DESC"
        
        limit = params.get('limit', 1000)
        query += f" LIMIT {limit}"
        
        return query
    
    def _post_process_patent_data(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Post-process patent search results"""
        
        # Clean and standardize data
        processed_data = data.copy()
        
        # Handle missing values
        processed_data = processed_data.fillna({
            'person_name': 'Unknown',
            'person_ctry_code': 'XX',
            'ipc_class_symbol': 'Unknown',
            'appln_title': 'No Title'
        })
        
        # Standardize country codes
        processed_data['person_ctry_code'] = processed_data['person_ctry_code'].str.upper()
        
        # Add derived fields
        processed_data['filing_decade'] = (processed_data['appln_filing_year'] // 10) * 10
        processed_data['is_recent'] = processed_data['appln_filing_year'] >= (datetime.now().year - 5)
        
        # Remove duplicates
        processed_data = processed_data.drop_duplicates(subset=['appln_id'])
        
        return processed_data
    
    def _post_process_applicant_data(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Post-process applicant search results"""
        
        processed_data = data.copy()
        
        # Calculate additional metrics
        if 'patent_families' in processed_data.columns and 'applications' in processed_data.columns:
            processed_data['family_to_app_ratio'] = (
                processed_data['patent_families'] / processed_data['applications']
            ).round(3)
        
        if 'earliest_filing_year' in processed_data.columns and 'latest_filing_year' in processed_data.columns:
            processed_data['filing_span_years'] = (
                processed_data['latest_filing_year'] - processed_data['earliest_filing_year']
            )
        
        return processed_data
    
    def _post_process_classification_data(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Post-process classification search results"""
        
        processed_data = data.copy()
        
        # Add IPC section information
        if 'ipc_class_symbol' in processed_data.columns:
            processed_data['ipc_section'] = processed_data['ipc_class_symbol'].str[0]
            processed_data['ipc_class'] = processed_data['ipc_class_symbol'].str[:3]
        
        return processed_data
    
    def _post_process_citation_data(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Post-process citation analysis results"""
        
        processed_data = data.copy()
        
        # Calculate citation metrics
        if 'forward_citations' in processed_data.columns and 'backward_citations' in processed_data.columns:
            processed_data['total_citations'] = (
                processed_data['forward_citations'] + processed_data['backward_citations']
            )
            processed_data['citation_balance'] = (
                processed_data['forward_citations'] - processed_data['backward_citations']
            )
        
        return processed_data
    
    def _validate_patent_search_params(self, params: Dict[str, Any]) -> bool:
        """Validate patent search parameters"""
        
        # Check for at least one search criterion
        search_criteria = [
            'filing_years', 'technology_area', 'ipc_classes', 
            'countries', 'applicant_names', 'date_from', 'date_to'
        ]
        
        if not any(param in params for param in search_criteria):
            self.logger.error("At least one search criterion is required")
            return False
        
        # Validate year ranges
        if 'filing_years' in params:
            years = params['filing_years']
            if not isinstance(years, list) or not all(isinstance(y, int) for y in years):
                self.logger.error("filing_years must be a list of integers")
                return False
            
            current_year = datetime.now().year
            if any(year < 1980 or year > current_year for year in years):
                self.logger.error(f"Filing years must be between 1980 and {current_year}")
                return False
        
        return True
    
    def _validate_applicant_search_params(self, params: Dict[str, Any]) -> bool:
        """Validate applicant search parameters"""
        
        if 'applicant_name' not in params and 'country_codes' not in params:
            self.logger.error("Either applicant_name or country_codes is required")
            return False
        
        return True
    
    def _validate_classification_search_params(self, params: Dict[str, Any]) -> bool:
        """Validate classification search parameters"""
        
        if 'ipc_classes' not in params and 'filing_years' not in params:
            self.logger.error("Either ipc_classes or filing_years is required")
            return False
        
        return True
    
    def _validate_citation_analysis_params(self, params: Dict[str, Any]) -> bool:
        """Validate citation analysis parameters"""
        
        if 'family_ids' not in params:
            self.logger.error("family_ids parameter is required for citation analysis")
            return False
        
        return True
    
    def _validate_custom_sql_params(self, params: Dict[str, Any]) -> bool:
        """Validate custom SQL parameters"""
        
        if 'sql_query' not in params:
            self.logger.error("sql_query parameter is required")
            return False
        
        return True
    
    def _load_optimized_queries(self) -> Dict[str, str]:
        """Load pre-optimized queries for common operations"""
        
        return {
            'top_applicants_by_year': """
                SELECT 
                    p.person_name,
                    p.person_ctry_code,
                    a.appln_filing_year,
                    COUNT(DISTINCT a.docdb_family_id) as families
                FROM TLS206_PERSON p
                JOIN TLS207_PERS_APPLN pa ON p.person_id = pa.person_id
                JOIN TLS201_APPLN a ON pa.appln_id = a.appln_id
                WHERE a.appln_filing_year >= {year}
                GROUP BY p.person_name, p.person_ctry_code, a.appln_filing_year
                HAVING families >= {min_families}
                ORDER BY families DESC
                LIMIT {limit}
            """,
            
            'technology_trends': """
                SELECT 
                    ipc.ipc_class_symbol,
                    a.appln_filing_year,
                    COUNT(DISTINCT a.docdb_family_id) as families
                FROM TLS209_APPLN_IPC ipc
                JOIN TLS201_APPLN a ON ipc.appln_id = a.appln_id
                WHERE ipc.ipc_position = 'F'
                    AND a.appln_filing_year >= {start_year}
                    AND a.appln_filing_year <= {end_year}
                GROUP BY ipc.ipc_class_symbol, a.appln_filing_year
                ORDER BY a.appln_filing_year, families DESC
            """
        }
    
    def _get_database_version(self) -> str:
        """Get PATSTAT database version"""
        try:
            version_query = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_name LIKE 'TLS%' LIMIT 1"
            result = self.patstat_client.sql_query(version_query)
            if result is not None and not result.empty:
                return "PATSTAT 2023 Autumn"  # Could be determined more precisely
            else:
                return "Unknown"
        except:
            return "Unknown"
    
    def _run_connection_test(self) -> Dict[str, Any]:
        """Run PATSTAT-specific connection test"""
        try:
            # Test basic table access (using correct uppercase table names)
            test_queries = [
                ("Application table", "SELECT COUNT(*) as count FROM TLS201_APPLN LIMIT 1"),
                ("Person table", "SELECT COUNT(*) as count FROM TLS206_PERSON LIMIT 1"),
                ("IPC table", "SELECT COUNT(*) as count FROM TLS209_APPLN_IPC LIMIT 1")
            ]
            
            test_results = {}
            for test_name, query in test_queries:
                try:
                    result = self.patstat_client.sql_query(query)
                    test_results[test_name] = "✅ Accessible" if result is not None else "❌ Failed"
                except Exception as e:
                    test_results[test_name] = f"❌ Error: {str(e)[:50]}"
            
            return {
                'database_tests': test_results,
                'connection_quality': 'Good' if all('✅' in status for status in test_results.values()) else 'Limited'
            }
            
        except Exception as e:
            return {'connection_test': f"Failed: {str(e)}"}
    
    def _update_performance_stats(self, query_time: float, record_count: int):
        """Update PATSTAT-specific performance statistics"""
        
        self.query_performance['total_execution_time'] += query_time
        self.query_performance['total_records_retrieved'] += record_count
        
        if self.query_performance['total_execution_time'] > 0:
            self.query_performance['average_records_per_second'] = (
                self.query_performance['total_records_retrieved'] / 
                self.query_performance['total_execution_time']
            )
    
    def get_patstat_summary(self) -> Dict[str, Any]:
        """Get PATSTAT-specific provider summary"""
        
        base_summary = self.get_provider_summary()
        
        # Add PATSTAT-specific metrics
        patstat_metrics = {
            'database_environment': self.patstat_config['environment'],
            'total_records_retrieved': self.query_performance['total_records_retrieved'],
            'average_records_per_second': f"{self.query_performance['average_records_per_second']:.0f}",
            'performance_stats': self.query_performance,
            'optimized_queries_available': len(self.optimized_queries)
        }
        
        return {**base_summary, **patstat_metrics}