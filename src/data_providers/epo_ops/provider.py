"""
EPO OPS Data Provider - Real Implementation
Production-ready EPO Open Patent Services API access
"""

import pandas as pd
import requests
import json
import time
import os
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import base64
import logging

from ..base import DataProvider, DataProviderResult, DataProviderType, register_provider

# Setup logging
logger = logging.getLogger(__name__)


@register_provider(DataProviderType.EPO_OPS)
class EPOOpsDataProvider(DataProvider):
    """
    EPO Open Patent Services (OPS) API provider
    Provides access to patent data, full-text documents, and images
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize EPO OPS provider
        
        Args:
            config: OPS configuration including credentials and endpoints
        """
        super().__init__(config)
        
        # Default configuration with environment variable support
        self.default_config = {
            'base_url': 'https://ops.epo.org/3.2',
            'auth_url': 'https://ops.epo.org/3.2/auth/accesstoken',
            'consumer_key': os.getenv('OPS_KEY'),
            'consumer_secret': os.getenv('OPS_SECRET'),
            'rate_limit_per_minute': int(os.getenv('OPS_RATE_LIMIT_PER_MINUTE', '30')),
            'rate_limit_per_week': int(os.getenv('OPS_RATE_LIMIT_PER_WEEK', '4000')),
            'timeout': 30,
            'max_retries': 3,
            'retry_delay': 1,
            'cache_tokens': True
        }
        
        # Merge configurations
        self.ops_config = {**self.default_config, **self.config}
        
        # Authentication
        self.access_token = None
        self.token_expires = None
        self.auth_header = None
        
        # Rate limiting
        self.request_times = []
        self.weekly_requests = 0
        self.weekly_reset_time = datetime.now() + timedelta(days=7)
        
        # Available services
        self.available_services = [
            'published-data/search',
            'published-data/publication',
            'published-data/images',
            'family/publication',
            'legal',
            'register'
        ]
    
    def _get_provider_type(self) -> DataProviderType:
        """Return EPO OPS provider type"""
        return DataProviderType.EPO_OPS
    
    def connect(self) -> bool:
        """
        Establish connection to EPO OPS API
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Check credentials
            if not self.ops_config.get('consumer_key') or not self.ops_config.get('consumer_secret'):
                self.logger.error("❌ EPO OPS credentials not configured")
                return False
            
            # Authenticate
            if self._authenticate():
                self.is_connected = True
                self.connection_metadata = {
                    'base_url': self.ops_config['base_url'],
                    'connection_time': datetime.now().isoformat(),
                    'rate_limits': {
                        'per_minute': self.ops_config['rate_limit_per_minute'],
                        'per_week': self.ops_config['rate_limit_per_week']
                    },
                    'available_services': self.available_services
                }
                
                self.logger.info("✅ EPO OPS connection established")
                return True
            else:
                self.logger.error("❌ EPO OPS authentication failed")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ EPO OPS connection failed: {e}")
            return False
    
    def disconnect(self):
        """Close EPO OPS connection"""
        self.access_token = None
        self.token_expires = None
        self.auth_header = None
        self.is_connected = False
        self.logger.info("📡 EPO OPS connection closed")
    
    def validate_query_params(self, query_params: Dict[str, Any]) -> bool:
        """
        Validate EPO OPS query parameters
        
        Args:
            query_params: Query parameters to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(query_params, dict):
            self.logger.error("Query parameters must be a dictionary")
            return False
        
        # Check for required service type
        if 'service' not in query_params:
            self.logger.error("Missing required 'service' parameter")
            return False
        
        service = query_params['service']
        
        # Validate specific services
        if service == 'search':
            return self._validate_search_params(query_params)
        elif service == 'publication':
            return self._validate_publication_params(query_params)
        elif service == 'family':
            return self._validate_family_params(query_params)
        elif service == 'images':
            return self._validate_images_params(query_params)
        elif service == 'legal':
            return self._validate_legal_params(query_params)
        else:
            self.logger.error(f"Unknown service: {service}")
            return False
    
    def query(self, query_params: Dict[str, Any], **kwargs) -> DataProviderResult:
        """
        Execute EPO OPS query
        
        Args:
            query_params: OPS query parameters
            **kwargs: Additional query options
            
        Returns:
            DataProviderResult with OPS data
        """
        start_time = time.time()
        
        # Validate connection
        if not self.is_connected:
            if not self.connect():
                return DataProviderResult(
                    data=pd.DataFrame(),
                    metadata={'provider': 'EPO_OPS', 'error': 'Connection failed'},
                    status='failed',
                    errors=['Failed to connect to EPO OPS']
                )
        
        # Validate parameters
        if not self.validate_query_params(query_params):
            return DataProviderResult(
                data=pd.DataFrame(),
                metadata={'provider': 'EPO_OPS', 'query_params': query_params},
                status='failed',
                errors=['Invalid query parameters']
            )
        
        # Check rate limits
        if not self._check_rate_limits():
            return DataProviderResult(
                data=pd.DataFrame(),
                metadata={'provider': 'EPO_OPS'},
                status='failed',
                errors=['Rate limit exceeded']
            )
        
        try:
            # Execute query based on service
            service = query_params['service']
            
            if service == 'search':
                result = self._execute_search(query_params, **kwargs)
            elif service == 'publication':
                result = self._execute_publication_query(query_params, **kwargs)
            elif service == 'family':
                result = self._execute_family_query(query_params, **kwargs)
            elif service == 'images':
                result = self._execute_images_query(query_params, **kwargs)
            elif service == 'legal':
                result = self._execute_legal_query(query_params, **kwargs)
            else:
                raise ValueError(f"Unsupported service: {service}")
            
            # Update performance metrics
            query_time = time.time() - start_time
            self._update_query_metrics(query_time, result.is_successful)
            
            return result
            
        except Exception as e:
            query_time = time.time() - start_time
            error_msg = f"EPO OPS query failed: {str(e)}"
            self.logger.error(error_msg)
            
            self._update_query_metrics(query_time, False)
            
            return DataProviderResult(
                data=pd.DataFrame(),
                metadata={
                    'provider': 'EPO_OPS',
                    'query_params': query_params,
                    'query_time': query_time,
                    'error': str(e)
                },
                status='failed',
                errors=[error_msg]
            )
    
    def _authenticate(self) -> bool:
        """Authenticate with EPO OPS API"""
        try:
            # Check if token is still valid
            if self.access_token and self.token_expires:
                if datetime.now() < self.token_expires:
                    return True
            
            # Prepare authentication request
            consumer_key = self.ops_config['consumer_key']
            consumer_secret = self.ops_config['consumer_secret']
            
            # Create basic auth header
            credentials = f"{consumer_key}:{consumer_secret}"
            encoded_credentials = base64.b64encode(credentials.encode()).decode()
            
            headers = {
                'Authorization': f'Basic {encoded_credentials}',
                'Content-Type': 'application/x-www-form-urlencoded'
            }
            
            data = {'grant_type': 'client_credentials'}
            
            # Make authentication request
            response = requests.post(
                self.ops_config['auth_url'],
                headers=headers,
                data=data,
                timeout=self.ops_config['timeout']
            )
            
            if response.status_code == 200:
                token_data = response.json()
                self.access_token = token_data['access_token']
                
                # Calculate token expiration (subtract buffer time)
                expires_in = int(token_data.get('expires_in', 3600))
                self.token_expires = datetime.now() + timedelta(seconds=expires_in - 60)
                
                # Prepare auth header for API requests
                self.auth_header = f"Bearer {self.access_token}"
                
                self.logger.info("✅ EPO OPS authentication successful")
                return True
            else:
                self.logger.error(f"❌ EPO OPS authentication failed: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ EPO OPS authentication error: {e}")
            return False
    
    def _check_rate_limits(self) -> bool:
        """Check if request can be made within rate limits"""
        current_time = datetime.now()
        
        # Clean old request times (older than 1 minute)
        minute_ago = current_time - timedelta(minutes=1)
        self.request_times = [t for t in self.request_times if t > minute_ago]
        
        # Check per-minute limit
        if len(self.request_times) >= self.ops_config['rate_limit_per_minute']:
            self.logger.warning("⚠️ EPO OPS per-minute rate limit reached")
            return False
        
        # Check weekly limit
        if current_time > self.weekly_reset_time:
            self.weekly_requests = 0
            self.weekly_reset_time = current_time + timedelta(days=7)
        
        if self.weekly_requests >= self.ops_config['rate_limit_per_week']:
            self.logger.warning("⚠️ EPO OPS weekly rate limit reached")
            return False
        
        return True
    
    def _make_api_request(self, endpoint: str, params: Dict[str, Any] = None) -> requests.Response:
        """Make authenticated API request to EPO OPS"""
        
        # Ensure authentication
        if not self.access_token or datetime.now() >= self.token_expires:
            if not self._authenticate():
                raise Exception("Authentication failed")
        
        # Prepare request
        url = f"{self.ops_config['base_url']}/{endpoint}"
        headers = {
            'Authorization': self.auth_header,
            'Accept': 'application/json'
        }
        
        # Record request time
        self.request_times.append(datetime.now())
        self.weekly_requests += 1
        
        # Make request with retries
        for attempt in range(self.ops_config['max_retries']):
            try:
                response = requests.get(
                    url,
                    headers=headers,
                    params=params,
                    timeout=self.ops_config['timeout']
                )
                
                if response.status_code == 200:
                    return response
                elif response.status_code == 429:  # Rate limited
                    self.logger.warning(f"Rate limited, attempt {attempt + 1}")
                    time.sleep(self.ops_config['retry_delay'] * (attempt + 1))
                    continue
                elif response.status_code == 401:  # Authentication expired
                    self.logger.info("Token expired, re-authenticating...")
                    if self._authenticate():
                        headers['Authorization'] = self.auth_header
                        continue
                    else:
                        raise Exception("Re-authentication failed")
                else:
                    response.raise_for_status()
                    
            except requests.exceptions.RequestException as e:
                if attempt == self.ops_config['max_retries'] - 1:
                    raise e
                time.sleep(self.ops_config['retry_delay'] * (attempt + 1))
        
        raise Exception("Max retries exceeded")
    
    def _execute_search(self, params: Dict[str, Any], **kwargs) -> DataProviderResult:
        """Execute patent search via OPS"""
        
        # Build search query
        query_string = self._build_search_query(params)
        
        # Prepare search parameters
        search_params = {
            'q': query_string,
            'Range': f"1-{params.get('limit', 100)}"
        }
        
        self.logger.info(f"🔍 Executing EPO OPS search: {query_string[:100]}...")
        
        try:
            response = self._make_api_request('rest-services/published-data/search', search_params)
            data = response.json()
            
            # Process search results
            if 'ops:world-patent-data' in data:
                search_results = self._process_search_results(data, params)
                
                metadata = {
                    'provider': 'EPO_OPS',
                    'service': 'search',
                    'query_string': query_string,
                    'search_params': params,
                    'total_results': len(search_results)
                }
                
                return DataProviderResult(
                    data=search_results,
                    metadata=metadata,
                    status='success'
                )
            else:
                return DataProviderResult(
                    data=pd.DataFrame(),
                    metadata={'provider': 'EPO_OPS', 'service': 'search'},
                    status='no_data',
                    warnings=['No search results found']
                )
                
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            raise
    
    def _execute_publication_query(self, params: Dict[str, Any], **kwargs) -> DataProviderResult:
        """Execute publication data query"""
        
        publication_id = params.get('publication_id')
        if not publication_id:
            raise ValueError("publication_id required for publication query")
        
        # Build endpoint
        endpoint = f"rest-services/published-data/publication/epodoc/{publication_id}/biblio"
        
        self.logger.info(f"📄 Retrieving publication data for: {publication_id}")
        
        try:
            response = self._make_api_request(endpoint)
            data = response.json()
            
            # Process publication data
            publication_data = self._process_publication_data(data, params)
            
            metadata = {
                'provider': 'EPO_OPS',
                'service': 'publication',
                'publication_id': publication_id
            }
            
            return DataProviderResult(
                data=publication_data,
                metadata=metadata,
                status='success'
            )
            
        except Exception as e:
            self.logger.error(f"Publication query failed: {e}")
            raise
    
    def _execute_family_query(self, params: Dict[str, Any], **kwargs) -> DataProviderResult:
        """Execute patent family query"""
        
        family_id = params.get('family_id')
        if not family_id:
            raise ValueError("family_id required for family query")
        
        endpoint = f"rest-services/family/publication/epodoc/{family_id}"
        
        self.logger.info(f"👨‍👩‍👧‍👦 Retrieving family data for: {family_id}")
        
        try:
            response = self._make_api_request(endpoint)
            data = response.json()
            
            # Process family data
            family_data = self._process_family_data(data, params)
            
            metadata = {
                'provider': 'EPO_OPS',
                'service': 'family',
                'family_id': family_id,
                'family_size': len(family_data)
            }
            
            return DataProviderResult(
                data=family_data,
                metadata=metadata,
                status='success'
            )
            
        except Exception as e:
            self.logger.error(f"Family query failed: {e}")
            raise
    
    def _execute_images_query(self, params: Dict[str, Any], **kwargs) -> DataProviderResult:
        """Execute images query"""
        
        publication_id = params.get('publication_id')
        if not publication_id:
            raise ValueError("publication_id required for images query")
        
        endpoint = f"rest-services/published-data/images/{publication_id}"
        
        self.logger.info(f"🖼️ Retrieving images for: {publication_id}")
        
        try:
            response = self._make_api_request(endpoint)
            
            # For images, we might get different content types
            if response.headers.get('content-type', '').startswith('application/json'):
                data = response.json()
                images_data = self._process_images_data(data, params)
            else:
                # Binary image data
                images_data = pd.DataFrame([{
                    'publication_id': publication_id,
                    'content_type': response.headers.get('content-type'),
                    'content_length': len(response.content),
                    'image_data': response.content
                }])
            
            metadata = {
                'provider': 'EPO_OPS',
                'service': 'images',
                'publication_id': publication_id
            }
            
            return DataProviderResult(
                data=images_data,
                metadata=metadata,
                status='success'
            )
            
        except Exception as e:
            self.logger.error(f"Images query failed: {e}")
            raise
    
    def _execute_legal_query(self, params: Dict[str, Any], **kwargs) -> DataProviderResult:
        """Execute legal status query"""
        
        publication_id = params.get('publication_id')
        if not publication_id:
            raise ValueError("publication_id required for legal query")
        
        endpoint = f"rest-services/legal/{publication_id}"
        
        self.logger.info(f"⚖️ Retrieving legal status for: {publication_id}")
        
        try:
            response = self._make_api_request(endpoint)
            data = response.json()
            
            # Process legal data
            legal_data = self._process_legal_data(data, params)
            
            metadata = {
                'provider': 'EPO_OPS',
                'service': 'legal',
                'publication_id': publication_id
            }
            
            return DataProviderResult(
                data=legal_data,
                metadata=metadata,
                status='success'
            )
            
        except Exception as e:
            self.logger.error(f"Legal query failed: {e}")
            raise
    
    def _build_search_query(self, params: Dict[str, Any]) -> str:
        """Build OPS search query string"""
        
        query_parts = []
        
        # Text search
        if 'text' in params:
            query_parts.append(f'txt="{params["text"]}"')
        
        # Title search
        if 'title' in params:
            query_parts.append(f'ti="{params["title"]}"')
        
        # Abstract search
        if 'abstract' in params:
            query_parts.append(f'ab="{params["abstract"]}"')
        
        # Applicant search
        if 'applicant' in params:
            query_parts.append(f'pa="{params["applicant"]}"')
        
        # Inventor search
        if 'inventor' in params:
            query_parts.append(f'in="{params["inventor"]}"')
        
        # Classification search
        if 'ipc_class' in params:
            query_parts.append(f'cl="{params["ipc_class"]}"')
        
        if 'cpc_class' in params:
            query_parts.append(f'cl="{params["cpc_class"]}"')
        
        # Publication date range
        if 'date_from' in params or 'date_to' in params:
            date_from = params.get('date_from', '1900-01-01')
            date_to = params.get('date_to', '2100-12-31')
            query_parts.append(f'pd="{date_from}:{date_to}"')
        
        # Priority date range
        if 'priority_date_from' in params or 'priority_date_to' in params:
            date_from = params.get('priority_date_from', '1900-01-01')
            date_to = params.get('priority_date_to', '2100-12-31')
            query_parts.append(f'pr="{date_from}:{date_to}"')
        
        # Publication number
        if 'publication_number' in params:
            query_parts.append(f'num="{params["publication_number"]}"')
        
        # Combine query parts
        if not query_parts:
            raise ValueError("At least one search criterion is required")
        
        return ' AND '.join(query_parts)
    
    def _process_search_results(self, data: Dict[str, Any], params: Dict[str, Any]) -> pd.DataFrame:
        """Process OPS search results into DataFrame"""
        
        results = []
        
        try:
            # Navigate the OPS response structure
            world_data = data.get('ops:world-patent-data', {})
            search_result = world_data.get('ops:biblio-search', {})
            query_result = search_result.get('ops:query', {})
            publications = query_result.get('ops:publication-reference', [])
            
            if not isinstance(publications, list):
                publications = [publications]
            
            for pub in publications:
                result_item = {}
                
                # Extract document ID
                doc_id = pub.get('document-id', {})
                if isinstance(doc_id, list):
                    doc_id = doc_id[0]
                
                result_item['country'] = doc_id.get('country', {}).get('$', '')
                result_item['doc_number'] = doc_id.get('doc-number', {}).get('$', '')
                result_item['kind'] = doc_id.get('kind', {}).get('$', '')
                result_item['date'] = doc_id.get('date', {}).get('$', '')
                
                # Create publication ID
                result_item['publication_id'] = f"{result_item['country']}{result_item['doc_number']}{result_item['kind']}"
                
                results.append(result_item)
            
            return pd.DataFrame(results)
            
        except Exception as e:
            self.logger.warning(f"Error processing search results: {e}")
            return pd.DataFrame()
    
    def _process_publication_data(self, data: Dict[str, Any], params: Dict[str, Any]) -> pd.DataFrame:
        """Process publication data into DataFrame"""
        
        try:
            # Extract bibliographic data
            world_data = data.get('ops:world-patent-data', {})
            pub_data = world_data.get('ops:patent-document', {})
            
            result = {
                'publication_id': params.get('publication_id'),
                'title': '',
                'abstract': '',
                'applicants': [],
                'inventors': [],
                'ipc_classes': [],
                'cpc_classes': [],
                'filing_date': '',
                'publication_date': ''
            }
            
            # Extract title
            biblio_data = pub_data.get('ops:bibliographic-data', {})
            titles = biblio_data.get('invention-title', [])
            if titles and isinstance(titles, list):
                for title in titles:
                    if title.get('@lang') == 'en':
                        result['title'] = title.get('$', '')
                        break
            elif isinstance(titles, dict):
                result['title'] = titles.get('$', '')
            
            # Extract abstract
            abstract_data = pub_data.get('ops:abstract', {})
            if abstract_data:
                abs_text = abstract_data.get('p', {})
                if isinstance(abs_text, dict):
                    result['abstract'] = abs_text.get('$', '')
            
            # Extract applicants and inventors
            parties = biblio_data.get('parties', {})
            if parties:
                applicants = parties.get('applicants', {}).get('applicant', [])
                if not isinstance(applicants, list):
                    applicants = [applicants]
                result['applicants'] = [app.get('applicant-name', {}).get('name', {}).get('$', '') for app in applicants]
                
                inventors = parties.get('inventors', {}).get('inventor', [])
                if not isinstance(inventors, list):
                    inventors = [inventors]
                result['inventors'] = [inv.get('inventor-name', {}).get('name', {}).get('$', '') for inv in inventors]
            
            # Extract classifications
            classifications = biblio_data.get('classifications-ipcr', {}).get('classification-ipcr', [])
            if not isinstance(classifications, list):
                classifications = [classifications]
            result['ipc_classes'] = [cls.get('text', {}).get('$', '') for cls in classifications if cls.get('text')]
            
            return pd.DataFrame([result])
            
        except Exception as e:
            self.logger.warning(f"Error processing publication data: {e}")
            return pd.DataFrame()
    
    def _process_family_data(self, data: Dict[str, Any], params: Dict[str, Any]) -> pd.DataFrame:
        """Process family data into DataFrame"""
        
        try:
            results = []
            
            world_data = data.get('ops:world-patent-data', {})
            family_data = world_data.get('ops:patent-family', {})
            members = family_data.get('ops:family-member', [])
            
            if not isinstance(members, list):
                members = [members]
            
            for member in members:
                result_item = {
                    'family_id': params.get('family_id'),
                    'country': '',
                    'doc_number': '',
                    'kind': '',
                    'date': '',
                    'publication_id': ''
                }
                
                # Extract publication reference
                pub_ref = member.get('publication-reference', {})
                doc_id = pub_ref.get('document-id', {})
                
                if isinstance(doc_id, list):
                    doc_id = doc_id[0]
                
                result_item['country'] = doc_id.get('country', {}).get('$', '')
                result_item['doc_number'] = doc_id.get('doc-number', {}).get('$', '')
                result_item['kind'] = doc_id.get('kind', {}).get('$', '')
                result_item['date'] = doc_id.get('date', {}).get('$', '')
                result_item['publication_id'] = f"{result_item['country']}{result_item['doc_number']}{result_item['kind']}"
                
                results.append(result_item)
            
            return pd.DataFrame(results)
            
        except Exception as e:
            self.logger.warning(f"Error processing family data: {e}")
            return pd.DataFrame()
    
    def _process_images_data(self, data: Dict[str, Any], params: Dict[str, Any]) -> pd.DataFrame:
        """Process images data into DataFrame"""
        
        try:
            # Process JSON response for images metadata
            result = {
                'publication_id': params.get('publication_id'),
                'images_available': True,
                'image_types': [],
                'image_count': 0
            }
            
            # Extract image information if available
            world_data = data.get('ops:world-patent-data', {})
            images_data = world_data.get('ops:document-images', {})
            
            if images_data:
                result['image_count'] = len(images_data)
                result['image_types'] = list(images_data.keys())
            
            return pd.DataFrame([result])
            
        except Exception as e:
            self.logger.warning(f"Error processing images data: {e}")
            return pd.DataFrame()
    
    def _process_legal_data(self, data: Dict[str, Any], params: Dict[str, Any]) -> pd.DataFrame:
        """Process legal status data into DataFrame"""
        
        try:
            results = []
            
            world_data = data.get('ops:world-patent-data', {})
            legal_data = world_data.get('ops:legal-data', {})
            events = legal_data.get('ops:legal-event', [])
            
            if not isinstance(events, list):
                events = [events]
            
            for event in events:
                result_item = {
                    'publication_id': params.get('publication_id'),
                    'event_code': event.get('code', {}).get('$', ''),
                    'event_description': event.get('description', {}).get('$', ''),
                    'event_date': event.get('date', {}).get('$', ''),
                    'status': event.get('status', {}).get('$', '')
                }
                
                results.append(result_item)
            
            return pd.DataFrame(results)
            
        except Exception as e:
            self.logger.warning(f"Error processing legal data: {e}")
            return pd.DataFrame()
    
    def _validate_search_params(self, params: Dict[str, Any]) -> bool:
        """Validate search parameters"""
        
        search_fields = ['text', 'title', 'abstract', 'applicant', 'inventor', 
                        'ipc_class', 'cpc_class', 'publication_number']
        
        if not any(field in params for field in search_fields):
            self.logger.error("At least one search field is required")
            return False
        
        return True
    
    def _validate_publication_params(self, params: Dict[str, Any]) -> bool:
        """Validate publication parameters"""
        
        if 'publication_id' not in params:
            self.logger.error("publication_id is required for publication service")
            return False
        
        return True
    
    def _validate_family_params(self, params: Dict[str, Any]) -> bool:
        """Validate family parameters"""
        
        if 'family_id' not in params:
            self.logger.error("family_id is required for family service")
            return False
        
        return True
    
    def _validate_images_params(self, params: Dict[str, Any]) -> bool:
        """Validate images parameters"""
        
        if 'publication_id' not in params:
            self.logger.error("publication_id is required for images service")
            return False
        
        return True
    
    def _validate_legal_params(self, params: Dict[str, Any]) -> bool:
        """Validate legal parameters"""
        
        if 'publication_id' not in params:
            self.logger.error("publication_id is required for legal service")
            return False
        
        return True
    
    def _run_connection_test(self) -> Dict[str, Any]:
        """Run EPO OPS-specific connection test"""
        try:
            # Test basic search
            test_query = {
                'service': 'search',
                'text': 'patent',
                'limit': 1
            }
            
            result = self.query(test_query)
            
            return {
                'search_test': '✅ Successful' if result.is_successful else '❌ Failed',
                'rate_limits': {
                    'requests_this_minute': len(self.request_times),
                    'weekly_requests': self.weekly_requests,
                    'weekly_limit': self.ops_config['rate_limit_per_week']
                },
                'authentication': '✅ Valid token' if self.access_token else '❌ No token'
            }
            
        except Exception as e:
            return {'connection_test': f"Failed: {str(e)}"}
    
    def get_ops_summary(self) -> Dict[str, Any]:
        """Get EPO OPS-specific provider summary"""
        
        base_summary = self.get_provider_summary()
        
        # Add OPS-specific metrics
        ops_metrics = {
            'available_services': self.available_services,
            'rate_limit_status': {
                'requests_this_minute': len(self.request_times),
                'weekly_requests': self.weekly_requests,
                'weekly_limit': self.ops_config['rate_limit_per_week'],
                'weekly_remaining': self.ops_config['rate_limit_per_week'] - self.weekly_requests
            },
            'authentication_status': 'authenticated' if self.access_token else 'not_authenticated',
            'token_expires': self.token_expires.isoformat() if self.token_expires else None
        }
        
        return {**base_summary, **ops_metrics}