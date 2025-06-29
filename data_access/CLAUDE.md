# Data Access Module - CLAUDE.md

**Developer Documentation for AI-Assisted Development**

## Module Overview

The data_access module provides production-ready data layer functionality for the Patent Intelligence Platform, including PATSTAT database connectivity, EPO OPS API integration, geographic intelligence, and performance-optimized caching. This documentation details all classes, functions, and interfaces for AI-assisted development.

## Core Classes

### PatstatClient

**Location**: `data_access/patstat_client.py`

**Purpose**: Advanced PATSTAT database connectivity with zero-exception architecture

#### Constructor

```python
PatstatClient(environment: str = 'PROD', config: ConfigurationManager = None, enable_context_manager: bool = True)
```

**Parameters**:
- `environment` (str): Database environment ('PROD', 'TEST', 'DEV'). Defaults to 'PROD'
- `config` (ConfigurationManager, optional): Configuration manager instance
- `enable_context_manager` (bool): Enable context manager support. Defaults to True

**Initialization Process**:
1. Validates environment parameter
2. Loads database configuration
3. Establishes EPO TIP connection to PATSTAT
4. Sets up connection registry for lifecycle management
5. Configures models and SQL functions
6. Implements garbage collection protection

#### Primary Properties

##### `db` - Database Session
**Type**: SQLAlchemy Session  
**Purpose**: Direct database session for query execution  
**Usage**: `session = patstat_client.db`

##### `models` - PATSTAT Table Models
**Type**: Dictionary of SQLAlchemy models  
**Purpose**: Access to PATSTAT table definitions  
**Available Models**:
- `TLS201_APPLN` - Patent applications
- `TLS212_CITATION` - Citations
- `TLS228_DOCDB_FAM_CITN` - Family citations
- `TLS209_APPLN_IPC` - IPC classifications
- `TLS224_APPLN_CPC` - CPC classifications
- `TLS206_PERSON` - Applicant/inventor data
- `TLS207_PERS_APPLN` - Person-application relationships

##### `sql_funcs` - SQL Functions
**Type**: Dictionary of SQLAlchemy functions  
**Purpose**: Database-specific SQL functions  
**Available Functions**: `func`, `and_`, `or_`, `distinct`, `case`

#### Context Manager Support

```python
with PatstatClient(environment='PROD') as client:
    session = client.db
    results = session.query(client.models['TLS201_APPLN']).limit(10).all()
    # Automatic cleanup on exit
```

#### Error Handling

**PatstatConnectionError** - Raised when database connection fails  
**PatstatQueryError** - Raised when query execution fails  
**EnvironmentError** - Raised when invalid environment specified

### EPOOPSClient

**Location**: `data_access/ops_client.py`

**Purpose**: Production-ready EPO Open Patent Services API integration

#### Constructor

```python
EPOOPSClient(consumer_key: str = None, consumer_secret: str = None, config: ConfigurationManager = None)
```

**Parameters**:
- `consumer_key` (str, optional): EPO OPS consumer key. Uses ENV:OPS_KEY if not provided
- `consumer_secret` (str, optional): EPO OPS consumer secret. Uses ENV:OPS_SECRET if not provided
- `config` (ConfigurationManager, optional): Configuration manager instance

**Authentication Process**:
1. Validates API credentials
2. Performs OAuth2 authentication with EPO OPS
3. Sets up automatic token refresh
4. Configures rate limiting parameters
5. Initializes request retry logic

#### Primary Methods

##### `search_patents(query: str, limit: int = 100) -> List[Dict]`

**Purpose**: Search patents using CQL (Common Query Language)

**Parameters**:
- `query` (str): CQL search query
- `limit` (int): Maximum number of results. Defaults to 100, max 1000

**Returns**: List of patent dictionaries with bibliographic data

**Example**:
```python
results = ops_client.search_patents('ta=(artificial intelligence)', limit=500)
```

##### `get_patent_details(patent_id: str, sections: List[str] = None) -> Dict`

**Purpose**: Retrieve detailed patent information

**Parameters**:
- `patent_id` (str): Patent identifier (publication number)
- `sections` (List[str], optional): Sections to retrieve ['biblio', 'abstract', 'claims']

**Returns**: Dictionary with patent details

##### `get_patent_family(patent_id: str) -> Dict`

**Purpose**: Retrieve patent family information

**Parameters**:
- `patent_id` (str): Patent identifier

**Returns**: Dictionary with family members and relationships

##### `get_patent_citations(patent_id: str, citation_type: str = 'both') -> Dict`

**Purpose**: Retrieve citation data for patent

**Parameters**:
- `patent_id` (str): Patent identifier
- `citation_type` (str): Type of citations ('forward', 'backward', 'both')

**Returns**: Dictionary with citation relationships

#### Rate Limiting

**Automatic Throttling**: Built-in rate limiting respects EPO OPS guidelines  
**Requests Per Minute**: Configurable (default: 30 rpm)  
**Burst Handling**: Intelligent burst detection and backoff  
**Retry Logic**: Exponential backoff with jitter

### PatentCountryMapper

**Location**: `data_access/country_mapper.py`

**Purpose**: Enhanced geographic intelligence mapping with strategic positioning

#### Constructor

```python
PatentCountryMapper(config: ConfigurationManager = None)
```

**Parameters**:
- `config` (ConfigurationManager, optional): Configuration manager instance

**Initialization**:
1. Loads country mapping database (249 countries)
2. Sets up regional groupings (IP5, EU, OECD, etc.)
3. Configures strategic positioning coordinates
4. Initializes continent and region classifications

#### Primary Methods

##### `get_country_info(country_code: str) -> Dict[str, Any]`

**Purpose**: Get comprehensive country information

**Parameters**:
- `country_code` (str): ISO country code or PATSTAT code

**Returns**: Dictionary with country details:
```python
{
    'name': str,              # Country name
    'iso_code': str,          # ISO 3166-1 alpha-2 code
    'continent': str,         # Continent name
    'region': str,            # Regional classification
    'coordinates': Dict,      # Lat/lon coordinates
    'strategic_groups': List, # IP5, EU, OECD memberships
    'market_classification': str  # Market tier classification
}
```

##### `enhance_country_data(df: pd.DataFrame, country_column: str) -> pd.DataFrame`

**Purpose**: Enhance DataFrame with comprehensive country intelligence

**Parameters**:
- `df` (pd.DataFrame): Input DataFrame
- `country_column` (str): Column name containing country codes

**Returns**: Enhanced DataFrame with additional columns:
- `country_name` - Full country name
- `continent` - Continent classification
- `region` - Regional grouping
- `coordinates_lat` - Latitude coordinate
- `coordinates_lon` - Longitude coordinate
- `strategic_groups` - Strategic group memberships

##### `get_regional_grouping(group_name: str) -> List[str]`

**Purpose**: Get countries in specific regional grouping

**Parameters**:
- `group_name` (str): Group name ('ip5_offices', 'eu_countries', 'oecd_countries', etc.)

**Returns**: List of country codes in the group

#### Strategic Intelligence Features

**IP5 Offices**: US, EP, JP, CN, KR - Major patent offices  
**EU Countries**: All European Union member states  
**OECD Countries**: OECD member nations  
**Major Economies**: G7, G20, BRICS classifications  
**Market Tiers**: Developed, emerging, developing classifications

### NUTSMapper

**Location**: `data_access/nuts_mapper.py`

**Purpose**: EU hierarchical geographic mapping for detailed regional analysis

#### Constructor

```python
NUTSMapper(patstat_client: PatstatClient = None, fallback_to_csv: bool = True)
```

**Parameters**:
- `patstat_client` (PatstatClient, optional): PATSTAT client for real-time data
- `fallback_to_csv` (bool): Use local CSV if PATSTAT unavailable. Defaults to True

#### Primary Methods

##### `get_nuts_hierarchy(nuts_code: str) -> List[str]`

**Purpose**: Get complete NUTS hierarchy for a region

**Parameters**:
- `nuts_code` (str): NUTS code (e.g., 'DE111')

**Returns**: List of NUTS codes from country to specific region
```python
['DE', 'DE1', 'DE11', 'DE111']  # Country to district level
```

##### `get_nuts_info(nuts_code: str) -> Dict[str, Any]`

**Purpose**: Get detailed information about NUTS region

**Parameters**:
- `nuts_code` (str): NUTS code

**Returns**: Dictionary with region details:
```python
{
    'nuts_code': str,      # NUTS code
    'nuts_label': str,     # Region name
    'nuts_level': int,     # Hierarchy level (0-3)
    'country_code': str,   # Parent country
    'parent_nuts': str     # Parent NUTS region
}
```

##### `aggregate_by_nuts_level(df: pd.DataFrame, target_level: int, nuts_column: str = 'nuts_code') -> pd.DataFrame`

**Purpose**: Aggregate patent data by NUTS level

**Parameters**:
- `df` (pd.DataFrame): Input DataFrame with patent data
- `target_level` (int): Target NUTS level (0-3)
- `nuts_column` (str): Column containing NUTS codes

**Returns**: Aggregated DataFrame grouped by target NUTS level

#### NUTS Level Structure
- **Level 0**: Countries (DE, FR, IT)
- **Level 1**: Major regions (DE1 = Baden-Württemberg)
- **Level 2**: Basic regions (DE11 = Stuttgart region)
- **Level 3**: Small regions (DE111 = Stuttgart district)

### PatentDataCache

**Location**: `data_access/cache_manager.py`

**Purpose**: Multi-level intelligent caching system for performance optimization

#### Constructor

```python
PatentDataCache(cache_dir: str = './cache', default_ttl: int = 3600)
```

**Parameters**:
- `cache_dir` (str): Directory for cache storage. Defaults to './cache'
- `default_ttl` (int): Default time-to-live in seconds. Defaults to 3600 (1 hour)

#### Primary Methods

##### `set(namespace: str, key: str, value: Any, ttl: int = None) -> bool`

**Purpose**: Store data in cache with optional TTL

**Parameters**:
- `namespace` (str): Cache namespace ('patstat', 'epo_ops', 'analysis')
- `key` (str): Unique cache key
- `value` (Any): Data to cache (must be JSON serializable)
- `ttl` (int, optional): Time-to-live override

**Returns**: True if successfully cached

##### `get(namespace: str, key: str) -> Any`

**Purpose**: Retrieve data from cache

**Parameters**:
- `namespace` (str): Cache namespace
- `key` (str): Cache key

**Returns**: Cached data or None if not found/expired

##### `get_stats() -> Dict[str, Any]`

**Purpose**: Get cache performance statistics

**Returns**: Dictionary with cache metrics:
```python
{
    'total_requests': int,    # Total cache requests
    'cache_hits': int,        # Successful cache hits
    'cache_misses': int,      # Cache misses
    'hit_rate': float,        # Hit rate percentage
    'total_size_mb': float,   # Total cache size in MB
    'namespaces': Dict        # Per-namespace statistics
}
```

## Factory Functions

### setup_patstat_connection(environment: str = 'PROD', config: ConfigurationManager = None) -> Tuple[PatstatClient, PatentSearcher]

**Purpose**: Initialize PATSTAT connection with patent searcher

**Parameters**:
- `environment` (str): PATSTAT environment
- `config` (ConfigurationManager, optional): Configuration instance

**Returns**: Tuple of (PatstatClient, PatentSearcher)

### setup_epo_ops_client(config: ConfigurationManager = None) -> Tuple[EPOOPSClient, PatentValidator]

**Purpose**: Initialize EPO OPS client with validator

**Parameters**:
- `config` (ConfigurationManager, optional): Configuration instance

**Returns**: Tuple of (EPOOPSClient, PatentValidator)

### create_country_mapper(config: ConfigurationManager = None) -> PatentCountryMapper

**Purpose**: Create configured country mapper

**Parameters**:
- `config` (ConfigurationManager, optional): Configuration instance

**Returns**: Configured PatentCountryMapper

### create_nuts_mapper(patstat_client: PatstatClient = None) -> NUTSMapper

**Purpose**: Create NUTS mapper with PATSTAT integration

**Parameters**:
- `patstat_client` (PatstatClient, optional): PATSTAT client for real-time data

**Returns**: Configured NUTSMapper

### create_cache_manager(cache_dir: str = './cache') -> PatentDataCache

**Purpose**: Create cache manager with default settings

**Parameters**:
- `cache_dir` (str): Cache directory path

**Returns**: Configured PatentDataCache

### setup_full_pipeline(cache_dir: str = './cache', environment: str = 'PROD') -> Dict[str, Any]

**Purpose**: Initialize complete data access pipeline

**Parameters**:
- `cache_dir` (str): Cache directory path
- `environment` (str): PATSTAT environment

**Returns**: Dictionary with all initialized components:
```python
{
    'patstat_client': PatstatClient,
    'ops_client': EPOOPSClient,
    'country_mapper': PatentCountryMapper,
    'nuts_mapper': NUTSMapper,
    'cache_manager': PatentDataCache,
    'patent_searcher': PatentSearcher
}
```

## Specialized Classes

### PatentSearcher

**Location**: `data_access/patstat_client.py` (helper class)

**Purpose**: Intelligent patent search with configurable strategies

#### Constructor

```python
PatentSearcher(patstat_client: PatstatClient, config: ConfigurationManager = None)
```

#### Key Methods

##### `execute_technology_specific_search(technology_areas: List[str], start_date: str, end_date: str, focused_search: bool = True) -> pd.DataFrame`

**Purpose**: Execute search for specific technology areas

**Parameters**:
- `technology_areas` (List[str]): Technology area identifiers
- `start_date` (str): Start date (YYYY-MM-DD)
- `end_date` (str): End date (YYYY-MM-DD)
- `focused_search` (bool): Use focused (500) vs comprehensive (5000) results

**Returns**: DataFrame with patent search results

##### `search_by_cpc_codes(cpc_codes: List[str], limit: int = 1000) -> pd.DataFrame`

**Purpose**: Search patents by CPC classification codes

**Parameters**:
- `cpc_codes` (List[str]): List of CPC codes
- `limit` (int): Maximum results

**Returns**: DataFrame with CPC-based search results

### CitationAnalyzer

**Location**: `data_access/patstat_client.py` (helper class)

**Purpose**: Family-level and application-level citation analysis

#### Constructor

```python
CitationAnalyzer(patstat_client: PatstatClient)
```

#### Key Methods

##### `get_forward_citations(family_ids: List[int]) -> pd.DataFrame`

**Purpose**: Get forward citations (who cites these patents)

**Parameters**:
- `family_ids` (List[int]): List of DOCDB family IDs

**Returns**: DataFrame with forward citation relationships

##### `get_backward_citations(family_ids: List[int]) -> pd.DataFrame`

**Purpose**: Get backward citations (what these patents cite)

**Parameters**:
- `family_ids` (List[int]): List of DOCDB family IDs

**Returns**: DataFrame with backward citation relationships

##### `analyze_citation_network(family_ids: List[int]) -> Dict[str, Any]`

**Purpose**: Comprehensive citation network analysis

**Parameters**:
- `family_ids` (List[int]): List of DOCDB family IDs

**Returns**: Dictionary with network metrics and relationships

## Data Models

### PATSTAT Table Relationships

**Core Application Data**:
- `TLS201_APPLN` - Central patent application data
- `TLS202_APPLN_TITLE` - Patent titles
- `TLS203_APPLN_ABSTR` - Patent abstracts

**Classification Data**:
- `TLS209_APPLN_IPC` - IPC classifications
- `TLS224_APPLN_CPC` - CPC classifications

**Citation Data**:
- `TLS212_CITATION` - Application-level citations
- `TLS228_DOCDB_FAM_CITN` - Family-level citations
- `TLS215_CITN_CATEG` - Citation categories

**People/Organization Data**:
- `TLS206_PERSON` - Person/organization master data
- `TLS207_PERS_APPLN` - Person-application relationships

### Geographic Data Models

**Country Information**:
```python
{
    'iso_code': str,          # ISO 3166-1 alpha-2
    'name': str,              # Country name
    'continent': str,         # Continent
    'region': str,            # Sub-regional classification
    'coordinates': {          # Geographic coordinates
        'lat': float,
        'lon': float
    },
    'strategic_groups': List[str],  # IP5, EU, OECD memberships
    'market_tier': str              # Economic classification
}
```

**NUTS Region Information**:
```python
{
    'nuts_code': str,         # NUTS code (e.g., 'DE111')
    'nuts_label': str,        # Region name
    'nuts_level': int,        # Hierarchy level (0-3)
    'country_code': str,      # Parent country ISO code
    'parent_nuts': str,       # Parent NUTS region
    'hierarchy': List[str]    # Complete hierarchy path
}
```

## Performance Characteristics

### PATSTAT Performance
- **Connection Time**: 2-3 seconds for initial connection
- **Query Performance**: Optimized for production workloads
- **Concurrent Connections**: Thread-safe with connection pooling
- **Memory Management**: Zero garbage collection issues

### EPO OPS Performance
- **Rate Limiting**: 30 requests/minute with burst handling
- **Authentication**: Automatic token refresh
- **Response Time**: 1-3 seconds per request
- **Error Recovery**: Exponential backoff with retry logic

### Cache Performance
- **Hit Rate**: 80-90% for repeated operations
- **Storage**: Compressed JSON/pickle format
- **Memory Usage**: LRU eviction with configurable limits
- **Persistence**: Survives application restarts

### Geographic Mapping Performance
- **Country Lookup**: <1ms per lookup
- **NUTS Hierarchy**: <5ms for complete hierarchy
- **Data Enhancement**: ~100ms per 1000 records
- **Memory Footprint**: <10MB for complete mapping data

## Error Handling Patterns

### Connection Errors
```python
try:
    with PatstatClient(environment='PROD') as client:
        # Database operations
except PatstatConnectionError as e:
    # Handle connection failure
    logger.error(f"PATSTAT connection failed: {e}")
```

### API Errors
```python
try:
    results = ops_client.search_patents(query)
except EPOOPSError as e:
    if e.status_code == 429:  # Rate limit
        # Implement backoff
    elif e.status_code == 403:  # Authentication
        # Refresh credentials
```

### Cache Errors
```python
try:
    cached_data = cache.get('namespace', 'key')
except CacheError as e:
    # Fall back to live data
    live_data = fetch_from_source()
```

## Testing Interface

### Test Functions

**test_patstat_connection()** - Validates PATSTAT connectivity  
**test_epo_ops_client()** - Tests EPO OPS authentication and search  
**test_country_mapper()** - Validates country mapping functionality  
**test_nuts_mapper()** - Tests NUTS geographic hierarchy  
**test_cache_functionality()** - Validates caching operations  
**test_citation_analysis()** - Tests citation data retrieval  
**test_geographic_integration()** - Tests geographic data enhancement  
**test_setup_functions()** - Validates factory functions

### Running Tests
```bash
# Complete test suite
./test_data_access.sh

# Individual tests
python -c "from data_access.test_data_access import test_patstat_connection; test_patstat_connection()"
```

---

**Last Updated**: 2025-06-29  
**Module Version**: 1.0  
**API Stability**: Stable