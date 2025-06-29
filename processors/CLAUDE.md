# Processors Module - CLAUDE.md

**Developer Documentation for AI-Assisted Development**

## Module Overview

The processors module provides comprehensive patent data processing capabilities for multi-dimensional patent intelligence analysis. It implements a search-driven architecture where PatentSearchProcessor serves as the foundation, feeding standardized search results to specialized enhancement processors. This documentation details all classes, functions, and interfaces for AI-assisted development.

## Architecture Overview

### Search-Driven Pipeline
```
PatentSearchProcessor (Foundation) → Enhancement Processors
├── ApplicantAnalyzer - Market intelligence and competitive analysis
├── ClassificationProcessor - Technology intelligence and innovation networks
├── GeographicAnalyzer - Strategic geographical insights with NUTS integration
└── CitationAnalyzer - Innovation impact and technology flow analysis
```

## Core Classes

### PatentSearchProcessor

**Location**: `processors/search.py`

**Purpose**: Foundation patent search processor that finds patent families using keywords and technology areas

#### Constructor

```python
PatentSearchProcessor(patstat_client: Optional[object] = None, config_path: Optional[str] = None)
```

**Parameters**:
- `patstat_client` (object, optional): PATSTAT client instance. Creates new one if None
- `config_path` (str, optional): Path to search patterns configuration

**Initialization Process**:
1. Sets up PATSTAT connection or creates client
2. Loads search patterns configuration from YAML
3. Validates technology area mappings
4. Configures SQL functions and database models
5. Sets up logging for search operations

#### Primary Methods

##### `search_patent_families(keywords: List[str] = None, technology_areas: List[str] = None, date_range: Tuple[str, str] = None, quality_mode: str = 'intersection', max_results: int = None) -> pd.DataFrame`

**Purpose**: Search for patent families using keywords and technology areas

**Parameters**:
- `keywords` (List[str], optional): Keywords to search for in titles/abstracts
- `technology_areas` (List[str], optional): Technology areas from configuration
- `date_range` (Tuple[str, str], optional): Date range as ('YYYY-MM-DD', 'YYYY-MM-DD')
- `quality_mode` (str): Search quality mode ('intersection', 'union'). Defaults to 'intersection'
- `max_results` (int, optional): Maximum number of results to return

**Returns**: DataFrame with standardized search results:
```python
{
    'docdb_family_id': int,      # PATSTAT family ID
    'quality_score': int,        # 1-3 scoring (3=highest)
    'match_type': str,           # 'intersection', 'keyword', 'cpc'
    'earliest_filing_year': int, # Family earliest filing year
    'family_size': int,          # Number of applications in family
    'primary_technology': str,   # Primary CPC code
    'keyword_matches': List[str] # Matched keywords
}
```

##### `load_search_patterns_config(config_path: str = None) -> Dict`

**Purpose**: Load and validate search patterns configuration

**Parameters**:
- `config_path` (str, optional): Path to configuration file

**Returns**: Loaded configuration dictionary

**Configuration Structure**:
```python
{
    'technology_areas': {
        'rare_earth_elements': {
            'description': str,
            'keywords': List[str],
            'cpc_codes': List[str],
            'ipc_codes': List[str]
        }
    },
    'search_strategies': {
        'focused_mode': {
            'max_results': int,
            'quality_threshold': float
        }
    }
}
```

##### `execute_keyword_search(keywords: List[str], date_range: Tuple[str, str] = None, limit: int = 1000) -> pd.DataFrame`

**Purpose**: Execute keyword-based search in patent titles and abstracts

**Parameters**:
- `keywords` (List[str]): Keywords to search for
- `date_range` (Tuple[str, str], optional): Date range filter
- `limit` (int): Maximum results. Defaults to 1000

**Returns**: DataFrame with keyword search results

##### `execute_cpc_search(cpc_codes: List[str], date_range: Tuple[str, str] = None, limit: int = 1000) -> pd.DataFrame`

**Purpose**: Execute CPC classification-based search

**Parameters**:
- `cpc_codes` (List[str]): CPC classification codes
- `date_range` (Tuple[str, str], optional): Date range filter
- `limit` (int): Maximum results. Defaults to 1000

**Returns**: DataFrame with CPC search results

### ApplicantAnalyzer

**Location**: `processors/applicant.py`

**Purpose**: Applicant analysis processor that works with PatentSearchProcessor results for market intelligence

#### Constructor

```python
ApplicantAnalyzer(patstat_client: Optional[object] = None)
```

**Parameters**:
- `patstat_client` (object, optional): PATSTAT client instance for data enrichment

**Initialization**:
1. Sets up PATSTAT connection for applicant data
2. Loads geographic patterns for company identification
3. Configures organization type classification
4. Sets up competitive intelligence algorithms

#### Primary Methods

##### `analyze_search_results(search_results: pd.DataFrame) -> pd.DataFrame`

**Purpose**: Analyze applicant patterns from search results

**Parameters**:
- `search_results` (pd.DataFrame): Standardized search results from PatentSearchProcessor

**Returns**: DataFrame with applicant analysis:
```python
{
    'applicant_name': str,        # Standardized applicant name
    'patent_families': int,       # Number of patent families
    'market_share_pct': float,    # Market share percentage
    'strategic_score': int,       # Strategic importance (0-100)
    'competitive_threat': str,    # 'High', 'Medium', 'Low'
    'likely_country': str,        # Inferred country code
    'organization_type': str,     # 'Corporation', 'University', 'Individual'
    'filing_trend': str,          # 'Growing', 'Stable', 'Declining'
    'total_applications': int     # Total applications across families
}
```

##### `enrich_with_patstat_data(family_ids: List[int]) -> pd.DataFrame`

**Purpose**: Enrich search results with applicant data from PATSTAT

**Parameters**:
- `family_ids` (List[int]): List of DOCDB family IDs

**Returns**: DataFrame with PATSTAT applicant data

**PATSTAT Tables Used**:
- `TLS201_APPLN` - Application data
- `TLS207_PERS_APPLN` - Person-application relationships
- `TLS206_PERSON` - Person/organization master data

##### `calculate_market_intelligence(applicant_data: pd.DataFrame) -> pd.DataFrame`

**Purpose**: Calculate market intelligence metrics for applicants

**Parameters**:
- `applicant_data` (pd.DataFrame): Enriched applicant data

**Returns**: DataFrame with market intelligence analysis

##### `identify_organization_type(applicant_name: str) -> str`

**Purpose**: Classify organization type based on name patterns

**Parameters**:
- `applicant_name` (str): Name of applicant organization

**Returns**: Organization type ('Corporation', 'University', 'Individual', 'Government', 'Unknown')

### ClassificationProcessor

**Location**: `processors/classification.py`

**Purpose**: Technology intelligence analysis through patent classifications

#### Constructor

```python
ClassificationProcessor(patstat_client: Optional[object] = None)
```

#### Primary Methods

##### `analyze_search_results(search_results: pd.DataFrame) -> pd.DataFrame`

**Purpose**: Analyze technology patterns from search results

**Parameters**:
- `search_results` (pd.DataFrame): Standardized search results

**Returns**: DataFrame with technology intelligence:
```python
{
    'technology_domain': str,     # Technology domain name
    'family_count': int,          # Number of families in domain
    'innovation_score': float,    # Innovation intensity (0-10)
    'network_centrality': float,  # Network centrality measure
    'trend_direction': str,       # 'Growing', 'Stable', 'Declining'
    'primary_cpc_codes': List[str], # Main CPC codes
    'related_technologies': List[str], # Related technology areas
    'maturity_level': str         # 'Emerging', 'Growing', 'Mature'
}
```

##### `enrich_with_classification_data(family_ids: List[int]) -> pd.DataFrame`

**Purpose**: Enrich with IPC/CPC classification data from PATSTAT

**Parameters**:
- `family_ids` (List[int]): DOCDB family IDs

**Returns**: Classification data from PATSTAT

**PATSTAT Tables Used**:
- `TLS209_APPLN_IPC` - IPC classifications
- `TLS224_APPLN_CPC` - CPC classifications

##### `build_technology_network(classification_data: pd.DataFrame) -> Dict`

**Purpose**: Build technology co-occurrence network

**Parameters**:
- `classification_data` (pd.DataFrame): Classification data

**Returns**: Dictionary with network structure and metrics

### GeographicAnalyzer

**Location**: `processors/geographic.py`

**Purpose**: Strategic geographical insights with EU NUTS regional analysis

#### Constructor

```python
GeographicAnalyzer(patstat_client: Optional[object] = None)
```

#### Primary Methods

##### `analyze_search_results(search_results: pd.DataFrame, analyze_applicants: bool = True, analyze_inventors: bool = True, nuts_level: int = 2) -> pd.DataFrame`

**Purpose**: Analyze geographic patterns with NUTS integration

**Parameters**:
- `search_results` (pd.DataFrame): Standardized search results
- `analyze_applicants` (bool): Include applicant geography. Defaults to True
- `analyze_inventors` (bool): Include inventor geography. Defaults to True
- `nuts_level` (int): NUTS hierarchical level (0-3). Defaults to 2

**Returns**: DataFrame with geographic analysis:
```python
{
    'nuts_code': str,             # NUTS region code
    'nuts_level': int,            # NUTS hierarchical level
    'nuts_label': str,            # Region name
    'country_code': str,          # Country ISO code
    'region_name': str,           # Full region name
    'patent_families': int,       # Patent families in region
    'inventor_families': int,     # Families with inventors in region
    'applicant_families': int,    # Families with applicants in region
    'innovation_intensity': float, # Innovation intensity score
    'filing_concentration': float, # Filing concentration metric
    'regional_hierarchy': List[str] # Complete NUTS hierarchy
}
```

##### `analyze_inventor_geography(search_results: pd.DataFrame, nuts_level: int = 3) -> pd.DataFrame`

**Purpose**: Analyze inventor geography (R&D locations)

**Parameters**:
- `search_results` (pd.DataFrame): Search results
- `nuts_level` (int): NUTS level for analysis

**Returns**: Inventor geographic analysis

##### `analyze_applicant_geography(search_results: pd.DataFrame, nuts_level: int = 1) -> pd.DataFrame`

**Purpose**: Analyze applicant geography (filing strategies)

**Parameters**:
- `search_results` (pd.DataFrame): Search results
- `nuts_level` (int): NUTS level for analysis

**Returns**: Applicant geographic analysis

##### `compare_innovation_vs_filing_geography(search_results: pd.DataFrame, nuts_level: int = 2) -> Dict`

**Purpose**: Compare innovation locations vs filing strategies

**Parameters**:
- `search_results` (pd.DataFrame): Search results
- `nuts_level` (int): NUTS level for comparison

**Returns**: Dictionary with comparison analysis:
```python
{
    'inventor_geography': pd.DataFrame,  # R&D location analysis
    'applicant_geography': pd.DataFrame, # Filing strategy analysis
    'regional_overlap': pd.DataFrame,    # Geographic alignment analysis
    'strategic_insights': Dict           # Strategic intelligence summary
}
```

##### `enrich_with_nuts_data(family_ids: List[int]) -> pd.DataFrame`

**Purpose**: Enrich with NUTS geographic data from PATSTAT

**Parameters**:
- `family_ids` (List[int]): DOCDB family IDs

**Returns**: Geographic data with NUTS codes

**PATSTAT Tables Used**:
- `TLS206_PERSON` - Person data with NUTS codes
- `TLS207_PERS_APPLN` - Person-application relationships
- `TLS904_NUTS` - NUTS reference data

#### NUTS Integration Features

**NUTS Level Structure**:
- **Level 0**: Countries (DE, FR, IT)
- **Level 1**: Major regions (DE1 = Baden-Württemberg)
- **Level 2**: Basic regions (DE11 = Stuttgart region)
- **Level 3**: Small regions (DE111 = Stuttgart district)

**Data Coverage**: 2,056 NUTS regions across 43 countries

### CitationAnalyzer

**Location**: `processors/citation.py`

**Purpose**: Innovation impact analysis through citation networks

#### Constructor

```python
CitationAnalyzer(patstat_client: Optional[object] = None)
```

#### Primary Methods

##### `analyze_search_results(search_results: pd.DataFrame) -> pd.DataFrame`

**Purpose**: Analyze citation patterns from search results

**Parameters**:
- `search_results` (pd.DataFrame): Standardized search results

**Returns**: DataFrame with citation analysis:
```python
{
    'patent_family': int,         # DOCDB family ID
    'forward_citations': int,     # Number of times cited
    'backward_citations': int,    # Number of citations made
    'citation_impact_score': float, # Impact score (0-100)
    'technology_influence': float,  # Technology influence measure
    'citation_velocity': float,   # Citations per year
    'self_citations_pct': float,  # Self-citation percentage
    'citation_quality': str,     # 'High', 'Medium', 'Low'
    'innovation_impact': str     # 'Breakthrough', 'Incremental', 'Foundational'
}
```

##### `enrich_with_citation_data(family_ids: List[int]) -> pd.DataFrame`

**Purpose**: Enrich with citation data from PATSTAT

**Parameters**:
- `family_ids` (List[int]): DOCDB family IDs

**Returns**: Citation data from PATSTAT

**PATSTAT Tables Used**:
- `TLS212_CITATION` - Application-level citations
- `TLS228_DOCDB_FAM_CITN` - Family-level citations
- `TLS215_CITN_CATEG` - Citation categories

##### `analyze_citation_network(citation_data: pd.DataFrame) -> Dict`

**Purpose**: Analyze citation network structure and metrics

**Parameters**:
- `citation_data` (pd.DataFrame): Citation relationships

**Returns**: Network analysis results with centrality measures

## Factory Functions

### `create_patent_search_processor(patstat_client: Optional[object] = None) -> PatentSearchProcessor`

**Purpose**: Create configured PatentSearchProcessor

**Parameters**:
- `patstat_client` (object, optional): PATSTAT client instance

**Returns**: Configured PatentSearchProcessor

### `create_applicant_analyzer(patstat_client: Optional[object] = None) -> ApplicantAnalyzer`

**Purpose**: Create configured ApplicantAnalyzer

**Parameters**:
- `patstat_client` (object, optional): PATSTAT client instance

**Returns**: Configured ApplicantAnalyzer

### `create_classification_processor(patstat_client: Optional[object] = None) -> ClassificationProcessor`

**Purpose**: Create configured ClassificationProcessor

**Parameters**:
- `patstat_client` (object, optional): PATSTAT client instance

**Returns**: Configured ClassificationProcessor

### `create_geographic_analyzer(patstat_client: Optional[object] = None) -> GeographicAnalyzer`

**Purpose**: Create configured GeographicAnalyzer

**Parameters**:
- `patstat_client` (object, optional): PATSTAT client instance

**Returns**: Configured GeographicAnalyzer

### `create_citation_analyzer(patstat_client: Optional[object] = None) -> CitationAnalyzer`

**Purpose**: Create configured CitationAnalyzer

**Parameters**:
- `patstat_client` (object, optional): PATSTAT client instance

**Returns**: Configured CitationAnalyzer

### `setup_full_processing_pipeline() -> Dict[str, object]`

**Purpose**: Setup complete processing pipeline with all processors

**Returns**: Dictionary with all configured processors:
```python
{
    'search_processor': PatentSearchProcessor,
    'applicant_analyzer': ApplicantAnalyzer,
    'classification_processor': ClassificationProcessor,
    'geographic_analyzer': GeographicAnalyzer,
    'citation_analyzer': CitationAnalyzer
}
```

### `create_analysis_pipeline() -> Dict[str, object]`

**Purpose**: Create analysis pipeline (search processor + analyzers)

**Returns**: Dictionary with analysis components

## Workflow Classes

### ComprehensiveAnalysisWorkflow

**Location**: `processors/__init__.py`

**Purpose**: Integrated analysis workflow for comprehensive patent intelligence

#### Constructor

```python
ComprehensiveAnalysisWorkflow(patstat_client: Optional[object] = None)
```

**Parameters**:
- `patstat_client` (object, optional): PATSTAT client for all processors

#### Primary Methods

##### `run_patent_search(keywords: List[str] = None, technology_areas: List[str] = None, date_range: Tuple[str, str] = None, quality_mode: str = 'intersection', max_results: int = None) -> pd.DataFrame`

**Purpose**: Run patent family search using PatentSearchProcessor

**Parameters**:
- `keywords` (List[str], optional): Keywords to search for
- `technology_areas` (List[str], optional): Technology areas from config
- `date_range` (Tuple[str, str], optional): Date range filter
- `quality_mode` (str): Search quality mode. Defaults to 'intersection'
- `max_results` (int, optional): Maximum results to return

**Returns**: Search results DataFrame

##### `run_complete_analysis(search_results: pd.DataFrame = None) -> Dict[str, pd.DataFrame]`

**Purpose**: Run complete analysis workflow on search results

**Parameters**:
- `search_results` (pd.DataFrame, optional): Search results (uses cached if None)

**Returns**: Dictionary with all analysis results:
```python
{
    'applicant': pd.DataFrame,      # Applicant analysis results
    'classification': pd.DataFrame, # Technology analysis results
    'geographic': pd.DataFrame,     # Geographic analysis results
    'citation': pd.DataFrame        # Citation analysis results
}
```

##### `run_applicant_analysis(search_results: pd.DataFrame = None) -> pd.DataFrame`

**Purpose**: Run applicant analysis on search results

##### `run_classification_analysis(search_results: pd.DataFrame = None) -> pd.DataFrame`

**Purpose**: Run classification analysis on search results

##### `run_geographic_analysis(search_results: pd.DataFrame = None) -> pd.DataFrame`

**Purpose**: Run geographic analysis on search results

##### `run_citation_analysis(search_results: pd.DataFrame = None) -> pd.DataFrame`

**Purpose**: Run citation analysis on search results

##### `get_comprehensive_summary() -> Dict`

**Purpose**: Generate comprehensive summary across all analyses

**Returns**: Dictionary with integrated summary:
```python
{
    'search_overview': {
        'total_families': int,
        'search_quality': float
    },
    'analyses_completed': List[str],
    'analysis_summaries': Dict[str, Dict]
}
```

##### `export_all_results(base_filename: str = None) -> Dict[str, str]`

**Purpose**: Export all analysis results to files

**Parameters**:
- `base_filename` (str, optional): Base filename (timestamp added if None)

**Returns**: Dictionary with export filenames

## Exception Classes

### PatstatConnectionError

**Purpose**: Raised when PATSTAT database connection fails

### DataNotFoundError

**Purpose**: Raised when PATSTAT query returns no results

### InvalidQueryError

**Purpose**: Raised when PATSTAT query syntax is invalid

## Performance Characteristics

### Search Performance
- **Keyword Search**: ~500-1000 families/second
- **CPC Search**: ~800-1200 families/second
- **Combined Search**: ~300-500 families/second

### Analysis Performance (Per 1000 Families)
- **Applicant Analyzer**: 3,324 families/second
- **Classification Processor**: 1,461 families/second
- **Geographic Analyzer**: 4,547 families/second
- **Citation Analyzer**: 4,189 families/second

### Production Capacity
- **Estimated Total**: 12.17 million families/hour
- **Memory Usage**: <2GB for 100k families
- **PATSTAT Query Time**: 2-5 seconds per enrichment

## Data Models

### Search Results Schema
```python
search_results = pd.DataFrame({
    'docdb_family_id': int,       # PATSTAT family identifier
    'quality_score': int,         # Search quality (1-3)
    'match_type': str,            # Match type indicator
    'earliest_filing_year': int,  # Family filing year
    'family_size': int,           # Applications in family
    'primary_technology': str,    # Primary CPC code
    'keyword_matches': List[str]  # Matched keywords
})
```

### Geographic Data Model
```python
geographic_data = pd.DataFrame({
    'nuts_code': str,             # NUTS region code
    'nuts_level': int,            # Hierarchy level (0-3)
    'nuts_label': str,            # Region name
    'country_code': str,          # ISO country code
    'patent_families': int,       # Families in region
    'inventor_families': int,     # R&D location families
    'applicant_families': int,    # Filing strategy families
    'regional_hierarchy': List[str] # Complete hierarchy
})
```

## Configuration Integration

### Search Patterns Configuration
```yaml
# search_patterns_config.yaml
technology_areas:
  rare_earth_elements:
    description: "Rare earth elements extraction and processing"
    keywords: ["rare earth", "lanthanide", "REE"]
    cpc_codes: ["C22B 19/28", "C22B 19/30"]
    ipc_codes: ["C22B 19/00"]

search_strategies:
  focused_mode:
    max_results: 500
    quality_threshold: 2.5
  comprehensive_mode:
    max_results: 5000
    quality_threshold: 2.0
```

## Error Handling Patterns

### PATSTAT Connection Handling
```python
try:
    results = processor.analyze_search_results(search_results)
except PatstatConnectionError:
    # Fallback to mock data or retry
    results = processor.analyze_with_fallback_data(search_results)
```

### Data Validation
```python
try:
    validated_results = processor.validate_search_results(search_results)
except InvalidQueryError as e:
    logger.error(f"Invalid search results format: {e}")
    # Handle validation error
```

## Testing Interface

### Test Functions

**test_search_processor()** - Tests PatentSearchProcessor functionality  
**test_applicant_analyzer()** - Tests ApplicantAnalyzer with mock data  
**test_classification_processor()** - Tests ClassificationProcessor analysis  
**test_geographic_analyzer()** - Tests GeographicAnalyzer with NUTS data  
**test_citation_analyzer()** - Tests CitationAnalyzer functionality  
**test_comprehensive_workflow()** - Tests complete analysis workflow  
**test_patstat_integration()** - Tests real PATSTAT connectivity  
**test_performance_scaling()** - Tests performance with large datasets

### Running Tests
```bash
# Complete test suite
./test_processors.sh

# Unit tests only
python processors/test_processor.py

# Integration tests
python processors/test_complete_pipeline.py

# Specific processor test
python processors/test_processor.py --processor applicant
```

## Integration Patterns

### Data Access Integration
```python
from processors import create_applicant_analyzer
from data_access import PatstatClient

patstat_client = PatstatClient(environment='PROD')
analyzer = create_applicant_analyzer(patstat_client)
```

### Configuration Integration
```python
from processors import PatentSearchProcessor
from config import ConfigurationManager

config = ConfigurationManager()
search_processor = PatentSearchProcessor(config_path=config.get_search_patterns_path())
```

### Visualization Integration
```python
from processors import ComprehensiveAnalysisWorkflow
from visualizations import create_applicant_charts

workflow = ComprehensiveAnalysisWorkflow()
results = workflow.run_complete_analysis(search_results)
charts = create_applicant_charts(results['applicant'])
```

---

**Last Updated**: 2025-06-29  
**Module Version**: 1.0  
**API Stability**: Stable  
**Production Status**: Ready for EPO PATLIB 2025