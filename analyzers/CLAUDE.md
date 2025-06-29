# Analyzers Module - CLAUDE.md

**Developer Documentation for AI-Assisted Development**

## Module Overview

The analyzers module provides high-level intelligence analysis that combines multiple data processing capabilities to generate comprehensive patent intelligence insights. It delivers strategic intelligence across regional competitive analysis, technology landscape mapping, and temporal trends forecasting with cross-dimensional analysis and strategic synthesis. This documentation details all classes, functions, and interfaces for AI-assisted development.

## Core Classes

### RegionalAnalyzer

**Location**: `analyzers/regional.py`

**Purpose**: Regional competitive analysis and market dynamics intelligence

#### Constructor

```python
RegionalAnalyzer()
```

**Initialization**:
1. Sets up regional analysis algorithms
2. Initializes competitive intelligence frameworks
3. Configures market dynamics analysis
4. Sets up strategic opportunity identification

#### Primary Methods

##### `analyze_regional_dynamics(unified_data: pd.DataFrame) -> pd.DataFrame`

**Purpose**: Analyze regional competitive dynamics from unified processor data

**Parameters**:
- `unified_data` (pd.DataFrame): Unified data from processor results

**Expected Data Structure**:
```python
unified_data = pd.DataFrame({
    'region': str,                    # Geographic region
    'country_name': str,             # Country name  
    'docdb_family_id': int,          # Patent family ID
    'earliest_filing_year': int,     # Filing year
    'applicant_name': str,           # Applicant organization
    'patent_families': int           # Number of families
})
```

**Returns**: Regional analysis DataFrame with competitive metrics

##### `generate_regional_intelligence_report() -> Dict[str, Any]`

**Purpose**: Generate comprehensive regional intelligence report

**Returns**: Regional intelligence dictionary:
```python
{
    'executive_summary': {
        'market_leader': str,           # Leading region/country
        'leader_share': float,          # Market share percentage
        'competitive_intensity': str,   # 'High', 'Medium', 'Low'
        'emerging_regions': int,        # Number of emerging regions
        'strategic_recommendation': str # Key strategic recommendation
    },
    'competitive_landscape': {
        'competitive_tiers': Dict,      # Multi-tier competitive analysis
        'market_concentration': float,  # Market concentration metric
        'regional_dynamics': Dict       # Regional competition patterns
    },
    'strategic_opportunities': List[str], # Strategic opportunities
    'risk_assessment': Dict             # Competitive risk analysis
}
```

##### `create_competitive_matrix() -> pd.DataFrame`

**Purpose**: Create competitive positioning matrix

**Returns**: DataFrame with competitive matrix showing regional positioning

##### `identify_market_leaders() -> Dict[str, Any]`

**Purpose**: Identify market leaders and competitive positioning

**Returns**: Market leadership analysis with leader identification

##### `assess_competitive_threats() -> Dict[str, Any]`

**Purpose**: Assess competitive threats and market risks

**Returns**: Competitive threat assessment with risk analysis

### TechnologyAnalyzer

**Location**: `analyzers/technology.py`

**Purpose**: Technology landscape analysis and innovation networks with NetworkX

#### Constructor

```python
TechnologyAnalyzer()
```

**Initialization**:
1. Sets up technology classification frameworks
2. Initializes NetworkX for innovation network analysis
3. Configures technology convergence detection
4. Sets up emerging technology identification

#### Primary Methods

##### `analyze_technology_landscape(unified_data: pd.DataFrame) -> pd.DataFrame`

**Purpose**: Analyze technology landscape from unified processor data

**Parameters**:
- `unified_data` (pd.DataFrame): Unified data with technology classifications

**Expected Data Structure**:
```python
unified_data = pd.DataFrame({
    'family_id': int,                # Patent family ID
    'filing_year': int,              # Filing year
    'IPC_1': str,                    # Primary IPC classification
    'CPC_codes': List[str],          # CPC classification codes
    'technology_domain': str,        # Technology domain
    'classification_codes': List[str] # All classification codes
})
```

**Returns**: Technology analysis DataFrame with innovation metrics

##### `build_technology_network(tech_data: pd.DataFrame) -> nx.Graph`

**Purpose**: Build technology co-occurrence and convergence network

**Parameters**:
- `tech_data` (pd.DataFrame): Technology analysis results

**Returns**: NetworkX Graph with technology relationships

**Network Features**:
- Nodes represent technology classifications
- Edges represent co-occurrence relationships
- Edge weights indicate relationship strength
- Node attributes include innovation metrics

##### `generate_technology_intelligence() -> Dict[str, Any]`

**Purpose**: Generate comprehensive technology intelligence report

**Returns**: Technology intelligence dictionary:
```python
{
    'executive_summary': {
        'dominant_area': str,           # Leading technology domain
        'emerging_technologies': int,   # Number of emerging tech areas
        'innovation_intensity': float,  # Innovation intensity score
        'convergence_patterns': List,   # Technology convergence patterns
        'strategic_focus': str          # Recommended strategic focus
    },
    'technology_clusters': Dict,        # Technology cluster analysis
    'innovation_networks': Dict,        # Network analysis results
    'emerging_technologies': List,      # Emerging technology identification
    'cross_domain_opportunities': List  # Cross-domain innovation opportunities
}
```

##### `identify_innovation_opportunities() -> List[Dict[str, Any]]`

**Purpose**: Identify innovation opportunities from technology analysis

**Returns**: List of innovation opportunities with detailed analysis

**Innovation Indicators**:
- **Cross-Domain**: Technologies bridging multiple domains (weight: 3)
- **Convergence**: Multiple technologies converging (weight: 2)
- **Emergence**: New technology areas appearing (weight: 2)
- **Acceleration**: Rapid growth in activity (weight: 1)

##### `analyze_technology_convergence(tech_data: pd.DataFrame) -> Dict[str, Any]`

**Purpose**: Analyze technology convergence patterns

**Parameters**:
- `tech_data` (pd.DataFrame): Technology data for convergence analysis

**Returns**: Technology convergence analysis results

##### `calculate_innovation_metrics(tech_data: pd.DataFrame) -> Dict[str, float]`

**Purpose**: Calculate innovation intensity and technology metrics

**Parameters**:
- `tech_data` (pd.DataFrame): Technology data

**Returns**: Dictionary with innovation metrics

### TrendsAnalyzer

**Location**: `analyzers/trends.py`

**Purpose**: Temporal trends analysis and predictive forecasting

#### Constructor

```python
TrendsAnalyzer()
```

**Initialization**:
1. Sets up temporal analysis algorithms
2. Initializes forecasting models
3. Configures cycle detection algorithms
4. Sets up market momentum analysis

#### Primary Methods

##### `analyze_temporal_trends(unified_data: pd.DataFrame) -> pd.DataFrame`

**Purpose**: Analyze temporal trends from unified processor data

**Parameters**:
- `unified_data` (pd.DataFrame): Unified data with temporal information

**Expected Data Structure**:
```python
unified_data = pd.DataFrame({
    'family_id': int,                # Patent family ID
    'filing_year': int,              # Filing year
    'application_count': int,        # Application count
    'earliest_filing_year': int,     # Earliest filing
    'latest_filing_year': int        # Latest filing
})
```

**Returns**: Temporal trends analysis DataFrame

##### `generate_trends_intelligence_report() -> Dict[str, Any]`

**Purpose**: Generate comprehensive trends intelligence report

**Returns**: Trends intelligence dictionary:
```python
{
    'executive_summary': {
        'market_momentum': str,         # 'Strong Growth', 'Moderate Growth', etc.
        'yoy_growth': float,           # Year-over-year growth percentage
        'filing_velocity': float,      # Filing velocity metric
        'cycle_phase': str,            # Innovation cycle phase
        'forecasting_confidence': str   # Forecasting confidence level
    },
    'growth_patterns': Dict,           # Growth pattern analysis
    'innovation_cycles': Dict,         # Innovation cycle analysis
    'market_forecasting': Dict,        # Future trend projections
    'volatility_analysis': Dict        # Market volatility assessment
}
```

##### `analyze_innovation_cycles() -> Dict[str, Any]`

**Purpose**: Analyze innovation cycles and technology maturity

**Returns**: Innovation cycle analysis with phase identification

##### `forecast_trends(periods: int = 5) -> Dict[str, Any]`

**Purpose**: Forecast future trends and market projections

**Parameters**:
- `periods` (int): Number of future periods to forecast. Defaults to 5

**Returns**: Forecasting results with confidence intervals

##### `calculate_market_momentum() -> str`

**Purpose**: Calculate market momentum from trend analysis

**Returns**: Market momentum classification ('Strong Growth', 'Moderate Growth', 'Stable', 'Declining')

##### `detect_seasonal_patterns() -> Dict[str, Any]`

**Purpose**: Detect seasonal and cyclical patterns

**Returns**: Seasonal pattern analysis results

## Workflow Functions

### `setup_complete_analysis_suite() -> Dict[str, object]`

**Purpose**: Setup complete analysis suite with all analyzers

**Returns**: Dictionary with all configured analyzer instances:
```python
{
    'regional_analyzer': RegionalAnalyzer,
    'technology_analyzer': TechnologyAnalyzer,
    'trends_analyzer': TrendsAnalyzer
}
```

### `run_comprehensive_intelligence_analysis(patent_data: pd.DataFrame, analysis_config: Dict = None) -> Dict[str, Any]`

**Purpose**: Run comprehensive patent intelligence analysis across all dimensions

**Parameters**:
- `patent_data` (pd.DataFrame): Unified DataFrame with all patent data from processors
- `analysis_config` (Dict, optional): Configuration for analysis parameters

**Expected Input**: Unified data from processors with required columns for each analyzer

**Returns**: Dictionary with comprehensive intelligence results:
```python
{
    'analysis_metadata': {
        'timestamp': str,              # Analysis timestamp
        'data_records': int,           # Number of records analyzed
        'analysis_scope': List[str],   # Analyses performed
        'completion_status': str       # 'Success' or 'Error'
    },
    'intelligence_reports': {
        'regional': Dict,              # Regional intelligence report
        'technology': Dict,            # Technology intelligence report
        'trends': Dict                 # Trends intelligence report
    },
    'cross_dimensional_insights': Dict, # Cross-dimensional analysis
    'strategic_synthesis': Dict        # Executive strategic synthesis
}
```

**Analysis Configuration**:
```python
analysis_config = {
    'regional_analysis': bool,         # Enable regional analysis
    'technology_analysis': bool,       # Enable technology analysis
    'trends_analysis': bool,          # Enable trends analysis
    'cross_analysis': bool            # Enable cross-dimensional analysis
}
```

### `generate_cross_dimensional_insights(intelligence_reports: Dict) -> Dict[str, Any]`

**Purpose**: Generate insights from cross-dimensional analysis

**Parameters**:
- `intelligence_reports` (Dict): Intelligence reports from different analyzers

**Returns**: Cross-dimensional insights:
```python
{
    'regional_technology_convergence': {
        'dominant_region': str,         # Leading region
        'dominant_technology': str,     # Leading technology
        'convergence_opportunity': str, # 'High', 'Moderate', 'Low'
        'strategic_alignment': str      # Strategic alignment assessment
    },
    'temporal_regional_dynamics': {
        'market_momentum': str,         # Market momentum status
        'regional_competitive_intensity': int, # Competitive regions count
        'dynamic_assessment': str,      # Market dynamics assessment
        'strategic_timing': str         # 'Optimal', 'Cautious', etc.
    },
    'technology_lifecycle_regional_patterns': Dict,
    'integrated_competitive_landscape': Dict
}
```

### `generate_strategic_synthesis(analysis_results: Dict) -> Dict[str, Any]`

**Purpose**: Generate strategic synthesis from all analysis results

**Parameters**:
- `analysis_results` (Dict): Complete analysis results dictionary

**Returns**: Strategic synthesis:
```python
{
    'executive_overview': {
        'analysis_scope': List[str],    # Completed analyses
        'intelligence_confidence': str, # 'High', 'Moderate', 'Low'
        'strategic_readiness': str      # Decision-making readiness
    },
    'key_findings': List[str],          # Key strategic findings
    'strategic_priorities': List[str],  # Priority strategic actions
    'recommended_actions': List[str],   # Specific recommended actions
    'risk_assessment': Dict             # Strategic risk assessment
}
```

## Integrated Platform Class

### IntegratedPatentIntelligence

**Location**: `analyzers/__init__.py`

**Purpose**: Integrated patent intelligence platform combining all analyzers

#### Constructor

```python
IntegratedPatentIntelligence()
```

**Initialization**:
1. Sets up all analyzer instances
2. Initializes results cache
3. Configures analysis history tracking
4. Sets up integrated workflows

#### Primary Methods

##### `run_full_intelligence_analysis(data_sources: pd.DataFrame, analysis_scope: str = 'full') -> Dict[str, Any]`

**Purpose**: Run comprehensive intelligence analysis with caching

**Parameters**:
- `data_sources` (pd.DataFrame): Unified data from processors
- `analysis_scope` (str): Scope of analysis ('full', 'quick', 'custom')

**Analysis Scopes**:
- **'full'**: Complete analysis (regional, technology, trends, cross-analysis)
- **'quick'**: Quick overview (regional, trends only)
- **'custom'**: Custom configuration (regional, technology, cross-analysis)

**Returns**: Comprehensive intelligence results with caching

##### `get_analysis_summary(analysis_id: str = None) -> Dict[str, Any]`

**Purpose**: Get summary of analysis results

**Parameters**:
- `analysis_id` (str, optional): Analysis ID (uses latest if None)

**Returns**: Analysis summary from strategic synthesis

##### `export_intelligence_report(analysis_id: str = None, format: str = 'dict') -> Dict[str, Any]`

**Purpose**: Export intelligence report in specified format

**Parameters**:
- `analysis_id` (str, optional): Analysis ID
- `format` (str): Export format ('summary', 'detailed', 'dict')

**Returns**: Intelligence report in requested format

## Factory Functions

### `create_regional_analyzer() -> RegionalAnalyzer`

**Purpose**: Create configured RegionalAnalyzer

### `create_technology_analyzer() -> TechnologyAnalyzer`

**Purpose**: Create configured TechnologyAnalyzer

### `create_trends_analyzer() -> TrendsAnalyzer`

**Purpose**: Create configured TrendsAnalyzer

### `create_integrated_intelligence_platform() -> IntegratedPatentIntelligence`

**Purpose**: Create configured integrated intelligence platform

## Data Models

### Unified Input Data Schema

```python
# Expected unified data from processors
unified_data = pd.DataFrame({
    # Regional Analysis Requirements
    'region': str,                    # Geographic region
    'country_name': str,             # Country name
    'docdb_family_id': int,          # Patent family ID
    'earliest_filing_year': int,     # Filing year
    
    # Technology Analysis Requirements
    'family_id': int,                # Patent family ID
    'filing_year': int,              # Filing year
    'IPC_1': str,                    # Primary IPC classification
    'technology_domain': str,        # Technology domain
    
    # Trends Analysis Requirements
    'family_id': int,                # Patent family ID
    'filing_year': int,              # Filing year
    'application_count': int         # Application count
})
```

### Intelligence Report Schema

```python
# Complete intelligence results structure
intelligence_results = {
    'analysis_metadata': {
        'timestamp': str,            # ISO timestamp
        'data_records': int,         # Number of records
        'analysis_scope': List[str], # Completed analyses
        'completion_status': str     # 'Success' or 'Error'
    },
    'intelligence_reports': {
        'regional': {
            'intelligence_report': Dict,     # Regional intelligence
            'competitive_matrix': Dict,      # Competitive matrix
            'analysis_status': str           # 'Complete' or 'Skipped'
        },
        'technology': {
            'intelligence_report': Dict,     # Technology intelligence
            'network_metrics': Dict,         # Network analysis metrics
            'innovation_opportunities': List, # Innovation opportunities
            'analysis_status': str           # 'Complete' or 'Skipped'
        },
        'trends': {
            'intelligence_report': Dict,     # Trends intelligence
            'innovation_cycles': Dict,       # Innovation cycles
            'forecasting_data': Dict,        # Forecasting results
            'analysis_status': str           # 'Complete' or 'Skipped'
        }
    },
    'cross_dimensional_insights': Dict,      # Cross-dimensional analysis
    'strategic_synthesis': Dict              # Strategic synthesis
}
```

## Performance Characteristics

### Analysis Performance
- **Regional Analysis**: 2-5 seconds (10k records)
- **Technology Analysis**: 5-15 seconds (with NetworkX)
- **Trends Analysis**: 3-8 seconds (with forecasting)
- **Cross-Dimensional**: 1-3 seconds
- **Strategic Synthesis**: <1 second

### Memory Usage
- **Regional Analyzer**: 10-50 MB
- **Technology Networks**: 50-200 MB (complex networks)
- **Trends Analysis**: 20-80 MB (with forecasting)
- **Complete Platform**: 100-500 MB

### Scalability
- **Small Datasets**: <1k records, sub-second
- **Medium Datasets**: 1k-10k records, 5-30 seconds
- **Large Datasets**: 10k+ records, 30-120 seconds

## Error Handling Patterns

### Analysis Validation
```python
try:
    results = run_comprehensive_intelligence_analysis(patent_data)
except ValueError as e:
    # Handle invalid data structure
    logger.error(f"Data validation failed: {e}")
```

### Missing Data Handling
```python
# Graceful degradation for missing columns
if all(col in patent_data.columns for col in required_columns):
    analysis_results = analyzer.analyze_data(patent_data)
else:
    analysis_results = {'analysis_status': 'Skipped - Missing columns'}
```

### Network Analysis Errors
```python
try:
    network = technology_analyzer.build_technology_network(tech_data)
except Exception as e:
    # Fallback to simple analysis
    network = create_simplified_network(tech_data)
```

## Configuration Integration

### Analysis Configuration
```python
# Default analysis configuration
default_config = {
    'regional_analysis': True,
    'technology_analysis': True,
    'trends_analysis': True,
    'cross_analysis': True,
    'forecasting_periods': 5,
    'network_threshold': 0.1,
    'competitive_tiers': 3
}
```

### Network Analysis Configuration
```python
# Technology network configuration
network_config = {
    'edge_threshold': 0.1,           # Minimum edge weight
    'max_nodes': 1000,               # Maximum network nodes
    'centrality_measures': ['betweenness', 'closeness', 'degree'],
    'clustering_algorithm': 'louvain'
}
```

## Testing Interface

### Test Functions

**test_regional_analyzer()** - Tests RegionalAnalyzer functionality  
**test_technology_analyzer()** - Tests TechnologyAnalyzer with NetworkX  
**test_trends_analyzer()** - Tests TrendsAnalyzer forecasting  
**test_cross_dimensional_analysis()** - Tests cross-dimensional insights  
**test_strategic_synthesis()** - Tests strategic synthesis generation  
**test_integrated_platform()** - Tests complete platform workflow  
**test_comprehensive_analysis()** - Tests end-to-end analysis

### Running Tests
```bash
# Complete analyzer tests
./test_analyzers.sh

# Individual tests
python analyzers/test_analyzers.py

# Specific analyzer test
python -c "from analyzers.test_analyzers import test_regional_analyzer; test_regional_analyzer()"
```

## Integration Patterns

### Processor Integration
```python
from processors import ComprehensiveAnalysisWorkflow
from analyzers import run_comprehensive_intelligence_analysis

# Get processor results
workflow = ComprehensiveAnalysisWorkflow()
processor_results = workflow.run_complete_analysis(search_results)

# Convert to unified data format
unified_data = combine_processor_results(processor_results)

# Run intelligence analysis
intelligence_results = run_comprehensive_intelligence_analysis(unified_data)
```

### Visualization Integration
```python
from analyzers import IntegratedPatentIntelligence
from visualizations import create_strategic_dashboards

platform = IntegratedPatentIntelligence()
intelligence_results = platform.run_full_intelligence_analysis(unified_data)

# Create strategic visualizations
dashboards = create_strategic_dashboards(intelligence_results)
```

---

**Last Updated**: 2025-06-29  
**Module Version**: 1.0  
**API Stability**: Stable  
**Production Status**: Ready for EPO PATLIB 2025