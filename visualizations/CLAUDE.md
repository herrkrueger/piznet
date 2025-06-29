# Visualizations Module - CLAUDE.md

**Developer Documentation for AI-Assisted Development**

## Module Overview

The visualizations module provides comprehensive chart creation, interactive dashboards, and geographic mapping capabilities for patent intelligence analysis. Built with Plotly for interactive visualizations, it integrates seamlessly with the production-ready patent analysis platform. This documentation details all classes, functions, and interfaces for AI-assisted development.

## Core Classes

### ProductionChartCreator

**Location**: `visualizations/charts.py`

**Purpose**: Production-ready chart creation for patent intelligence with configuration-driven styling

#### Constructor

```python
ProductionChartCreator(config_manager: ConfigurationManager = None, theme: str = None)
```

**Parameters**:
- `config_manager` (ConfigurationManager, optional): Configuration manager instance
- `theme` (str, optional): Override theme from configuration

**Initialization Process**:
1. Loads configuration manager or creates default
2. Sets up theme and styling from configuration
3. Initializes country mapper for geographic data
4. Loads chart styling configuration
5. Sets up chart counter for unique IDs

#### Primary Methods

##### `create_applicant_chart(applicant_data: pd.DataFrame, chart_type: str = 'market_share', title: str = None, **kwargs) -> go.Figure`

**Purpose**: Create applicant analysis charts with market intelligence

**Parameters**:
- `applicant_data` (pd.DataFrame): Applicant analysis results from ApplicantAnalyzer
- `chart_type` (str): Chart type ('market_share', 'competitive_landscape', 'portfolio_analysis', 'organization_breakdown')
- `title` (str, optional): Chart title override
- `**kwargs`: Additional chart customization options

**Returns**: Plotly Figure object with applicant visualization

**Expected Data Structure**:
```python
applicant_data = pd.DataFrame({
    'applicant_name': str,        # Organization name
    'patent_families': int,       # Number of families
    'market_share_pct': float,    # Market share percentage
    'strategic_score': int,       # Strategic importance (0-100)
    'competitive_threat': str,    # 'High', 'Medium', 'Low'
    'organization_type': str,     # 'Corporation', 'University', etc.
    'likely_country': str         # Country code
})
```

**Chart Types**:
- `'market_share'`: Pie chart with market share distribution
- `'competitive_landscape'`: Bubble scatter plot with strategic positioning
- `'portfolio_analysis'`: Bar chart with portfolio size and quality
- `'organization_breakdown'`: Stacked bar chart by organization type

##### `create_geographic_chart(geographic_data: pd.DataFrame, chart_type: str = 'choropleth', nuts_level: int = 2, title: str = None, **kwargs) -> go.Figure`

**Purpose**: Create geographic visualization charts with NUTS integration

**Parameters**:
- `geographic_data` (pd.DataFrame): Geographic analysis results from GeographicAnalyzer
- `chart_type` (str): Chart type ('choropleth', 'bubble_map', 'regional_comparison', 'nuts_hierarchy')
- `nuts_level` (int): NUTS hierarchical level (0-3)
- `title` (str, optional): Chart title override
- `**kwargs`: Additional chart customization options

**Returns**: Plotly Figure object with geographic visualization

**Expected Data Structure**:
```python
geographic_data = pd.DataFrame({
    'country_code': str,          # ISO country code
    'nuts_code': str,             # NUTS region code
    'nuts_level': int,            # NUTS hierarchy level
    'patent_families': int,       # Families in region
    'innovation_intensity': float, # Innovation intensity score
    'filing_concentration': float, # Filing concentration metric
    'coordinates_lat': float,     # Latitude
    'coordinates_lon': float      # Longitude
})
```

##### `create_technology_chart(classification_data: pd.DataFrame, chart_type: str = 'network', title: str = None, **kwargs) -> go.Figure`

**Purpose**: Create technology intelligence charts from classification analysis

**Parameters**:
- `classification_data` (pd.DataFrame): Classification results from ClassificationProcessor
- `chart_type` (str): Chart type ('network', 'treemap', 'innovation_trends', 'maturity_analysis')
- `title` (str, optional): Chart title override
- `**kwargs`: Additional customization options

**Returns**: Plotly Figure object with technology visualization

##### `create_citation_chart(citation_data: pd.DataFrame, chart_type: str = 'impact_analysis', title: str = None, **kwargs) -> go.Figure`

**Purpose**: Create citation analysis charts for innovation impact

**Parameters**:
- `citation_data` (pd.DataFrame): Citation results from CitationAnalyzer
- `chart_type` (str): Chart type ('impact_analysis', 'network_flow', 'velocity_trends', 'quality_assessment')
- `title` (str, optional): Chart title override
- `**kwargs`: Additional customization options

**Returns**: Plotly Figure object with citation visualization

##### `create_temporal_chart(temporal_data: pd.DataFrame, chart_type: str = 'time_series', title: str = None, **kwargs) -> go.Figure`

**Purpose**: Create time-based analysis charts

**Parameters**:
- `temporal_data` (pd.DataFrame): Time series data with date columns
- `chart_type` (str): Chart type ('time_series', 'seasonal_patterns', 'trend_analysis', 'forecast')
- `title` (str, optional): Chart title override
- `**kwargs`: Additional customization options

**Returns**: Plotly Figure object with temporal visualization

##### `export_chart(figure: go.Figure, filename: str, formats: List[str] = ['html'], **kwargs) -> Dict[str, str]`

**Purpose**: Export chart to multiple formats

**Parameters**:
- `figure` (go.Figure): Plotly figure to export
- `filename` (str): Base filename (without extension)
- `formats` (List[str]): Export formats ('html', 'png', 'svg', 'pdf')
- `**kwargs`: Format-specific options (width, height, resolution)

**Returns**: Dictionary mapping format to exported filename

### ProductionDashboardCreator

**Location**: `visualizations/dashboards.py`

**Purpose**: Multi-panel interactive dashboard creation for comprehensive patent analysis

#### Constructor

```python
ProductionDashboardCreator(config_manager: ConfigurationManager = None, theme: str = None)
```

#### Primary Methods

##### `create_executive_dashboard(processor_results: Dict[str, pd.DataFrame], title: str = "Executive Patent Intelligence Dashboard", **kwargs) -> go.Figure`

**Purpose**: Create executive-level dashboard with high-level insights

**Parameters**:
- `processor_results` (Dict[str, pd.DataFrame]): Results from all processors
- `title` (str): Dashboard title
- `**kwargs`: Layout and styling options

**Expected Input Structure**:
```python
processor_results = {
    'applicant': pd.DataFrame,      # ApplicantAnalyzer results
    'geographic': pd.DataFrame,     # GeographicAnalyzer results
    'classification': pd.DataFrame, # ClassificationProcessor results
    'citation': pd.DataFrame        # CitationAnalyzer results
}
```

**Returns**: Plotly Figure with multi-panel dashboard

**Dashboard Panels**:
1. **Market Overview**: Top applicants and market share
2. **Geographic Distribution**: Key regions and filing patterns
3. **Technology Landscape**: Main technology areas and trends
4. **Innovation Impact**: Citation metrics and influence

##### `create_comprehensive_dashboard(processor_results: Dict[str, pd.DataFrame], layout: str = 'standard', **kwargs) -> go.Figure`

**Purpose**: Create comprehensive dashboard with detailed analytical views

**Parameters**:
- `processor_results` (Dict[str, pd.DataFrame]): Complete processor results
- `layout` (str): Dashboard layout ('standard', 'detailed', 'custom')
- `**kwargs**: Customization options

**Returns**: Multi-panel comprehensive dashboard

##### `create_technical_dashboard(processor_results: Dict[str, pd.DataFrame], focus_area: str = 'technology', **kwargs) -> go.Figure`

**Purpose**: Create technical dashboard focused on specific analysis area

**Parameters**:
- `processor_results` (Dict[str, pd.DataFrame]): Processor results
- `focus_area` (str): Focus area ('technology', 'geographic', 'competitive', 'citation')
- `**kwargs**: Technical dashboard options

**Returns**: Specialized technical dashboard

##### `export_dashboard(dashboard: go.Figure, filename: str, formats: List[str] = ['html'], include_data: bool = True, **kwargs) -> Dict[str, str]`

**Purpose**: Export dashboard to multiple formats

**Parameters**:
- `dashboard` (go.Figure): Dashboard figure to export
- `filename` (str): Base filename
- `formats` (List[str]): Export formats
- `include_data` (bool): Include underlying data in export
- `**kwargs`: Export options

**Returns**: Dictionary with exported file information

### ProductionMapsCreator

**Location**: `visualizations/maps.py`

**Purpose**: Geographic visualization creation with NUTS integration and strategic mapping

#### Constructor

```python
ProductionMapsCreator(config_manager: ConfigurationManager = None, country_mapper: PatentCountryMapper = None)
```

#### Primary Methods

##### `create_choropleth_map(geographic_data: pd.DataFrame, value_column: str = 'patent_families', nuts_level: int = 0, title: str = None, **kwargs) -> go.Figure`

**Purpose**: Create choropleth map for geographic patent distribution

**Parameters**:
- `geographic_data` (pd.DataFrame): Geographic analysis results
- `value_column` (str): Column to map as color values
- `nuts_level` (int): NUTS hierarchical level for EU regions
- `title` (str, optional): Map title
- `**kwargs**: Map styling options

**Returns**: Plotly Figure with choropleth map

**Map Features**:
- Country-level or NUTS-level geographic mapping
- Color scaling based on patent metrics
- Interactive hover information
- Country boundary definitions
- Strategic positioning overlays

##### `create_strategic_map(geographic_data: pd.DataFrame, applicant_data: pd.DataFrame = None, focus_regions: List[str] = None, **kwargs) -> go.Figure`

**Purpose**: Create strategic positioning map with innovation hubs

**Parameters**:
- `geographic_data` (pd.DataFrame): Geographic analysis results
- `applicant_data` (pd.DataFrame, optional): Applicant analysis for overlay
- `focus_regions` (List[str], optional): Regions to highlight
- `**kwargs**: Strategic map options

**Returns**: Strategic positioning map with competitive intelligence

##### `create_regional_comparison(geographic_data: pd.DataFrame, comparison_metric: str = 'innovation_intensity', regions: List[str] = None, **kwargs) -> go.Figure`

**Purpose**: Create regional comparison visualization

**Parameters**:
- `geographic_data` (pd.DataFrame): Geographic data for comparison
- `comparison_metric` (str): Metric for comparison
- `regions` (List[str], optional): Specific regions to compare
- `**kwargs**: Comparison options

**Returns**: Regional comparison visualization

##### `enhance_with_country_intelligence(geographic_data: pd.DataFrame) -> pd.DataFrame`

**Purpose**: Enhance geographic data with country intelligence

**Parameters**:
- `geographic_data` (pd.DataFrame): Raw geographic data

**Returns**: Enhanced data with country mappings, coordinates, and strategic groups

### PatentVisualizationFactory

**Location**: `visualizations/factory.py`

**Purpose**: Integrated visualization factory orchestrating all visualization modules

#### Constructor

```python
PatentVisualizationFactory(config_manager: ConfigurationManager = None)
```

#### Primary Methods

##### `create_comprehensive_analysis(processor_results: Dict[str, pd.DataFrame], analysis_type: str = 'executive', **kwargs) -> Dict[str, Any]`

**Purpose**: Create comprehensive visualization analysis suite

**Parameters**:
- `processor_results` (Dict[str, pd.DataFrame]): Results from all processors
- `analysis_type` (str): Analysis type ('executive', 'technical', 'full')
- `**kwargs`: Factory options

**Returns**: Dictionary with complete visualization suite:
```python
{
    'charts': Dict[str, go.Figure],      # Individual charts by type
    'dashboards': Dict[str, go.Figure],  # Dashboard visualizations
    'maps': Dict[str, go.Figure],        # Geographic maps
    'exports': Dict[str, str],           # Exported file information
    'metadata': Dict[str, Any]           # Analysis metadata
}
```

##### `create_full_analysis(search_results: pd.DataFrame, **kwargs) -> Dict[str, Any]`

**Purpose**: Create full analysis from search results through complete pipeline

**Parameters**:
- `search_results` (pd.DataFrame): PatentSearchProcessor results
- `**kwargs`: Analysis options

**Returns**: Complete analysis with visualizations

##### `create_executive_analysis(processor_results: Dict[str, pd.DataFrame], **kwargs) -> Dict[str, Any]`

**Purpose**: Create executive-focused analysis and visualizations

##### `create_technical_analysis(processor_results: Dict[str, pd.DataFrame], focus_area: str = None, **kwargs) -> Dict[str, Any]`

**Purpose**: Create technical analysis focused on specific domain

## Factory Functions

### Chart Factory Functions

#### `create_production_chart_creator(config_manager: ConfigurationManager = None, theme: str = None) -> ProductionChartCreator`

**Purpose**: Create configured chart creator

#### `create_applicant_chart(applicant_data: pd.DataFrame, **kwargs) -> go.Figure`

**Purpose**: Quick applicant chart creation

#### `create_geographic_chart(geographic_data: pd.DataFrame, **kwargs) -> go.Figure`

**Purpose**: Quick geographic chart creation

#### `create_technology_chart(classification_data: pd.DataFrame, **kwargs) -> go.Figure`

**Purpose**: Quick technology chart creation

#### `create_citation_chart(citation_data: pd.DataFrame, **kwargs) -> go.Figure`

**Purpose**: Quick citation chart creation

#### `create_temporal_chart(temporal_data: pd.DataFrame, **kwargs) -> go.Figure`

**Purpose**: Quick temporal chart creation

### Dashboard Factory Functions

#### `create_production_dashboard_creator(config_manager: ConfigurationManager = None) -> ProductionDashboardCreator`

**Purpose**: Create configured dashboard creator

#### `create_executive_dashboard(processor_results: Dict[str, pd.DataFrame], **kwargs) -> go.Figure`

**Purpose**: Quick executive dashboard creation

#### `create_comprehensive_dashboard(processor_results: Dict[str, pd.DataFrame], **kwargs) -> go.Figure`

**Purpose**: Quick comprehensive dashboard creation

### Maps Factory Functions

#### `create_production_maps_creator(config_manager: ConfigurationManager = None) -> ProductionMapsCreator`

**Purpose**: Create configured maps creator

#### `create_choropleth_map(geographic_data: pd.DataFrame, **kwargs) -> go.Figure`

**Purpose**: Quick choropleth map creation

#### `create_strategic_map(geographic_data: pd.DataFrame, **kwargs) -> go.Figure`

**Purpose**: Quick strategic map creation

#### `create_regional_comparison(geographic_data: pd.DataFrame, **kwargs) -> go.Figure`

**Purpose**: Quick regional comparison creation

### Complete Factory Functions

#### `create_visualization_factory(config_manager: ConfigurationManager = None) -> PatentVisualizationFactory`

**Purpose**: Create complete visualization factory

#### `create_patent_visualizations(search_results: pd.DataFrame, analysis_type: str = 'executive', **kwargs) -> Dict[str, Any]`

**Purpose**: Main entry point for creating patent visualizations

**Parameters**:
- `search_results` (pd.DataFrame): Patent search results
- `analysis_type` (str): Analysis type ('executive', 'technical', 'full')
- `**kwargs`: Visualization options

**Returns**: Complete visualization suite

## Legacy Compatibility Functions

### Legacy Chart Functions

#### `create_chart_creator() -> ProductionChartCreator`

**Purpose**: Legacy chart creator for backward compatibility

#### `quick_scatter(data: pd.DataFrame, x: str, y: str, **kwargs) -> go.Figure`

**Purpose**: Quick scatter plot creation

#### `quick_bar(data: pd.DataFrame, x: str, y: str, **kwargs) -> go.Figure`

**Purpose**: Quick bar chart creation

#### `quick_pie(data: pd.DataFrame, values: str, names: str, **kwargs) -> go.Figure`

**Purpose**: Quick pie chart creation

#### `quick_timeseries(data: pd.DataFrame, x: str, y: str, **kwargs) -> go.Figure`

**Purpose**: Quick time series chart creation

## Convenience Functions

### `create_quick_executive_dashboard(processor_results: Dict[str, pd.DataFrame], **kwargs) -> go.Figure`

**Purpose**: Quick executive dashboard creation

### `create_quick_patent_map(processor_results: Dict[str, pd.DataFrame], **kwargs) -> go.Figure`

**Purpose**: Quick patent choropleth map creation

### `create_quick_market_analysis(processor_results: Dict[str, pd.DataFrame], **kwargs) -> go.Figure`

**Purpose**: Quick market analysis chart creation

## Configuration Integration

### Visualization Configuration Structure

```python
# From visualization_config.yaml
{
    'general': {
        'themes': {
            'default_theme': str,
            'available_themes': List[str]
        },
        'output': {
            'default_format': str,
            'supported_formats': List[str],
            'image_resolution': int
        },
        'branding': {
            'title_prefix': str,
            'subtitle_format': str,
            'watermark': str
        }
    },
    'charts': {
        'layout': Dict,           # Default dimensions and styling
        'color_schemes': Dict,    # Color palettes
        'export': Dict            # Export settings
    },
    'maps': {
        'choropleth': Dict,       # Map styling
        'strategic_maps': Dict,   # Strategic map settings
        'country_coordinates': Dict  # Coordinate mappings
    },
    'dashboards': {
        'layouts': Dict,          # Dashboard layouts
        'panels': Dict            # Panel configurations
    }
}
```

### Theme Configuration

```python
theme_config = {
    'patent_intelligence': {
        'colors': {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'accent': '#2ca02c',
            'background': '#ffffff',
            'text': '#2e2e2e'
        },
        'fonts': {
            'title': 'Arial Bold, 18px',
            'axis': 'Arial, 14px',
            'legend': 'Arial, 12px'
        }
    }
}
```

## Performance Characteristics

### Chart Generation Performance
- **Simple Charts**: <1 second (1k records)
- **Complex Dashboards**: 2-5 seconds (multi-panel)
- **Geographic Maps**: 3-8 seconds (with enhancement)
- **Network Visualizations**: 5-15 seconds (large networks)

### Export Performance
- **HTML Export**: <2 seconds, interactive preserved
- **PNG Export**: 2-5 seconds, 300 DPI resolution
- **SVG Export**: 1-3 seconds, vector format
- **PDF Export**: 3-8 seconds, publication ready

### Memory Usage
- **Chart Objects**: 5-20 MB per complex chart
- **Dashboard Memory**: 50-200 MB comprehensive
- **Export Files**: 500KB-5MB per file

## Error Handling Patterns

### Chart Creation Errors
```python
try:
    chart = chart_creator.create_applicant_chart(data)
except ValueError as e:
    # Handle invalid data structure
    logger.error(f"Chart creation failed: {e}")
    chart = create_fallback_chart(data)
```

### Export Errors
```python
try:
    exports = chart_creator.export_chart(figure, filename, formats=['png', 'svg'])
except Exception as e:
    # Fallback to HTML export
    exports = {'html': figure.write_html(f"{filename}.html")}
```

### Geographic Data Errors
```python
try:
    enhanced_data = country_mapper.enhance_data(geographic_data)
except KeyError as e:
    # Handle missing country mappings
    enhanced_data = handle_missing_countries(geographic_data)
```

## Testing Interface

### Test Functions

**test_chart_creation()** - Tests chart factory functions  
**test_dashboard_assembly()** - Tests dashboard creation  
**test_map_generation()** - Tests geographic visualization  
**test_export_functionality()** - Tests multi-format exports  
**test_factory_integration()** - Tests complete factory workflow  
**test_configuration_loading()** - Tests theme and config integration

### Running Tests
```bash
# Complete visualization tests
./test_visualizations.sh

# Individual component tests
python visualizations/test_visualizations.py

# Specific test functions
python -c "from visualizations.test_visualizations import test_chart_creation; test_chart_creation()"
```

## Integration Patterns

### Processor Integration
```python
from processors import ComprehensiveAnalysisWorkflow
from visualizations import PatentVisualizationFactory

workflow = ComprehensiveAnalysisWorkflow()
results = workflow.run_complete_analysis(search_results)

viz_factory = PatentVisualizationFactory()
visualizations = viz_factory.create_comprehensive_analysis(results)
```

### Configuration Integration
```python
from config import ConfigurationManager
from visualizations import create_production_chart_creator

config = ConfigurationManager()
chart_creator = create_production_chart_creator(config)
```

### Export Integration
```python
# Export complete analysis
exports = viz_factory.export_all_visualizations(
    visualizations,
    base_filename="patent_analysis_2025",
    formats=['html', 'png', 'pdf']
)
```

---

**Last Updated**: 2025-06-29  
**Module Version**: 1.0  
**API Stability**: Stable  
**Production Status**: Ready for EPO PATLIB 2025