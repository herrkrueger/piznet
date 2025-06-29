"""
Production Dashboards Module for Patent Analysis Interactive Visualizations
Integrated with production config, processors, data access, and visualization layers

This module provides comprehensive dashboard creation capabilities for patent intelligence
with multi-panel layouts, interactive features, business intelligence formatting,
and seamless integration with the production-ready patent analysis platform.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import networkx as nx
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from datetime import datetime
import json
from pathlib import Path

# Production imports
from config import ConfigurationManager
from data_access.country_mapper import PatentCountryMapper
from .charts import ProductionChartCreator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductionDashboardCreator:
    """
    Production-ready dashboard creation for comprehensive patent intelligence analysis.
    Integrates seamlessly with the production patent analysis platform.
    """
    
    def __init__(self, config_manager: ConfigurationManager = None, theme: str = None):
        """
        Initialize production dashboard creator with configuration management.
        
        Args:
            config_manager: Configuration manager instance
            theme: Override theme from configuration
        """
        self.config = config_manager or ConfigurationManager()
        self.theme = theme or self.config.get('visualization', 'general.themes.default_theme')
        self.dashboard_counter = 0
        self.country_mapper = PatentCountryMapper()
        
        # Initialize sub-creators with same configuration
        self.chart_creator = ProductionChartCreator(self.config, self.theme)
        
        # Load configuration-driven settings
        self._load_dashboard_config()
    
    def _load_dashboard_config(self):
        """Load dashboard configuration from YAML files."""
        viz_config = self.config.get('visualization')
        
        # Dashboard layouts from configuration
        self.dashboard_layouts = viz_config.get('dashboards', {}).get('layouts', {})
        
        # Panel configurations
        self.panel_configs = viz_config.get('dashboards', {}).get('panels', {})
        
        # Get branding configuration
        self.branding = viz_config.get('general', {}).get('branding', {})
    
    def create_executive_dashboard(self, processor_results: Dict[str, Dict[str, Any]],
                                 title: str = "Patent Intelligence - Executive Dashboard") -> go.Figure:
        """
        Create comprehensive executive dashboard using processor results.
        
        Args:
            processor_results: Dictionary with results from all four processors
            title: Dashboard title
            
        Returns:
            Plotly figure with multi-panel executive dashboard
        """
        logger.debug(f"📊 Creating executive dashboard: {title}")
        
        # Get layout configuration
        layout = self.dashboard_layouts.get('executive_summary', {
            'rows': 2, 'cols': 2, 'height': 800,
            'panel_titles': ['Market Leaders', 'Market Share', 'Geographic Distribution', 'Activity Timeline'],
            'horizontal_spacing': 0.1, 'vertical_spacing': 0.15
        })
        
        # Create subplot structure with configuration
        specs = [[{"secondary_y": False}, {"type": "pie"}], [{"type": "bar"}, {"type": "scatter"}]]
        
        fig = make_subplots(
            rows=layout['rows'], 
            cols=layout['cols'],
            specs=specs,
            subplot_titles=layout['panel_titles'],
            horizontal_spacing=layout.get('horizontal_spacing', 0.1),
            vertical_spacing=layout.get('vertical_spacing', 0.15)
        )
        
        # Panel 1: Market Leaders (Bubble Scatter) from ApplicantProcessor
        if ('applicant' in processor_results and 
            not processor_results['applicant'].empty):
            
            # Handle both DataFrame and dict structures
            if isinstance(processor_results['applicant'], pd.DataFrame):
                applicant_data = processor_results['applicant'].head(15)
            else:
                applicant_data = processor_results['applicant'].get('applicant_ranking', pd.DataFrame()).head(15)
            
            # Extract data with proper column handling
            if not applicant_data.empty:
                x_data = applicant_data['patent_families'] if 'patent_families' in applicant_data.columns else applicant_data.iloc[:, 1]
                size_data = applicant_data['market_share_pct'] if 'market_share_pct' in applicant_data.columns else x_data
                color_data = size_data
                text_data = applicant_data['applicant_name'] if 'applicant_name' in applicant_data.columns else applicant_data.iloc[:, 0]
                
                # Ensure data is numeric and non-empty
                if hasattr(size_data, 'max') and size_data.max() > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=x_data,
                            y=list(range(len(applicant_data))),
                            mode='markers+text',
                            marker=dict(
                                size=size_data * 2,  # Scale up for visibility
                                sizeref=2. * size_data.max() / (35**2) if size_data.max() > 0 else 1,
                                sizemin=8,
                                color=color_data,
                                colorscale='Viridis',
                                showscale=True,
                                colorbar=dict(x=0.48, len=0.4, title="Market Share %")
                            ),
                            text=[str(name)[:20] for name in text_data],
                            textposition='middle right',
                            textfont=dict(size=9),
                            name='Patent Leaders',
                            hovertemplate=(
                                "<b>%{text}</b><br>" +
                                "Patent Families: %{x}<br>" +
                                "Market Share: %{marker.color:.1f}%<br>" +
                                "<extra></extra>"
                            )
                        ),
                        row=1, col=1
                    )
        
        # Panel 2: Market Share (Pie Chart) from ApplicantProcessor
        if ('applicant' in processor_results and 
            not processor_results['applicant'].empty):
            
            # Handle both DataFrame and dict structures
            if isinstance(processor_results['applicant'], pd.DataFrame):
                top_applicants = processor_results['applicant'].head(8)
            else:
                top_applicants = processor_results['applicant'].get('applicant_ranking', pd.DataFrame()).head(8)
            
            if not top_applicants.empty:
                share_col = 'market_share_pct' if 'market_share_pct' in top_applicants.columns else 'patent_families'
                name_col = 'applicant_name' if 'applicant_name' in top_applicants.columns else top_applicants.columns[0]
                
                # Create top 8 + others
                if share_col in top_applicants.columns:
                    top_share = top_applicants[share_col].sum()
                    others_share = max(0, 100 - top_share) if top_share < 100 else 0
                    
                    pie_values = list(top_applicants[share_col]) + ([others_share] if others_share > 1 else [])
                    pie_labels = [str(name)[:15] for name in top_applicants[name_col]] + (['Others'] if others_share > 1 else [])
                    
                    fig.add_trace(
                        go.Pie(
                            values=pie_values,
                            labels=pie_labels,
                            hole=0.3,
                            textinfo='label+percent',
                            textposition='auto',
                            textfont=dict(size=10),
                            marker=dict(line=dict(color='white', width=1)),
                            name='Market Share'
                        ),
                        row=1, col=2
                    )
        
        # Panel 3: Geographic Distribution (Bar Chart) from GeographicProcessor
        if ('geographic' in processor_results and 
            not processor_results['geographic'].empty):
            
            # Handle both DataFrame and dict structures
            if isinstance(processor_results['geographic'], pd.DataFrame):
                geo_data = processor_results['geographic'].head(10)
            else:
                geo_data = processor_results['geographic'].get('country_summary', pd.DataFrame()).head(10)
            
            if not geo_data.empty:
                # Use country mapper for enhanced country data if possible
                try:
                    if 'country_code' in geo_data.columns:
                        geo_data = self.country_mapper.enhance_country_data(geo_data, 'country_code')
                except:
                    pass  # Continue with original data if enhancement fails
                
                # Determine best columns to use
                country_col = 'country_name' if 'country_name' in geo_data.columns else 'nuts_region_name' if 'nuts_region_name' in geo_data.columns else geo_data.columns[0]
                value_col = 'patent_families' if 'patent_families' in geo_data.columns else 'total_applications' if 'total_applications' in geo_data.columns else geo_data.columns[1]
                
                # Create geographic summary
                if country_col in geo_data.columns and value_col in geo_data.columns:
                    geo_summary = geo_data.groupby(country_col)[value_col].sum().sort_values(ascending=True).tail(12)
                    
                    # Only add trace if we have valid data
                    if len(geo_summary) > 0 and geo_summary.sum() > 0:
                        fig.add_trace(
                            go.Bar(
                                x=geo_summary.values,
                                y=[str(name)[:20] for name in geo_summary.index],  # Truncate long names
                                orientation='h',
                                marker=dict(
                                    color=geo_summary.values,
                                    colorscale='Blues',
                                    showscale=False
                                ),
                                name='Geographic Distribution',
                                hovertemplate=(
                                    "<b>%{y}</b><br>" +
                                    f"{value_col.replace('_', ' ').title()}: %{{x}}<br>" +
                                    "<extra></extra>"
                                )
                            ),
                            row=2, col=1
                        )
        
        # Panel 4: Activity Timeline (Line Chart) from any processor with temporal data
        temporal_data = None
        for processor_name, results in processor_results.items():
            if isinstance(results, pd.DataFrame) and not results.empty:
                # Look for year-related columns in any processor results
                year_columns = [col for col in results.columns if 'year' in col.lower()]
                if year_columns:
                    temporal_data = results
                    break
            elif isinstance(results, dict):
                if 'temporal_summary' in results:
                    temporal_data = results['temporal_summary']
                    break
                elif 'annual_activity' in results:
                    temporal_data = results['annual_activity']
                    break
        
        if temporal_data is not None and len(temporal_data) > 0:
            try:
                # Find appropriate year and count columns
                year_columns = [col for col in temporal_data.columns if 'year' in col.lower()]
                count_columns = [col for col in temporal_data.columns if any(word in col.lower() for word in ['families', 'applications', 'count', 'patents'])]
                
                if year_columns and count_columns:
                    year_col = year_columns[0]
                    count_col = count_columns[0]
                    
                    # Create annual activity summary
                    annual_activity = temporal_data.groupby(year_col)[count_col].sum().reset_index()
                    annual_activity = annual_activity.sort_values(year_col)
                    
                    if len(annual_activity) > 0:
                        fig.add_trace(
                            go.Scatter(
                                x=annual_activity[year_col],
                                y=annual_activity[count_col],
                                mode='lines+markers',
                                line=dict(color='red', width=3),
                                marker=dict(size=8, color='darkred'),
                                name='Filing Activity',
                                hovertemplate=(
                                    "Year: %{x}<br>" +
                                    f"{count_col.replace('_', ' ').title()}: %{{y}}<br>" +
                                    "<extra></extra>"
                                )
                            ),
                            row=2, col=2
                        )
            except Exception as e:
                logger.debug(f"Could not create temporal chart: {e}")
        
        # Update layout with configuration
        title_prefix = self.branding.get('title_prefix', '🎯 ')
        
        fig.update_layout(
            title=f"{title_prefix}{title}",
            title_font_size=20,
            height=layout.get('height', 800),
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='#f8f9fa',
            annotations=[
                dict(
                    text=self.branding.get('watermark', 'Generated with Claude Code'),
                    xref="paper", yref="paper",
                    x=1.0, y=0.0, xanchor='right', yanchor='bottom',
                    showarrow=False, font=dict(size=8, color='gray')
                )
            ]
        )
        
        return fig
    
    def create_comprehensive_analysis_dashboard(self, processor_results: Dict[str, Dict[str, Any]],
                                              title: str = "Comprehensive Patent Analysis Dashboard") -> go.Figure:
        """
        Create comprehensive multi-processor analysis dashboard.
        
        Args:
            processor_results: Dictionary with results from all four processors
            title: Dashboard title
            
        Returns:
            Plotly figure with comprehensive analysis dashboard
        """
        logger.debug(f"📊 Creating comprehensive analysis dashboard: {title}")
        
        # Create 3x2 layout for comprehensive analysis
        fig = make_subplots(
            rows=3, cols=2,
            specs=[
                [{"type": "scatter"}, {"type": "pie"}],
                [{"type": "bar"}, {"type": "scatter"}],
                [{"type": "geo", "colspan": 2}, None]
            ],
            subplot_titles=[
                'Applicant Network', 'Technology Distribution',
                'Geographic Ranking', 'Temporal Trends',
                'Global Patent Landscape'
            ],
            horizontal_spacing=0.1,
            vertical_spacing=0.12
        )
        
        # Panel 1: Applicant Network (if available)
        if ('applicant' in processor_results and 
            'applicant_ranking' in processor_results['applicant']):
            
            data = processor_results['applicant']['applicant_ranking'].head(12)
            
            fig.add_trace(
                go.Scatter(
                    x=data.iloc[:, 1],  # Patent families
                    y=data.get('Market_Share_Pct', range(len(data))),
                    mode='markers+text',
                    marker=dict(size=15, color='blue', opacity=0.7),
                    text=data.iloc[:, 0].str[:10],
                    textposition='top center',
                    name='Applicants'
                ),
                row=1, col=1
            )
        
        # Panel 2: Technology Distribution
        if ('classification' in processor_results and 
            'cpc_distribution' in processor_results['classification']):
            
            tech_data = processor_results['classification']['cpc_distribution'].head(8)
            
            fig.add_trace(
                go.Pie(
                    values=tech_data.iloc[:, 1],
                    labels=tech_data.iloc[:, 0],
                    hole=0.3,
                    name='Technology'
                ),
                row=1, col=2
            )
        
        # Panel 3: Geographic Ranking
        if ('geographic' in processor_results and 
            'country_summary' in processor_results['geographic']):
            
            geo_data = processor_results['geographic']['country_summary']
            country_col = 'country_name' if 'country_name' in geo_data.columns else geo_data.columns[0]
            value_col = 'unique_families' if 'unique_families' in geo_data.columns else geo_data.columns[1]
            
            top_countries = geo_data.groupby(country_col)[value_col].sum().sort_values(ascending=True).tail(10)
            
            fig.add_trace(
                go.Bar(
                    x=top_countries.values,
                    y=top_countries.index,
                    orientation='h',
                    marker_color='lightgreen',
                    name='Countries'
                ),
                row=2, col=1
            )
        
        # Panel 4: Temporal Trends
        temporal_data = None
        for processor_name, results in processor_results.items():
            if 'temporal_summary' in results:
                temporal_data = results['temporal_summary']
                break
        
        if temporal_data is not None and len(temporal_data) > 0:
            year_col = temporal_data.columns[0]
            count_col = temporal_data.columns[1]
            
            fig.add_trace(
                go.Scatter(
                    x=temporal_data[year_col],
                    y=temporal_data[count_col],
                    mode='lines+markers',
                    line=dict(color='purple', width=3),
                    name='Trends'
                ),
                row=2, col=2
            )
        
        # Panel 5: Global Map (if geographic data available)
        if ('geographic' in processor_results and 
            'country_summary' in processor_results['geographic']):
            
            geo_data = processor_results['geographic']['country_summary']
            
            # Simple country mapping for choropleth
            iso_mapping = {
                'China': 'CHN', 'United States': 'USA', 'Japan': 'JPN', 'Germany': 'DEU',
                'South Korea': 'KOR', 'France': 'FRA', 'United Kingdom': 'GBR'
            }
            
            if 'country_name' in geo_data.columns:
                geo_data['iso_code'] = geo_data['country_name'].map(iso_mapping)
                geo_filtered = geo_data[geo_data['iso_code'].notna()]
                
                if len(geo_filtered) > 0:
                    value_col = 'unique_families' if 'unique_families' in geo_filtered.columns else geo_filtered.columns[1]
                    
                    fig.add_trace(
                        go.Choropleth(
                            locations=geo_filtered['iso_code'],
                            z=geo_filtered[value_col],
                            locationmode='ISO-3',
                            colorscale='Blues',
                            showscale=False
                        ),
                        row=3, col=1
                    )
        
        # Update layout
        title_prefix = self.branding.get('title_prefix', '🎯 ')
        
        fig.update_layout(
            title=f"{title_prefix}{title}",
            title_font_size=18,
            height=1000,
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='#f8f9fa'
        )
        
        # Update geo styling
        fig.update_geos(
            projection_type='natural earth',
            showframe=False,
            showcoastlines=True
        )
        
        return fig
    
    def export_dashboard_data(self, dashboard_data: Dict[str, Any], 
                            filename_prefix: str = "patent_intelligence") -> Dict[str, str]:
        """
        Export dashboard data to multiple formats for business intelligence.
        
        Args:
            dashboard_data: Dictionary with dashboard data
            filename_prefix: Prefix for exported files
            
        Returns:
            Dictionary with export file paths
        """
        logger.debug(f"💾 Exporting dashboard data: {filename_prefix}")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        export_files = {}
        
        try:
            # Export summary to JSON
            summary_file = f"{filename_prefix}_dashboard_summary_{timestamp}.json"
            with open(summary_file, 'w') as f:
                # Convert DataFrames to dict for JSON serialization
                json_data = {}
                for key, value in dashboard_data.items():
                    if isinstance(value, pd.DataFrame):
                        json_data[key] = value.to_dict('records')
                    elif isinstance(value, (dict, list, str, int, float)):
                        json_data[key] = value
                    else:
                        json_data[key] = str(value)
                
                json.dump(json_data, f, indent=2, default=str)
            
            export_files['summary'] = summary_file
            
            # Export detailed data to Excel (if pandas DataFrames present)
            excel_data = {k: v for k, v in dashboard_data.items() if isinstance(v, pd.DataFrame)}
            if excel_data:
                excel_file = f"{filename_prefix}_dashboard_data_{timestamp}.xlsx"
                with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                    for sheet_name, df in excel_data.items():
                        # Limit sheet name length
                        clean_sheet_name = sheet_name[:30].replace('/', '_')
                        df.to_excel(writer, sheet_name=clean_sheet_name, index=False)
                
                export_files['detailed'] = excel_file
            
            logger.debug(f"✅ Dashboard export complete: {len(export_files)} files created")
            
        except Exception as e:
            logger.error(f"❌ Dashboard export failed: {e}")
            export_files['error'] = str(e)
        
        return export_files

def create_production_dashboard_creator(config_manager: ConfigurationManager = None, theme: str = None) -> ProductionDashboardCreator:
    """
    Factory function to create configured production dashboard creator.
    
    Args:
        config_manager: Configuration manager instance
        theme: Styling theme override
        
    Returns:
        Configured ProductionDashboardCreator instance
    """
    return ProductionDashboardCreator(config_manager, theme)

# Production convenience functions for processor integration
def create_executive_dashboard(processor_results: Dict[str, Dict[str, Any]], **kwargs) -> go.Figure:
    """Create executive dashboard from processor results."""
    creator = create_production_dashboard_creator()
    return creator.create_executive_dashboard(processor_results, **kwargs)

def create_comprehensive_dashboard(processor_results: Dict[str, Dict[str, Any]], **kwargs) -> go.Figure:
    """Create comprehensive analysis dashboard from processor results."""
    creator = create_production_dashboard_creator()
    return creator.create_comprehensive_analysis_dashboard(processor_results, **kwargs)

# Legacy compatibility
def create_dashboard_creator(theme: str = 'patent_intelligence') -> ProductionDashboardCreator:
    """Legacy factory function for backward compatibility."""
    return create_production_dashboard_creator(theme=theme)

# Production integration example
def demo_production_dashboard_creation():
    """Demonstrate production dashboard creation with processor integration."""
    logger.debug("🚀 Production Dashboard Creation Demo")
    
    try:
        # Initialize production components
        config_manager = ConfigurationManager()
        creator = create_production_dashboard_creator(config_manager)
        
        # Create sample processor results structure
        np.random.seed(42)
        sample_results = {
            'applicant': {
                'applicant_ranking': pd.DataFrame({
                    'Applicant': [f'Company_{i}' for i in range(1, 16)],
                    'Patent_Families': np.random.randint(10, 200, 15),
                    'Market_Share_Pct': np.random.uniform(1, 25, 15)
                })
            },
            'geographic': {
                'country_summary': pd.DataFrame({
                    'country_name': ['China', 'United States', 'Japan', 'Germany', 'South Korea', 'France'],
                    'unique_families': np.random.randint(100, 800, 6)
                })
            },
            'classification': {
                'cpc_distribution': pd.DataFrame({
                    'cpc_class': ['H01M', 'C22B', 'G06F', 'H01L', 'C09K', 'B01J'],
                    'patent_count': np.random.randint(50, 300, 6)
                })
            },
            'citation': {
                'temporal_summary': pd.DataFrame({
                    'filing_year': list(range(2010, 2024)),
                    'patent_count': np.random.randint(20, 200, 14)
                })
            }
        }
        
        # Create production dashboards
        executive_dashboard = creator.create_executive_dashboard(
            sample_results, title="Patent Intelligence Executive Dashboard"
        )
        
        comprehensive_dashboard = creator.create_comprehensive_analysis_dashboard(
            sample_results, title="Comprehensive Patent Analysis"
        )
        
        logger.debug("✅ Production demo dashboards created successfully")
        
        return executive_dashboard, comprehensive_dashboard
        
    except Exception as e:
        logger.error(f"❌ Dashboard demo failed: {e}")
        return None, None

if __name__ == "__main__":
    demo_production_dashboard_creation()