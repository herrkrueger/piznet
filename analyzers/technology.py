"""
Technology Analysis Module for REE Patent Intelligence
Enhanced from EPO PATLIB 2025 Live Demo Code

This module provides comprehensive technology analysis capabilities including
innovation networks, technology convergence, and emerging technology identification.
"""

import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Tuple, Union, Set
import logging
from datetime import datetime
from collections import defaultdict, Counter
import itertools
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from data_access.classification_config import get_classification_config
except ImportError:
    get_classification_config = None

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TechnologyAnalyzer:
    """
    Comprehensive technology analysis for patent intelligence with innovation networks.
    Uses official CPC/IPC classification data for precise technology domain analysis.
    """
    
    # Innovation indicators
    INNOVATION_INDICATORS = {
        'Cross-Domain': {'weight': 3, 'description': 'Technologies bridging multiple domains'},
        'Convergence': {'weight': 2, 'description': 'Multiple technologies converging'},
        'Emergence': {'weight': 2, 'description': 'New technology areas appearing'},
        'Acceleration': {'weight': 1, 'description': 'Rapid growth in activity'}
    }
    
    def __init__(self):
        """Initialize technology analyzer with official classification data."""
        self.analyzed_data = None
        self.technology_network = None
        self.innovation_landscape = None
        self.technology_intelligence = None
        
        # Initialize classification system
        self.classification_config = None
        self.classification_client = None
        self._initialize_classification_system()
    
    def _initialize_classification_system(self):
        """Initialize CPC/IPC classification system."""
        try:
            if get_classification_config:
                self.classification_config = get_classification_config()
                self.classification_client = self.classification_config.get_client()
                logger.info(f"🎯 Technology analyzer using {self.classification_config.system.upper()} classification system")
            else:
                logger.warning("⚠️ Classification system not available, using basic analysis")
        except Exception as e:
            logger.warning(f"⚠️ Could not initialize classification system: {e}")
    
    def analyze_technology_landscape(self, patent_data: pd.DataFrame,
                                   ipc_col: str = 'IPC_1',
                                   ipc2_col: str = 'IPC_2',
                                   family_col: str = 'family_id',
                                   year_col: str = 'filing_year',
                                   domain_col: str = 'domain_1') -> pd.DataFrame:
        """
        Comprehensive technology landscape analysis with innovation metrics.
        
        Args:
            patent_data: DataFrame with technology classification data
            ipc_col: Column name for primary IPC code
            ipc2_col: Column name for secondary IPC code (for co-occurrence)
            family_col: Column name for patent family IDs
            year_col: Column name for filing years
            domain_col: Column name for technology domains
            
        Returns:
            Enhanced DataFrame with technology intelligence
        """
        logger.debug("🔬 Starting comprehensive technology landscape analysis...")
        
        df = patent_data.copy()
        
        # Validate required columns
        required_cols = [ipc_col, family_col, year_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Add official technology classification
        df = self._add_technology_classification(df, ipc_col)
        
        # Analyze technology evolution
        df = self._analyze_technology_evolution(df, year_col, ipc_col, family_col)
        
        # Calculate innovation metrics
        df = self._calculate_innovation_metrics(df, ipc_col, ipc2_col, year_col)
        
        # Identify emerging technologies
        df = self._identify_emerging_technologies(df, year_col, ipc_col)
        
        # Analyze technology convergence
        if ipc2_col in df.columns:
            df = self._analyze_technology_convergence(df, ipc_col, ipc2_col, domain_col)
        
        self.analyzed_data = df
        logger.debug(f"✅ Technology analysis complete for {len(df)} records")
        
        return df
    
    def _add_technology_classification(self, df: pd.DataFrame, ipc_col: str) -> pd.DataFrame:
        """Add official technology classification using CPC/IPC data."""
        logger.debug("🏷️ Adding official technology classification...")
        
        def classify_technology(code: str) -> Tuple[str, str, str, str]:
            """Classify code using official CPC/IPC system."""
            if pd.isna(code) or not str(code).strip():
                return 'Other', 'Unknown', 'Unknown', 'Unknown'
            
            code_str = str(code).strip()
            
            # Get official description
            if self.classification_client and self.classification_client.available:
                try:
                    # Get subclass (first 4 characters for CPC/IPC)
                    subclass = code_str[:4] if len(code_str) >= 4 else code_str
                    description = self.classification_client.get_description(subclass)
                    
                    if description and description != f'Unknown {self.classification_config.system.upper()}: {subclass}':
                        # Extract technology area from subclass
                        technology_area = self._extract_technology_area(subclass, description)
                        subcategory = self._extract_subcategory(code_str, description)
                        maturity = self._estimate_maturity(subclass)
                        strategic_value = self._estimate_strategic_value(subclass)
                        
                        return technology_area, subcategory, maturity, strategic_value
                except Exception as e:
                    logger.debug(f"Classification lookup failed for {code_str}: {e}")
            
            # Fallback to basic section-based classification
            return self._basic_section_classification(code_str)
        
        # Apply official classification
        classification_results = df[ipc_col].apply(classify_technology)
        df['technology_area'] = [result[0] for result in classification_results]
        df['technology_subcategory'] = [result[1] for result in classification_results]
        df['technology_maturity'] = [result[2] for result in classification_results]
        df['strategic_value'] = [result[3] for result in classification_results]
        
        return df
    
    def _extract_technology_area(self, subclass: str, description: str) -> str:
        """Extract technology area from subclass and description."""
        # Map common subclasses to technology areas
        area_mapping = {
            'A01': 'Agriculture & Food',
            'A61': 'Medical & Healthcare', 
            'B01': 'Chemical Processing',
            'C01': 'Inorganic Chemistry',
            'C07': 'Organic Chemistry',
            'C08': 'Macromolecules & Polymers',
            'C22': 'Metallurgy & Extraction',
            'C04': 'Ceramics & Materials',
            'H01': 'Electronics & Semiconductors',
            'H02': 'Electrical Engineering',
            'G01': 'Measuring & Testing',
            'G06': 'Computing & Data Processing',
            'Y02': 'Climate & Sustainability',
            'F01': 'Mechanical Engineering',
            'F16': 'Engineering Elements'
        }
        
        # Check for direct mapping
        for prefix, area in area_mapping.items():
            if subclass.startswith(prefix):
                return area
        
        # Fallback to section-based classification
        if subclass:
            section = subclass[0]
            section_mapping = {
                'A': 'Human Necessities',
                'B': 'Operations & Transport', 
                'C': 'Chemistry & Metallurgy',
                'D': 'Textiles & Paper',
                'E': 'Construction & Mining',
                'F': 'Mechanical Engineering',
                'G': 'Physics & Instruments',
                'H': 'Electricity & Electronics',
                'Y': 'Emerging Technologies'
            }
            return section_mapping.get(section, 'Other Technologies')
        
        return 'Other Technologies'
    
    def _extract_subcategory(self, code: str, description: str) -> str:
        """Extract subcategory from full code and description."""
        # Use first meaningful part of description as subcategory
        if description and len(description) > 10:
            # Take first part before comma or semicolon
            subcategory = description.split(',')[0].split(';')[0].strip()
            if len(subcategory) > 50:
                subcategory = subcategory[:47] + '...'
            return subcategory
        
        return 'General'
    
    def _estimate_maturity(self, subclass: str) -> str:
        """Estimate technology maturity based on subclass."""
        # Y-section codes are typically emerging/environmental
        if subclass.startswith('Y'):
            return 'Emerging'
        
        # Traditional manufacturing and chemistry are mature
        mature_prefixes = ['C22', 'C01', 'F01', 'F16', 'B21', 'B23']
        if any(subclass.startswith(prefix) for prefix in mature_prefixes):
            return 'Mature'
        
        # Electronics and computing are rapidly evolving
        growing_prefixes = ['H01', 'H02', 'G06', 'G01']
        if any(subclass.startswith(prefix) for prefix in growing_prefixes):
            return 'Growing'
        
        return 'Established'
    
    def _estimate_strategic_value(self, subclass: str) -> str:
        """Estimate strategic value based on subclass."""
        # Critical technologies for modern industry
        critical_prefixes = ['C22', 'H01', 'Y02', 'A61']
        if any(subclass.startswith(prefix) for prefix in critical_prefixes):
            return 'Critical'
        
        # High-value technologies
        high_value_prefixes = ['G06', 'C07', 'C08', 'H02']
        if any(subclass.startswith(prefix) for prefix in high_value_prefixes):
            return 'High'
        
        # Strategic but not critical
        strategic_prefixes = ['G01', 'F01', 'B01']
        if any(subclass.startswith(prefix) for prefix in strategic_prefixes):
            return 'Strategic'
        
        return 'Standard'
    
    def _basic_section_classification(self, code: str) -> Tuple[str, str, str, str]:
        """Fallback classification based on IPC/CPC section."""
        if not code:
            return 'Other', 'Unknown', 'Unknown', 'Unknown'
            
        section = code[0] if code else 'Z'
        
        section_info = {
            'A': ('Human Necessities', 'General', 'Established', 'Standard'),
            'B': ('Operations & Transport', 'General', 'Mature', 'Standard'), 
            'C': ('Chemistry & Metallurgy', 'General', 'Mature', 'High'),
            'D': ('Textiles & Paper', 'General', 'Mature', 'Standard'),
            'E': ('Construction & Mining', 'General', 'Established', 'Standard'),
            'F': ('Mechanical Engineering', 'General', 'Mature', 'Standard'),
            'G': ('Physics & Instruments', 'General', 'Growing', 'High'),
            'H': ('Electricity & Electronics', 'General', 'Growing', 'Critical'),
            'Y': ('Emerging Technologies', 'General', 'Emerging', 'Critical')
        }
        
        return section_info.get(section, ('Other', 'Unknown', 'Unknown', 'Unknown'))
    
    def _analyze_technology_evolution(self, df: pd.DataFrame, year_col: str, 
                                    ipc_col: str, family_col: str) -> pd.DataFrame:
        """Analyze technology evolution patterns over time."""
        logger.debug("📈 Analyzing technology evolution...")
        
        # Calculate technology lifecycle metrics
        tech_evolution = df.groupby(['technology_area', year_col]).agg({
            family_col: 'nunique',
            ipc_col: 'nunique'
        }).reset_index()
        
        tech_evolution.columns = ['technology_area', 'year', 'patent_families', 'unique_classifications']
        
        # Calculate growth rates and trends
        def calculate_evolution_metrics(group):
            if len(group) < 2:
                return pd.Series({
                    'cagr': 0,
                    'trend_direction': 'Stable',
                    'peak_year': group['year'].iloc[0],
                    'lifecycle_stage': 'Unknown'
                })
            
            # Calculate CAGR
            first_year_patents = group.iloc[0]['patent_families']
            last_year_patents = group.iloc[-1]['patent_families']
            years_span = group.iloc[-1]['year'] - group.iloc[0]['year']
            
            if first_year_patents > 0 and years_span > 0:
                cagr = ((last_year_patents / first_year_patents) ** (1/years_span) - 1) * 100
            else:
                cagr = 0
            
            # Determine trend direction
            recent_avg = group.tail(3)['patent_families'].mean()
            early_avg = group.head(3)['patent_families'].mean()
            
            if recent_avg > early_avg * 1.2:
                trend_direction = 'Growing'
            elif recent_avg < early_avg * 0.8:
                trend_direction = 'Declining'
            else:
                trend_direction = 'Stable'
            
            # Find peak year
            peak_year = group.loc[group['patent_families'].idxmax(), 'year']
            
            # Determine lifecycle stage
            max_patents = group['patent_families'].max()
            current_patents = group.iloc[-1]['patent_families']
            
            if current_patents >= max_patents * 0.8:
                lifecycle_stage = 'Growth/Maturity'
            elif current_patents >= max_patents * 0.5:
                lifecycle_stage = 'Stable'
            else:
                lifecycle_stage = 'Decline'
            
            return pd.Series({
                'cagr': cagr,
                'trend_direction': trend_direction,
                'peak_year': peak_year,
                'lifecycle_stage': lifecycle_stage
            })
        
        evolution_metrics = tech_evolution.groupby('technology_area').apply(calculate_evolution_metrics).reset_index()
        
        # Merge evolution metrics back to main dataframe
        df = df.merge(
            evolution_metrics,
            left_on='technology_area',
            right_on='technology_area',
            how='left'
        )
        
        return df
    
    def _calculate_innovation_metrics(self, df: pd.DataFrame, ipc_col: str, 
                                    ipc2_col: str, year_col: str) -> pd.DataFrame:
        """Calculate comprehensive innovation metrics."""
        logger.debug("💡 Calculating innovation metrics...")
        
        # Innovation diversity within technology areas
        diversity_metrics = df.groupby('technology_area').agg({
            ipc_col: 'nunique',
            'technology_subcategory': 'nunique',
            'family_id': 'nunique'
        }).rename(columns={
            ipc_col: 'classification_diversity',
            'technology_subcategory': 'subcategory_diversity',
            'family_id': 'total_families'
        })
        
        # Calculate innovation intensity
        diversity_metrics['innovation_intensity'] = (
            diversity_metrics['classification_diversity'] / diversity_metrics['total_families']
        ).round(3)
        
        # Innovation novelty (recent technologies with few historical patents)
        current_year = datetime.now().year
        recent_threshold = current_year - 3
        
        def calculate_novelty_score(group):
            total_patents = len(group)
            recent_patents = len(group[group[year_col] >= recent_threshold])
            
            if total_patents == 0:
                return 0
            
            # High novelty = high recent activity with low historical base
            novelty_ratio = recent_patents / total_patents
            if total_patents < 10 and novelty_ratio > 0.6:
                return 0.9  # High novelty
            elif total_patents < 50 and novelty_ratio > 0.4:
                return 0.6  # Medium novelty
            else:
                return 0.3  # Low novelty
        
        novelty_scores = df.groupby(ipc_col).apply(calculate_novelty_score)
        df['novelty_score'] = df[ipc_col].map(novelty_scores)
        
        # Cross-domain innovation (if second IPC available)
        if ipc2_col in df.columns and 'domain_1' in df.columns and 'domain_2' in df.columns:
            df['cross_domain_innovation'] = df['domain_1'] != df['domain_2']
            df['innovation_complexity'] = df['cross_domain_innovation'].map({
                True: 'High (Cross-Domain)',
                False: 'Standard (Single Domain)'
            })
        else:
            df['cross_domain_innovation'] = False
            df['innovation_complexity'] = 'Standard (Single Domain)'
        
        # Merge diversity metrics
        df = df.merge(
            diversity_metrics.add_suffix('_area'),
            left_on='technology_area',
            right_index=True,
            how='left'
        )
        
        return df
    
    def _identify_emerging_technologies(self, df: pd.DataFrame, year_col: str, ipc_col: str) -> pd.DataFrame:
        """Identify emerging technologies and innovation patterns."""
        logger.debug("🌱 Identifying emerging technologies...")
        
        current_year = datetime.now().year
        emergence_window = 5  # Look at last 5 years for emergence
        
        # Calculate emergence indicators
        emergence_analysis = df.groupby(ipc_col).apply(
            lambda group: pd.Series({
                'first_appearance': group[year_col].min(),
                'total_families': group['family_id'].nunique(),
                'recent_activity': len(group[group[year_col] >= current_year - emergence_window]),
                'activity_acceleration': self._calculate_activity_acceleration(group, year_col),
                'geographic_spread': group['country_name'].nunique() if 'country_name' in group.columns else 1
            })
        )
        
        # Classify emergence characteristics
        def classify_emergence(row):
            if row['first_appearance'] >= current_year - emergence_window:
                if row['total_families'] >= 5 and row['activity_acceleration'] > 0.2:
                    return 'Breakthrough Technology'
                elif row['total_families'] >= 2:
                    return 'Emerging Technology'
                else:
                    return 'Experimental Technology'
            elif row['activity_acceleration'] > 0.5 and row['recent_activity'] >= 3:
                return 'Accelerating Technology'
            else:
                return 'Established Technology'
        
        emergence_analysis['emergence_classification'] = emergence_analysis.apply(classify_emergence, axis=1)
        
        # Calculate emergence score
        emergence_analysis['emergence_score'] = (
            emergence_analysis['activity_acceleration'] * 0.4 +
            (emergence_analysis['recent_activity'] / emergence_analysis['total_families']) * 0.3 +
            emergence_analysis['geographic_spread'] / 10 * 0.3
        ).round(3)
        
        # Merge back to main dataframe
        df = df.merge(
            emergence_analysis[['emergence_classification', 'emergence_score']],
            left_on=ipc_col,
            right_index=True,
            how='left'
        )
        
        return df
    
    def _calculate_activity_acceleration(self, group: pd.DataFrame, year_col: str) -> float:
        """Calculate activity acceleration for a technology group."""
        if len(group) < 3:
            return 0.0
        
        yearly_activity = group.groupby(year_col).size().sort_index()
        
        if len(yearly_activity) < 3:
            return 0.0
        
        # Calculate acceleration using linear regression slope
        years = yearly_activity.index.values
        activities = yearly_activity.values
        
        # Simple acceleration: difference between recent and early average
        split_point = len(yearly_activity) // 2
        early_avg = yearly_activity.iloc[:split_point].mean()
        recent_avg = yearly_activity.iloc[split_point:].mean()
        
        return (recent_avg - early_avg) / early_avg if early_avg > 0 else 0.0
    
    def _analyze_technology_convergence(self, df: pd.DataFrame, ipc_col: str, 
                                      ipc2_col: str, domain_col: str) -> pd.DataFrame:
        """Analyze technology convergence patterns."""
        logger.debug("🔄 Analyzing technology convergence...")
        
        # Technology convergence matrix
        convergence_patterns = df.groupby([ipc_col, ipc2_col]).agg({
            'family_id': 'nunique',
            'filing_year': 'count'
        }).rename(columns={
            'family_id': 'unique_families',
            'filing_year': 'total_occurrences'
        }).reset_index()
        
        # Calculate convergence strength
        max_convergence = convergence_patterns['total_occurrences'].max()
        convergence_patterns['convergence_strength'] = (
            convergence_patterns['total_occurrences'] / max_convergence
        ).round(3)
        
        # Identify strong convergence patterns
        strong_convergence = convergence_patterns[
            convergence_patterns['convergence_strength'] >= 0.1
        ].copy()
        
        # Classify convergence types
        def classify_convergence_type(row):
            tech1_area = df[df[ipc_col] == row[ipc_col]]['technology_area'].iloc[0] if len(df[df[ipc_col] == row[ipc_col]]) > 0 else 'Unknown'
            tech2_area = df[df[ipc_col] == row[ipc2_col]]['technology_area'].iloc[0] if len(df[df[ipc_col] == row[ipc2_col]]) > 0 else 'Unknown'
            
            if tech1_area == tech2_area:
                return 'Within-Domain Convergence'
            else:
                return 'Cross-Domain Convergence'
        
        strong_convergence['convergence_type'] = strong_convergence.apply(classify_convergence_type, axis=1)
        
        # Add convergence indicators to main dataframe
        convergence_indicators = strong_convergence.groupby(ipc_col).agg({
            'convergence_strength': 'mean',
            'convergence_type': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown'
        })
        
        df = df.merge(
            convergence_indicators.add_suffix('_convergence'),
            left_on=ipc_col,
            right_index=True,
            how='left'
        )
        
        # Fill missing convergence data
        df['convergence_strength_convergence'] = df['convergence_strength_convergence'].fillna(0)
        df['convergence_type_convergence'] = df['convergence_type_convergence'].fillna('Isolated Technology')
        
        return df
    
    def build_technology_network(self, df: Optional[pd.DataFrame] = None,
                               min_strength: float = 0.05) -> nx.Graph:
        """
        Build technology network graph showing innovation connections.
        
        Args:
            df: DataFrame to analyze (uses self.analyzed_data if None)
            min_strength: Minimum convergence strength for network edges
            
        Returns:
            NetworkX graph with technology network
        """
        if df is None:
            df = self.analyzed_data
        
        if df is None or df.empty:
            raise ValueError("No data available for network building. Run analyze_technology_landscape first.")
        
        # Check if required columns exist (they're added by analyze_technology_landscape)
        required_columns = ['convergence_strength_convergence', 'IPC_2']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}. Run analyze_technology_landscape first.")
        
        logger.debug("🕸️ Building technology network...")
        
        # Filter for technologies with convergence data
        network_data = df[
            (df['convergence_strength_convergence'] >= min_strength) & 
            (df['IPC_2'].notna())
        ].copy()
        
        # Create network graph
        G = nx.Graph()
        
        # Add nodes with technology attributes
        for ipc_code in pd.concat([network_data['IPC_1'], network_data['IPC_2']]).unique():
            tech_data = df[df['IPC_1'] == ipc_code]
            if len(tech_data) > 0:
                node_data = tech_data.iloc[0]
                G.add_node(ipc_code, 
                          technology_area=node_data['technology_area'],
                          maturity=node_data['technology_maturity'],
                          strategic_value=node_data['strategic_value'],
                          emergence_score=node_data.get('emergence_score', 0),
                          innovation_intensity=node_data.get('innovation_intensity_area', 0))
        
        # Add edges with convergence information
        for _, row in network_data.iterrows():
            if G.has_node(row['IPC_1']) and G.has_node(row['IPC_2']):
                G.add_edge(
                    row['IPC_1'], 
                    row['IPC_2'],
                    weight=row['convergence_strength_convergence'],
                    convergence_type=row['convergence_type_convergence'],
                    families=row['family_id'] if 'family_id' in row else 1
                )
        
        # Calculate network metrics
        if G.number_of_nodes() > 0:
            centrality = nx.betweenness_centrality(G)
            clustering = nx.clustering(G)
            degree_centrality = nx.degree_centrality(G)
            
            # Add network metrics as node attributes
            for node in G.nodes():
                G.nodes[node]['centrality'] = centrality.get(node, 0)
                G.nodes[node]['clustering'] = clustering.get(node, 0)
                G.nodes[node]['degree_centrality'] = degree_centrality.get(node, 0)
        
        self.technology_network = G
        logger.debug(f"✅ Technology network built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        return G
    
    def generate_technology_intelligence(self, df: Optional[pd.DataFrame] = None) -> Dict:
        """
        Generate comprehensive technology intelligence report.
        
        Args:
            df: DataFrame to analyze (uses self.analyzed_data if None)
            
        Returns:
            Dictionary with technology intelligence insights
        """
        if df is None:
            df = self.analyzed_data
        
        if df is None:
            raise ValueError("No analyzed data available. Run analyze_technology_landscape first.")
        
        logger.debug("📋 Generating technology intelligence report...")
        
        # Technology area overview
        tech_overview = df.groupby('technology_area').agg({
            'family_id': 'nunique',
            'IPC_1': 'nunique',
            'novelty_score': 'mean',
            'emergence_score': 'mean',
            'cagr': 'first',
            'trend_direction': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown'
        }).round(3)
        
        tech_overview.columns = ['total_families', 'unique_classifications', 'avg_novelty', 
                               'avg_emergence', 'growth_rate', 'trend_direction']
        tech_overview = tech_overview.sort_values('total_families', ascending=False)
        
        # Emerging technologies analysis
        emerging_techs = df[df['emergence_classification'].isin(['Breakthrough Technology', 'Emerging Technology'])]
        emerging_analysis = emerging_techs.groupby('emergence_classification').agg({
            'IPC_1': 'nunique',
            'family_id': 'nunique',
            'emergence_score': 'mean'
        }).round(3)
        
        # Innovation hotspots
        innovation_hotspots = df.groupby('technology_area').agg({
            'cross_domain_innovation': 'sum',
            'innovation_intensity_area': 'first',
            'convergence_strength_convergence': 'mean'
        }).round(3)
        
        innovation_hotspots['innovation_index'] = (
            innovation_hotspots['cross_domain_innovation'] * 0.4 +
            innovation_hotspots['innovation_intensity_area'] * 100 * 0.3 +
            innovation_hotspots['convergence_strength_convergence'] * 100 * 0.3
        ).round(2)
        
        innovation_hotspots = innovation_hotspots.sort_values('innovation_index', ascending=False)
        
        # Technology convergence patterns
        convergence_patterns = df['convergence_type_convergence'].value_counts().to_dict()
        
        # Network analysis (if network exists)
        network_insights = {}
        if self.technology_network and self.technology_network.number_of_nodes() > 0:
            # Find central technologies
            centrality_scores = nx.betweenness_centrality(self.technology_network)
            top_central = sorted(centrality_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            
            # Find technology clusters
            clusters = list(nx.connected_components(self.technology_network))
            
            network_insights = {
                'network_density': float(nx.density(self.technology_network)),
                'total_nodes': self.technology_network.number_of_nodes(),
                'total_edges': self.technology_network.number_of_edges(),
                'central_technologies': [{'technology': tech, 'centrality': score} for tech, score in top_central],
                'technology_clusters': len(clusters),
                'largest_cluster_size': max(len(cluster) for cluster in clusters) if clusters else 0
            }
        
        # Strategic recommendations
        strategic_recommendations = self._generate_technology_recommendations(df)
        
        intelligence_report = {
            'executive_summary': {
                'total_technology_areas': len(tech_overview),
                'dominant_area': tech_overview.index[0] if len(tech_overview) > 0 else 'N/A',
                'fastest_growing': tech_overview[tech_overview['growth_rate'] == tech_overview['growth_rate'].max()].index[0] if len(tech_overview) > 0 else 'N/A',
                'emerging_technologies': len(emerging_techs['IPC_1'].unique()),
                'cross_domain_innovations': int(df['cross_domain_innovation'].sum())
            },
            'technology_landscape': tech_overview.to_dict('index'),
            'emerging_technologies': emerging_analysis.to_dict('index'),
            'innovation_hotspots': innovation_hotspots.to_dict('index'),
            'convergence_patterns': convergence_patterns,
            'network_analysis': network_insights,
            'strategic_recommendations': strategic_recommendations,
            'technology_specific_insights': self._analyze_technology_specific_trends(df)
        }
        
        self.technology_intelligence = intelligence_report
        return intelligence_report
    
    def _generate_technology_recommendations(self, df: pd.DataFrame) -> List[str]:
        """Generate strategic technology recommendations."""
        recommendations = []
        
        # Emerging technology opportunities
        breakthrough_techs = df[df['emergence_classification'] == 'Breakthrough Technology']['technology_area'].unique()
        if len(breakthrough_techs) > 0:
            recommendations.append(f"Investigate breakthrough opportunities in: {', '.join(breakthrough_techs)}")
        
        # High growth areas
        high_growth = df[df['cagr'] > 20]['technology_area'].unique()
        if len(high_growth) > 0:
            recommendations.append(f"Monitor high-growth technology areas: {', '.join(high_growth)}")
        
        # Cross-domain innovation
        cross_domain_areas = df[df['cross_domain_innovation'] == True]['technology_area'].value_counts().head(3)
        if len(cross_domain_areas) > 0:
            recommendations.append(f"Explore cross-domain innovation in: {', '.join(cross_domain_areas.index)}")
        
        # Strategic value opportunities
        critical_areas = df[df['strategic_value'] == 'Critical']['technology_area'].unique()
        if len(critical_areas) > 0:
            recommendations.append(f"Prioritize critical technology areas: {', '.join(critical_areas)}")
        
        return recommendations
    
    def _analyze_technology_specific_trends(self, df: pd.DataFrame) -> Dict:
        """Analyze technology-specific trends and patterns using official classification."""
        tech_trends = {}
        
        # Technology maturity distribution
        maturity_dist = df.drop_duplicates('technology_area')['technology_maturity'].value_counts().to_dict()
        
        # Strategic value distribution
        strategic_dist = df.drop_duplicates('technology_area')['strategic_value'].value_counts().to_dict()
        
        # Technology area evolution
        area_evolution = df.groupby('technology_area').agg({
            'cagr': 'first',
            'trend_direction': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown',
            'lifecycle_stage': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown'
        }).to_dict('index')
        
        tech_trends = {
            'maturity_distribution': maturity_dist,
            'strategic_value_distribution': strategic_dist,
            'technology_evolution': area_evolution,
            'innovation_leaders': df.groupby('technology_area')['innovation_intensity_area'].first().to_dict()
        }
        
        return tech_trends
    
    def identify_innovation_opportunities(self, df: Optional[pd.DataFrame] = None) -> Dict:
        """
        Identify innovation opportunities and white spaces.
        
        Args:
            df: DataFrame to analyze (uses self.analyzed_data if None)
            
        Returns:
            Dictionary with innovation opportunity analysis
        """
        if df is None:
            df = self.analyzed_data
        
        if df is None:
            raise ValueError("No analyzed data available. Run analyze_technology_landscape first.")
        
        logger.debug("🔍 Identifying innovation opportunities...")
        
        opportunities = {
            'emerging_opportunities': {},
            'convergence_opportunities': {},
            'white_space_analysis': {},
            'acceleration_opportunities': {}
        }
        
        # Emerging technology opportunities
        emerging_techs = df[df['emergence_score'] > 0.5]
        if len(emerging_techs) > 0:
            opportunities['emerging_opportunities'] = {
                'high_potential_technologies': emerging_techs.groupby('IPC_1').agg({
                    'emergence_score': 'first',
                    'technology_area': 'first',
                    'family_id': 'nunique'
                }).sort_values('emergence_score', ascending=False).head(10).to_dict('index'),
                'emerging_areas_summary': emerging_techs['technology_area'].value_counts().to_dict()
            }
        
        # Convergence opportunities
        high_convergence = df[df['convergence_strength_convergence'] > 0.3]
        if len(high_convergence) > 0:
            opportunities['convergence_opportunities'] = {
                'high_convergence_technologies': high_convergence.groupby('IPC_1').agg({
                    'convergence_strength_convergence': 'first',
                    'convergence_type_convergence': 'first',
                    'technology_area': 'first'
                }).sort_values('convergence_strength_convergence', ascending=False).head(10).to_dict('index')
            }
        
        # White space analysis (low activity areas with potential)
        tech_activity = df.groupby('technology_area').agg({
            'family_id': 'nunique',
            'strategic_value': 'first'
        })
        
        # Low activity but high strategic value = white space
        white_spaces = tech_activity[
            (tech_activity['family_id'] < tech_activity['family_id'].median()) &
            (tech_activity['strategic_value'].isin(['Critical', 'Strategic']))
        ]
        
        opportunities['white_space_analysis'] = {
            'underexplored_strategic_areas': white_spaces.to_dict('index'),
            'potential_impact': 'High strategic value with low current activity'
        }
        
        # Technology acceleration opportunities
        accelerating_areas = df[df['trend_direction'] == 'Growing']['technology_area'].value_counts()
        opportunities['acceleration_opportunities'] = {
            'growing_technology_areas': accelerating_areas.to_dict(),
            'recommendation': 'Consider increased investment in growing areas'
        }
        
        return opportunities

def create_technology_analyzer() -> TechnologyAnalyzer:
    """
    Factory function to create configured technology analyzer.
    
    Returns:
        Configured TechnologyAnalyzer instance
    """
    return TechnologyAnalyzer()

# Example usage and demo functions
def demo_technology_analysis():
    """Demonstrate technology analysis capabilities."""
    logger.debug("🚀 Technology Analysis Demo")
    
    # Create sample data
    np.random.seed(42)
    sample_data = []
    
    ipc_codes = ['C22B19/28', 'C04B18/04', 'H01M10/54', 'C09K11/01', 'H01J09/52', 'Y02W30/52']
    domains = ['Metallurgy & Extraction', 'Ceramics & Materials', 'Batteries & Energy Storage', 
               'Phosphors & Luminescence', 'Electronic Devices', 'Recycling & Sustainability']
    
    for i in range(100):
        family_id = 100000 + i
        filing_year = np.random.randint(2010, 2023)
        ipc1 = np.random.choice(ipc_codes)
        ipc2 = np.random.choice(ipc_codes)
        domain1 = domains[ipc_codes.index(ipc1)]
        
        # Ensure different IPC codes
        while ipc2 == ipc1:
            ipc2 = np.random.choice(ipc_codes)
        
        domain2 = domains[ipc_codes.index(ipc2)]
        
        sample_data.append({
            'family_id': family_id,
            'filing_year': filing_year,
            'IPC_1': ipc1,
            'IPC_2': ipc2,
            'domain_1': domain1,
            'domain_2': domain2,
            'country_name': np.random.choice(['China', 'US', 'Japan', 'Germany'])
        })
    
    df = pd.DataFrame(sample_data)
    
    # Analyze technology landscape
    analyzer = create_technology_analyzer()
    analyzed_df = analyzer.analyze_technology_landscape(df)
    
    # Build network
    network = analyzer.build_technology_network(analyzed_df)
    
    # Generate intelligence
    intelligence = analyzer.generate_technology_intelligence()
    opportunities = analyzer.identify_innovation_opportunities()
    
    logger.debug("✅ Demo analysis complete")
    logger.debug(f"🔬 Dominant technology area: {intelligence['executive_summary']['dominant_area']}")
    logger.debug(f"🕸️ Network nodes: {network.number_of_nodes()}")
    
    return analyzer, analyzed_df, intelligence

if __name__ == "__main__":
    demo_technology_analysis()