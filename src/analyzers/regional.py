"""
Regional Intelligence Analyzer - Clean Implementation
Demonstrates clean architecture with standardized intelligence analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
import time

from .base import BaseAnalyzer, AnalyzerResult


class RegionalAnalyzer(BaseAnalyzer):
    """
    Clean implementation of regional competitive intelligence analysis
    Demonstrates standardized interface and intelligence generation
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize regional analyzer
        
        Args:
            config: Regional analysis configuration
        """
        super().__init__(config)
        
        # Default configuration
        self.default_config = {
            'min_region_significance': 0.05,  # 5% minimum market share
            'competitiveness_threshold': 0.1,  # 10% for competitive assessment
            'geographic_aggregation': 'country',  # 'country' or 'region'
            'include_strategic_analysis': True,
            'market_concentration_method': 'hhi'  # Herfindahl-Hirschman Index
        }
        
        # Merge with provided config
        self.regional_config = {**self.default_config, **self.config}
    
    def analyze(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], 
               analysis_params: Dict[str, Any] = None, **kwargs) -> AnalyzerResult:
        """
        Perform regional competitive intelligence analysis
        
        Args:
            data: Input data (DataFrame or dict of processor results)
            analysis_params: Regional analysis parameters
            **kwargs: Additional analysis options
            
        Returns:
            AnalyzerResult with regional intelligence and insights
        """
        start_time = time.time()
        
        if analysis_params is None:
            analysis_params = {}
        
        self.logger.info("🌍 Starting regional competitive intelligence analysis")
        
        try:
            # Input validation
            if not self._validate_input(data):
                raise ValueError("Invalid input data for regional analysis")
            
            # Extract and prepare data
            if isinstance(data, dict):
                analysis_data = self._extract_data_from_processors(data, ['search', 'applicant', 'geographic'])
            else:
                analysis_data = data.copy()
            
            if analysis_data.empty:
                raise ValueError("No data available for regional analysis")
            
            # Perform regional analysis
            regional_intelligence = self._perform_regional_analysis(analysis_data, analysis_params)
            
            # Generate strategic insights
            strategic_insights = self._generate_strategic_insights(regional_intelligence, analysis_data)
            
            analysis_time = time.time() - start_time
            
            # Create metadata
            metadata = self._create_metadata(
                input_data=data,
                analysis_time=analysis_time,
                analysis_params=analysis_params,
                additional_metadata={
                    'regions_analyzed': len(regional_intelligence.get('regional_breakdown', {})),
                    'market_concentration': regional_intelligence.get('market_metrics', {}).get('concentration_hhi', 0),
                    'dominant_region': regional_intelligence.get('executive_summary', {}).get('market_leader', 'Unknown')
                }
            )
            
            # Update performance metrics
            self._update_performance_metrics(analysis_time, len(analysis_data))
            
            self.logger.info(f"✅ Regional analysis completed in {analysis_time:.2f}s")
            
            return AnalyzerResult(
                intelligence=regional_intelligence,
                insights=strategic_insights,
                metadata=metadata,
                status="completed"
            )
            
        except Exception as e:
            analysis_time = time.time() - start_time
            error_msg = f"Regional analysis failed: {str(e)}"
            self.logger.error(error_msg)
            
            return AnalyzerResult(
                intelligence={},
                insights=[],
                metadata=self._create_metadata(data, analysis_time, analysis_params, {'error': str(e)}),
                status="failed",
                warnings=[error_msg]
            )
    
    def _perform_regional_analysis(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive regional competitive analysis
        
        Args:
            data: Prepared analysis data
            params: Analysis parameters
            
        Returns:
            Complete regional intelligence structure
        """
        # Determine geographic column
        geo_column = self._identify_geographic_column(data)
        if not geo_column:
            raise ValueError("No geographic data found for regional analysis")
        
        # Regional breakdown analysis
        regional_breakdown = self._analyze_regional_breakdown(data, geo_column)
        
        # Market concentration analysis
        market_metrics = self._calculate_market_concentration(regional_breakdown)
        
        # Competitive landscape analysis
        competitive_landscape = self._analyze_competitive_landscape(data, geo_column, regional_breakdown)
        
        # Strategic opportunities analysis
        strategic_opportunities = self._identify_strategic_opportunities(regional_breakdown, market_metrics)
        
        # Executive summary
        executive_summary = self._generate_regional_executive_summary(
            regional_breakdown, market_metrics, competitive_landscape
        )
        
        return {
            'executive_summary': executive_summary,
            'regional_breakdown': regional_breakdown,
            'market_metrics': market_metrics,
            'competitive_landscape': competitive_landscape,
            'strategic_opportunities': strategic_opportunities,
            'analysis_scope': {
                'total_records': len(data),
                'geographic_coverage': len(regional_breakdown),
                'time_period': self._extract_time_period(data),
                'data_quality_score': self._assess_data_quality(data, geo_column)
            }
        }
    
    def _identify_geographic_column(self, data: pd.DataFrame) -> Optional[str]:
        """Identify the best geographic column for analysis"""
        geo_columns = ['person_ctry_code', 'country_code', 'region', 'country_name', 'country']
        
        for col in geo_columns:
            if col in data.columns and not data[col].isna().all():
                self.logger.info(f"Using geographic column: {col}")
                return col
        
        return None
    
    def _analyze_regional_breakdown(self, data: pd.DataFrame, geo_column: str) -> Dict[str, Dict[str, Any]]:
        """Analyze patent activity by region"""
        regional_stats = {}
        
        # Group by geographic region
        regional_groups = data.groupby(geo_column).agg({
            'docdb_family_id': 'nunique',
            'person_id': 'nunique',
            'appln_filing_year': ['min', 'max', 'count']
        }).fillna(0)
        
        total_families = data['docdb_family_id'].nunique()
        total_applicants = data['person_id'].nunique() if 'person_id' in data.columns else 0
        
        for region in regional_groups.index:
            region_data = regional_groups.loc[region]
            
            patent_families = region_data[('docdb_family_id', 'nunique')]
            applicants = region_data[('person_id', 'nunique')] if ('person_id', 'nunique') in region_data.index else 0
            applications = region_data[('appln_filing_year', 'count')]
            
            market_share = patent_families / total_families if total_families > 0 else 0
            
            regional_stats[region] = {
                'patent_families': int(patent_families),
                'unique_applicants': int(applicants),
                'total_applications': int(applications),
                'market_share': float(market_share),
                'market_share_pct': float(market_share * 100),
                'innovation_intensity': float(patent_families / applicants) if applicants > 0 else 0,
                'filing_period': {
                    'earliest': int(region_data[('appln_filing_year', 'min')]) if not pd.isna(region_data[('appln_filing_year', 'min')]) else None,
                    'latest': int(region_data[('appln_filing_year', 'max')]) if not pd.isna(region_data[('appln_filing_year', 'max')]) else None
                }
            }
        
        # Sort by market share
        return dict(sorted(regional_stats.items(), key=lambda x: x[1]['market_share'], reverse=True))
    
    def _calculate_market_concentration(self, regional_breakdown: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate market concentration metrics"""
        market_shares = [region['market_share'] for region in regional_breakdown.values()]
        
        # Herfindahl-Hirschman Index
        hhi = sum(share ** 2 for share in market_shares)
        
        # Market concentration classification
        if hhi < 0.1:
            concentration_level = "Low (Highly Competitive)"
        elif hhi < 0.18:
            concentration_level = "Moderate"
        else:
            concentration_level = "High (Concentrated)"
        
        # Top region concentration
        top_3_share = sum(sorted(market_shares, reverse=True)[:3])
        top_5_share = sum(sorted(market_shares, reverse=True)[:5])
        
        return {
            'concentration_hhi': float(hhi),
            'concentration_level': concentration_level,
            'top_3_market_share': float(top_3_share),
            'top_5_market_share': float(top_5_share),
            'number_of_regions': len(regional_breakdown),
            'effective_competitors': int(1 / hhi) if hhi > 0 else len(regional_breakdown)
        }
    
    def _analyze_competitive_landscape(self, data: pd.DataFrame, geo_column: str, 
                                     regional_breakdown: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze competitive dynamics between regions"""
        
        # Competitive tiers
        market_shares = [(region, stats['market_share']) for region, stats in regional_breakdown.items()]
        market_shares.sort(key=lambda x: x[1], reverse=True)
        
        competitive_tiers = {
            'tier_1': [region for region, share in market_shares[:2] if share >= 0.15],
            'tier_2': [region for region, share in market_shares if 0.05 <= share < 0.15],
            'tier_3': [region for region, share in market_shares if share < 0.05]
        }
        
        # Regional growth patterns
        growth_analysis = self._analyze_regional_growth(data, geo_column)
        
        # Innovation hubs identification
        innovation_hubs = self._identify_innovation_hubs(regional_breakdown)
        
        return {
            'competitive_tiers': competitive_tiers,
            'market_leaders': market_shares[:3],
            'emerging_regions': growth_analysis.get('high_growth_regions', []),
            'innovation_hubs': innovation_hubs,
            'competitive_intensity': self._calculate_competitive_intensity(regional_breakdown),
            'regional_specialization': self._analyze_regional_specialization(data, geo_column)
        }
    
    def _analyze_regional_growth(self, data: pd.DataFrame, geo_column: str) -> Dict[str, Any]:
        """Analyze growth patterns by region"""
        if 'appln_filing_year' not in data.columns:
            return {'high_growth_regions': [], 'growth_patterns': {}}
        
        # Calculate year-over-year growth for each region
        recent_years = data['appln_filing_year'].max() - 2
        recent_data = data[data['appln_filing_year'] >= recent_years]
        
        growth_regions = []
        for region in data[geo_column].unique():
            region_data = recent_data[recent_data[geo_column] == region]
            if len(region_data) >= 5:  # Minimum data for growth analysis
                yearly_counts = region_data.groupby('appln_filing_year')['docdb_family_id'].nunique()
                if len(yearly_counts) >= 2:
                    growth_rate = (yearly_counts.iloc[-1] - yearly_counts.iloc[0]) / yearly_counts.iloc[0] if yearly_counts.iloc[0] > 0 else 0
                    if growth_rate > 0.2:  # 20% growth threshold
                        growth_regions.append((region, growth_rate))
        
        return {
            'high_growth_regions': [region for region, _ in sorted(growth_regions, key=lambda x: x[1], reverse=True)[:5]],
            'growth_patterns': dict(growth_regions)
        }
    
    def _identify_innovation_hubs(self, regional_breakdown: Dict[str, Dict[str, Any]]) -> List[str]:
        """Identify regions with high innovation intensity"""
        innovation_threshold = np.mean([stats['innovation_intensity'] for stats in regional_breakdown.values()])
        
        innovation_hubs = [
            region for region, stats in regional_breakdown.items()
            if stats['innovation_intensity'] > innovation_threshold and stats['market_share'] > 0.05
        ]
        
        return sorted(innovation_hubs, key=lambda r: regional_breakdown[r]['innovation_intensity'], reverse=True)
    
    def _calculate_competitive_intensity(self, regional_breakdown: Dict[str, Dict[str, Any]]) -> str:
        """Calculate overall competitive intensity"""
        num_significant_regions = sum(1 for stats in regional_breakdown.values() if stats['market_share'] > 0.05)
        
        if num_significant_regions >= 5:
            return "Very High"
        elif num_significant_regions >= 3:
            return "High"
        elif num_significant_regions >= 2:
            return "Moderate"
        else:
            return "Low"
    
    def _analyze_regional_specialization(self, data: pd.DataFrame, geo_column: str) -> Dict[str, List[str]]:
        """Analyze technology specialization by region"""
        if 'ipc_class_symbol' not in data.columns:
            return {}
        
        specialization = {}
        for region in data[geo_column].unique():
            region_data = data[data[geo_column] == region]
            tech_counts = region_data['ipc_class_symbol'].value_counts()
            top_techs = tech_counts.head(3).index.tolist()
            specialization[region] = top_techs
        
        return specialization
    
    def _identify_strategic_opportunities(self, regional_breakdown: Dict[str, Dict[str, Any]], 
                                        market_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Identify strategic opportunities based on regional analysis"""
        opportunities = {
            'market_entry': [],
            'partnership_targets': [],
            'competitive_threats': [],
            'expansion_opportunities': []
        }
        
        # Market entry opportunities (low market share regions with growth potential)
        for region, stats in regional_breakdown.items():
            if stats['market_share'] < 0.1 and stats['unique_applicants'] > 5:
                opportunities['market_entry'].append({
                    'region': region,
                    'current_share': stats['market_share_pct'],
                    'reason': 'Underrepresented market with active innovation'
                })
        
        # Partnership targets (high innovation intensity regions)
        innovation_leaders = sorted(
            regional_breakdown.items(), 
            key=lambda x: x[1]['innovation_intensity'], 
            reverse=True
        )[:3]
        
        for region, stats in innovation_leaders:
            if stats['innovation_intensity'] > 2.0:  # High innovation threshold
                opportunities['partnership_targets'].append({
                    'region': region,
                    'innovation_intensity': stats['innovation_intensity'],
                    'reason': 'High innovation density indicates strong R&D capabilities'
                })
        
        return opportunities
    
    def _generate_regional_executive_summary(self, regional_breakdown: Dict[str, Dict[str, Any]], 
                                           market_metrics: Dict[str, Any], 
                                           competitive_landscape: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary for regional analysis"""
        
        # Market leader
        market_leader = list(regional_breakdown.keys())[0] if regional_breakdown else "Unknown"
        leader_share = list(regional_breakdown.values())[0]['market_share_pct'] if regional_breakdown else 0
        
        # Key insights
        total_regions = len(regional_breakdown)
        significant_regions = sum(1 for stats in regional_breakdown.values() if stats['market_share'] > 0.05)
        
        return {
            'market_leader': market_leader,
            'leader_market_share': float(leader_share),
            'market_concentration': market_metrics['concentration_level'],
            'geographic_diversity': f"{significant_regions} significant regions out of {total_regions} total",
            'competitive_intensity': competitive_landscape['competitive_intensity'],
            'key_finding': f"{market_leader} dominates with {leader_share:.1f}% market share in a {market_metrics['concentration_level'].lower()} market",
            'strategic_priority': self._determine_strategic_priority(market_metrics, competitive_landscape)
        }
    
    def _determine_strategic_priority(self, market_metrics: Dict[str, Any], 
                                    competitive_landscape: Dict[str, Any]) -> str:
        """Determine strategic priority based on analysis"""
        concentration = market_metrics['concentration_hhi']
        intensity = competitive_landscape['competitive_intensity']
        
        if concentration > 0.18 and intensity == "High":
            return "Market defense - Maintain position in concentrated market"
        elif concentration < 0.1 and intensity == "Very High":
            return "Differentiation - Stand out in fragmented market"
        elif len(competitive_landscape['emerging_regions']) > 2:
            return "Geographic expansion - Capitalize on emerging markets"
        else:
            return "Market development - Strengthen competitive position"
    
    def _generate_strategic_insights(self, intelligence: Dict[str, Any], data: pd.DataFrame) -> List[str]:
        """Generate strategic insights from regional analysis"""
        insights = []
        
        exec_summary = intelligence.get('executive_summary', {})
        market_metrics = intelligence.get('market_metrics', {})
        competitive_landscape = intelligence.get('competitive_landscape', {})
        opportunities = intelligence.get('strategic_opportunities', {})
        
        # Market leadership insight
        if exec_summary.get('market_leader'):
            insights.append(
                f"{exec_summary['market_leader']} leads the market with "
                f"{exec_summary['leader_market_share']:.1f}% share, indicating "
                f"{'strong dominance' if exec_summary['leader_market_share'] > 30 else 'competitive leadership'}"
            )
        
        # Market concentration insight
        concentration_level = market_metrics.get('concentration_level', '')
        if 'High' in concentration_level:
            insights.append(
                "Market shows high concentration - focus on differentiation and niche strategies"
            )
        elif 'Low' in concentration_level:
            insights.append(
                "Highly fragmented market presents opportunities for consolidation and market share gains"
            )
        
        # Geographic diversity insight
        if market_metrics.get('number_of_regions', 0) > 5:
            insights.append(
                f"Innovation activity spans {market_metrics['number_of_regions']} regions, "
                "indicating global technology relevance"
            )
        
        # Competitive intensity insight
        intensity = competitive_landscape.get('competitive_intensity', '')
        if intensity in ['High', 'Very High']:
            insights.append(
                f"Competition intensity is {intensity.lower()}, requiring strong IP strategy and rapid innovation"
            )
        
        # Emerging markets insight
        emerging_regions = competitive_landscape.get('emerging_regions', [])
        if emerging_regions:
            insights.append(
                f"High-growth regions ({', '.join(emerging_regions[:3])}) present expansion opportunities"
            )
        
        # Innovation hubs insight
        innovation_hubs = competitive_landscape.get('innovation_hubs', [])
        if innovation_hubs:
            insights.append(
                f"Innovation concentrated in {', '.join(innovation_hubs[:3])}, "
                "suggesting optimal locations for R&D partnerships"
            )
        
        return insights
    
    def _extract_time_period(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Extract time period information from data"""
        if 'appln_filing_year' in data.columns:
            return {
                'start_year': int(data['appln_filing_year'].min()),
                'end_year': int(data['appln_filing_year'].max()),
                'span_years': int(data['appln_filing_year'].max() - data['appln_filing_year'].min())
            }
        return {'start_year': None, 'end_year': None, 'span_years': 0}
    
    def _assess_data_quality(self, data: pd.DataFrame, geo_column: str) -> float:
        """Assess quality of geographic data"""
        total_records = len(data)
        valid_geo = data[geo_column].notna().sum()
        completeness = valid_geo / total_records if total_records > 0 else 0
        
        # Convert to 0-100 scale
        return float(completeness * 100)