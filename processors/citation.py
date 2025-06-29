"""
Citation Analysis Processor for Patent Intelligence
Enhanced from EPO PATLIB 2025 Live Demo Code

This module processes search results from PatentSearchProcessor to analyze citation patterns,
impact metrics, and innovation networks. Works with PATSTAT data to extract citation intelligence.
"""

import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Tuple, Union, Set
from collections import defaultdict
import logging
from datetime import datetime

# Import exception classes
from . import PatstatConnectionError, DataNotFoundError, InvalidQueryError

# Import PATSTAT client and models for citation data enrichment
try:
    from epo.tipdata.patstat import PatstatClient
    from epo.tipdata.patstat.database.models import (
        TLS201_APPLN, TLS212_CITATION
    )
    from sqlalchemy import func, and_, distinct
    PATSTAT_AVAILABLE = True
except ImportError:
    PATSTAT_AVAILABLE = False
    logging.warning("PATSTAT integration not available")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CitationAnalyzer:
    """
    Citation analyzer that works with PatentSearchProcessor results.
    
    Takes patent family search results and enriches them with citation data from PATSTAT,
    then performs comprehensive citation impact and network analysis.
    """
    
    # Citation quality categories for assessment
    CITATION_QUALITY_CATEGORIES = {
        'high_impact': 'Citations indicating significant technological influence',
        'moderate_impact': 'Citations showing moderate technological relevance',
        'low_impact': 'Citations with limited technological significance',
        'self_citation': 'Citations within same applicant/inventor group',
        'examiner_citation': 'Citations added by patent examiners',
        'applicant_citation': 'Citations provided by patent applicants'
    }
    
    # Citation aging patterns for temporal analysis
    CITATION_AGING_PERIODS = {
        'immediate': (0, 2),      # 0-2 years after publication
        'early': (3, 5),         # 3-5 years after publication  
        'mature': (6, 10),       # 6-10 years after publication
        'legacy': (11, float('inf'))  # 11+ years after publication
    }
    
    def __init__(self, patstat_client: Optional[object] = None):
        """
        Initialize citation analyzer.
        
        Args:
            patstat_client: PATSTAT client instance for data enrichment
        """
        self.patstat_client = patstat_client
        self.session = None
        self.models = {}
        self.sql_funcs = {}
        self.analyzed_data = None
        self.citation_data = None
        self.citation_network = None
        self.citation_intelligence = None
        self.citation_config = self._load_citation_config()
        
        # Initialize PATSTAT connection
        if PATSTAT_AVAILABLE and self.patstat_client is None:
            try:
                # Import our working PatstatClient
                from data_access import PatstatClient as DataAccessPatstatClient
                self.patstat_client = DataAccessPatstatClient(environment='PROD')
                logger.debug("✅ Connected to PATSTAT for citation data enrichment")
            except Exception as e:
                logger.error(f"❌ Failed to connect to PATSTAT: {e}")
                self.patstat_client = None
        
        if self.patstat_client:
            try:
                # Use the db session from our PatstatClient
                if hasattr(self.patstat_client, 'db') and self.patstat_client.db is not None:
                    self.session = self.patstat_client.db
                    # Get models and SQL functions from our client
                    if hasattr(self.patstat_client, 'models'):
                        self.models = self.patstat_client.models
                    if hasattr(self.patstat_client, 'sql_funcs'):
                        self.sql_funcs = self.patstat_client.sql_funcs
                    logger.debug("✅ PATSTAT session initialized for citation analysis")
                elif hasattr(self.patstat_client, 'orm') and callable(self.patstat_client.orm):
                    # Fallback to EPO PatstatClient orm method
                    self.session = self.patstat_client.db
                    # Import models directly for fallback
                    try:
                        from epo.tipdata.patstat.database.models import (
                            TLS201_APPLN, TLS228_DOCDB_FAM_CITN, TLS212_CITATION, TLS215_CITN_CATEG
                        )
                        from sqlalchemy import func, and_, or_
                        self.models = {
                            'TLS201_APPLN': TLS201_APPLN,
                            'TLS228_DOCDB_FAM_CITN': TLS228_DOCDB_FAM_CITN,
                            'TLS212_CITATION': TLS212_CITATION,
                            'TLS215_CITN_CATEG': TLS215_CITN_CATEG
                        }
                        self.sql_funcs = {'func': func, 'and_': and_, 'or_': or_}
                    except ImportError:
                        logger.warning("⚠️ Could not import PATSTAT models for fallback")
                    logger.debug("✅ PATSTAT session initialized for citation analysis (via orm)")
                else:
                    logger.error("❌ No valid PATSTAT session method found")
                    self.session = None
            except Exception as e:
                logger.error(f"❌ Failed to initialize PATSTAT session: {e}")
        
    def analyze_search_results(self, search_results: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze patent search results to extract citation intelligence.
        
        Args:
            search_results: DataFrame from PatentSearchProcessor with columns:
                           ['docdb_family_id', 'quality_score', 'match_type', 'earliest_filing_year', etc.]
                           
        Returns:
            Enhanced DataFrame with citation intelligence
        """
        logger.debug(f"🔗 Starting citation analysis of {len(search_results)} patent families...")
        
        if search_results.empty:
            logger.warning("⚠️ No search results to analyze")
            return pd.DataFrame()
        
        # Step 1: Enrich search results with citation data from PATSTAT
        logger.debug("📊 Step 1: Enriching with citation data from PATSTAT...")
        citation_data = self._enrich_with_citation_data(search_results)
        
        if citation_data.empty:
            logger.warning("⚠️ No citation data found for the search results")
            return pd.DataFrame()
        
        # Step 2: Analyze citation impact patterns
        logger.debug("🎯 Step 2: Analyzing citation impact patterns...")
        impact_analysis = self._analyze_citation_impact(citation_data)
        
        # Step 3: Build citation networks and calculate centrality
        logger.debug("🕸️ Step 3: Building citation networks...")
        network_analysis = self._build_citation_networks(citation_data)
        
        # Step 4: Advanced citation analysis (NEW - based on training insights)
        logger.debug("📊 Step 4a: Advanced citation quality analysis...")
        quality_analysis = self._analyze_citation_categories(citation_data)
        
        logger.debug("📅 Step 4b: Temporal trend analysis...")
        temporal_trends = self._analyze_temporal_citation_trends(citation_data)
        
        # Step 5: Generate citation intelligence insights
        logger.debug("💡 Step 5: Generating citation intelligence insights...")
        intelligence_analysis = self._generate_citation_intelligence(impact_analysis, network_analysis)
        
        self.analyzed_data = intelligence_analysis
        self.citation_data = citation_data
        self.quality_analysis = quality_analysis
        self.temporal_trends = temporal_trends
        
        logger.debug(f"✅ Citation analysis complete: {len(intelligence_analysis)} citation patterns analyzed")
        
        return intelligence_analysis
    
    def _enrich_with_citation_data(self, search_results: pd.DataFrame) -> pd.DataFrame:
        """
        Enrich search results with citation data from PATSTAT using optimized batching.
        
        Uses TLS212_CITATION table to get forward and backward citations.
        Performance optimized for large datasets (30K+ families).
        """
        if not self.session:
            logger.error("❌ No PATSTAT session available for citation enrichment")
            return pd.DataFrame()
        
        family_ids = search_results['docdb_family_id'].tolist()
        logger.debug(f"   Enriching {len(family_ids)} families with citation data...")
        
        try:
            # Get models from the client
            TLS201_APPLN = self.models.get('TLS201_APPLN')
            TLS212_CITATION = self.models.get('TLS212_CITATION')
            TLS228_DOCDB_FAM_CITN = self.models.get('TLS228_DOCDB_FAM_CITN')
            
            if not TLS201_APPLN:
                logger.error("❌ TLS201_APPLN model not available")
                return pd.DataFrame()
            
            # Try family-level citations first (TLS228) as it's more efficient
            if TLS228_DOCDB_FAM_CITN:
                logger.debug("   📈 Querying family-level citations (TLS228)...")
                return self._get_family_level_citations(family_ids, TLS228_DOCDB_FAM_CITN, TLS201_APPLN)
            
            # Fallback to application-level citations (TLS212) with batching
            if not TLS212_CITATION:
                logger.error("❌ No citation tables available")
                return pd.DataFrame()
            
            logger.debug("   📈 Using optimized batched citation queries...")
            return self._get_application_level_citations_optimized(family_ids, TLS212_CITATION, TLS201_APPLN, search_results)
            
        except Exception as e:
            logger.error(f"❌ Failed to enrich with citation data: {e}")
            return pd.DataFrame()
    
    def _get_application_level_citations_optimized(self, family_ids: List[int], 
                                                 TLS212_CITATION, TLS201_APPLN, 
                                                 search_results: pd.DataFrame) -> pd.DataFrame:
        """Optimized application-level citation processing with batching."""
        
        # Batch size optimization for database performance
        batch_size = min(1000, len(family_ids) // 10 + 100)  # Adaptive batching
        logger.debug(f"   Using batch size: {batch_size} for {len(family_ids)} families")
        
        all_citation_records = []
        
        # Process in batches to avoid database timeouts and memory issues
        for i in range(0, len(family_ids), batch_size):
            batch_families = family_ids[i:i+batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(family_ids) + batch_size - 1) // batch_size
            
            logger.debug(f"   Processing batch {batch_num}/{total_batches} ({len(batch_families)} families)...")
            
            try:
                # Single optimized query combining forward and backward citations with joins
                combined_query = self.session.query(
                    TLS201_APPLN.docdb_family_id.label('source_family_id'),
                    TLS201_APPLN.earliest_filing_year.label('source_year'),
                    TLS212_CITATION.cited_appln_id,
                    TLS212_CITATION.citing_appln_id, 
                    TLS212_CITATION.citn_gener_auth,
                    TLS212_CITATION.citn_origin,
                    # Join to get target family info in single query
                    TLS201_APPLN.docdb_family_id.label('target_family_id'),
                    TLS201_APPLN.earliest_filing_year.label('target_year')
                ).select_from(TLS201_APPLN)\
                .join(TLS212_CITATION, 
                     (TLS201_APPLN.appln_id == TLS212_CITATION.cited_appln_id) |
                     (TLS201_APPLN.appln_id == TLS212_CITATION.citing_appln_id))\
                .filter(TLS201_APPLN.docdb_family_id.in_(batch_families))
                
                batch_results = combined_query.limit(50000).all()  # Reasonable limit per batch
                
                # Process batch results efficiently
                for result in batch_results:
                    source_family = result[0]
                    source_year = result[1] 
                    cited_appln = result[2]
                    citing_appln = result[3]
                    auth = result[4]
                    category = result[5]
                    
                    # Determine citation direction and create record
                    if source_family in batch_families:
                        # This is a citation involving our target families
                        all_citation_records.append({
                            'cited_family_id': source_family if cited_appln else None,
                            'citing_family_id': source_family if citing_appln else None,
                            'cited_year': source_year,
                            'citing_year': source_year,
                            'citation_direction': 'forward' if cited_appln else 'backward',
                            'citation_authority': auth,
                            'citation_category': category
                        })
                        
            except Exception as e:
                logger.warning(f"   ⚠️ Batch {batch_num} failed, trying simplified query: {e}")
                # Fallback to simpler batch processing
                batch_records = self._process_batch_simple(batch_families, TLS212_CITATION, TLS201_APPLN)
                all_citation_records.extend(batch_records)
        
        if not all_citation_records:
            logger.warning(f"⚠️ No citation data found for {len(family_ids)} families")
            return pd.DataFrame()
        
        # Convert to DataFrame efficiently
        citation_df = pd.DataFrame(all_citation_records)
        
        # Clean invalid records
        citation_df = citation_df.dropna(subset=['cited_family_id', 'citing_family_id'])
        
        # Merge with search results efficiently using pandas operations
        enriched_data = self._merge_citation_data_optimized(search_results, citation_df)
        
        logger.debug(f"   ✅ Processed {len(all_citation_records)} citation records")
        logger.debug(f"   🏢 Final enriched dataset: {len(enriched_data)} relationships")
        
        return enriched_data
    
    def _process_batch_simple(self, batch_families: List[int], TLS212_CITATION, TLS201_APPLN) -> List[Dict]:
        """Simplified batch processing fallback."""
        records = []
        
        # Simple forward citations query
        forward_query = self.session.query(
            TLS201_APPLN.docdb_family_id,
            TLS201_APPLN.earliest_filing_year,
            TLS212_CITATION.citn_gener_auth
        ).select_from(TLS201_APPLN)\
        .join(TLS212_CITATION, TLS201_APPLN.appln_id == TLS212_CITATION.cited_appln_id)\
        .filter(TLS201_APPLN.docdb_family_id.in_(batch_families))\
        .limit(10000)
        
        for result in forward_query.all():
            records.append({
                'cited_family_id': result[0],
                'citing_family_id': result[0],  # Simplified
                'cited_year': result[1],
                'citing_year': result[1],
                'citation_direction': 'forward',
                'citation_authority': result[2],
                'citation_category': 'simplified'
            })
        
        return records
    
    def _merge_citation_data_optimized(self, search_results: pd.DataFrame, citation_df: pd.DataFrame) -> pd.DataFrame:
        """Optimized merge operation avoiding deprecated append()."""
        
        # Forward citations merge
        forward_merge = search_results.merge(
            citation_df[citation_df['citation_direction'] == 'forward'],
            left_on='docdb_family_id',
            right_on='cited_family_id',
            how='inner',
            suffixes=('', '_forward')
        )
        
        # Backward citations merge
        backward_merge = search_results.merge(
            citation_df[citation_df['citation_direction'] == 'backward'],
            left_on='docdb_family_id',
            right_on='citing_family_id',
            how='inner',
            suffixes=('', '_backward')
        )
        
        # Combine using concat instead of deprecated append
        if not forward_merge.empty and not backward_merge.empty:
            enriched_data = pd.concat([forward_merge, backward_merge], ignore_index=True)
        elif not forward_merge.empty:
            enriched_data = forward_merge
        elif not backward_merge.empty:
            enriched_data = backward_merge
        else:
            enriched_data = pd.DataFrame()
        
        # Remove duplicates efficiently
        if not enriched_data.empty:
            enriched_data = enriched_data.drop_duplicates().reset_index(drop=True)
            enriched_data = self._clean_citation_data_patstat(enriched_data)
        
        return enriched_data
    
    def _load_citation_config(self) -> Dict:
        """Load citation analysis configuration from YAML."""
        try:
            import yaml
            with open('config/search_patterns_config.yaml', 'r') as f:
                config = yaml.safe_load(f)
                return config.get('citation_analysis', {})
        except Exception as e:
            logger.warning(f"⚠️ Could not load citation config: {e}")
            return {}
    
    def _get_family_level_citations(self, family_ids: List[int], TLS228_DOCDB_FAM_CITN, TLS201_APPLN) -> pd.DataFrame:
        """Optimized family-level citation processing with batching."""
        
        batch_size = min(2000, len(family_ids) // 5 + 200)  # Larger batches for family-level
        logger.debug(f"   Using family-level batch size: {batch_size} for {len(family_ids)} families")
        
        all_citation_records = []
        year_cache = {}  # Cache to avoid repeated year lookups
        
        try:
            # Process in optimized batches
            for i in range(0, len(family_ids), batch_size):
                batch_families = family_ids[i:i+batch_size]
                batch_num = i // batch_size + 1
                total_batches = (len(family_ids) + batch_size - 1) // batch_size
                
                logger.debug(f"   Family batch {batch_num}/{total_batches} ({len(batch_families)} families)...")
                
                # Forward citations for this batch
                forward_query = self.session.query(
                    TLS228_DOCDB_FAM_CITN.cited_docdb_family_id.label('cited_family_id'),
                    TLS228_DOCDB_FAM_CITN.docdb_family_id.label('citing_family_id')
                ).filter(
                    TLS228_DOCDB_FAM_CITN.cited_docdb_family_id.in_(batch_families)
                ).limit(10000)  # Reasonable limit per batch
                
                forward_results = forward_query.all()
                
                # Backward citations for this batch 
                backward_query = self.session.query(
                    TLS228_DOCDB_FAM_CITN.cited_docdb_family_id.label('cited_family_id'),
                    TLS228_DOCDB_FAM_CITN.docdb_family_id.label('citing_family_id')
                ).filter(
                    TLS228_DOCDB_FAM_CITN.docdb_family_id.in_(batch_families)
                ).limit(10000)  # Reasonable limit per batch
                
                backward_results = backward_query.all()
                
                # Process forward citations efficiently
                for result in forward_results:
                    all_citation_records.append({
                        'cited_family_id': result.cited_family_id,
                        'citing_family_id': result.citing_family_id,
                        'citation_direction': 'forward',
                        'citation_authority': 'X',  # Default for family-level
                        'citation_category': 'family_level'
                    })
                
                # Process backward citations efficiently
                for result in backward_results:
                    all_citation_records.append({
                        'cited_family_id': result.cited_family_id,
                        'citing_family_id': result.citing_family_id,
                        'citation_direction': 'backward',
                        'citation_authority': 'X',  # Default for family-level
                        'citation_category': 'family_level'
                    })
            
            if not all_citation_records:
                logger.warning(f"⚠️ No family-level citations found for {len(family_ids)} families")
                return pd.DataFrame()
            
            citation_df = pd.DataFrame(all_citation_records)
            
            # Optimized year lookup with batching
            all_family_ids = list(set(citation_df['cited_family_id'].tolist() + citation_df['citing_family_id'].tolist()))
            
            if all_family_ids:
                # Batch year lookups to avoid memory issues
                year_batch_size = 5000
                for i in range(0, len(all_family_ids), year_batch_size):
                    year_batch = all_family_ids[i:i+year_batch_size]
                    
                    year_query = self.session.query(
                        TLS201_APPLN.docdb_family_id,
                        TLS201_APPLN.earliest_filing_year
                    ).filter(
                        TLS201_APPLN.docdb_family_id.in_(year_batch)
                    ).distinct()
                    
                    for result in year_query.all():
                        year_cache[result.docdb_family_id] = result.earliest_filing_year
                
                # Map years efficiently
                citation_df['cited_year'] = citation_df['cited_family_id'].map(year_cache)
                citation_df['citing_year'] = citation_df['citing_family_id'].map(year_cache)
            
            # Create base search results
            search_results_base = pd.DataFrame({
                'docdb_family_id': family_ids,
                'quality_score': [3.0] * len(family_ids),
                'match_type': ['family_citation'] * len(family_ids),
                'earliest_filing_year': [year_cache.get(fid, 2018) for fid in family_ids]
            })
            
            # Optimized merge operations
            forward_data = search_results_base.merge(
                citation_df[citation_df['citation_direction'] == 'forward'],
                left_on='docdb_family_id',
                right_on='cited_family_id',
                how='inner'
            )
            
            backward_data = search_results_base.merge(
                citation_df[citation_df['citation_direction'] == 'backward'],
                left_on='docdb_family_id',
                right_on='citing_family_id',
                how='inner'
            )
            
            # Combine efficiently
            enriched_data_parts = []
            if not forward_data.empty:
                enriched_data_parts.append(forward_data)
            if not backward_data.empty:
                enriched_data_parts.append(backward_data)
            
            if enriched_data_parts:
                enriched_data = pd.concat(enriched_data_parts, ignore_index=True)
                enriched_data = enriched_data.drop_duplicates().reset_index(drop=True)
                enriched_data = self._clean_citation_data_patstat(enriched_data)
            else:
                enriched_data = pd.DataFrame()
            
            logger.debug(f"   ✅ Family-level processing: {len(all_citation_records)} citation records")
            logger.debug(f"   🏢 Final enriched: {len(enriched_data)} relationships")
            
            return enriched_data
            
        except Exception as e:
            logger.error(f"❌ Optimized family-level citation query failed: {e}")
            return pd.DataFrame()
    
    def _clean_citation_data_patstat(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean citation data from PATSTAT."""
        logger.debug("🧹 Cleaning PATSTAT citation data...")
        
        # Remove self-citations (same family citing itself)
        df = df[df['cited_family_id'] != df['citing_family_id']].copy()
        
        # Calculate citation lag
        df['citation_lag_years'] = df['citing_year'] - df['cited_year']
        
        # Remove invalid lags (negative or excessive)
        df = df[df['citation_lag_years'].between(0, 50)].copy()
        
        # Classify citation aging patterns
        def classify_citation_age(lag_years):
            for category, (min_years, max_years) in self.CITATION_AGING_PERIODS.items():
                if min_years <= lag_years <= max_years:
                    return category
            return 'unknown'
        
        df['citation_age_category'] = df['citation_lag_years'].apply(classify_citation_age)
        
        # Add citation quality based on authority
        quality_mapping = {
            'X': 0.8,  # Examiner citation - high quality
            'Y': 0.8,  # Examiner citation - high quality
            'A': 0.6,  # Applicant citation - medium quality
            'I': 0.6,  # Applicant citation - medium quality
            'P': 0.6,  # Applicant citation - medium quality
        }
        
        df['citation_quality_score'] = df['citation_authority'].map(quality_mapping).fillna(0.4)
        
        return df
    
    def _analyze_citation_impact(self, citation_data: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze citation impact patterns and metrics.
        """
        logger.debug("🎯 Analyzing citation impact patterns...")
        
        # Check available year columns
        year_col = None
        for col in ['earliest_filing_year_x', 'earliest_filing_year_y', 'earliest_filing_year']:
            if col in citation_data.columns:
                year_col = col
                break
        
        if year_col is None:
            logger.error("❌ No filing year column found for citation analysis")
            return pd.DataFrame()
        
        logger.debug(f"   Using year column: {year_col}")
        
        # Aggregate citation metrics by family
        agg_dict = {
            'citing_family_id': 'nunique',
            'cited_family_id': 'nunique', 
            'citation_quality_score': 'mean',
            'citation_lag_years': 'mean',
            year_col: ['min', 'max']
        }
        
        # Add optional columns if they exist
        if 'quality_score' in citation_data.columns:
            agg_dict['quality_score'] = 'mean'
        
        # Group by the original search families
        impact_analysis = citation_data.groupby('docdb_family_id').agg(agg_dict).reset_index()
        
        # Flatten column names
        flattened_columns = []
        for col in impact_analysis.columns:
            if isinstance(col, tuple):
                if col[1] == '':
                    flattened_columns.append(col[0])
                elif col[1] == 'min':
                    flattened_columns.append('first_citation_year')
                elif col[1] == 'max':
                    flattened_columns.append('latest_citation_year')
                else:
                    flattened_columns.append(f"{col[0]}_{col[1]}")
            else:
                flattened_columns.append(col)
        
        impact_analysis.columns = flattened_columns
        
        # Map column names
        column_mapping = {
            'citing_family_id_nunique': 'forward_citations',
            'cited_family_id_nunique': 'backward_citations',
            'citation_quality_score_mean': 'avg_citation_quality',
            'citation_lag_years_mean': 'avg_citation_lag',
            'quality_score_mean': 'avg_search_quality'
        }
        
        for old_name, new_name in column_mapping.items():
            if old_name in impact_analysis.columns:
                impact_analysis = impact_analysis.rename(columns={old_name: new_name})
        
        # Add missing columns with defaults
        if 'avg_search_quality' not in impact_analysis.columns:
            impact_analysis['avg_search_quality'] = 2.0
        
        # Calculate impact metrics
        impact_analysis['citation_span'] = impact_analysis['latest_citation_year'] - impact_analysis['first_citation_year'] + 1
        impact_analysis['total_citations'] = impact_analysis['forward_citations'] + impact_analysis['backward_citations']
        
        # Apply YAML-configured quality thresholds or fallback to defaults
        quality_metrics = self.citation_config.get('quality_metrics', {})
        impact_levels = quality_metrics.get('impact_levels', {
            'low_impact': 1, 'medium_impact': 2, 'high_impact': 5, 'breakthrough_impact': 15
        })
        
        # Dynamic impact classification based on config
        bins = [0, impact_levels.get('low_impact', 1), 
                impact_levels.get('medium_impact', 2), 
                impact_levels.get('high_impact', 5), float('inf')]
        labels = ['Low Impact', 'Moderate Impact', 'High Impact', 'Breakthrough Impact']
        
        impact_analysis['impact_category'] = pd.cut(
            impact_analysis['forward_citations'],
            bins=bins,
            labels=labels
        )
        
        # Sort by total citations
        impact_analysis = impact_analysis.sort_values('total_citations', ascending=False).reset_index(drop=True)
        impact_analysis['citation_rank'] = range(1, len(impact_analysis) + 1)
        
        logger.debug(f"   ✅ Analyzed citation impact for {len(impact_analysis)} families")
        logger.debug(f"   🏆 Top cited family: {impact_analysis.iloc[0]['docdb_family_id']} ({impact_analysis.iloc[0]['total_citations']} citations)")
        
        return impact_analysis
    
    def _build_citation_networks(self, citation_data: pd.DataFrame) -> Dict:
        """
        Build citation networks and calculate network metrics.
        """
        logger.debug("🕸️ Building citation networks...")
        
        # Build directed citation network
        G = nx.DiGraph()
        
        for _, row in citation_data.iterrows():
            cited_id = row['cited_family_id']
            citing_id = row['citing_family_id']
            
            # Add edge with attributes
            edge_attrs = {
                'citation_lag': row.get('citation_lag_years', 0),
                'quality_score': row.get('citation_quality_score', 0.5),
                'direction': row.get('citation_direction', 'unknown')
            }
            
            G.add_edge(cited_id, citing_id, **edge_attrs)
        
        # Calculate network metrics
        network_metrics = {
            'total_nodes': G.number_of_nodes(),
            'total_edges': G.number_of_edges(),
            'network_density': nx.density(G) if G.number_of_nodes() > 1 else 0,
            'strongly_connected_components': nx.number_strongly_connected_components(G),
            'weakly_connected_components': nx.number_weakly_connected_components(G)
        }
        
        # Calculate centrality measures
        centrality_metrics = {}
        if G.number_of_nodes() > 0:
            try:
                pagerank = nx.pagerank(G)
                betweenness = nx.betweenness_centrality(G)
                in_degree = dict(G.in_degree())
                out_degree = dict(G.out_degree())
                
                # Top influential nodes
                top_pagerank = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10]
                top_cited = sorted(in_degree.items(), key=lambda x: x[1], reverse=True)[:10]
                top_citing = sorted(out_degree.items(), key=lambda x: x[1], reverse=True)[:10]
                
                centrality_metrics = {
                    'top_influential_families': top_pagerank,
                    'top_cited_families': top_cited,
                    'top_citing_families': top_citing,
                    'pagerank_scores': pagerank,
                    'betweenness_scores': betweenness
                }
            except Exception as e:
                logger.warning(f"⚠️ Centrality calculation failed: {e}")
                centrality_metrics = {'top_influential_families': [], 'top_cited_families': [], 'top_citing_families': []}
        
        self.citation_network = G
        
        network_analysis = {
            'network_metrics': network_metrics,
            'centrality_metrics': centrality_metrics
        }
        
        logger.debug(f"   ✅ Built citation network with {network_metrics['total_nodes']} nodes and {network_metrics['total_edges']} edges")
        logger.debug(f"   📊 Network density: {network_metrics['network_density']:.3f}")
        
        return network_analysis
    
    def _generate_citation_intelligence(self, impact_analysis: pd.DataFrame, network_analysis: Dict) -> pd.DataFrame:
        """
        Generate comprehensive citation intelligence insights.
        """
        logger.debug("💡 Generating citation intelligence insights...")
        
        # Enhance impact analysis with network insights
        enhanced_analysis = impact_analysis.copy()
        
        # Add network centrality measures
        if network_analysis['centrality_metrics'].get('pagerank_scores'):
            pagerank_dict = network_analysis['centrality_metrics']['pagerank_scores']
            enhanced_analysis['network_centrality'] = enhanced_analysis['docdb_family_id'].apply(
                lambda x: pagerank_dict.get(x, 0)
            )
        else:
            enhanced_analysis['network_centrality'] = 0
        
        # Calculate citation influence score
        max_citations = enhanced_analysis['total_citations'].max() if not enhanced_analysis.empty else 1
        enhanced_analysis['influence_score'] = (
            (enhanced_analysis['forward_citations'] / max_citations) * 0.4 +
            enhanced_analysis['avg_citation_quality'] * 0.3 +
            (1 / (enhanced_analysis['avg_citation_lag'] + 1)) * 0.2 +
            enhanced_analysis['network_centrality'] * 0.1
        ).round(3)
        
        # Citation maturity assessment
        def assess_citation_maturity(row):
            if row['citation_span'] > 15 and row['forward_citations'] > 10:
                return 'Established'
            elif row['citation_span'] > 10:
                return 'Mature'
            elif row['citation_span'] > 5:
                return 'Developing'
            else:
                return 'Emerging'
        
        enhanced_analysis['citation_maturity'] = enhanced_analysis.apply(assess_citation_maturity, axis=1)
        
        # Strategic citation importance
        def calculate_citation_importance(row):
            score = 0
            # Citation volume weight
            if row['forward_citations'] > 20:
                score += 4
            elif row['forward_citations'] > 10:
                score += 3
            elif row['forward_citations'] > 5:
                score += 2
            elif row['forward_citations'] > 0:
                score += 1
            
            # Quality weight
            if row['avg_citation_quality'] > 0.7:
                score += 2
            elif row['avg_citation_quality'] > 0.5:
                score += 1
            
            # Network importance weight
            if row['network_centrality'] > 0.01:
                score += 2
            elif row['network_centrality'] > 0.001:
                score += 1
            
            return score
        
        enhanced_analysis['citation_importance_score'] = enhanced_analysis.apply(calculate_citation_importance, axis=1)
        
        # Strategic category
        def assign_citation_category(score: int) -> str:
            if score >= 7:
                return 'Seminal Technology'
            elif score >= 5:
                return 'Influential Technology'
            elif score >= 3:
                return 'Notable Technology'
            else:
                return 'Emerging Technology'
        
        enhanced_analysis['citation_category'] = enhanced_analysis['citation_importance_score'].apply(assign_citation_category)
        
        return enhanced_analysis
    
    def _analyze_citation_categories(self, citation_data: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze citation categories using TLS215_CITN_CATEG for quality assessment.
        Based on PATSTAT training best practices.
        """
        if not self.session or 'TLS215_CITN_CATEG' not in self.models:
            logger.warning("⚠️ TLS215_CITN_CATEG not available for quality analysis")
            return citation_data
        
        logger.debug("📊 Analyzing citation categories for quality assessment...")
        
        try:
            TLS215_CITN_CATEG = self.models.get('TLS215_CITN_CATEG')
            TLS212_CITATION = self.models.get('TLS212_CITATION')
            
            if not TLS215_CITN_CATEG or not TLS212_CITATION:
                return citation_data
            
            # Extract publication IDs from citation data
            pub_ids = []
            if 'citing_publn_id' in citation_data.columns:
                pub_ids.extend(citation_data['citing_publn_id'].dropna().unique())
            if 'cited_pat_publn_id' in citation_data.columns:
                pub_ids.extend(citation_data['cited_pat_publn_id'].dropna().unique())
            
            if not pub_ids:
                return citation_data
            
            # Query citation categories (following TLS215 training patterns)
            category_query = self.session.query(
                TLS212_CITATION.pat_publn_id,
                TLS212_CITATION.citn_id,
                TLS215_CITN_CATEG.citn_categ,
                TLS215_CITN_CATEG.relevant_claim
            ).join(
                TLS215_CITN_CATEG,
                (TLS212_CITATION.pat_publn_id == TLS215_CITN_CATEG.pat_publn_id) &
                (TLS212_CITATION.citn_id == TLS215_CITN_CATEG.citn_id)
            ).filter(
                TLS212_CITATION.pat_publn_id.in_(pub_ids[:1000])  # Limit for performance
            )
            
            category_results = category_query.all()
            
            if category_results:
                category_df = pd.DataFrame(category_results, columns=[
                    'pat_publn_id', 'citn_id', 'citn_categ', 'relevant_claim'
                ])
                
                # Map citation quality based on training insights
                quality_mapping = {
                    'X': 0.9,  # Prejudices novelty alone (highest quality)
                    'Y': 0.8,  # Prejudices inventive step when combined
                    'A': 0.6,  # Technological background
                    'I': 0.9,  # Post-2011 inventive step (equivalent to X)
                    'D': 0.4,  # Document cited in application
                    'L': 0.3,  # Other reasons (lowest quality)
                    'O': 0.5,  # Oral/non-written disclosure
                    'P': 0.5,  # Intermediate document
                    'T': 0.5,  # Theory or principle
                    'E': 0.7   # Earlier application (same technical field)
                }
                
                # Single letter categories (rich citations)
                rich_citations = category_df[category_df['citn_categ'].str.len() == 1].copy()
                if not rich_citations.empty:
                    rich_citations['quality_score'] = rich_citations['citn_categ'].map(quality_mapping).fillna(0.4)
                    
                    # Add aggregated quality metrics to citation_data
                    quality_summary = rich_citations.groupby('pat_publn_id').agg({
                        'quality_score': ['mean', 'max', 'count'],
                        'citn_categ': lambda x: ''.join(sorted(set(x)))  # Combined categories
                    }).round(3)
                    
                    quality_summary.columns = ['avg_citation_quality', 'max_citation_quality', 
                                             'total_categorized_citations', 'citation_categories']
                    quality_summary = quality_summary.reset_index()
                    
                    logger.debug(f"   ✅ Analyzed {len(quality_summary)} publications with citation categories")
                    return quality_summary
            
            logger.debug("   📊 No citation categories found for current dataset")
            return citation_data
            
        except Exception as e:
            logger.warning(f"⚠️ Citation category analysis failed: {e}")
            return citation_data
    
    def _analyze_temporal_citation_trends(self, citation_data: pd.DataFrame) -> Dict:
        """
        Analyze temporal citation trends based on PATSTAT training best practices.
        """
        logger.debug("📅 Analyzing temporal citation trends...")
        
        trends = {}
        
        if 'citation_lag_years' in citation_data.columns:
            # Citation velocity analysis (from training)
            lag_stats = citation_data['citation_lag_years'].describe()
            trends['citation_velocity'] = {
                'avg_lag_years': float(lag_stats['mean']),
                'median_lag_years': float(lag_stats['50%']),
                'fastest_citation_lag': float(lag_stats['min']),
                'slowest_citation_lag': float(lag_stats['max'])
            }
            
            # Innovation velocity classification (from training insights)
            avg_lag = lag_stats['mean']
            if avg_lag < 3:
                trends['innovation_velocity'] = 'High - Rapidly adopted technology'
            elif avg_lag < 7:
                trends['innovation_velocity'] = 'Medium - Standard adoption rate'
            else:
                trends['innovation_velocity'] = 'Low - Slow adoption or niche technology'
        
        if 'citing_year' in citation_data.columns and 'cited_year' in citation_data.columns:
            # Temporal citation patterns (like the training examples)
            citation_data['citing_decade'] = (citation_data['citing_year'] // 10) * 10
            citation_data['cited_decade'] = (citation_data['cited_year'] // 10) * 10
            
            decade_analysis = citation_data.groupby(['cited_decade', 'citing_decade']).size().reset_index()
            decade_analysis.columns = ['cited_decade', 'citing_decade', 'citation_count']
            
            trends['decade_patterns'] = decade_analysis.to_dict('records')
            
        return trends
    
    def _apply_yaml_query_templates(self, family_ids: List[int], query_type: str = 'forward_citations') -> pd.DataFrame:
        """
        Use YAML-configured query templates for citation analysis.
        Implements the configured templates from search_patterns_config.yaml
        """
        family_citations = self.citation_config.get('family_citations', {})
        query_config = family_citations.get(query_type, {})
        
        if not query_config:
            logger.warning(f"⚠️ No YAML config found for {query_type}")
            return pd.DataFrame()
        
        try:
            # Format the template with actual family IDs
            template = query_config.get('template', '')
            family_ids_str = ','.join(map(str, family_ids[:1000]))  # Limit for performance
            
            formatted_query = template.format(family_ids=family_ids_str)
            
            # Execute raw SQL query (following training patterns)
            if hasattr(self.session, 'execute'):
                result = self.session.execute(formatted_query)
                
                # Convert to DataFrame
                columns = result.keys() if hasattr(result, 'keys') else ['cited_family_id', 'citing_family_id']
                data = [dict(row) for row in result]
                
                if data:
                    df = pd.DataFrame(data)
                    df['query_source'] = query_config.get('description', query_type)
                    logger.debug(f"   ✅ YAML query '{query_type}' returned {len(df)} results")
                    return df
            
            logger.debug(f"   📊 YAML query '{query_type}' returned no results")
            return pd.DataFrame()
            
        except Exception as e:
            logger.warning(f"⚠️ YAML query execution failed for {query_type}: {e}")
            return pd.DataFrame()
    
    def _clean_citation_data(self, df: pd.DataFrame, 
                           citing_family_col: str, cited_family_col: str) -> pd.DataFrame:
        """Clean and validate citation data."""
        logger.debug("🧹 Cleaning citation data...")
        
        initial_count = len(df)
        
        # Remove null values
        df = df[df[citing_family_col].notna()].copy()
        df = df[df[cited_family_col].notna()].copy()
        
        # Remove self-citations (same family citing itself)
        df = df[df[citing_family_col] != df[cited_family_col]].copy()
        
        # Ensure numeric family IDs
        df[citing_family_col] = pd.to_numeric(df[citing_family_col], errors='coerce')
        df[cited_family_col] = pd.to_numeric(df[cited_family_col], errors='coerce')
        
        # Remove rows with conversion errors
        df = df[df[citing_family_col].notna()].copy()
        df = df[df[cited_family_col].notna()].copy()
        
        # Remove duplicates
        df = df.drop_duplicates(subset=[citing_family_col, cited_family_col]).copy()
        
        cleaned_count = len(df)
        removed_count = initial_count - cleaned_count
        if removed_count > 0:
            logger.debug(f"📊 Removed {removed_count} invalid citation records")
        
        return df
    
    def _analyze_temporal_citation_patterns(self, df: pd.DataFrame,
                                          citing_year_col: str, cited_year_col: str) -> pd.DataFrame:
        """Analyze temporal patterns in citation behavior."""
        logger.debug("📅 Analyzing temporal citation patterns...")
        
        # Calculate citation lag (time between cited and citing patents)
        df['citation_lag_years'] = df[citing_year_col] - df[cited_year_col]
        
        # Remove invalid lags (negative or excessive)
        df = df[df['citation_lag_years'].between(0, 50)].copy()
        
        # Classify citation aging patterns
        def classify_citation_age(lag_years):
            for category, (min_years, max_years) in self.CITATION_AGING_PERIODS.items():
                if min_years <= lag_years <= max_years:
                    return category
            return 'unknown'
        
        df['citation_age_category'] = df['citation_lag_years'].apply(classify_citation_age)
        
        # Add temporal trend indicators
        df['citing_decade'] = (df[citing_year_col] // 10) * 10
        df['cited_decade'] = (df[cited_year_col] // 10) * 10
        
        # Cross-decade citation analysis
        df['is_cross_decade'] = df['citing_decade'] != df['cited_decade']
        
        return df
    
    def _assess_citation_quality(self, df: pd.DataFrame, citation_type_col: str) -> pd.DataFrame:
        """Assess citation quality based on type and context."""
        logger.debug("🎯 Assessing citation quality...")
        
        # Standardize citation types
        citation_type_mapping = {
            'X': 'examiner_citation',
            'Y': 'examiner_citation', 
            'A': 'applicant_citation',
            'I': 'applicant_citation',
            'P': 'applicant_citation',
            'E': 'examiner_citation'
        }
        
        df['citation_source'] = df[citation_type_col].map(citation_type_mapping).fillna('unknown')
        
        # Quality scoring based on source
        quality_scores = {
            'examiner_citation': 0.8,  # High quality - examiner validated
            'applicant_citation': 0.6,  # Medium quality - applicant provided
            'unknown': 0.4  # Lower quality - source unknown
        }
        
        df['citation_quality_score'] = df['citation_source'].map(quality_scores)
        
        # Quality categories
        df['citation_quality_category'] = pd.cut(
            df['citation_quality_score'],
            bins=[0, 0.4, 0.6, 1.0],
            labels=['low_impact', 'moderate_impact', 'high_impact']
        )
        
        return df
    
    def _calculate_citation_impact_metrics(self, df: pd.DataFrame,
                                         citing_family_col: str, cited_family_col: str) -> pd.DataFrame:
        """Calculate comprehensive citation impact metrics."""
        logger.debug("📊 Calculating citation impact metrics...")
        
        # Forward citations (how often each patent is cited)
        forward_citations = df.groupby(cited_family_col).agg({
            citing_family_col: 'nunique',
            'citation_quality_score': 'mean' if 'citation_quality_score' in df.columns else 'count'
        }).rename(columns={
            citing_family_col: 'forward_citation_count',
            'citation_quality_score': 'avg_forward_quality'
        })
        
        # Backward citations (how many patents each patent cites)
        backward_citations = df.groupby(citing_family_col).agg({
            cited_family_col: 'nunique',
            'citation_quality_score': 'mean' if 'citation_quality_score' in df.columns else 'count'
        }).rename(columns={
            cited_family_col: 'backward_citation_count',
            'citation_quality_score': 'avg_backward_quality'
        })
        
        # Calculate impact scores
        max_forward = forward_citations['forward_citation_count'].max() if len(forward_citations) > 0 else 1
        max_backward = backward_citations['backward_citation_count'].max() if len(backward_citations) > 0 else 1
        
        forward_citations['forward_impact_score'] = (
            forward_citations['forward_citation_count'] / max_forward
        ).round(3)
        
        backward_citations['backward_impact_score'] = (
            backward_citations['backward_citation_count'] / max_backward
        ).round(3)
        
        # Impact classification
        forward_citations['impact_category'] = pd.cut(
            forward_citations['forward_impact_score'],
            bins=[0, 0.1, 0.3, 0.6, 1.0],
            labels=['Low Impact', 'Moderate Impact', 'High Impact', 'Breakthrough Impact']
        )
        
        # Add metrics back to main dataframe
        df = df.merge(
            forward_citations.add_suffix('_forward'),
            left_on=cited_family_col,
            right_index=True,
            how='left'
        )
        
        df = df.merge(
            backward_citations.add_suffix('_backward'),
            left_on=citing_family_col,
            right_index=True,
            how='left'
        )
        
        return df
    
    def _calculate_network_position_metrics(self, df: pd.DataFrame,
                                          citing_family_col: str, cited_family_col: str) -> pd.DataFrame:
        """Calculate network position and centrality metrics."""
        logger.debug("🕸️ Calculating network position metrics...")
        
        # Build citation network
        G = nx.DiGraph()
        for _, row in df.iterrows():
            G.add_edge(row[cited_family_col], row[citing_family_col])
        
        # Calculate centrality measures
        try:
            pagerank = nx.pagerank(G)
            betweenness = nx.betweenness_centrality(G)
            in_degree = dict(G.in_degree())
            out_degree = dict(G.out_degree())
            
            # Add centrality metrics to dataframe
            all_families = set(df[citing_family_col].unique()) | set(df[cited_family_col].unique())
            
            centrality_df = pd.DataFrame({
                'family_id': list(all_families),
                'pagerank_centrality': [pagerank.get(fid, 0) for fid in all_families],
                'betweenness_centrality': [betweenness.get(fid, 0) for fid in all_families],
                'in_degree_centrality': [in_degree.get(fid, 0) for fid in all_families],
                'out_degree_centrality': [out_degree.get(fid, 0) for fid in all_families]
            })
            
            # Network position classification
            centrality_df['network_position'] = pd.cut(
                centrality_df['pagerank_centrality'],
                bins=[0, 0.001, 0.01, 0.1, 1.0],
                labels=['Peripheral', 'Connected', 'Influential', 'Central Hub']
            )
            
            # Merge centrality data
            df = df.merge(
                centrality_df.add_suffix('_citing'),
                left_on=citing_family_col,
                right_on='family_id_citing',
                how='left'
            )
            
            df = df.merge(
                centrality_df.add_suffix('_cited'),
                left_on=cited_family_col,
                right_on='family_id_cited',
                how='left'
            )
            
            self.citation_network = G
            
        except Exception as e:
            logger.warning(f"⚠️ Network centrality calculation failed: {e}")
            # Add default values if network analysis fails
            df['pagerank_centrality_citing'] = 0
            df['network_position_citing'] = 'Unknown'
        
        return df
    
    def build_citation_network(self, df: Optional[pd.DataFrame] = None,
                             min_citations: int = 2) -> nx.DiGraph:
        """
        Build directed citation network graph.
        
        Args:
            df: DataFrame to analyze (uses self.analyzed_data if None)
            min_citations: Minimum citation threshold for inclusion
            
        Returns:
            NetworkX directed graph with citation network
        """
        if df is None:
            df = self.analyzed_data
        
        if df is None:
            raise ValueError("No analyzed data available. Run analyze_citation_patterns first.")
        
        logger.debug("🕸️ Building citation network...")
        
        # Filter by minimum citations
        citation_counts = df['cited_family_id'].value_counts()
        qualified_patents = citation_counts[citation_counts >= min_citations].index
        filtered_df = df[df['cited_family_id'].isin(qualified_patents)].copy()
        
        # Build directed network
        G = nx.DiGraph()
        
        for _, row in filtered_df.iterrows():
            cited = row['cited_family_id']
            citing = row['citing_family_id']
            
            # Add edge with attributes
            edge_attrs = {
                'citation_lag': row.get('citation_lag_years', 0),
                'quality_score': row.get('citation_quality_score', 0.5),
                'age_category': row.get('citation_age_category', 'unknown')
            }
            
            G.add_edge(cited, citing, **edge_attrs)
        
        # Add node attributes
        for node in G.nodes():
            G.nodes[node]['in_degree'] = G.in_degree(node)
            G.nodes[node]['out_degree'] = G.out_degree(node)
            G.nodes[node]['total_degree'] = G.degree(node)
        
        self.citation_network = G
        logger.debug(f"✅ Citation network built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        return G
    
    def generate_citation_intelligence_summary(self) -> Dict:
        """
        Generate comprehensive citation intelligence summary.
        
        Returns:
            Dictionary with citation intelligence insights
        """
        if self.analyzed_data is None:
            raise ValueError("No analyzed data available. Run analyze_search_results first.")
        
        df = self.analyzed_data
        logger.debug("📋 Generating citation intelligence summary...")
        
        total_families = len(df)
        top_cited = df.iloc[0] if len(df) > 0 else None
        
        summary = {
            'citation_overview': {
                'total_patent_families': int(total_families),
                'total_forward_citations': int(df['forward_citations'].sum()),
                'total_backward_citations': int(df['backward_citations'].sum()),
                'avg_forward_citations': float(df['forward_citations'].mean()),
                'avg_backward_citations': float(df['backward_citations'].mean()),
                'top_cited_family': int(top_cited['docdb_family_id']) if top_cited is not None else None,
                'top_cited_count': int(top_cited['forward_citations']) if top_cited is not None else 0
            },
            'impact_distribution': df['impact_category'].value_counts().to_dict(),
            'citation_maturity': df['citation_maturity'].value_counts().to_dict(),
            'citation_categories': df['citation_category'].value_counts().to_dict(),
            'citation_metrics': {
                'avg_citation_quality': float(df['avg_citation_quality'].mean()),
                'avg_citation_lag': float(df['avg_citation_lag'].mean()),
                'avg_influence_score': float(df['influence_score'].mean()),
                'breakthrough_technologies': len(df[df['impact_category'] == 'Breakthrough Impact']),
                'seminal_technologies': len(df[df['citation_category'] == 'Seminal Technology'])
            },
            'temporal_insights': {
                'earliest_citation': int(df['first_citation_year'].min()) if not df.empty else None,
                'latest_citation': int(df['latest_citation_year'].max()) if not df.empty else None,
                'avg_citation_span': float(df['citation_span'].mean()),
                'most_sustained_family': int(df.loc[df['citation_span'].idxmax(), 'docdb_family_id']) if not df.empty else None
            }
        }
        
        # Add network insights if available
        if hasattr(self, 'citation_network') and self.citation_network:
            network_metrics = {
                'network_nodes': self.citation_network.number_of_nodes(),
                'network_edges': self.citation_network.number_of_edges(),
                'network_density': nx.density(self.citation_network),
                'strong_components': nx.number_strongly_connected_components(self.citation_network),
                'weak_components': nx.number_weakly_connected_components(self.citation_network)
            }
            summary['network_analysis'] = network_metrics
        
        self.citation_intelligence = summary
        return summary
    
    def get_innovation_influencers(self, top_n: int = 20, min_citations: int = 1) -> pd.DataFrame:
        """
        Get top innovation influencers based on citation patterns.
        
        Args:
            top_n: Number of top influencers to return
            min_citations: Minimum number of forward citations required
            
        Returns:
            DataFrame with top innovation influencers
        """
        if self.analyzed_data is None:
            raise ValueError("No analyzed data available. Run analyze_search_results first.")
        
        filtered_df = self.analyzed_data[self.analyzed_data['forward_citations'] >= min_citations].copy()
        return filtered_df.head(top_n)
    
    def get_citation_hotspots(self, top_n: int = 5) -> pd.DataFrame:
        """
        Get citation hotspots based on multiple criteria.
        
        Args:
            top_n: Number of hotspots to return
            
        Returns:
            DataFrame with citation hotspots
        """
        if self.analyzed_data is None:
            raise ValueError("No analyzed data available. Run analyze_search_results first.")
        
        # Sort by influence score and total citations
        hotspots = self.analyzed_data.sort_values(
            ['influence_score', 'total_citations'], 
            ascending=False
        ).head(top_n)
        
        return hotspots[['docdb_family_id', 'forward_citations', 'backward_citations', 'influence_score', 'citation_category', 'citation_maturity']]
    
    def build_citation_network(self, min_citations: int = 2) -> nx.DiGraph:
        """
        Build and return the citation network.
        
        Args:
            min_citations: Minimum citation threshold for inclusion
            
        Returns:
            NetworkX directed graph of citation relationships
        """
        if self.citation_network is None:
            logger.warning("⚠️ No citation network available. Run analyze_search_results first.")
            return nx.DiGraph()
        
        # Filter network by minimum citations
        filtered_graph = nx.DiGraph()
        for u, v, data in self.citation_network.edges(data=True):
            if self.citation_network.in_degree(v) >= min_citations:
                filtered_graph.add_edge(u, v, **data)
        
        return filtered_graph

    # Legacy method for backward compatibility (deprecated)
    def generate_citation_intelligence(self, df: Optional[pd.DataFrame] = None) -> Dict:
        """
        Generate comprehensive citation intelligence summary.
        
        Args:
            df: DataFrame to analyze (uses self.analyzed_data if None)
            
        Returns:
            Dictionary with citation intelligence insights
        """
        if df is None:
            df = self.analyzed_data
        
        if df is None:
            raise ValueError("No analyzed data available. Run analyze_citation_patterns first.")
        
        logger.debug("📋 Generating citation intelligence...")
        
        # Overall citation statistics
        total_citations = len(df)
        unique_citing = df['citing_family_id'].nunique()
        unique_cited = df['cited_family_id'].nunique()
        
        # Impact analysis
        forward_citation_stats = df.groupby('cited_family_id')['citing_family_id'].nunique()
        
        # Quality analysis
        quality_distribution = {}
        if 'citation_quality_category' in df.columns:
            quality_distribution = df['citation_quality_category'].value_counts().to_dict()
        
        # Temporal analysis
        temporal_patterns = {}
        if 'citation_age_category' in df.columns:
            temporal_patterns = df['citation_age_category'].value_counts().to_dict()
        
        # Network analysis
        network_stats = {}
        if self.citation_network:
            network_stats = {
                'total_nodes': self.citation_network.number_of_nodes(),
                'total_edges': self.citation_network.number_of_edges(),
                'density': float(nx.density(self.citation_network)),
                'strongly_connected_components': nx.number_strongly_connected_components(self.citation_network),
                'weakly_connected_components': nx.number_weakly_connected_components(self.citation_network)
            }
        
        # Innovation flow analysis
        flow_analysis = self._analyze_innovation_flow(df)
        
        intelligence = {
            'overview': {
                'total_citations': int(total_citations),
                'unique_citing_families': int(unique_citing),
                'unique_cited_families': int(unique_cited),
                'avg_citations_per_patent': float(total_citations / max(unique_cited, 1)),
                'citation_density': float(total_citations / max(unique_citing * unique_cited, 1))
            },
            'impact_metrics': {
                'highly_cited_patents': int(len(forward_citation_stats[forward_citation_stats >= 10])),
                'moderately_cited_patents': int(len(forward_citation_stats[forward_citation_stats.between(3, 9)])),
                'rarely_cited_patents': int(len(forward_citation_stats[forward_citation_stats <= 2])),
                'max_citations_received': int(forward_citation_stats.max()) if len(forward_citation_stats) > 0 else 0,
                'avg_citations_received': float(forward_citation_stats.mean()) if len(forward_citation_stats) > 0 else 0
            },
            'quality_distribution': quality_distribution,
            'temporal_patterns': temporal_patterns,
            'network_intelligence': network_stats,
            'innovation_flow': flow_analysis
        }
        
        self.citation_intelligence = intelligence
        return intelligence
    
    def _analyze_innovation_flow(self, df: pd.DataFrame) -> Dict:
        """Analyze innovation flow patterns from citation data."""
        flow_analysis = {}
        
        if 'citation_lag_years' in df.columns:
            avg_citation_lag = df['citation_lag_years'].mean()
            
            # Technology adoption speed analysis
            immediate_adoption = len(df[df['citation_lag_years'] <= 2]) / len(df) * 100
            slow_adoption = len(df[df['citation_lag_years'] >= 10]) / len(df) * 100
            
            flow_analysis = {
                'avg_citation_lag_years': float(avg_citation_lag),
                'immediate_adoption_rate': float(immediate_adoption),
                'slow_adoption_rate': float(slow_adoption),
                'innovation_velocity': 'High' if avg_citation_lag < 5 else 'Medium' if avg_citation_lag < 10 else 'Low'
            }
        
        return flow_analysis
    
    def get_innovation_influencers(self, df: Optional[pd.DataFrame] = None, top_n: int = 20) -> pd.DataFrame:
        """
        Identify top innovation influencers based on citation patterns.
        
        Args:
            df: DataFrame to analyze (uses self.analyzed_data if None)
            top_n: Number of top influencers to return
            
        Returns:
            DataFrame with top innovation influencers
        """
        if df is None:
            df = self.analyzed_data
        
        if df is None:
            raise ValueError("No analyzed data available. Run analyze_citation_patterns first.")
        
        # Calculate comprehensive influence metrics
        influence_metrics = df.groupby('cited_family_id').agg({
            'citing_family_id': 'nunique',
            'citation_quality_score': 'mean' if 'citation_quality_score' in df.columns else 'count',
            'citation_lag_years': 'mean' if 'citation_lag_years' in df.columns else 'count',
            'pagerank_centrality_cited': 'first' if 'pagerank_centrality_cited' in df.columns else 'count'
        }).round(3)
        
        influence_metrics.columns = [
            'total_citations',
            'avg_citation_quality',
            'avg_citation_lag',
            'network_centrality'
        ]
        
        # Calculate composite influence score
        max_citations = influence_metrics['total_citations'].max() if len(influence_metrics) > 0 else 1
        influence_metrics['influence_score'] = (
            (influence_metrics['total_citations'] / max_citations) * 0.4 +
            influence_metrics['avg_citation_quality'] * 0.3 +
            (1 / (influence_metrics['avg_citation_lag'] + 1)) * 0.2 +
            influence_metrics['network_centrality'] * 0.1
        ).round(3)
        
        # Rank and classify
        influence_metrics = influence_metrics.sort_values('influence_score', ascending=False)
        influence_metrics['influence_rank'] = range(1, len(influence_metrics) + 1)
        
        influence_metrics['influence_category'] = pd.cut(
            influence_metrics['influence_score'],
            bins=[0, 0.3, 0.6, 0.8, 1.0],
            labels=['Moderate Influence', 'Significant Influence', 'High Influence', 'Breakthrough Influence']
        )
        
        return influence_metrics.head(top_n)

class CitationDataProcessor:
    """
    Data processor for cleaning and preparing citation data from PATSTAT.
    """
    
    def __init__(self):
        """Initialize citation data processor."""
        self.processed_data = None
    
    def process_patstat_citation_data(self, raw_data: List[Tuple]) -> pd.DataFrame:
        """
        Process raw PATSTAT citation query results.
        
        Args:
            raw_data: Raw query results from PATSTAT citation analysis
            
        Returns:
            Processed DataFrame ready for citation analysis
        """
        logger.debug(f"📊 Processing {len(raw_data)} raw citation records...")
        
        # Convert to DataFrame
        df = pd.DataFrame(raw_data, columns=[
            'citing_family_id', 'cited_family_id', 'citing_year', 'cited_year'
        ])
        
        # Data cleaning
        df = self._clean_citation_relationships(df)
        df = self._validate_citation_data(df)
        df = self._standardize_citation_data(df)
        
        logger.debug(f"✅ Processed to {len(df)} clean citation records")
        self.processed_data = df
        
        return df
    
    def _clean_citation_relationships(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean citation relationship data."""
        logger.debug("🧹 Cleaning citation relationships...")
        
        # Remove null values
        df = df.dropna().copy()
        
        # Ensure proper data types
        df['citing_family_id'] = pd.to_numeric(df['citing_family_id'], errors='coerce')
        df['cited_family_id'] = pd.to_numeric(df['cited_family_id'], errors='coerce')
        df['citing_year'] = pd.to_numeric(df['citing_year'], errors='coerce')
        df['cited_year'] = pd.to_numeric(df['cited_year'], errors='coerce')
        
        # Remove rows with conversion errors
        df = df.dropna().copy()
        
        return df
    
    def _validate_citation_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate citation data quality."""
        logger.debug("🔍 Validating citation data...")
        
        initial_count = len(df)
        
        # Remove self-citations
        df = df[df['citing_family_id'] != df['cited_family_id']].copy()
        
        # Validate years
        current_year = datetime.now().year
        df = df[df['citing_year'].between(1980, current_year)].copy()
        df = df[df['cited_year'].between(1980, current_year)].copy()
        
        # Ensure citing year >= cited year
        df = df[df['citing_year'] >= df['cited_year']].copy()
        
        # Remove excessive citation lags (> 50 years)
        df = df[(df['citing_year'] - df['cited_year']) <= 50].copy()
        
        removed_count = initial_count - len(df)
        if removed_count > 0:
            logger.debug(f"📊 Removed {removed_count} invalid citation records")
        
        return df
    
    def _standardize_citation_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize citation data format."""
        logger.debug("🏷️ Standardizing citation data...")
        
        # Ensure integer family IDs
        df['citing_family_id'] = df['citing_family_id'].astype(int)
        df['cited_family_id'] = df['cited_family_id'].astype(int)
        df['citing_year'] = df['citing_year'].astype(int)
        df['cited_year'] = df['cited_year'].astype(int)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['citing_family_id', 'cited_family_id']).copy()
        
        # Sort by citing year for temporal analysis
        df = df.sort_values(['citing_year', 'citing_family_id']).reset_index(drop=True)
        
        return df

def create_citation_analyzer(patstat_client: Optional[object] = None) -> CitationAnalyzer:
    """
    Factory function to create configured citation analyzer.
    
    Args:
        patstat_client: Optional PATSTAT client instance
        
    Returns:
        Configured CitationAnalyzer instance
    """
    return CitationAnalyzer(patstat_client=patstat_client)

def create_citation_processor() -> CitationDataProcessor:
    """
    Factory function to create configured citation data processor.
    
    Returns:
        Configured CitationDataProcessor instance
    """
    return CitationDataProcessor()

