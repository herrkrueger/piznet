"""
Base Analyzer Classes - Clean Architecture Foundation
Standardized interfaces for all intelligence analysis components
"""

import pandas as pd
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import time

# Setup logging
logger = logging.getLogger(__name__)


class AnalyzerResult:
    """
    Standardized result container for all analyzers
    Contains intelligence insights, strategic analysis, and metadata
    """
    
    def __init__(self, intelligence: Dict[str, Any], insights: Union[List[str], Dict[str, Any]], 
                 metadata: Dict[str, Any], status: str = "completed", warnings: List[str] = None):
        """
        Initialize analyzer result
        
        Args:
            intelligence: Structured intelligence analysis
            insights: Key insights and recommendations
            metadata: Analysis metadata and parameters
            status: Analysis status ('completed', 'partial', 'failed')
            warnings: List of any warnings encountered
        """
        self.intelligence = intelligence
        self.insights = insights
        self.metadata = metadata or {}
        self.status = status
        self.warnings = warnings or []
        self.timestamp = datetime.now().isoformat()
        
        # Add basic statistics
        self.metadata.update({
            'intelligence_sections': len(intelligence) if isinstance(intelligence, dict) else 0,
            'insight_count': len(insights) if isinstance(insights, (list, dict)) else 0,
            'analysis_timestamp': self.timestamp
        })
    
    @property
    def is_successful(self) -> bool:
        """Check if analysis was successful"""
        return self.status == "completed"
    
    def add_warning(self, warning_message: str):
        """Add a warning to the result"""
        self.warnings.append(warning_message)
    
    def get_executive_summary(self) -> Dict[str, Any]:
        """Extract executive summary from intelligence"""
        if isinstance(self.intelligence, dict) and 'executive_summary' in self.intelligence:
            return self.intelligence['executive_summary']
        return {}


class BaseAnalyzer(ABC):
    """
    Abstract base class for all analyzers
    Ensures standardized interface and common intelligence functionality
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize base analyzer
        
        Args:
            config: Analysis configuration parameters
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Analysis tracking
        self.analysis_history = []
        self.total_analyses = 0
        self.average_analysis_time = 0.0
    
    @abstractmethod
    def analyze(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], 
               analysis_params: Dict[str, Any] = None, **kwargs) -> AnalyzerResult:
        """
        Perform intelligence analysis and return standardized result
        
        Args:
            data: Input data to analyze (DataFrame or dict of DataFrames)
            analysis_params: Analysis-specific parameters
            **kwargs: Additional analysis options
            
        Returns:
            AnalyzerResult with intelligence, insights, and metadata
        """
        pass
    
    def _validate_input(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]) -> bool:
        """
        Validate input data before analysis
        
        Args:
            data: Input data to validate
            
        Returns:
            True if valid, False otherwise
        """
        if isinstance(data, pd.DataFrame):
            if data.empty:
                self.logger.warning("Input DataFrame is empty")
                return False
            return True
        
        elif isinstance(data, dict):
            if not data:
                self.logger.warning("Input dictionary is empty")
                return False
            
            # Check that dict contains DataFrames
            for key, df in data.items():
                if not isinstance(df, pd.DataFrame):
                    self.logger.error(f"Dictionary value '{key}' is not a DataFrame: {type(df)}")
                    return False
                if df.empty:
                    self.logger.warning(f"DataFrame '{key}' is empty")
            
            return True
        
        else:
            self.logger.error(f"Invalid input type: {type(data)}. Expected DataFrame or dict of DataFrames")
            return False
    
    def _create_metadata(self, input_data: Union[pd.DataFrame, Dict], analysis_time: float,
                        analysis_params: Dict[str, Any] = None, 
                        additional_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create standardized metadata for analysis results
        
        Args:
            input_data: Original input data
            analysis_time: Time taken for analysis
            analysis_params: Analysis parameters used
            additional_metadata: Additional analyzer-specific metadata
            
        Returns:
            Complete metadata dictionary
        """
        metadata = {
            'analyzer_type': self.__class__.__name__,
            'analysis_timestamp': datetime.now().isoformat(),
            'analysis_time': analysis_time,
            'configuration': self.config.copy(),
            'analysis_parameters': analysis_params or {}
        }
        
        # Add input data statistics
        if isinstance(input_data, pd.DataFrame):
            metadata.update({
                'input_type': 'dataframe',
                'input_records': len(input_data),
                'input_columns': list(input_data.columns)
            })
        elif isinstance(input_data, dict):
            metadata.update({
                'input_type': 'dictionary',
                'input_dataframes': list(input_data.keys()),
                'total_input_records': sum(len(df) for df in input_data.values() if isinstance(df, pd.DataFrame))
            })
        
        if additional_metadata:
            metadata.update(additional_metadata)
        
        return metadata
    
    def _update_performance_metrics(self, analysis_time: float, data_size: int):
        """Update analyzer performance metrics"""
        self.analysis_history.append({
            'timestamp': datetime.now().isoformat(),
            'analysis_time': analysis_time,
            'data_size': data_size
        })
        
        self.total_analyses += 1
        
        # Calculate moving average
        recent_times = [h['analysis_time'] for h in self.analysis_history[-10:]]
        self.average_analysis_time = sum(recent_times) / len(recent_times)
    
    def _extract_data_from_processors(self, data: Dict[str, Any], preferred_processors: List[str] = None) -> pd.DataFrame:
        """
        Extract and combine data from processor results
        
        Args:
            data: Dictionary of processor results
            preferred_processors: List of preferred processor types to use
            
        Returns:
            Combined DataFrame for analysis
        """
        if preferred_processors is None:
            preferred_processors = ['search', 'applicant', 'geographic', 'classification']
        
        dataframes = []
        
        for processor_name in preferred_processors:
            if processor_name in data:
                processor_result = data[processor_name]
                
                # Extract DataFrame from ProcessorResult or direct DataFrame
                if hasattr(processor_result, 'data'):
                    df = processor_result.data
                elif isinstance(processor_result, pd.DataFrame):
                    df = processor_result
                else:
                    continue
                
                if not df.empty:
                    dataframes.append(df)
                    self.logger.info(f"Using data from {processor_name}: {len(df)} records")
        
        if not dataframes:
            # Fallback: use any DataFrame found
            for key, value in data.items():
                if isinstance(value, pd.DataFrame) and not value.empty:
                    dataframes.append(value)
                    break
        
        # Combine DataFrames
        if dataframes:
            combined_df = dataframes[0].copy()
            
            # Merge additional DataFrames if they have common columns
            for df in dataframes[1:]:
                common_cols = list(set(combined_df.columns) & set(df.columns))
                if common_cols and len(common_cols) > 0:
                    # Use most specific ID column for merging
                    merge_col = None
                    for preferred_col in ['docdb_family_id', 'person_id', 'appln_id']:
                        if preferred_col in common_cols:
                            merge_col = preferred_col
                            break
                    
                    if merge_col:
                        try:
                            combined_df = combined_df.merge(df, on=merge_col, how='left', suffixes=('', '_additional'))
                        except Exception as e:
                            self.logger.warning(f"Failed to merge DataFrame: {e}")
            
            return combined_df
        
        else:
            self.logger.warning("No suitable DataFrames found for analysis")
            return pd.DataFrame()
    
    def _generate_executive_summary(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate executive summary from analysis results
        Should be overridden by specific analyzers
        
        Args:
            analysis_data: Analysis results data
            
        Returns:
            Executive summary dictionary
        """
        return {
            'analysis_type': self.__class__.__name__,
            'analysis_completed': True,
            'data_quality': 'Good' if analysis_data else 'Limited',
            'recommendations': ['Review detailed analysis for specific insights']
        }
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of analyzer performance and configuration"""
        return {
            'analyzer_type': self.__class__.__name__,
            'configuration': self.config,
            'total_analyses': self.total_analyses,
            'average_analysis_time': self.average_analysis_time,
            'recent_analyses': self.analysis_history[-5:] if self.analysis_history else []
        }