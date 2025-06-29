"""
Base Processor Classes - Clean Architecture Foundation
Standardized interfaces for all data processing components
"""

import pandas as pd
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import time

# Setup logging
logger = logging.getLogger(__name__)


class ProcessorResult:
    """
    Standardized result container for all processors
    """
    
    def __init__(self, data: pd.DataFrame, metadata: Dict[str, Any], 
                 status: str = "completed", errors: List[str] = None):
        """
        Initialize processor result
        
        Args:
            data: Processed DataFrame
            metadata: Processing metadata and statistics
            status: Processing status ('completed', 'partial', 'failed')
            errors: List of any errors encountered
        """
        self.data = data
        self.metadata = metadata or {}
        self.status = status
        self.errors = errors or []
        self.timestamp = datetime.now().isoformat()
        
        # Add basic statistics
        if isinstance(data, pd.DataFrame):
            self.metadata.update({
                'record_count': len(data),
                'column_count': len(data.columns),
                'memory_usage': data.memory_usage(deep=True).sum()
            })
    
    @property
    def is_successful(self) -> bool:
        """Check if processing was successful"""
        return self.status == "completed" and not self.errors
    
    def add_error(self, error_message: str):
        """Add an error to the result"""
        self.errors.append(error_message)
        if self.status == "completed":
            self.status = "partial"


class BaseProcessor(ABC):
    """
    Abstract base class for all processors
    Ensures standardized interface and common functionality
    """
    
    def __init__(self, data_provider=None, config: Dict[str, Any] = None):
        """
        Initialize base processor
        
        Args:
            data_provider: Data access provider (dependency injection)
            config: Configuration parameters
        """
        self.data_provider = data_provider
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Performance tracking
        self.processing_history = []
        self.total_records_processed = 0
        self.average_processing_time = 0.0
    
    @abstractmethod
    def process(self, data: pd.DataFrame, **kwargs) -> ProcessorResult:
        """
        Process data and return standardized result
        
        Args:
            data: Input DataFrame to process
            **kwargs: Additional processing parameters
            
        Returns:
            ProcessorResult with processed data and metadata
        """
        pass
    
    def _validate_input(self, data: pd.DataFrame) -> bool:
        """
        Validate input data before processing
        
        Args:
            data: Input DataFrame
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(data, pd.DataFrame):
            self.logger.error(f"Invalid input type: {type(data)}. Expected pandas DataFrame")
            return False
        
        if data.empty:
            self.logger.warning("Input DataFrame is empty")
            return False
        
        return True
    
    def _create_metadata(self, input_data: pd.DataFrame, processing_time: float, 
                        additional_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create standardized metadata for processing results
        
        Args:
            input_data: Original input data
            processing_time: Time taken for processing
            additional_metadata: Additional processor-specific metadata
            
        Returns:
            Complete metadata dictionary
        """
        metadata = {
            'processor_type': self.__class__.__name__,
            'processing_timestamp': datetime.now().isoformat(),
            'processing_time': processing_time,
            'input_records': len(input_data),
            'input_columns': list(input_data.columns),
            'configuration': self.config.copy(),
            'data_provider': str(type(self.data_provider).__name__) if self.data_provider else None
        }
        
        if additional_metadata:
            metadata.update(additional_metadata)
        
        return metadata
    
    def _update_performance_metrics(self, processing_time: float, record_count: int):
        """Update processor performance metrics"""
        self.processing_history.append({
            'timestamp': datetime.now().isoformat(),
            'processing_time': processing_time,
            'record_count': record_count
        })
        
        self.total_records_processed += record_count
        
        # Calculate moving average
        recent_times = [h['processing_time'] for h in self.processing_history[-10:]]
        self.average_processing_time = sum(recent_times) / len(recent_times)


class ProcessorPipeline:
    """
    Orchestrates multiple processors in sequence
    Provides standardized pipeline execution with error handling
    """
    
    def __init__(self, data_provider=None, config: Dict[str, Any] = None):
        """
        Initialize processor pipeline
        
        Args:
            data_provider: Shared data provider for all processors
            config: Global configuration
        """
        self.data_provider = data_provider
        self.config = config or {}
        self.processors: List[BaseProcessor] = []
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Pipeline metrics
        self.execution_history = []
    
    def add_processor(self, processor: BaseProcessor):
        """
        Add a processor to the pipeline
        
        Args:
            processor: Processor instance to add
        """
        if not isinstance(processor, BaseProcessor):
            raise ValueError(f"Processor must inherit from BaseProcessor, got {type(processor)}")
        
        # Inject dependencies if not already set
        if processor.data_provider is None:
            processor.data_provider = self.data_provider
        
        if not processor.config:
            processor.config = self.config.copy()
        
        self.processors.append(processor)
        self.logger.info(f"Added processor: {processor.__class__.__name__}")
    
    def execute(self, initial_data: pd.DataFrame, **kwargs) -> Dict[str, ProcessorResult]:
        """
        Execute all processors in the pipeline
        
        Args:
            initial_data: Initial data to process
            **kwargs: Additional execution parameters
            
        Returns:
            Dictionary of processor results by processor name
        """
        start_time = time.time()
        results = {}
        current_data = initial_data.copy()
        
        self.logger.info(f"Starting pipeline execution with {len(self.processors)} processors")
        
        for i, processor in enumerate(self.processors):
            processor_name = processor.__class__.__name__
            
            try:
                self.logger.info(f"Executing processor {i+1}/{len(self.processors)}: {processor_name}")
                
                # Execute processor
                processor_start = time.time()
                result = processor.process(current_data, **kwargs)
                processor_time = time.time() - processor_start
                
                # Store result
                results[processor_name] = result
                
                # Update current data for next processor (if successful)
                if result.is_successful and not result.data.empty:
                    current_data = result.data
                
                self.logger.info(f"✅ {processor_name} completed in {processor_time:.2f}s")
                
            except Exception as e:
                error_msg = f"Processor {processor_name} failed: {str(e)}"
                self.logger.error(error_msg)
                
                # Create error result
                results[processor_name] = ProcessorResult(
                    data=pd.DataFrame(),
                    metadata={'processor_type': processor_name, 'error': str(e)},
                    status='failed',
                    errors=[error_msg]
                )
        
        # Pipeline execution summary
        execution_time = time.time() - start_time
        successful_processors = sum(1 for r in results.values() if r.is_successful)
        
        execution_summary = {
            'timestamp': datetime.now().isoformat(),
            'execution_time': execution_time,
            'total_processors': len(self.processors),
            'successful_processors': successful_processors,
            'success_rate': successful_processors / len(self.processors) if self.processors else 0,
            'final_record_count': len(current_data)
        }
        
        self.execution_history.append(execution_summary)
        
        self.logger.info(f"Pipeline execution completed in {execution_time:.2f}s")
        self.logger.info(f"Success rate: {successful_processors}/{len(self.processors)} processors")
        
        return results
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get summary of pipeline configuration and performance"""
        return {
            'processor_count': len(self.processors),
            'processor_types': [p.__class__.__name__ for p in self.processors],
            'execution_count': len(self.execution_history),
            'average_execution_time': sum(h['execution_time'] for h in self.execution_history[-5:]) / min(5, len(self.execution_history)) if self.execution_history else 0,
            'latest_execution': self.execution_history[-1] if self.execution_history else None
        }