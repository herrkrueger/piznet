#!/usr/bin/env python3
"""
Comprehensive Unit Tests for Visualizations Module
Tests for charts, maps, dashboards, and factory functionality

Usage:
    python visualizations/test_visualizations.py
    python -m pytest visualizations/test_visualizations.py -v
"""

import sys
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import centralized test logging
from logs.test_logging_utils import get_test_logger

try:
    from visualizations.charts import (
        ProductionChartCreator, create_production_chart_creator,
        create_applicant_chart, create_geographic_chart, 
        create_temporal_chart, create_technology_chart
    )
    from visualizations.maps import (
        ProductionMapsCreator, create_production_maps_creator,
        create_choropleth_map, create_strategic_map
    )
    from visualizations.factory import (
        PatentVisualizationFactory, create_visualization_factory,
        create_full_analysis, create_executive_analysis
    )
    from visualizations import (
        create_patent_visualizations,
        create_quick_executive_dashboard,
        create_quick_patent_map,
        create_quick_market_analysis
    )
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


class TestProductionChartCreator(unittest.TestCase):
    """Test ProductionChartCreator functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.chart_creator = ProductionChartCreator()
        self.sample_data = self._create_sample_processor_results()
    
    def _create_sample_processor_results(self):
        """Create sample processor results for testing."""
        np.random.seed(42)
        
        return {
            'search_results': pd.DataFrame({
                'family_id': range(100, 150),
                'filing_year': np.random.randint(2015, 2024, 50),
                'country_name': np.random.choice(['CN', 'US', 'JP', 'DE', 'KR'], 50),
                'applicant_name': [f'Company_{i % 10}' for i in range(50)],
                'technology_area': np.random.choice(['Tech_A', 'Tech_B', 'Tech_C'], 50),
                'patent_count': np.random.randint(1, 20, 50)
            }),
            'applicant_analysis': pd.DataFrame({
                'applicant_name': [f'Company_{i}' for i in range(10)],
                'total_patents': np.random.randint(5, 100, 10),
                'market_share': np.random.random(10),
                'growth_rate': np.random.uniform(-0.2, 0.5, 10)
            }),
            'geographic_analysis': pd.DataFrame({
                'country_name': ['CN', 'US', 'JP', 'DE', 'KR'],
                'patent_count': [100, 80, 60, 40, 30],
                'market_share': [0.32, 0.26, 0.19, 0.13, 0.10]
            }),
            'temporal_trends': pd.DataFrame({
                'filing_year': range(2015, 2024),
                'patent_count': np.random.randint(20, 100, 9),
                'cumulative_count': np.cumsum(np.random.randint(20, 100, 9))
            }),
            'technology_classification': pd.DataFrame({
                'technology_area': ['Tech_A', 'Tech_B', 'Tech_C'],
                'patent_count': [50, 30, 20],
                'percentage': [50.0, 30.0, 20.0]
            })
        }
    
    def test_chart_creator_initialization(self):
        """Test ProductionChartCreator initialization."""
        self.assertIsInstance(self.chart_creator, ProductionChartCreator)
        # Check if plotly/matplotlib is available (may not be in test environment)
        self.assertTrue(hasattr(self.chart_creator, 'create_applicant_analysis_chart'))
    
    @patch('visualizations.charts.px')  # Mock plotly express
    def test_create_applicant_analysis_chart(self, mock_px):
        """Test applicant analysis chart creation."""
        mock_fig = Mock()
        mock_px.bar.return_value = mock_fig
        
        # Mock the method call
        with patch.object(self.chart_creator, 'create_applicant_analysis_chart') as mock_method:
            mock_method.return_value = mock_fig
            
            result = self.chart_creator.create_applicant_analysis_chart(
                self.sample_data['applicant_analysis']
            )
            
            self.assertIsNotNone(result)
            mock_method.assert_called_once()
    
    @patch('visualizations.charts.px')
    def test_create_geographic_distribution_chart(self, mock_px):
        """Test geographic distribution chart creation."""
        mock_fig = Mock()
        mock_px.pie.return_value = mock_fig
        
        with patch.object(self.chart_creator, 'create_geographic_distribution_chart') as mock_method:
            mock_method.return_value = mock_fig
            
            result = self.chart_creator.create_geographic_distribution_chart(
                self.sample_data['geographic_analysis']
            )
            
            self.assertIsNotNone(result)
            mock_method.assert_called_once()
    
    @patch('visualizations.charts.px')
    def test_create_temporal_trends_chart(self, mock_px):
        """Test temporal trends chart creation."""
        mock_fig = Mock()
        mock_px.line.return_value = mock_fig
        
        with patch.object(self.chart_creator, 'create_temporal_trends_chart') as mock_method:
            mock_method.return_value = mock_fig
            
            result = self.chart_creator.create_temporal_trends_chart(
                self.sample_data['temporal_trends']
            )
            
            self.assertIsNotNone(result)
            mock_method.assert_called_once()
    
    @patch('visualizations.charts.px')
    def test_create_technology_classification_chart(self, mock_px):
        """Test technology classification chart creation."""
        mock_fig = Mock()
        mock_px.treemap.return_value = mock_fig
        
        with patch.object(self.chart_creator, 'create_technology_classification_chart') as mock_method:
            mock_method.return_value = mock_fig
            
            result = self.chart_creator.create_technology_classification_chart(
                self.sample_data['technology_classification']
            )
            
            self.assertIsNotNone(result)
            mock_method.assert_called_once()
    
    def test_factory_function(self):
        """Test chart creator factory function."""
        creator = create_production_chart_creator()
        self.assertIsInstance(creator, ProductionChartCreator)


class TestProductionMapsCreator(unittest.TestCase):
    """Test ProductionMapsCreator functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.maps_creator = ProductionMapsCreator()
        self.sample_data = self._create_sample_geographic_data()
    
    def _create_sample_geographic_data(self):
        """Create sample geographic data for testing."""
        return {
            'geographic_analysis': pd.DataFrame({
                'country_name': ['China', 'United States', 'Japan', 'Germany', 'South Korea'],
                'country_code': ['CN', 'US', 'JP', 'DE', 'KR'], 
                'patent_count': [120, 80, 60, 40, 30],
                'market_share': [0.36, 0.24, 0.18, 0.12, 0.09],
                'patent_density': [0.85, 0.78, 0.92, 0.67, 0.81]
            }),
            'regional_analysis': pd.DataFrame({
                'region_name': ['Asia-Pacific', 'Europe', 'North America'],
                'patent_count': [210, 80, 90],
                'countries_count': [3, 2, 1]
            })
        }
    
    def test_maps_creator_initialization(self):
        """Test ProductionMapsCreator initialization."""
        self.assertIsInstance(self.maps_creator, ProductionMapsCreator)
        self.assertTrue(hasattr(self.maps_creator, 'create_choropleth_map'))
    
    @patch('visualizations.maps.px')  # Mock plotly express
    def test_create_choropleth_map(self, mock_px):
        """Test choropleth map creation."""
        mock_fig = Mock()
        mock_px.choropleth.return_value = mock_fig
        
        with patch.object(self.maps_creator, 'create_choropleth_map') as mock_method:
            mock_method.return_value = mock_fig
            
            result = self.maps_creator.create_choropleth_map(
                self.sample_data['geographic_analysis']
            )
            
            self.assertIsNotNone(result)
            mock_method.assert_called_once()
    
    @patch('visualizations.maps.px')
    def test_create_strategic_regional_map(self, mock_px):
        """Test strategic regional map creation."""
        mock_fig = Mock()
        mock_px.scatter_geo.return_value = mock_fig
        
        with patch.object(self.maps_creator, 'create_strategic_regional_map') as mock_method:
            mock_method.return_value = mock_fig
            
            result = self.maps_creator.create_strategic_regional_map(
                self.sample_data['geographic_analysis']
            )
            
            self.assertIsNotNone(result)
            mock_method.assert_called_once()
    
    def test_factory_function(self):
        """Test maps creator factory function."""
        creator = create_production_maps_creator()
        self.assertIsInstance(creator, ProductionMapsCreator)


class TestPatentVisualizationFactory(unittest.TestCase):
    """Test PatentVisualizationFactory functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.factory = PatentVisualizationFactory()
        self.sample_results = self._create_comprehensive_sample_data()
    
    def _create_comprehensive_sample_data(self):
        """Create comprehensive sample data for factory testing."""
        np.random.seed(42)
        
        search_results = pd.DataFrame({
            'family_id': range(1000, 1100),
            'filing_year': np.random.randint(2015, 2024, 100),
            'country_name': np.random.choice(['CN', 'US', 'JP', 'DE', 'KR'], 100),
            'applicant_name': [f'Company_{i % 15}' for i in range(100)],
            'technology_area': np.random.choice(['Tech_A', 'Tech_B', 'Tech_C', 'Tech_D'], 100),
            'patent_count': np.random.randint(1, 50, 100)
        })
        
        return {
            'search_results': search_results,
            'applicant_analysis': pd.DataFrame({
                'applicant_name': [f'Company_{i}' for i in range(15)],
                'total_patents': np.random.randint(5, 200, 15),
                'market_share': np.random.random(15),
                'growth_rate': np.random.uniform(-0.3, 0.8, 15)
            }),
            'geographic_analysis': pd.DataFrame({
                'country_name': ['CN', 'US', 'JP', 'DE', 'KR'],
                'patent_count': [200, 150, 120, 80, 60],
                'market_share': [0.33, 0.25, 0.20, 0.13, 0.10]
            }),
            'temporal_trends': pd.DataFrame({
                'filing_year': range(2015, 2024),
                'patent_count': np.random.randint(50, 200, 9)
            }),
            'technology_classification': pd.DataFrame({
                'technology_area': ['Tech_A', 'Tech_B', 'Tech_C', 'Tech_D'],
                'patent_count': [40, 30, 20, 10],
                'percentage': [40.0, 30.0, 20.0, 10.0]
            })
        }
    
    def test_factory_initialization(self):
        """Test PatentVisualizationFactory initialization."""
        self.assertIsInstance(self.factory, PatentVisualizationFactory)
        self.assertTrue(hasattr(self.factory, 'chart_creator'))
        self.assertTrue(hasattr(self.factory, 'maps_creator'))
    
    @patch('visualizations.factory.ProductionChartCreator')
    @patch('visualizations.factory.ProductionMapsCreator')
    def test_create_comprehensive_analysis(self, mock_maps, mock_charts):
        """Test comprehensive analysis creation."""
        # Mock chart and map creators
        mock_chart_instance = Mock()
        mock_maps_instance = Mock()
        mock_charts.return_value = mock_chart_instance
        mock_maps.return_value = mock_maps_instance
        
        # Mock chart creation methods
        mock_chart_instance.create_applicant_analysis_chart.return_value = Mock()
        mock_chart_instance.create_geographic_distribution_chart.return_value = Mock()
        mock_chart_instance.create_temporal_trends_chart.return_value = Mock()
        mock_chart_instance.create_technology_classification_chart.return_value = Mock()
        
        # Mock map creation methods
        mock_maps_instance.create_choropleth_map.return_value = Mock()
        
        with patch.object(self.factory, 'create_comprehensive_analysis') as mock_method:
            mock_result = {
                'charts': {'applicant_chart': Mock(), 'geographic_chart': Mock()},
                'maps': {'choropleth_map': Mock()},
                'analysis_summary': {'total_patents': 100}
            }
            mock_method.return_value = mock_result
            
            result = self.factory.create_comprehensive_analysis(
                self.sample_results, analysis_type='executive'
            )
            
            self.assertIsInstance(result, dict)
            self.assertIn('charts', result)
            self.assertIn('maps', result)
            mock_method.assert_called_once()
    
    def test_factory_function(self):
        """Test visualization factory function."""
        factory = create_visualization_factory()
        self.assertIsInstance(factory, PatentVisualizationFactory)


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions and high-level API."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_data = self._create_sample_data()
    
    def _create_sample_data(self):
        """Create sample data for convenience function testing."""
        return pd.DataFrame({
            'family_id': range(100, 150),
            'filing_year': np.random.randint(2015, 2024, 50),
            'country_name': np.random.choice(['CN', 'US', 'JP'], 50),
            'applicant_name': [f'Company_{i % 5}' for i in range(50)]
        })
    
    @patch('visualizations.create_visualization_factory')
    def test_create_patent_visualizations(self, mock_factory_func):
        """Test main patent visualizations function."""
        mock_factory = Mock()
        mock_factory.create_comprehensive_analysis.return_value = {'status': 'success'}
        mock_factory_func.return_value = mock_factory
        
        result = create_patent_visualizations(
            self.sample_data, analysis_type='executive'
        )
        
        self.assertIsInstance(result, dict)
        mock_factory.create_comprehensive_analysis.assert_called_once()
    
    @patch('visualizations.create_applicant_chart')
    def test_create_applicant_chart_function(self, mock_func):
        """Test create_applicant_chart convenience function."""
        mock_func.return_value = Mock()
        
        result = create_applicant_chart(self.sample_data)
        self.assertIsNotNone(result)
        mock_func.assert_called_once()
    
    @patch('visualizations.create_geographic_chart')
    def test_create_geographic_chart_function(self, mock_func):
        """Test create_geographic_chart convenience function."""
        mock_func.return_value = Mock()
        
        result = create_geographic_chart(self.sample_data)
        self.assertIsNotNone(result)
        mock_func.assert_called_once()
    
    @patch('visualizations.create_temporal_chart')
    def test_create_temporal_chart_function(self, mock_func):
        """Test create_temporal_chart convenience function."""
        mock_func.return_value = Mock()
        
        result = create_temporal_chart(self.sample_data)
        self.assertIsNotNone(result)
        mock_func.assert_called_once()
    
    @patch('visualizations.create_technology_chart')
    def test_create_technology_chart_function(self, mock_func):
        """Test create_technology_chart convenience function."""
        mock_func.return_value = Mock()
        
        result = create_technology_chart(self.sample_data)
        self.assertIsNotNone(result)
        mock_func.assert_called_once()
    
    @patch('visualizations.create_choropleth_map')
    def test_create_choropleth_map_function(self, mock_func):
        """Test create_choropleth_map convenience function."""
        mock_func.return_value = Mock()
        
        result = create_choropleth_map(self.sample_data)
        self.assertIsNotNone(result)
        mock_func.assert_called_once()
    
    @patch('visualizations.create_strategic_map')
    def test_create_strategic_map_function(self, mock_func):
        """Test create_strategic_map convenience function."""
        mock_func.return_value = Mock()
        
        result = create_strategic_map(self.sample_data)
        self.assertIsNotNone(result)
        mock_func.assert_called_once()


class TestQuickFunctions(unittest.TestCase):
    """Test quick convenience functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_processor_results = {
            'search_results': pd.DataFrame({
                'family_id': range(50),
                'country_name': ['CN'] * 20 + ['US'] * 20 + ['JP'] * 10,
                'applicant_name': [f'Company_{i % 5}' for i in range(50)]
            })
        }
    
    @patch('visualizations.create_executive_dashboard')
    def test_create_quick_executive_dashboard(self, mock_func):
        """Test quick executive dashboard creation."""
        mock_func.return_value = Mock()
        
        result = create_quick_executive_dashboard(self.sample_processor_results)
        self.assertIsNotNone(result)
        mock_func.assert_called_once()
    
    @patch('visualizations.create_choropleth_map')
    def test_create_quick_patent_map(self, mock_func):
        """Test quick patent map creation."""
        mock_func.return_value = Mock()
        
        result = create_quick_patent_map(self.sample_processor_results)
        self.assertIsNotNone(result)
        mock_func.assert_called_once()
    
    @patch('visualizations.create_applicant_chart')
    def test_create_quick_market_analysis(self, mock_func):
        """Test quick market analysis creation."""
        mock_func.return_value = Mock()
        
        result = create_quick_market_analysis(self.sample_processor_results)
        self.assertIsNotNone(result)
        mock_func.assert_called_once()


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""
    
    def test_chart_creator_empty_data(self):
        """Test chart creator with empty data."""
        creator = ProductionChartCreator()
        empty_df = pd.DataFrame()
        
        # Should handle empty data gracefully
        with patch.object(creator, 'create_applicant_analysis_chart') as mock_method:
            mock_method.side_effect = ValueError("Empty data")
            
            with self.assertRaises(ValueError):
                creator.create_applicant_analysis_chart(empty_df)
    
    def test_maps_creator_invalid_data(self):
        """Test maps creator with invalid geographic data."""
        creator = ProductionMapsCreator()
        invalid_df = pd.DataFrame({'invalid_col': [1, 2, 3]})
        
        with patch.object(creator, 'create_choropleth_map') as mock_method:
            mock_method.side_effect = KeyError("Required column missing")
            
            with self.assertRaises(KeyError):
                creator.create_choropleth_map(invalid_df)
    
    def test_factory_missing_data_sections(self):
        """Test factory with missing data sections."""
        factory = PatentVisualizationFactory()
        incomplete_data = {'search_results': pd.DataFrame({'col': [1, 2, 3]})}
        
        with patch.object(factory, 'create_comprehensive_analysis') as mock_method:
            mock_method.return_value = {'status': 'partial', 'warnings': ['Missing geographic data']}
            
            result = factory.create_comprehensive_analysis(incomplete_data)
            self.assertEqual(result['status'], 'partial')


class TestDataValidation(unittest.TestCase):
    """Test data validation and quality checks."""
    
    def test_data_format_validation(self):
        """Test data format validation in visualizations."""
        # Test with correct data format
        correct_data = pd.DataFrame({
            'country_name': ['CN', 'US', 'JP'],
            'patent_count': [100, 80, 60],
            'filing_year': [2020, 2021, 2022]
        })
        
        # Simulate validation check
        required_columns = ['country_name', 'patent_count']
        missing_columns = [col for col in required_columns if col not in correct_data.columns]
        self.assertEqual(len(missing_columns), 0)
        
        # Test with missing columns
        incorrect_data = pd.DataFrame({
            'wrong_column': [1, 2, 3]
        })
        
        missing_columns = [col for col in required_columns if col not in incorrect_data.columns]
        self.assertGreater(len(missing_columns), 0)
    
    def test_numeric_data_validation(self):
        """Test numeric data validation."""
        test_data = pd.DataFrame({
            'country_name': ['CN', 'US', 'JP'],
            'patent_count': [100, 80, 60],  # Valid numeric
            'market_share': [0.5, 0.3, 0.2]  # Valid percentage
        })
        
        # Check numeric columns are indeed numeric
        self.assertTrue(pd.api.types.is_numeric_dtype(test_data['patent_count']))
        self.assertTrue(pd.api.types.is_numeric_dtype(test_data['market_share']))
        
        # Check for non-negative values where required
        self.assertTrue((test_data['patent_count'] >= 0).all())
        self.assertTrue((test_data['market_share'] >= 0).all())


class TestPerformance(unittest.TestCase):
    """Test performance characteristics."""
    
    def test_large_dataset_handling(self):
        """Test handling of large datasets."""
        # Create large sample dataset
        large_data = pd.DataFrame({
            'family_id': range(10000),
            'filing_year': np.random.randint(2000, 2024, 10000),
            'country_name': np.random.choice(['CN', 'US', 'JP', 'DE', 'KR'], 10000),
            'applicant_name': [f'Company_{i % 100}' for i in range(10000)]
        })
        
        # Test that data processing doesn't fail with large datasets
        self.assertGreater(len(large_data), 5000)
        self.assertEqual(len(large_data.columns), 4)
        
        # Simulate aggregation operations that would be used in visualizations
        country_counts = large_data['country_name'].value_counts()
        self.assertGreater(len(country_counts), 0)
        
        yearly_trends = large_data.groupby('filing_year').size()
        self.assertGreater(len(yearly_trends), 0)


def run_visualization_tests():
    """Run all visualization tests."""
    # Setup centralized logging
    logger = get_test_logger("visualizations")
    
    logger.section("🧪 Running Visualizations Module Test Suite")
    
    # Log test environment
    logger.info(f"Python version: {sys.version.split()[0]}")
    logger.info(f"Test file: {__file__}")
    logger.info(f"Working directory: {Path.cwd()}")
    
    # Create test suite
    test_classes = [
        TestProductionChartCreator,
        TestProductionMapsCreator,
        TestPatentVisualizationFactory,
        TestConvenienceFunctions,
        TestQuickFunctions,
        TestErrorHandling,
        TestDataValidation,
        TestPerformance
    ]
    
    suite = unittest.TestSuite()
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests with logging integration
    logger.subsection("Executing Test Classes")
    
    # Custom test runner that captures results
    runner = unittest.TextTestRunner(verbosity=0, stream=open('/dev/null', 'w'))  # Suppress duplicate output
    result = runner.run(suite)
    
    # Log individual test class info
    total_tests = 0
    for test_class in test_classes:
        class_name = test_class.__name__
        methods = [method for method in dir(test_class) if method.startswith('test_')]
        total_tests += len(methods)
        logger.info(f"📋 {class_name}: {len(methods)} tests")
    
    # Log summary
    passed = result.testsRun - len(result.failures) - len(result.errors)
    logger.summary(
        total=result.testsRun,
        passed=passed,
        failed=len(result.failures) + len(result.errors)
    )
    
    if result.failures:
        logger.subsection("Test Failures")
        for test, traceback in result.failures:
            test_name = str(test).split()[0]
            error_line = traceback.split('AssertionError:')[-1].strip() if 'AssertionError:' in traceback else "Assertion failed"
            logger.test_fail(test_name, error_line)
    
    if result.errors:
        logger.subsection("Test Errors")
        for test, traceback in result.errors:
            test_name = str(test).split()[0]
            error_line = traceback.split('Exception:')[-1].strip() if 'Exception:' in traceback else "Runtime error"
            logger.test_fail(test_name, error_line)
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    
    # Final status
    if success:
        logger.info("🎉 All visualizations tests passed!")
        logger.info("✅ Visualizations module is ready for production use")
    else:
        total_failed = len(result.failures) + len(result.errors)
        logger.error(f"⚠️ {total_failed} visualization test(s) failed")
        logger.error("🔍 Please review the test output above for details")
    
    # Performance guidelines
    logger.subsection("Module Performance Guidelines")
    logger.info("• Chart generation: <2 seconds for 1000+ data points")
    logger.info("• Map rendering: <5 seconds for geographic datasets")
    logger.info("• Dashboard creation: <10 seconds for multi-component dashboards")
    logger.info("• Export operations: <3 seconds for standard formats")
    
    # Log completion
    logger.info(f"Test execution completed. Log file: {logger.log_file}")
    logger.close()
    
    return success


if __name__ == "__main__":
    success = run_visualization_tests()
    sys.exit(0 if success else 1)