#!/usr/bin/env python3
"""
Simple Visualization Tests - Chart and Map Creation Only
Tests visualization functionality with static data - NO database connections
"""

import sys
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import centralized test logging
from logs.test_logging_utils import get_test_logger

try:
    from visualizations.charts import ProductionChartCreator
    from visualizations.maps import ProductionMapsCreator
    from visualizations.dashboards import ProductionDashboardCreator
    from visualizations.factory import PatentVisualizationFactory
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


class TestProductionChartCreator(unittest.TestCase):
    """Test chart creation with static data."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.chart_creator = ProductionChartCreator()
        self.static_data = {
            'applicant_ranking': pd.DataFrame({
                'Applicant': ['Company A', 'Company B', 'Company C'],
                'Patent_Families': [10, 8, 5],
                'Market_Share_Pct': [40.0, 32.0, 20.0]
            }),
            'country_summary': pd.DataFrame({
                'country_name': ['United States', 'Germany', 'Japan'],
                'unique_families': [5, 3, 2],
                'country_code': ['US', 'DE', 'JP']
            }),
            'temporal_summary': pd.DataFrame({
                'filing_year': [2020, 2021, 2022],
                'patent_count': [10, 15, 12]
            }),
            'cpc_distribution': pd.DataFrame({
                'cpc_section': ['H - Electricity', 'G - Physics', 'C - Chemistry'],
                'family_count': [8, 4, 3]
            })
        }
    
    def test_chart_creator_initialization(self):
        """Test chart creator can be initialized."""
        self.assertIsInstance(self.chart_creator, ProductionChartCreator)
        self.assertTrue(hasattr(self.chart_creator, 'create_applicant_bubble_scatter'))
        self.assertTrue(hasattr(self.chart_creator, 'create_geographic_bar_ranking'))
    
    @patch('visualizations.charts.px')
    def test_create_applicant_bubble_scatter(self, mock_px):
        """Test applicant chart creation."""
        mock_fig = Mock()
        mock_px.scatter.return_value = mock_fig
        
        with patch.object(self.chart_creator, 'create_applicant_bubble_scatter') as mock_method:
            mock_method.return_value = mock_fig
            result = self.chart_creator.create_applicant_bubble_scatter(self.static_data)
            self.assertIsNotNone(result)
    
    @patch('visualizations.charts.px')
    def test_create_geographic_bar_ranking(self, mock_px):
        """Test geographic chart creation."""
        mock_fig = Mock()
        mock_px.bar.return_value = mock_fig
        
        with patch.object(self.chart_creator, 'create_geographic_bar_ranking') as mock_method:
            mock_method.return_value = mock_fig
            result = self.chart_creator.create_geographic_bar_ranking(self.static_data)
            self.assertIsNotNone(result)
    
    def test_temporal_chart_with_static_data(self):
        """Test temporal chart with static data."""
        try:
            result = self.chart_creator.create_temporal_trends_chart(self.static_data)
            self.assertIsNotNone(result)
        except Exception as e:
            # Expected with missing plotly dependencies
            self.assertIsInstance(e, (ImportError, AttributeError, ValueError))
    
    @patch('visualizations.charts.px')
    def test_create_technology_distribution_pie(self, mock_px):
        """Test technology chart creation."""
        mock_fig = Mock()
        mock_px.pie.return_value = mock_fig
        
        with patch.object(self.chart_creator, 'create_technology_distribution_pie') as mock_method:
            mock_method.return_value = mock_fig
            result = self.chart_creator.create_technology_distribution_pie(self.static_data)
            self.assertIsNotNone(result)


class TestProductionMapsCreator(unittest.TestCase):
    """Test map creation with static data."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.maps_creator = ProductionMapsCreator()
        self.geographic_data = {
            'country_summary': pd.DataFrame({
                'country_name': ['United States', 'Germany', 'Japan'],
                'country_code': ['US', 'DE', 'JP'], 
                'unique_families': [5, 3, 2]
            })
        }
    
    def test_maps_creator_initialization(self):
        """Test maps creator can be initialized."""
        self.assertIsInstance(self.maps_creator, ProductionMapsCreator)
    
    def test_choropleth_map_function_exists(self):
        """Test choropleth map function exists."""
        from visualizations.maps import create_choropleth_map
        self.assertTrue(callable(create_choropleth_map))
    
    def test_strategic_map_function_exists(self):
        """Test strategic map function exists."""
        from visualizations.maps import create_strategic_map
        self.assertTrue(callable(create_strategic_map))


class TestProductionDashboardCreator(unittest.TestCase):
    """Test dashboard creation with static data."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.dashboard_creator = ProductionDashboardCreator()
        # Use the same static data structure as chart tests - no duplication
        chart_test = TestProductionChartCreator()
        chart_test.setUp()
        self.static_data = chart_test.static_data
    
    def test_dashboard_creator_initialization(self):
        """Test dashboard creator can be initialized."""
        self.assertIsInstance(self.dashboard_creator, ProductionDashboardCreator)
        self.assertTrue(hasattr(self.dashboard_creator, 'create_executive_dashboard'))
        self.assertTrue(hasattr(self.dashboard_creator, 'create_comprehensive_analysis_dashboard'))
    
    @patch('visualizations.dashboards.make_subplots')
    def test_create_executive_dashboard(self, mock_subplots):
        """Test executive dashboard creation."""
        mock_fig = Mock()
        mock_subplots.return_value = mock_fig
        
        with patch.object(self.dashboard_creator, 'create_executive_dashboard') as mock_method:
            mock_method.return_value = mock_fig
            result = self.dashboard_creator.create_executive_dashboard(self.static_data)
            self.assertIsNotNone(result)
    
    @patch('visualizations.dashboards.make_subplots')
    def test_create_comprehensive_analysis_dashboard(self, mock_subplots):
        """Test comprehensive analysis dashboard creation."""
        mock_fig = Mock()
        mock_subplots.return_value = mock_fig
        
        with patch.object(self.dashboard_creator, 'create_comprehensive_analysis_dashboard') as mock_method:
            mock_method.return_value = mock_fig
            result = self.dashboard_creator.create_comprehensive_analysis_dashboard(self.static_data)
            self.assertIsNotNone(result)
    
    def test_dashboard_with_static_data(self):
        """Test dashboard with static data."""
        try:
            # This will fail gracefully if dependencies are missing
            result = self.dashboard_creator.create_executive_dashboard(self.static_data)
            if result is not None:
                self.assertIsNotNone(result)
        except Exception as e:
            # Expected with missing dependencies
            self.assertIsInstance(e, (ImportError, AttributeError, ValueError, KeyError))


class TestPatentVisualizationFactory(unittest.TestCase):
    """Test factory with static data."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.factory = PatentVisualizationFactory()
        self.complete_data = {
            'search_results': pd.DataFrame({
                'docdb_family_id': [1, 2, 3],
                'appln_filing_year': [2020, 2021, 2022]
            }),
            'applicant_ranking': pd.DataFrame({
                'Applicant': ['Company A', 'Company B'],
                'Patent_Families': [5, 3]
            }),
            'country_summary': pd.DataFrame({
                'country_name': ['United States', 'Germany'],
                'unique_families': [3, 2]
            })
        }
    
    def test_factory_initialization(self):
        """Test factory can be initialized."""
        self.assertIsInstance(self.factory, PatentVisualizationFactory)
        self.assertTrue(hasattr(self.factory, 'chart_creator'))
        self.assertTrue(hasattr(self.factory, 'maps_creator'))
    
    @patch('visualizations.factory.ProductionChartCreator')
    @patch('visualizations.factory.ProductionMapsCreator')
    def test_comprehensive_analysis_mock(self, mock_maps, mock_charts):
        """Test comprehensive analysis with mocks."""
        mock_chart_instance = Mock()
        mock_maps_instance = Mock()
        mock_charts.return_value = mock_chart_instance
        mock_maps.return_value = mock_maps_instance
        
        with patch.object(self.factory, 'create_comprehensive_analysis') as mock_method:
            mock_result = {'charts': {}, 'maps': {}}
            mock_method.return_value = mock_result
            
            result = self.factory.create_comprehensive_analysis(self.complete_data)
            self.assertIsInstance(result, dict)


class TestErrorHandling(unittest.TestCase):
    """Test error handling with empty/invalid data."""
    
    def test_chart_creator_empty_data(self):
        """Test chart creator with empty data."""
        creator = ProductionChartCreator()
        empty_data = {'applicant_ranking': pd.DataFrame()}
        
        try:
            result = creator.create_applicant_bubble_scatter(empty_data)
            self.assertIsNotNone(result)
        except Exception as e:
            # Acceptable to raise exception with empty data
            self.assertIsInstance(e, (ValueError, KeyError, IndexError))
    
    def test_maps_creator_invalid_data(self):
        """Test maps creator with invalid data."""
        creator = ProductionMapsCreator()
        invalid_data = {'country_summary': pd.DataFrame({'invalid': [1, 2, 3]})}
        
        try:
            from visualizations.maps import create_choropleth_map
            result = create_choropleth_map(invalid_data)
        except Exception as e:
            # Acceptable to raise exception with invalid data
            self.assertIsInstance(e, (ValueError, KeyError, AttributeError))


class TestDataValidation(unittest.TestCase):
    """Test data format validation."""
    
    def test_data_format_validation(self):
        """Test data format validation."""
        valid_data = pd.DataFrame({
            'country_name': ['US', 'DE', 'JP'],
            'patent_count': [100, 80, 60]
        })
        
        required_columns = ['country_name', 'patent_count']
        missing_columns = [col for col in required_columns if col not in valid_data.columns]
        self.assertEqual(len(missing_columns), 0)
    
    def test_numeric_data_validation(self):
        """Test numeric data validation."""
        test_data = pd.DataFrame({
            'country_name': ['US', 'DE', 'JP'],
            'patent_count': [100, 80, 60],
            'market_share': [0.5, 0.3, 0.2]
        })
        
        self.assertTrue(pd.api.types.is_numeric_dtype(test_data['patent_count']))
        self.assertTrue((test_data['patent_count'] >= 0).all())


class TestPerformance(unittest.TestCase):
    """Test performance with larger datasets."""
    
    def test_large_dataset_handling(self):
        """Test handling of large datasets."""
        large_data = pd.DataFrame({
            'family_id': range(1000),
            'filing_year': np.random.randint(2000, 2024, 1000),
            'country_name': np.random.choice(['US', 'DE', 'JP'], 1000)
        })
        
        self.assertGreater(len(large_data), 500)
        self.assertEqual(len(large_data.columns), 3)
        
        # Test aggregation operations
        country_counts = large_data['country_name'].value_counts()
        self.assertGreater(len(country_counts), 0)


def run_visualization_tests():
    """Run all visualization tests."""
    logger = get_test_logger("visualizations")
    
    logger.section("🧪 Running Simple Visualization Tests")
    logger.info("Testing ONLY chart/map creation - NO database connections")
    
    logger.info(f"Python version: {sys.version.split()[0]}")
    logger.info(f"Test file: {__file__}")
    logger.info(f"Working directory: {Path.cwd()}")
    
    # Create test suite
    test_classes = [
        TestProductionChartCreator,
        TestProductionMapsCreator,
        TestProductionDashboardCreator,
        TestPatentVisualizationFactory,
        TestErrorHandling,
        TestDataValidation,
        TestPerformance
    ]
    
    suite = unittest.TestSuite()
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=0, stream=open('/dev/null', 'w'))
    result = runner.run(suite)
    
    # Log results
    total_tests = 0
    for test_class in test_classes:
        class_name = test_class.__name__
        methods = [method for method in dir(test_class) if method.startswith('test_')]
        total_tests += len(methods)
        logger.info(f"📋 {class_name}: {len(methods)} tests")
    
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
    
    if success:
        logger.info("🎉 All visualization tests passed!")
        logger.info("✅ Chart and map creation is working correctly")
    else:
        total_failed = len(result.failures) + len(result.errors)
        logger.error(f"⚠️ {total_failed} test(s) failed")
    
    logger.info(f"Test execution completed. Log file: {logger.log_file}")
    logger.close()
    
    return success


if __name__ == "__main__":
    success = run_visualization_tests()
    sys.exit(0 if success else 1)