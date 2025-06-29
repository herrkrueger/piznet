#!/usr/bin/env python3
"""
Unit Tests for Individual Processors
Enhanced from EPO PATLIB 2025 Live Demo Code

This script tests each processor individually to ensure core functionality.
Use this for debugging specific processor issues.

Usage:
    python processors/test_unit.py
    python processors/test_unit.py --processor search
    python processors/test_unit.py --processor applicant
"""

import sys
import os
from pathlib import Path
import logging
import pandas as pd
import time
from typing import Dict, Any, List
import argparse

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import centralized test logging
from logs.test_logging_utils import get_test_logger

def setup_logging():
    """Setup test logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

def print_section(title: str, char: str = '=', width: int = 60):
    """Print a formatted section header."""
    print(f'\n{title}')
    print(char * width)

def print_subsection(title: str):
    """Print a formatted subsection header."""
    print(f'\n{title}')
    print('-' * 40)

def test_search_processor():
    """Test PatentSearchProcessor individually."""
    print_section('🔍 Search Processor Unit Test')
    
    try:
        from processors import create_patent_search_processor
        
        print_subsection('Initialization & Configuration')
        processor = create_patent_search_processor()
        config = processor.config
        print(f'   ✅ Processor created with {len(config)} config sections')
        
        print_subsection('Keyword Extraction')
        keywords = processor._get_all_configured_keywords()
        print(f'   ✅ Extracted {len(keywords)} keywords')
        
        print_subsection('PATSTAT Connection')
        if processor.patstat_client:
            print('   ✅ PATSTAT client available')
        else:
            print('   ⚠️ No PATSTAT client (testing mode)')
            
        print_subsection('Search Logic Test')
        empty_results = processor._combine_search_results([], 'intersection')
        print(f'   ✅ Empty result handling: {len(empty_results)} families')
        
        return True
        
    except Exception as e:
        print(f'❌ Search processor test failed: {e}')
        return False

def test_applicant_processor():
    """Test ApplicantAnalyzer individually."""
    print_section('👥 Applicant Processor Unit Test')
    
    try:
        from processors import create_applicant_analyzer
        import pandas as pd
        import numpy as np
        
        print_subsection('Initialization')
        analyzer = create_applicant_analyzer()
        print('   ✅ Applicant analyzer created')
        
        print_subsection('Real PATSTAT Data Processing')
        # Load real PATSTAT IDs from config
        try:
            from config import get_search_patterns_config
            real_families = get_search_patterns_config('demo_parameters.test_families')[:2]
            real_applications = get_search_patterns_config('demo_parameters.test_applications')[:2]
        except:
            # Fallback if config not available
            real_families = [72941368, 80821052]
            real_applications = [540061057, 596607294]
        
        search_results = pd.DataFrame({
            'appln_id': real_applications,  # Application IDs for PATSTAT JOINs
            'docdb_family_id': real_families,  # Family IDs for grouping
            'quality_score': [3, 2],
            'match_type': ['intersection', 'keyword'],
            'earliest_filing_year': [2019, 2020],  # Real filing years
            'family_size': [15, 15]  # Real family sizes
        })
        
        result = analyzer.analyze_search_results(search_results)
        print(f'   ✅ Processed {len(result)} applicant records')
        print(f'   📊 Columns: {len(result.columns)} attributes')
        
        print_subsection('Summary Generation')
        summary = analyzer.get_applicant_summary()
        print(f'   ✅ Summary generated: {summary["status"]}')
        
        return True
        
    except Exception as e:
        print(f'❌ Applicant processor test failed: {e}')
        return False

def test_classification_processor():
    """Test ClassificationProcessor individually."""
    print_section('🏷️ Classification Processor Unit Test')
    
    try:
        from processors import create_classification_processor
        import pandas as pd
        
        print_subsection('Initialization')
        processor = create_classification_processor()
        print('   ✅ Classification processor created')
        
        print_subsection('Real PATSTAT Data Processing')
        # Load real PATSTAT IDs from config
        try:
            from config import get_search_patterns_config
            real_families = get_search_patterns_config('demo_parameters.test_families')[:2]
            real_applications = get_search_patterns_config('demo_parameters.test_applications')[:2]
        except:
            # Fallback if config not available
            real_families = [72941368, 80821052]
            real_applications = [540061057, 596607294]
        
        search_results = pd.DataFrame({
            'appln_id': real_applications,  # Application IDs for PATSTAT JOINs
            'docdb_family_id': real_families,  # Family IDs for grouping
            'quality_score': [3, 2],
            'match_type': ['intersection', 'keyword'],
            'earliest_filing_year': [2019, 2020],  # Real filing years
            'family_size': [15, 15]  # Real family sizes
        })
        
        result = processor.analyze_search_results(search_results)
        print(f'   ✅ Processed {len(result)} classification records')
        if not result.empty:
            print(f'   📊 Columns: {len(result.columns)} attributes')
        
        return True
        
    except Exception as e:
        print(f'❌ Classification processor test failed: {e}')
        return False

def test_citation_processor():
    """Test CitationAnalyzer individually."""
    print_section('🔗 Citation Processor Unit Test')
    
    try:
        from processors import create_citation_analyzer
        import pandas as pd
        
        print_subsection('Initialization')
        analyzer = create_citation_analyzer()
        print('   ✅ Citation analyzer created')
        
        print_subsection('Real PATSTAT Data Processing')
        # Load real PATSTAT IDs from config
        try:
            from config import get_search_patterns_config
            real_families = get_search_patterns_config('demo_parameters.test_families')[:2]
            real_applications = get_search_patterns_config('demo_parameters.test_applications')[:2]
        except:
            # Fallback if config not available
            real_families = [72941368, 80821052]
            real_applications = [540061057, 596607294]
        
        search_results = pd.DataFrame({
            'appln_id': real_applications,  # Application IDs for PATSTAT JOINs
            'docdb_family_id': real_families,  # Family IDs for grouping
            'quality_score': [3, 2],
            'match_type': ['intersection', 'keyword'],
            'earliest_filing_year': [2019, 2020],  # Real filing years
            'family_size': [15, 15]  # Real family sizes
        })
        
        result = analyzer.analyze_search_results(search_results)
        print(f'   ✅ Processed {len(result)} citation records')
        if not result.empty:
            print(f'   📊 Columns: {len(result.columns)} attributes')
        else:
            print('   📝 No citation data (expected with mock family IDs)')
        
        return True
        
    except Exception as e:
        print(f'❌ Citation processor test failed: {e}')
        return False

def test_geographic_processor():
    """Test GeographicAnalyzer individually with NUTS and inventor support."""
    print_section('🌍 Geographic Processor Unit Test')
    
    try:
        from processors import create_geographic_analyzer
        import pandas as pd
        
        print_subsection('Initialization')
        analyzer = create_geographic_analyzer()
        print('   ✅ Geographic analyzer created')
        
        # Test NUTS mapper availability
        if hasattr(analyzer, 'nuts_mapper') and analyzer.nuts_mapper:
            print('   ✅ NUTS mapper available')
        else:
            print('   ⚠️ NUTS mapper not available (fallback mode)')
        
        # Test country mapper availability  
        if hasattr(analyzer, 'country_mapper') and analyzer.country_mapper:
            print('   ✅ Country mapper available')
        else:
            print('   ⚠️ Country mapper not available (fallback mode)')
        
        print_subsection('Role-Based Analysis Parameters')
        search_results = pd.DataFrame({
            'docdb_family_id': [72941368, 80821052],  # Real PATSTAT family IDs
            'quality_score': [3, 2],
            'match_type': ['intersection', 'keyword'],
            'earliest_filing_year': [2020, 2019],
            'family_size': [5, 3]
        })
        
        # Test applicant-only analysis
        print('   Testing applicant-only analysis...')
        result_applicants = analyzer.analyze_search_results(
            search_results, 
            analyze_applicants=True, 
            analyze_inventors=False,
            nuts_level=3
        )
        print(f'   ✅ Applicant analysis: {len(result_applicants)} records')
        
        # Test inventor-only analysis
        print('   Testing inventor-only analysis...')
        result_inventors = analyzer.analyze_search_results(
            search_results,
            analyze_applicants=False,
            analyze_inventors=True,
            nuts_level=3
        )
        print(f'   ✅ Inventor analysis: {len(result_inventors)} records')
        
        # Test combined analysis
        print('   Testing combined analysis...')
        result_combined = analyzer.analyze_search_results(
            search_results,
            analyze_applicants=True,
            analyze_inventors=True,
            nuts_level=2
        )
        print(f'   ✅ Combined analysis: {len(result_combined)} records')
        
        print_subsection('Specialized Geographic Methods')
        # Test specialized methods (they should handle empty data gracefully)
        try:
            inventor_geo = analyzer.analyze_inventor_geography(search_results, nuts_level=3)
            print(f'   ✅ Inventor geography method: {len(inventor_geo)} records')
        except Exception as e:
            print(f'   ⚠️ Inventor geography: {e}')
        
        try:
            applicant_geo = analyzer.analyze_applicant_geography(search_results, nuts_level=3)
            print(f'   ✅ Applicant geography method: {len(applicant_geo)} records')
        except Exception as e:
            print(f'   ⚠️ Applicant geography: {e}')
        
        try:
            comparison = analyzer.compare_innovation_vs_filing_geography(search_results, nuts_level=2)
            print(f'   ✅ Geographic comparison method: {len(comparison)} keys')
        except Exception as e:
            print(f'   ⚠️ Geographic comparison: {e}')
        
        print_subsection('NUTS Integration Testing')
        # Test NUTS helper methods if available
        if hasattr(analyzer, '_get_nuts_info'):
            nuts_info = analyzer._get_nuts_info('DE111')
            print(f'   ✅ NUTS info retrieval: {nuts_info.get("nuts_code", "Unknown")}')
        
        if hasattr(analyzer, '_get_target_nuts_from_hierarchy'):
            target_nuts = analyzer._get_target_nuts_from_hierarchy(['DE', 'DE1', 'DE11', 'DE111'], 2)
            print(f'   ✅ NUTS hierarchy navigation: {target_nuts}')
        
        print_subsection('Geographic Mapper Integration')
        country_info = analyzer._get_country_info('DE')
        print(f'   ✅ Country info: {country_info.get("name", "Unknown")}')
        
        return True
        
    except Exception as e:
        print(f'❌ Geographic processor test failed: {e}')
        import traceback
        traceback.print_exc()
        return False

def test_working_directory_independence():
    """Test that processors work regardless of current working directory (notebook scenario)."""
    print_section('📁 Working Directory Independence Test', '=', 60)
    print('Testing processor initialization from different working directories...')
    
    try:
        import os
        import tempfile
        import shutil
        
        # Save current working directory
        original_cwd = os.getcwd()
        
        # Create a temporary directory and change to it
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)
            print(f'   🔄 Changed working directory to: {temp_dir}')
            
            # Test that citation processor can still load config
            print('   🧪 Testing CitationAnalyzer config loading from different directory...')
            try:
                from processors import CitationAnalyzer
                from data_access import PatstatClient
                patstat = PatstatClient(environment='PROD')
                citation_analyzer = CitationAnalyzer(patstat)
                print('   ✅ CitationAnalyzer initialized successfully from different directory')
                
                # Check if config was loaded
                if hasattr(citation_analyzer, 'citation_config') and citation_analyzer.citation_config:
                    print('   ✅ Citation config loaded successfully')
                else:
                    print('   ⚠️ Citation config is empty (expected if citation_analysis not in YAML)')
                
                return True
                
            except Exception as e:
                print(f'   ❌ CitationAnalyzer failed from different directory: {e}')
                return False
            
            finally:
                # Always restore original working directory
                os.chdir(original_cwd)
                print(f'   🔄 Restored working directory to: {original_cwd}')
        
    except Exception as e:
        print(f'❌ Working directory test failed: {e}')
        return False

def run_all_unit_tests():
    """Run all unit tests."""
    tests = [
        ('Search Processor', test_search_processor),
        ('Applicant Processor', test_applicant_processor),
        ('Classification Processor', test_classification_processor),
        ('Citation Processor', test_citation_processor),
        ('Geographic Processor', test_geographic_processor),
        ('Working Directory Independence', test_working_directory_independence)
    ]
    
    results = {}
    for name, test_func in tests:
        results[name] = test_func()
    
    return results

def generate_unit_test_report(results: Dict[str, bool]) -> str:
    """Generate unit test report."""
    print_section('📋 Unit Test Results Summary', '=', 60)
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    failed_tests = total_tests - passed_tests
    
    print(f'Total Unit Tests: {total_tests}')
    print(f'Passed: {passed_tests} ✅')
    print(f'Failed: {failed_tests} ❌')
    print(f'Success Rate: {(passed_tests/total_tests)*100:.1f}%')
    
    print('\nDetailed Results:')
    for test_name, passed in results.items():
        status = '✅ PASS' if passed else '❌ FAIL'
        print(f'   {status} {test_name}')
    
    if failed_tests == 0:
        print('\n🎉 All unit tests passed!')
        return 'SUCCESS'
    else:
        print(f'\n⚠️ {failed_tests} unit test(s) failed.')
        return 'FAILURE'

def main():
    """Main test execution function."""
    parser = argparse.ArgumentParser(description='Unit tests for individual processors')
    parser.add_argument('--processor', choices=['search', 'applicant', 'classification', 'citation', 'geographic'], 
                       help='Test only a specific processor')
    args = parser.parse_args()
    
    # Setup centralized logging
    logger = get_test_logger("processors")
    
    logger.section('🔬 Patent Analysis Platform - Processors Unit Tests')
    logger.info('Enhanced from EPO PATLIB 2025 Live Demo Code')
    logger.info(f'Python version: {sys.version.split()[0]}')
    logger.info(f'Test file: {__file__}')
    logger.info(f'Working directory: {Path.cwd()}')
    
    if args.processor:
        # Test specific processor
        processor_tests = {
            'search': test_search_processor,
            'applicant': test_applicant_processor,
            'classification': test_classification_processor,
            'citation': test_citation_processor,
            'geographic': test_geographic_processor
        }
        
        test_func = processor_tests[args.processor]
        result = test_func()
        
        if result:
            logger.test_pass(f'{args.processor.title()} Processor')
            logger.info(f'\n🎉 {args.processor.title()} processor test passed!')
        else:
            logger.test_fail(f'{args.processor.title()} Processor')
            logger.error(f'\n❌ {args.processor.title()} processor test failed!')
        
        logger.close()
        return 0 if result else 1
    
    else:
        # Run all unit tests
        try:
            results = run_all_unit_tests()
            final_status = generate_unit_test_report(results)
            return 0 if final_status == 'SUCCESS' else 1
            
        except KeyboardInterrupt:
            print('\n⚠️ Unit tests interrupted by user')
            return 1
        except Exception as e:
            print(f'\n❌ Unit test execution failed: {e}')
            return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)