#!/usr/bin/env python3
"""
Comprehensive Notebook Cell Testing for CI/CD
Tests all notebook cells systematically to ensure demo readiness

This script validates that the notebook is ready for live demonstrations by:
1. Testing all code cells individually
2. Checking for undefined variables
3. Validating imports and dependencies
4. Ensuring proper error handling
5. Providing fixes for broken cells

Usage:
    python scripts/test_notebook_cells.py
    python scripts/test_notebook_cells.py --fix
    python scripts/test_notebook_cells.py --update-notebook
"""

import sys
import json
import logging
import argparse
import traceback
from pathlib import Path
from datetime import datetime

# Add parent directory to path (we're now in notebooks/ subdirectory)
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class NotebookTester:
    """Comprehensive notebook testing and validation."""
    
    def __init__(self, notebook_path=None):
        """Initialize notebook tester."""
        if notebook_path is None:
            notebook_path = Path(__file__).parent / 'Patent_Intelligence_Platform_Demo.ipynb'
        
        self.notebook_path = Path(notebook_path)
        self.test_results = {}
        self.execution_context = {}
        self.cell_dependencies = {}
        self.failed_cells = []
        self.passed_cells = []
        
    def load_notebook(self):
        """Load and parse the notebook."""
        logger.info(f"📖 Loading notebook: {self.notebook_path}")
        
        try:
            with open(self.notebook_path, 'r', encoding='utf-8') as f:
                self.notebook_data = json.load(f)
            
            # Extract code cells
            self.code_cells = {}
            for cell in self.notebook_data.get('cells', []):
                if cell.get('cell_type') == 'code':
                    cell_id = cell.get('id', f"cell_{len(self.code_cells)}")
                    source = ''.join(cell.get('source', []))
                    self.code_cells[cell_id] = source
            
            logger.info(f"✅ Loaded {len(self.code_cells)} code cells")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load notebook: {e}")
            return False
    
    def setup_test_environment(self):
        """Setup the test environment with all required imports."""
        logger.info("🔧 Setting up test environment...")
        
        try:
            # Standard library imports
            import pandas as pd
            import numpy as np
            import logging
            from datetime import datetime
            from pathlib import Path
            import sys
            import importlib
            
            # Add to context
            self.execution_context.update({
                'pd': pd, 'np': np, 'logging': logging, 'datetime': datetime,
                'Path': Path, 'sys': sys, 'importlib': importlib
            })
            
            # Platform imports with error handling
            try:
                from config import ConfigurationManager
                from data_access import PatstatClient, EPOOPSClient, PatentCountryMapper, PatentSearcher
                from processors import (ApplicantAnalyzer, GeographicAnalyzer,
                                       ClassificationProcessor, CitationAnalyzer)
                from visualizations import (ProductionChartCreator, ProductionDashboardCreator,
                                          ProductionMapsCreator)
                
                self.execution_context.update({
                    'ConfigurationManager': ConfigurationManager,
                    'PatstatClient': PatstatClient,
                    'EPOOPSClient': EPOOPSClient,
                    'PatentCountryMapper': PatentCountryMapper,
                    'PatentSearcher': PatentSearcher,
                    'ApplicantAnalyzer': ApplicantAnalyzer,
                    'GeographicAnalyzer': GeographicAnalyzer,
                    'ClassificationProcessor': ClassificationProcessor,
                    'CitationAnalyzer': CitationAnalyzer,
                    'ProductionChartCreator': ProductionChartCreator,
                    'ProductionDashboardCreator': ProductionDashboardCreator,
                    'ProductionMapsCreator': ProductionMapsCreator
                })
                
                logger.info("✅ Platform modules loaded successfully")
                
            except ImportError as e:
                logger.warning(f"⚠️ Some platform modules not available: {e}")
                return False
            
            logger.info("✅ Test environment setup complete")
            return True
            
        except Exception as e:
            logger.error(f"❌ Environment setup failed: {e}")
            return False
    
    def test_cell(self, cell_id, source_code):
        """Test an individual notebook cell."""
        logger.info(f"🧪 Testing cell: {cell_id}")
        
        try:
            # Create a copy of execution context for this test
            test_context = self.execution_context.copy()
            
            # Execute the cell code
            exec(source_code, test_context)
            
            # Update main context with successful execution results
            self.execution_context.update(test_context)
            
            # Record success
            self.test_results[cell_id] = {
                'status': 'passed',
                'error': None,
                'variables_created': list(set(test_context.keys()) - set(self.execution_context.keys())),
                'execution_time': datetime.now().isoformat()
            }
            
            self.passed_cells.append(cell_id)
            logger.info(f"✅ Cell {cell_id} passed")
            return True
            
        except Exception as e:
            error_msg = str(e)
            error_trace = traceback.format_exc()
            
            # Record failure
            self.test_results[cell_id] = {
                'status': 'failed',
                'error': error_msg,
                'trace': error_trace,
                'execution_time': datetime.now().isoformat()
            }
            
            self.failed_cells.append((cell_id, error_msg))
            logger.error(f"❌ Cell {cell_id} failed: {error_msg}")
            return False
    
    def test_all_cells(self):
        """Test all notebook cells in sequence."""
        logger.info("🚀 Testing all notebook cells...")
        
        if not self.setup_test_environment():
            logger.error("❌ Cannot proceed without test environment")
            return False
        
        total_cells = len(self.code_cells)
        for i, (cell_id, source_code) in enumerate(self.code_cells.items(), 1):
            logger.info(f"[{i}/{total_cells}] Testing {cell_id}")
            
            # Skip empty cells
            if not source_code.strip():
                logger.info(f"⏭️ Skipping empty cell: {cell_id}")
                continue
            
            self.test_cell(cell_id, source_code)
        
        return True
    
    def analyze_dependencies(self):
        """Analyze dependencies between cells."""
        logger.info("🔍 Analyzing cell dependencies...")
        
        for cell_id, source_code in self.code_cells.items():
            dependencies = []
            
            # Simple dependency analysis - look for variable usage
            lines = source_code.split('\n')
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Look for variable assignments in previous cells
                    for other_cell_id, other_source in self.code_cells.items():
                        if other_cell_id != cell_id:
                            # Check if this cell uses variables from other cells
                            other_lines = other_source.split('\n')
                            for other_line in other_lines:
                                if '=' in other_line and not other_line.strip().startswith('#'):
                                    var_name = other_line.split('=')[0].strip()
                                    if var_name in line and other_cell_id not in dependencies:
                                        dependencies.append(other_cell_id)
            
            self.cell_dependencies[cell_id] = dependencies
    
    def generate_test_report(self):
        """Generate comprehensive test report."""
        logger.info("📊 Generating test report...")
        
        total_cells = len(self.code_cells)
        passed_count = len(self.passed_cells)
        failed_count = len(self.failed_cells)
        success_rate = (passed_count / total_cells * 100) if total_cells > 0 else 0
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'notebook_path': str(self.notebook_path),
            'summary': {
                'total_cells': total_cells,
                'passed': passed_count,
                'failed': failed_count,
                'success_rate': success_rate
            },
            'cell_results': self.test_results,
            'failed_cells': [{'cell_id': cell_id, 'error': error} for cell_id, error in self.failed_cells],
            'dependencies': self.cell_dependencies
        }
        
        # Save report
        report_path = Path(__file__).parent / 'test_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📄 Report saved: {report_path}")
        return report
    
    def print_summary(self):
        """Print test summary."""
        total_cells = len(self.code_cells)
        passed_count = len(self.passed_cells)
        failed_count = len(self.failed_cells)
        success_rate = (passed_count / total_cells * 100) if total_cells > 0 else 0
        
        print("\n" + "=" * 60)
        print("📊 NOTEBOOK TEST SUMMARY")
        print("=" * 60)
        print(f"📖 Notebook: {self.notebook_path.name}")
        print(f"🧪 Total cells: {total_cells}")
        print(f"✅ Passed: {passed_count}")
        print(f"❌ Failed: {failed_count}")
        print(f"📈 Success rate: {success_rate:.1f}%")
        
        if self.failed_cells:
            print("\n❌ FAILED CELLS:")
            for cell_id, error in self.failed_cells:
                print(f"  - {cell_id}: {error}")
        
        if self.passed_cells:
            print(f"\n✅ PASSED CELLS: {', '.join(self.passed_cells[:5])}" + 
                  (f" and {len(self.passed_cells) - 5} more" if len(self.passed_cells) > 5 else ""))
        
        print("\n🎯 NOTEBOOK STATUS:")
        if success_rate == 100:
            print("🎉 EXCELLENT! Notebook is ready for live demo!")
        elif success_rate >= 80:
            print("⚠️ GOOD! Most cells work, fix remaining issues")
        elif success_rate >= 50:
            print("⚠️ NEEDS WORK! Multiple issues need fixing")
        else:
            print("❌ MAJOR ISSUES! Extensive fixes required")
        
        print("\n💡 RECOMMENDATIONS:")
        if failed_count > 0:
            print("1. Fix failed cells before demo")
            print("2. Run cells in sequence to ensure dependencies")
            print("3. Test with fresh kernel restart")
        print("4. Use notebook validation script before demos")
        print("5. Keep backup of working version")
    
    def fix_common_issues(self):
        """Provide fixes for common notebook issues."""
        logger.info("🔧 Analyzing common issues and providing fixes...")
        
        fixes = []
        
        # Check for the main issue we found
        for cell_id, source_code in self.code_cells.items():
            if 'patent_searcher.execute_technology_specific_search' in source_code:
                if 'patent_searcher = PatentSearcher(patstat)' not in source_code:
                    fixes.append({
                        'cell_id': cell_id,
                        'issue': 'PatentSearcher not initialized',
                        'fix': 'Add line: patent_searcher = PatentSearcher(patstat)',
                        'before_line': 'patent_searcher.execute_technology_specific_search'
                    })
        
        if fixes:
            print("\n🔧 AUTOMATIC FIXES AVAILABLE:")
            for fix in fixes:
                print(f"Cell: {fix['cell_id']}")
                print(f"Issue: {fix['issue']}")
                print(f"Fix: {fix['fix']}")
                print(f"Location: Before '{fix['before_line']}'")
                print("-" * 40)
        
        return fixes


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Test notebook cells for demo readiness')
    parser.add_argument('--notebook', help='Path to notebook file')
    parser.add_argument('--fix', action='store_true', help='Show fixes for common issues')
    parser.add_argument('--update-notebook', action='store_true', help='Update notebook with fixes')
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = NotebookTester(args.notebook)
    
    # Load notebook
    if not tester.load_notebook():
        return 1
    
    # Test all cells
    tester.test_all_cells()
    
    # Analyze dependencies
    tester.analyze_dependencies()
    
    # Generate report
    report = tester.generate_test_report()
    
    # Print summary
    tester.print_summary()
    
    # Show fixes if requested
    if args.fix:
        tester.fix_common_issues()
    
    # Return success/failure
    success_rate = report['summary']['success_rate']
    return 0 if success_rate >= 80 else 1


if __name__ == "__main__":
    sys.exit(main())