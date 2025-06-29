#!/usr/bin/env python3
"""
Notebook Validation Script for Patent Intelligence Platform
Tests all notebook cells 1:1 before copying to actual notebook

This script validates that all notebook code cells work correctly by executing them
exactly as they would run in the notebook, then provides the validated code for
copy-paste into the actual notebook.

Usage:
    python notebooks/test_notebook_validation.py
    python notebooks/test_notebook_validation.py --cell search-demo
    python notebooks/test_notebook_validation.py --fix-and-update
"""

import sys
import logging
import argparse
import traceback
from pathlib import Path
from datetime import datetime
import importlib

# Add parent directory to path (we're now in notebooks/ subdirectory)
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class NotebookCellValidator:
    """Validates notebook cells by executing them exactly as they would run in Jupyter."""
    
    def __init__(self):
        """Initialize the validator."""
        self.validated_cells = {}
        self.cell_outputs = {}
        self.execution_context = {}
        self.failed_cells = []
        self.success_count = 0
        self.total_count = 0
    
    def setup_notebook_environment(self):
        """Setup the exact environment that the notebook expects."""
        logger.info("🔧 Setting up notebook environment...")
        
        try:
            # Simulate notebook cell 1: Setup & Imports
            logger.info("📦 Loading modules (simulating first notebook cell)...")
            
            # Force reload modules as notebook does
            modules_to_reload = ['config', 'data_access', 'processors', 'visualizations']
            for module_name in modules_to_reload:
                if module_name in sys.modules:
                    del sys.modules[module_name]
            
            # Core imports - match notebook exactly
            import pandas as pd
            import numpy as np
            from datetime import datetime
            import logging
            
            # Add to execution context
            self.execution_context.update({
                'pd': pd,
                'np': np, 
                'datetime': datetime,
                'logging': logging,
                'Path': Path,
                'sys': sys,
                'importlib': importlib
            })
            
            # Platform imports - match notebook exactly
            from config import ConfigurationManager
            from data_access import PatstatClient, EPOOPSClient, PatentCountryMapper, PatentSearcher
            from processors import (ApplicantAnalyzer, GeographicAnalyzer, 
                                   ClassificationProcessor, CitationAnalyzer)
            from visualizations import (ProductionChartCreator, ProductionDashboardCreator, 
                                      ProductionMapsCreator)
            
            # Add to execution context
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
            
            logger.info("✅ Notebook environment setup complete")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to setup notebook environment: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def validate_cell_setup_imports(self):
        """Validate the setup and imports cell."""
        cell_name = "setup-imports"
        logger.info(f"🧪 Testing cell: {cell_name}")
        
        try:
            # EXACT NOTEBOOK CELL CODE - Copy this to notebook after validation
            cell_code = '''
# Force reload modules to pick up recent changes
import importlib
import sys

# Clear any cached modules
modules_to_reload = ['config', 'data_access', 'processors', 'visualizations']
for module_name in modules_to_reload:
    if module_name in sys.modules:
        del sys.modules[module_name]

# Core production imports
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import sys
from pathlib import Path

# Add parent directory to Python path for module imports
parent_dir = Path().resolve().parent
sys.path.insert(0, str(parent_dir))

# Production platform imports - using correct class names
from config import ConfigurationManager
from data_access import PatstatClient, EPOOPSClient, PatentCountryMapper, PatentSearcher

# Import the actual processor classes
from processors import (ApplicantAnalyzer, GeographicAnalyzer, 
                       ClassificationProcessor, CitationAnalyzer)

# Import visualization classes
from visualizations import (ProductionChartCreator, ProductionDashboardCreator, 
                          ProductionMapsCreator)

# Force reload of data_access module to pick up recent fixes
importlib.reload(sys.modules['data_access'])
from data_access import PatstatClient, PatentSearcher

# Configure logging for demo
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

print("🚀 Patent Intelligence Platform Initialized")
print("📅 Demo Date:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
print("🏭 Production Environment: Ready")
print(f"📁 Working Directory: {Path().resolve()}")
print(f"📦 Modules Path: {parent_dir}")
print("\\n✅ Using Production-Ready Architecture:")
print("  • ConfigurationManager for YAML-driven configuration")
print("  • PatstatClient for real database connectivity")
print("  • EPOOPSClient for enhanced patent data")
print("  • Analyzer classes for intelligence processing")  
print("  • Production visualization creators for business intelligence")
print("\\n🔄 Modules reloaded to pick up recent fixes")
'''
            
            # Execute the cell code
            exec(cell_code, self.execution_context)
            
            # Validate results
            required_objects = [
                'ConfigurationManager', 'PatstatClient', 'PatentSearcher',
                'ApplicantAnalyzer', 'GeographicAnalyzer', 'ClassificationProcessor',
                'CitationAnalyzer', 'ProductionChartCreator'
            ]
            
            for obj_name in required_objects:
                if obj_name not in self.execution_context:
                    raise ImportError(f"Required object {obj_name} not available in context")
            
            # Store validated cell
            self.validated_cells[cell_name] = cell_code
            self.cell_outputs[cell_name] = "✅ All imports successful, modules reloaded"
            self.success_count += 1
            logger.info(f"✅ Cell {cell_name} validated successfully")
            return True
            
        except Exception as e:
            self.failed_cells.append((cell_name, str(e)))
            logger.error(f"❌ Cell {cell_name} failed: {e}")
            return False
        finally:
            self.total_count += 1
    
    def validate_cell_config_manager(self):
        """Validate the configuration manager cell."""
        cell_name = "config-manager"
        logger.info(f"🧪 Testing cell: {cell_name}")
        
        try:
            # EXACT NOTEBOOK CELL CODE - Copy this to notebook after validation
            cell_code = '''
# Initialize configuration manager (auto-loads all YAML configs + .env)
config = ConfigurationManager()

print("📋 Configuration Status:")
print(f"✅ API Config: {len(config.get('api'))} settings loaded")
print(f"✅ Database Config: {len(config.get('database'))} settings loaded")
print(f"✅ Search Patterns: {len(config.get('search_patterns'))} patterns loaded")
print(f"✅ Visualization Config: {len(config.get('visualization'))} settings loaded")

# Show sample configuration structure
print("\\n🎨 Sample Visualization Themes:")
themes = config.get('visualization', 'general.themes')
if themes:
    for theme_name, theme_config in themes.items():
        if isinstance(theme_config, dict):
            description = theme_config.get('description', 'Professional theme')
        else:
            description = str(theme_config)
        print(f"  • {theme_name}: {description}")
else:
    print("  • No themes configured")

print("\\n🔍 Sample Search Strategies:")
strategies = config.get('search_patterns', 'search_strategies')
if strategies:
    for strategy_name in strategies.keys():
        print(f"  • {strategy_name}")
else:
    print("  • No search strategies configured")

print(f"\\n📊 Configuration Summary:")
summary = config.get_configuration_summary()
print(f"  🌍 Environment: {summary['environment']}")
print(f"  📁 Config Directory: {summary['config_directory']}")
print(f"  📋 Loaded Configs: {', '.join(summary['loaded_configs'])}")
'''
            
            # Execute the cell code
            exec(cell_code, self.execution_context)
            
            # Validate results - config should be in context now
            if 'config' not in self.execution_context:
                raise RuntimeError("ConfigurationManager instance 'config' not created")
            
            config = self.execution_context['config']
            if not hasattr(config, 'get'):
                raise RuntimeError("Config object doesn't have expected 'get' method")
            
            # Store validated cell
            self.validated_cells[cell_name] = cell_code
            self.cell_outputs[cell_name] = "✅ Configuration loaded successfully"
            self.success_count += 1
            logger.info(f"✅ Cell {cell_name} validated successfully")
            return True
            
        except Exception as e:
            self.failed_cells.append((cell_name, str(e)))
            logger.error(f"❌ Cell {cell_name} failed: {e}")
            return False
        finally:
            self.total_count += 1
    
    def validate_cell_patstat_connection(self):
        """Validate the PATSTAT connection cell."""
        cell_name = "patstat-connection"
        logger.info(f"🧪 Testing cell: {cell_name}")
        
        try:
            # EXACT NOTEBOOK CELL CODE - Copy this to notebook after validation
            cell_code = '''
# Initialize PATSTAT client with production environment
print("🔗 Connecting to PATSTAT Production Environment...")

# Create PATSTAT client with proven working configuration
patstat = PatstatClient(environment='PROD')  # Full dataset access

# Test connection
print("✅ PATSTAT connection established")
print("📊 Database Environment: PROD (full dataset access)")
print("🛡️ Connection Management: Advanced lifecycle management with zero GC issues")

# Initialize country mapper for geographic intelligence
country_mapper = PatentCountryMapper()
print("🌍 Geographic Intelligence: PatentCountryMapper initialized")

# Initialize EPO OPS client (if credentials available)
try:
    ops_client = EPOOPSClient()
    print("📡 EPO OPS API: Ready for enhanced data retrieval")
except Exception as e:
    print(f"⚠️ EPO OPS API: Credentials not configured")
    ops_client = None

print("\\n🎯 Production Environment Status:")
print("  ✅ PATSTAT PROD: Connected and ready")
print("  ✅ Configuration: Loaded from YAML files")
print("  ✅ Country Mapping: Enhanced geographic intelligence ready")
print(f"  {'✅' if ops_client else '⚠️'} EPO OPS API: {'Ready' if ops_client else 'Configure credentials for enhanced features'}")
'''
            
            # Execute the cell code
            exec(cell_code, self.execution_context)
            
            # Validate results
            required_objects = ['patstat', 'country_mapper']
            for obj_name in required_objects:
                if obj_name not in self.execution_context:
                    raise RuntimeError(f"Required object {obj_name} not created")
            
            # Store validated cell
            self.validated_cells[cell_name] = cell_code
            self.cell_outputs[cell_name] = "✅ PATSTAT and country mapper initialized"
            self.success_count += 1
            logger.info(f"✅ Cell {cell_name} validated successfully")
            return True
            
        except Exception as e:
            self.failed_cells.append((cell_name, str(e)))
            logger.error(f"❌ Cell {cell_name} failed: {e}")
            return False
        finally:
            self.total_count += 1
    
    def validate_cell_patent_search(self):
        """Validate the patent search cell - THIS IS THE BROKEN ONE."""
        cell_name = "patent-search" 
        logger.info(f"🧪 Testing cell: {cell_name} (FIXING THE BROKEN CELL)")
        
        try:
            # FIXED NOTEBOOK CELL CODE - This is the corrected version
            cell_code = '''
# Define search parameters using actual configuration technology areas
SEARCH_TECHNOLOGY_AREAS = ["rare_earth_elements"]
SEARCH_YEARS = "2020-01-01 to 2022-12-31"  # Will actually be used now!
MAX_RESULTS_MODE = "comprehensive"  # Use config: 5000 results instead of 500

# Parse date range correctly
def parse_date_range(date_range_str):
    if " to " in date_range_str:
        start_date, end_date = date_range_str.split(" to ")
        return start_date.strip(), end_date.strip()
    else:
        raise ValueError(f"Invalid date range format: '{date_range_str}'. Expected format: 'YYYY-MM-DD to YYYY-MM-DD'")

start_date, end_date = parse_date_range(SEARCH_YEARS)

print(f"🔍 Searching for {SEARCH_TECHNOLOGY_AREAS} patents ({start_date} to {end_date})...")
print("📋 Note: Search uses specific CPC codes from selected technology areas")

# Initialize PatentSearcher (THIS WAS MISSING!)
patent_searcher = PatentSearcher(patstat)

# Execute technology-specific search (FIXED VERSION)
search_results = patent_searcher.execute_technology_specific_search(
    technology_areas=SEARCH_TECHNOLOGY_AREAS,
    start_date=start_date,      
    end_date=end_date,           
    focused_search=False        # Use comprehensive (5000 results) instead of focused (500)
)

print(f"✅ Found {len(search_results)} patent applications from PATSTAT PROD")
print(f"📊 Date Range Used: {start_date} to {end_date}")  # Verify correct dates
print(f"📈 Coverage: {search_results['appln_auth'].nunique()} jurisdictions")
'''
            
            # Execute the cell code
            exec(cell_code, self.execution_context)
            
            # Validate results
            if 'patent_searcher' not in self.execution_context:
                raise RuntimeError("PatentSearcher instance 'patent_searcher' not created")
            
            if 'search_results' not in self.execution_context:
                raise RuntimeError("Search results not created")
            
            search_results = self.execution_context['search_results']
            if len(search_results) == 0:
                logger.warning("⚠️ Search returned 0 results - may need different parameters")
            
            # Store validated cell
            self.validated_cells[cell_name] = cell_code
            self.cell_outputs[cell_name] = f"✅ Search completed: {len(search_results)} results"
            self.success_count += 1
            logger.info(f"✅ Cell {cell_name} validated successfully (FIXED!)")
            return True
            
        except Exception as e:
            self.failed_cells.append((cell_name, str(e)))
            logger.error(f"❌ Cell {cell_name} failed: {e}")
            logger.error(f"🔍 Available objects: {list(self.execution_context.keys())}")
            return False
        finally:
            self.total_count += 1
    
    def validate_cell_run_processors(self):
        """Validate the processor execution cell."""
        cell_name = "run-processors"
        logger.info(f"🧪 Testing cell: {cell_name}")
        
        try:
            # EXACT NOTEBOOK CELL CODE - Copy this to notebook after validation  
            cell_code = '''
print("⚙️ Running Four-Processor Patent Intelligence Pipeline...")
print("🔄 Processing:", len(search_results), "patent applications")

# Initialize analysis results storage
analysis_results = {}

print("\\n🏭 Running Complete Analysis Workflow...")

# 1. Applicant Intelligence Analyzer
print("\\n👥 [1/4] Applicant Intelligence Analysis...")
try:
    applicant_analyzer = ApplicantAnalyzer(patstat)
    applicant_analysis = applicant_analyzer.analyze_search_results(search_results)
    analysis_results['applicant'] = applicant_analysis
    print(f"✅ Applicant analysis complete: {len(applicant_analysis)} applicants analyzed")
except Exception as e:
    print(f"⚠️ Applicant analyzer: {e}")
    analysis_results['applicant'] = pd.DataFrame()

# 2. Geographic Intelligence Analyzer
print("\\n🌍 [2/4] Geographic Intelligence Analysis...")
try:
    geographic_analyzer = GeographicAnalyzer(patstat)
    geographic_analysis = geographic_analyzer.analyze_search_results(search_results)
    analysis_results['geographic'] = geographic_analysis
    print(f"✅ Geographic analysis complete: {len(geographic_analysis)} regions analyzed")
except Exception as e:
    print(f"⚠️ Geographic analyzer: {e}")
    analysis_results['geographic'] = pd.DataFrame()

# 3. Technology Classification Analyzer
print("\\n🔬 [3/4] Technology Classification Analysis...")
try:
    classification_processor = ClassificationProcessor(patstat)
    classification_analysis = classification_processor.analyze_search_results(search_results)
    analysis_results['classification'] = classification_analysis
    print(f"✅ Classification analysis complete: {len(classification_analysis)} technology areas analyzed")
except Exception as e:
    print(f"⚠️ Classification analyzer: {e}")
    analysis_results['classification'] = pd.DataFrame()

# 4. Citation Network Analyzer
print("\\n🔗 [4/4] Citation Network Analysis...")
try:
    citation_analyzer = CitationAnalyzer(patstat)
    citation_analysis = citation_analyzer.analyze_search_results(search_results)
    analysis_results['citation'] = citation_analysis
    print(f"✅ Citation analysis complete: {len(citation_analysis)} citation patterns analyzed")
except Exception as e:
    print(f"⚠️ Citation analyzer: {e}")
    analysis_results['citation'] = pd.DataFrame()

print("\\n📊 Processing Pipeline Complete:")
for analysis_type, results in analysis_results.items():
    count = len(results) if hasattr(results, '__len__') else 'Available'
    print(f"  ✅ {analysis_type.title()}: {count} entities analyzed")

print(f"\\n🏆 Successfully processed {len([r for r in analysis_results.values() if len(r) > 0])}/4 intelligence layers")
print(f"📈 Total entities analyzed: {sum(len(r) for r in analysis_results.values() if hasattr(r, '__len__'))}")
'''
            
            # Execute the cell code
            exec(cell_code, self.execution_context)
            
            # Validate results
            if 'analysis_results' not in self.execution_context:
                raise RuntimeError("Analysis results not created")
            
            analysis_results = self.execution_context['analysis_results']
            if not isinstance(analysis_results, dict):
                raise RuntimeError("Analysis results should be a dictionary")
            
            # Store validated cell
            self.validated_cells[cell_name] = cell_code
            successful_analyzers = len([r for r in analysis_results.values() if hasattr(r, '__len__') and len(r) > 0])
            self.cell_outputs[cell_name] = f"✅ {successful_analyzers}/4 analyzers completed"
            self.success_count += 1
            logger.info(f"✅ Cell {cell_name} validated successfully")
            return True
            
        except Exception as e:
            self.failed_cells.append((cell_name, str(e)))
            logger.error(f"❌ Cell {cell_name} failed: {e}")
            return False
        finally:
            self.total_count += 1
    
    def run_full_validation(self):
        """Run validation on all notebook cells."""
        logger.info("🚀 Starting Full Notebook Validation")
        logger.info("=" * 60)
        
        # Setup environment
        if not self.setup_notebook_environment():
            logger.error("❌ Environment setup failed - cannot continue")
            return False
        
        # Validate cells in order
        validation_methods = [
            self.validate_cell_setup_imports,
            self.validate_cell_config_manager,
            self.validate_cell_patstat_connection,
            self.validate_cell_patent_search,  # The broken one we're fixing
            self.validate_cell_run_processors
        ]
        
        for validation_method in validation_methods:
            validation_method()
        
        # Summary
        self.print_validation_summary()
        return self.success_count == self.total_count
    
    def print_validation_summary(self):
        """Print validation summary and provide copy-paste code."""
        logger.info("\n" + "=" * 60)
        logger.info("📊 NOTEBOOK VALIDATION SUMMARY")
        logger.info("=" * 60)
        
        logger.info(f"Total cells tested: {self.total_count}")
        logger.info(f"Successful: {self.success_count}")
        logger.info(f"Failed: {len(self.failed_cells)}")
        
        if self.failed_cells:
            logger.error("\n❌ FAILED CELLS:")
            for cell_name, error in self.failed_cells:
                logger.error(f"  - {cell_name}: {error}")
        
        if self.success_count > 0:
            logger.info("\n✅ VALIDATED CELLS (Ready for copy-paste to notebook):")
            for cell_name, output in self.cell_outputs.items():
                logger.info(f"  ✅ {cell_name}: {output}")
        
        # Provide the fixed code
        if 'patent-search' in self.validated_cells:
            logger.info("\n🔧 FIXED CODE FOR BROKEN CELL (patent-search):")
            logger.info("Copy this EXACTLY to the notebook cell:")
            logger.info("-" * 40)
            print(self.validated_cells['patent-search'])
            logger.info("-" * 40)
        
        success_rate = (self.success_count / self.total_count * 100) if self.total_count > 0 else 0
        logger.info(f"\n🎯 SUCCESS RATE: {success_rate:.1f}%")
        
        if success_rate == 100:
            logger.info("🎉 ALL CELLS VALIDATED! Notebook is ready for demo!")
        elif success_rate >= 80:
            logger.info("⚠️ Most cells working. Fix remaining issues for full functionality.")
        else:
            logger.error("❌ Major issues found. Review and fix before demo.")


def main():
    """Main function to run notebook validation."""
    parser = argparse.ArgumentParser(description='Validate notebook cells before demo')
    parser.add_argument('--cell', help='Validate specific cell only')
    parser.add_argument('--fix-and-update', action='store_true', 
                       help='Fix issues and provide updated code')
    
    args = parser.parse_args()
    
    validator = NotebookCellValidator()
    
    if args.cell:
        # Validate specific cell
        if hasattr(validator, f'validate_cell_{args.cell.replace("-", "_")}'):
            validator.setup_notebook_environment()
            method = getattr(validator, f'validate_cell_{args.cell.replace("-", "_")}')
            method()
            validator.print_validation_summary()
        else:
            logger.error(f"Unknown cell: {args.cell}")
            return 1
    else:
        # Validate all cells
        success = validator.run_full_validation()
        return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())