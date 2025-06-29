"""
Data Providers Demo - Clean Architecture
Demonstration of all data providers working together
"""

import pandas as pd
import logging
from typing import Dict, Any
import time
import os
from pathlib import Path

# Load environment variables from .env file
def load_dotenv():
    env_path = Path('.env')
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value

load_dotenv()

# Import data providers and configuration
from src.data_providers.base import DataProviderFactory, DataProviderType
from src.data_providers.patstat.provider import PatstatDataProvider
from src.data_providers.epo_ops.provider import EPOOpsDataProvider
from src.data_providers.classification.ipc_provider import WipoIpcProvider
from src.config import ConfigurationManager, DataProviderConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demo_data_providers():
    """Demonstrate all data providers in action"""
    
    print("🚀 Patent Intelligence Platform - Data Providers Demo")
    print("=" * 70)
    
    # Initialize configuration manager
    config_manager = ConfigurationManager()
    provider_config = DataProviderConfig(config_manager)
    
    print("🔧 Configuration System:")
    config_summary = config_manager.get_summary()
    print(f"   📁 Config path: {config_summary['config_path']}")
    print(f"   📄 Config files: {len(config_summary['config_files_loaded'])} loaded")
    
    # Get configurations from YAML + environment
    demo_configs = provider_config.get_all_provider_configs()
    
    # Show provider status
    enabled_providers = provider_config.get_enabled_providers()
    print(f"   ✅ Enabled providers: {sum(enabled_providers.values())}/{len(enabled_providers)}")
    
    # Validate configuration
    validation = provider_config.validate_providers()
    if validation['valid']:
        print(f"   ✅ Configuration validation: PASSED")
    else:
        print(f"   ⚠️ Configuration validation: {len(validation['errors'])} errors")
    
    # Test each provider individually
    test_individual_providers(demo_configs)
    
    print("\n" + "=" * 70)
    print("🔗 INTEGRATED WORKFLOW DEMONSTRATION")
    print("=" * 70)
    
    # Demonstrate integrated workflow
    demo_integrated_workflow(demo_configs)


def test_individual_providers(configs: Dict[str, Any]):
    """Test each provider individually"""
    
    print("🧪 INDIVIDUAL PROVIDER TESTING")
    print("-" * 50)
    
    # Test WIPO IPC Provider (works offline)
    print("\n1️⃣ Testing WIPO IPC Classification Provider...")
    test_ipc_provider(configs['wipo_ipc_config'])
    
    # Test PATSTAT Provider (requires real connection)
    print("\n2️⃣ Testing PATSTAT Provider...")
    test_patstat_provider(configs['patstat_config'])
    
    # Test EPO OPS Provider (requires real credentials)
    print("\n3️⃣ Testing EPO OPS Provider...")
    test_epo_ops_provider(configs['epo_ops_config'])


def test_ipc_provider(config: Dict[str, Any]):
    """Test WIPO IPC provider"""
    
    try:
        # Create provider
        ipc_provider = WipoIpcProvider(config)
        
        # Test connection
        connection_result = ipc_provider.test_connection()
        print(f"   Connection: {connection_result.get('connection_status', 'unknown')}")
        
        if ipc_provider.connect():
            print("   ✅ IPC Provider connected successfully")
            
            # Test classification lookup
            lookup_params = {
                'query_type': 'classification_lookup',
                'ipc_codes': ['G06F', 'H01M', 'A61K']
            }
            
            result = ipc_provider.query(lookup_params)
            if result.is_successful:
                print(f"   ✅ Classification lookup: {len(result.data)} classifications found")
                if not result.data.empty:
                    sample = result.data.iloc[0]
                    print(f"      Example: {sample['ipc_code']} - {sample['title']}")
            
            # Test technology mapping
            tech_params = {
                'query_type': 'technology_mapping',
                'technology_area': 'energy storage',
                'include_related': True
            }
            
            result = ipc_provider.query(tech_params)
            if result.is_successful:
                print(f"   ✅ Technology mapping: {len(result.data)} IPC codes for 'energy storage'")
                if not result.data.empty:
                    top_code = result.data.iloc[0]
                    print(f"      Top match: {top_code['ipc_code']} (relevance: {top_code['relevance_score']:.2f})")
            
            # Get provider summary
            summary = ipc_provider.get_ipc_summary()
            print(f"   📊 Summary: {summary['total_queries']} queries, {summary['success_rate']} success rate")
            
        else:
            print("   ❌ IPC Provider connection failed")
            
    except Exception as e:
        print(f"   ❌ IPC Provider error: {e}")


def test_patstat_provider(config: Dict[str, Any]):
    """Test PATSTAT provider"""
    
    try:
        # Create provider
        patstat_provider = PatstatDataProvider(config)
        
        # Test connection
        print("   Attempting PATSTAT connection...")
        if patstat_provider.connect():
            print("   ✅ PATSTAT Provider connected successfully")
            
            # Test patent search
            search_params = {
                'query_type': 'patent_search',
                'technology_area': 'energy storage',
                'filing_years': [2022, 2023],
                'countries': ['US', 'DE', 'JP'],
                'limit': 100
            }
            
            result = patstat_provider.query(search_params)
            if result.is_successful:
                print(f"   ✅ Patent search: {len(result.data)} patents found")
                if not result.data.empty:
                    print(f"      Sample patent: {result.data.iloc[0]['person_name']} ({result.data.iloc[0]['person_ctry_code']})")
            
            # Test applicant search
            applicant_params = {
                'query_type': 'applicant_search',
                'applicant_name': 'Tesla',
                'min_patents': 5,
                'limit': 50
            }
            
            result = patstat_provider.query(applicant_params)
            if result.is_successful:
                print(f"   ✅ Applicant search: {len(result.data)} applicants found")
            
            summary = patstat_provider.get_patstat_summary()
            print(f"   📊 Summary: {summary['total_queries']} queries, Environment: {summary['database_environment']}")
            
        else:
            print("   ⚠️ PATSTAT Provider connection failed (may require real PATSTAT access)")
            
    except Exception as e:
        print(f"   ⚠️ PATSTAT Provider not available: {str(e)[:100]}...")


def test_epo_ops_provider(config: Dict[str, Any]):
    """Test EPO OPS provider"""
    
    try:
        # Create provider
        ops_provider = EPOOpsDataProvider(config)
        
        print("   Attempting EPO OPS connection...")
        if ops_provider.connect():
            print("   ✅ EPO OPS Provider connected successfully")
            
            # Test patent search
            search_params = {
                'service': 'search',
                'text': 'battery',
                'applicant': 'Tesla',
                'limit': 10
            }
            
            result = ops_provider.query(search_params)
            if result.is_successful:
                print(f"   ✅ OPS search: {len(result.data)} patents found")
                if not result.data.empty:
                    print(f"      Sample: {result.data.iloc[0]['publication_id']}")
            
            summary = ops_provider.get_ops_summary()
            print(f"   📊 Summary: {summary['total_queries']} queries, Rate limit remaining: {summary['rate_limit_status']['weekly_remaining']}")
            
        else:
            print("   ⚠️ EPO OPS Provider connection failed (requires real API credentials)")
            
    except Exception as e:
        print(f"   ⚠️ EPO OPS Provider not available: {str(e)[:100]}...")


def demo_integrated_workflow(configs: Dict[str, Any]):
    """Demonstrate integrated workflow using multiple providers"""
    
    print("🎯 Integrated Patent Intelligence Workflow")
    print("-" * 50)
    
    # Step 1: Technology Analysis with IPC
    print("\n1️⃣ STEP 1: Technology Classification Analysis")
    ipc_provider = WipoIpcProvider(configs['wipo_ipc_config'])
    
    if ipc_provider.connect():
        # Map technology to IPC codes
        tech_params = {
            'query_type': 'technology_mapping',
            'technology_area': 'artificial intelligence',
            'include_related': True
        }
        
        ipc_result = ipc_provider.query(tech_params)
        if ipc_result.is_successful and not ipc_result.data.empty:
            ipc_codes = ipc_result.data['ipc_code'].tolist()
            print(f"   ✅ Found {len(ipc_codes)} relevant IPC codes for AI: {ipc_codes[:3]}...")
            
            # Step 2: Patent Search with PATSTAT (if available)
            print("\n2️⃣ STEP 2: Patent Data Retrieval")
            patstat_provider = PatstatDataProvider(configs['patstat_config'])
            
            try:
                if patstat_provider.connect():
                    search_params = {
                        'query_type': 'patent_search',
                        'ipc_classes': ipc_codes[:3],  # Use top 3 IPC codes
                        'filing_years': [2022, 2023],
                        'limit': 200
                    }
                    
                    patent_result = patstat_provider.query(search_params)
                    if patent_result.is_successful:
                        print(f"   ✅ Retrieved {len(patent_result.data)} AI patents from PATSTAT")
                        
                        # Step 3: Analyze results
                        print("\n3️⃣ STEP 3: Intelligence Analysis")
                        analyze_integrated_results(ipc_result.data, patent_result.data)
                    else:
                        print("   ⚠️ PATSTAT search failed, using mock data for demo")
                        demo_with_mock_data(ipc_result.data)
                else:
                    print("   ⚠️ PATSTAT not available, using mock data for demo")
                    demo_with_mock_data(ipc_result.data)
                    
            except Exception as e:
                print(f"   ⚠️ PATSTAT integration failed: {e}")
                demo_with_mock_data(ipc_result.data)
        else:
            print("   ❌ IPC mapping failed")
    else:
        print("   ❌ IPC provider connection failed")


def analyze_integrated_results(ipc_data: pd.DataFrame, patent_data: pd.DataFrame):
    """Analyze results from multiple providers"""
    
    # IPC Analysis
    top_classifications = ipc_data.head(5)
    print(f"   🏷️ Top IPC Classifications:")
    for _, row in top_classifications.iterrows():
        print(f"      • {row['ipc_code']}: {row['title']} (relevance: {row['relevance_score']:.2f})")
    
    # Patent Analysis
    if not patent_data.empty:
        country_distribution = patent_data['person_ctry_code'].value_counts().head(5)
        print(f"\n   🌍 Top Countries by Patent Activity:")
        for country, count in country_distribution.items():
            print(f"      • {country}: {count} patents")
        
        recent_applicants = patent_data['person_name'].value_counts().head(3)
        print(f"\n   🏢 Top Applicants:")
        for applicant, count in recent_applicants.items():
            print(f"      • {applicant}: {count} patents")
        
        # Innovation trends
        if 'appln_filing_year' in patent_data.columns:
            yearly_trends = patent_data.groupby('appln_filing_year').size()
            print(f"\n   📈 Innovation Trends:")
            for year, count in yearly_trends.items():
                print(f"      • {year}: {count} patents")
    
    print(f"\n   🎯 INTEGRATION SUCCESS: Combined classification and patent data analysis complete!")


def demo_with_mock_data(ipc_data: pd.DataFrame):
    """Demo analysis with mock patent data when PATSTAT unavailable"""
    
    # Create mock patent data
    import numpy as np
    np.random.seed(42)
    
    mock_patents = pd.DataFrame({
        'person_name': ['Google LLC', 'Microsoft Corp', 'IBM Corp', 'Amazon Inc', 'Tesla Inc'] * 20,
        'person_ctry_code': np.random.choice(['US', 'DE', 'CN', 'JP', 'KR'], 100),
        'appln_filing_year': np.random.choice([2021, 2022, 2023], 100),
        'ipc_class_symbol': np.random.choice(ipc_data['ipc_code'].tolist()[:5], 100)
    })
    
    print(f"   🎬 Using mock data: {len(mock_patents)} sample patents")
    analyze_integrated_results(ipc_data, mock_patents)


def main():
    """Main demo execution"""
    demo_data_providers()
    
    print("\n" + "=" * 70)
    print("✨ DATA PROVIDERS DEMO COMPLETE")
    print("=" * 70)
    print("🏗️ This demonstrates the clean architecture benefits:")
    print("   ✅ Standardized interfaces across all data sources")
    print("   ✅ Independent provider testing and validation")
    print("   ✅ Seamless integration between providers")
    print("   ✅ Graceful handling of unavailable services")
    print("   ✅ Comprehensive error handling and logging")
    print("   ✅ Performance monitoring and rate limiting")
    print("\n🚀 Ready for production with real API credentials!")


if __name__ == "__main__":
    main()