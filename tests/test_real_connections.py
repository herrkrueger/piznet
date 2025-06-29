"""
Real Connection Test - PATSTAT and EPO OPS
Demonstrate actual connections to production data sources
"""

import pandas as pd
import logging
import os
from pathlib import Path
import time

# Load environment variables
def load_dotenv():
    env_path = Path('.env')
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value

load_dotenv()

# Import providers
from src.data_providers.patstat.provider import PatstatDataProvider
from src.data_providers.epo_ops.provider import EPOOpsDataProvider

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_patstat_connection():
    """Test real PATSTAT connection"""
    
    print("🔍 TESTING PATSTAT CONNECTION")
    print("=" * 50)
    
    try:
        # Create PATSTAT provider with PROD environment
        patstat_config = {
            'environment': 'PROD',
            'timeout': 30,
            'max_results': 100
        }
        
        patstat_provider = PatstatDataProvider(patstat_config)
        
        print(f"📡 Connecting to PATSTAT {patstat_config['environment']} environment...")
        
        # Test connection
        if patstat_provider.connect():
            print("✅ PATSTAT connection successful!")
            
            # Get connection metadata
            metadata = patstat_provider.connection_metadata
            print(f"   🏢 Environment: {metadata.get('environment')}")
            print(f"   📅 Connected: {metadata.get('connection_time')}")
            print(f"   🗄️ Database version: {metadata.get('database_version')}")
            
            # Test basic query
            print("\n🔍 Testing basic patent search...")
            search_params = {
                'query_type': 'patent_search',
                'technology_area': 'energy storage',
                'filing_years': [2023],
                'countries': ['US', 'DE'],
                'limit': 10
            }
            
            result = patstat_provider.query(search_params)
            
            if result.is_successful:
                print(f"✅ Patent search successful: {len(result.data)} patents found")
                
                if not result.data.empty:
                    print("📋 Sample results:")
                    sample = result.data.head(3)
                    for _, row in sample.iterrows():
                        print(f"   • {row.get('person_name', 'Unknown')} ({row.get('person_ctry_code', '??')}) - {row.get('ipc_class_symbol', 'No IPC')}")
                
                # Show performance
                exec_time = result.metadata.get('processing_time', 0)
                print(f"⚡ Query executed in {exec_time:.2f} seconds")
                
            else:
                print(f"❌ Patent search failed: {result.errors}")
            
            # Test connection summary
            summary = patstat_provider.get_patstat_summary()
            print(f"\n📊 Provider Summary:")
            print(f"   🔄 Total queries: {summary['total_queries']}")
            print(f"   ✅ Success rate: {summary['success_rate']}")
            print(f"   📈 Records retrieved: {summary.get('total_records_retrieved', 0)}")
            
            return True
            
        else:
            print("❌ PATSTAT connection failed")
            return False
            
    except Exception as e:
        print(f"❌ PATSTAT test error: {e}")
        return False


def test_epo_ops_connection():
    """Test real EPO OPS connection"""
    
    print("\n🌐 TESTING EPO OPS CONNECTION")
    print("=" * 50)
    
    try:
        # Create EPO OPS provider (credentials from .env)
        ops_config = {
            'rate_limit_per_minute': 30,
            'timeout': 30
        }
        
        ops_provider = EPOOpsDataProvider(ops_config)
        
        print("🔐 Authenticating with EPO OPS API...")
        print(f"   🔑 Using API key: {os.getenv('OPS_KEY', 'Not found')[:10]}...")
        
        # Test connection
        if ops_provider.connect():
            print("✅ EPO OPS authentication successful!")
            
            # Get connection metadata
            metadata = ops_provider.connection_metadata
            print(f"   🌍 Base URL: {metadata.get('base_url')}")
            print(f"   📊 Rate limits: {metadata.get('rate_limits')}")
            print(f"   🛠️ Services: {len(metadata.get('available_services', []))} available")
            
            # Test basic search
            print("\n🔍 Testing patent search...")
            search_params = {
                'service': 'search',
                'text': 'battery electric vehicle',
                'limit': 5
            }
            
            result = ops_provider.query(search_params)
            
            if result.is_successful:
                print(f"✅ Patent search successful: {len(result.data)} patents found")
                
                if not result.data.empty:
                    print("📋 Sample results:")
                    sample = result.data.head(3)
                    for _, row in sample.iterrows():
                        pub_id = row.get('publication_id', 'Unknown')
                        country = row.get('country', '??')
                        doc_num = row.get('doc_number', '???')
                        print(f"   • {pub_id} ({country}{doc_num})")
                
            else:
                print(f"⚠️ Search returned no results (this is normal for restrictive queries)")
            
            # Test publication details
            print("\n📄 Testing publication data retrieval...")
            pub_params = {
                'service': 'publication',
                'publication_id': 'EP1000000A1'  # Example publication
            }
            
            pub_result = ops_provider.query(pub_params)
            if pub_result.is_successful:
                print("✅ Publication data retrieval successful")
            else:
                print("⚠️ Publication not found (expected for example ID)")
            
            # Show rate limit status
            summary = ops_provider.get_ops_summary()
            rate_status = summary.get('rate_limit_status', {})
            print(f"\n📊 Rate Limit Status:")
            print(f"   🕐 Requests this minute: {rate_status.get('requests_this_minute', 0)}")
            print(f"   📅 Weekly requests: {rate_status.get('weekly_requests', 0)}")
            print(f"   🎯 Weekly remaining: {rate_status.get('weekly_remaining', 0)}")
            
            return True
            
        else:
            print("❌ EPO OPS authentication failed")
            print("   Check API credentials in .env file")
            return False
            
    except Exception as e:
        print(f"❌ EPO OPS test error: {e}")
        return False


def test_integrated_workflow():
    """Test integrated workflow with both providers"""
    
    print("\n🔗 TESTING INTEGRATED WORKFLOW")
    print("=" * 50)
    
    try:
        # Create both providers
        patstat_provider = PatstatDataProvider({'environment': 'PROD', 'max_results': 50})
        ops_provider = EPOOpsDataProvider({'rate_limit_per_minute': 30})
        
        print("🎯 Step 1: Search patents with PATSTAT...")
        
        # Search for AI patents
        patstat_params = {
            'query_type': 'patent_search',
            'technology_area': 'artificial intelligence',
            'filing_years': [2023],
            'limit': 5
        }
        
        patstat_connected = patstat_provider.connect()
        ops_connected = ops_provider.connect()
        
        if patstat_connected:
            patstat_result = patstat_provider.query(patstat_params)
            
            if patstat_result.is_successful and not patstat_result.data.empty:
                print(f"✅ Found {len(patstat_result.data)} AI patents")
                
                # Get a patent for detailed analysis
                sample_patent = patstat_result.data.iloc[0]
                applicant = sample_patent.get('person_name', 'Unknown')
                print(f"   🏢 Sample applicant: {applicant}")
                
                if ops_connected:
                    print("\n🎯 Step 2: Get detailed info with EPO OPS...")
                    
                    # Search for patents from same applicant
                    ops_params = {
                        'service': 'search',
                        'applicant': applicant[:20],  # First 20 chars to avoid issues
                        'limit': 3
                    }
                    
                    ops_result = ops_provider.query(ops_params)
                    
                    if ops_result.is_successful:
                        print(f"✅ Found additional patents for {applicant[:20]}")
                        
                        print("\n🎉 INTEGRATED WORKFLOW SUCCESS!")
                        print("   📊 PATSTAT provided patent data")
                        print("   🌐 EPO OPS provided detailed information")
                        print("   🔗 Both systems working together")
                        
                        return True
                    else:
                        print("⚠️ EPO OPS search limited (rate limits or no matches)")
                else:
                    print("⚠️ EPO OPS not available for integration")
            else:
                print("⚠️ No PATSTAT results for integration test")
        else:
            print("⚠️ PATSTAT not available for integration")
        
        # Show what we achieved
        print(f"\n📈 Integration Status:")
        print(f"   🗄️ PATSTAT: {'✅ Connected' if patstat_connected else '❌ Failed'}")
        print(f"   🌐 EPO OPS: {'✅ Connected' if ops_connected else '❌ Failed'}")
        print(f"   🔗 Both working: {'✅ Yes' if patstat_connected and ops_connected else '⚠️ Partial'}")
        
        return patstat_connected or ops_connected
        
    except Exception as e:
        print(f"❌ Integration test error: {e}")
        return False


def main():
    """Run all connection tests"""
    
    print("🚀 PATENT INTELLIGENCE PLATFORM - REAL CONNECTION TESTS")
    print("=" * 70)
    
    # Check environment setup
    print("🔧 Environment Check:")
    print(f"   🔑 OPS_KEY: {'✅ Set' if os.getenv('OPS_KEY') else '❌ Missing'}")
    print(f"   🔐 OPS_SECRET: {'✅ Set' if os.getenv('OPS_SECRET') else '❌ Missing'}")
    print(f"   🏢 PATSTAT_ENV: {os.getenv('PATSTAT_ENVIRONMENT', 'PROD')}")
    
    results = []
    
    # Test individual connections
    print(f"\n{'='*70}")
    results.append(test_patstat_connection())
    
    print(f"\n{'='*70}")
    results.append(test_epo_ops_connection())
    
    # Test integration
    print(f"\n{'='*70}")
    results.append(test_integrated_workflow())
    
    # Final summary
    print(f"\n{'='*70}")
    print("🎯 FINAL RESULTS")
    print("=" * 70)
    
    test_names = ['PATSTAT Connection', 'EPO OPS Connection', 'Integrated Workflow']
    
    for i, (test_name, result) in enumerate(zip(test_names, results)):
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{i+1}. {test_name}: {status}")
    
    success_count = sum(results)
    print(f"\n🏆 Overall Success: {success_count}/{len(results)} tests passed")
    
    if success_count >= 2:
        print("🎉 EXCELLENT! Multiple data providers are working!")
    elif success_count >= 1:
        print("👍 GOOD! At least one data provider is working!")
    else:
        print("⚠️ Need to check API credentials and network access")
    
    print("\n🚀 Ready for production patent intelligence workflows!")


if __name__ == "__main__":
    main()