#!/usr/bin/env python3
"""
Check what CPC description tables/columns are available in PATSTAT.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def check_cpc_tables():
    """Check available CPC tables and description fields in PATSTAT."""
    try:
        from data_access.patstat_client import PatstatClient
        
        # Create PATSTAT client
        patstat = PatstatClient(environment='PROD')
        
        if not patstat.is_connected():
            print("❌ PATSTAT connection failed")
            return
        
        print("🔍 CHECKING CPC TABLES IN PATSTAT")
        print("=" * 50)
        
        # Check available models
        models = patstat.models
        cpc_models = [name for name in models.keys() if 'CPC' in name.upper()]
        
        print(f"📊 Available CPC models: {cpc_models}")
        
        # Check TLS224_APPLN_CPC structure
        if 'TLS224_APPLN_CPC' in models:
            model = models['TLS224_APPLN_CPC']
            columns = [column.name for column in model.__table__.columns]
            print(f"📋 TLS224_APPLN_CPC columns: {columns}")
            
            # Check for description columns
            desc_cols = [col for col in columns if 'desc' in col.lower() or 'title' in col.lower() or 'label' in col.lower()]
            print(f"🏷️ Description columns: {desc_cols}")
        
        # Look for CPC definition/description tables
        all_models = list(models.keys())
        cpc_def_models = [name for name in all_models if 'CPC' in name.upper() and any(x in name.upper() for x in ['DEF', 'DESC', 'TITLE', 'LABEL'])]
        print(f"📚 CPC definition tables: {cpc_def_models}")
        
        # Sample some actual CPC codes to see format
        print(f"\n🔬 SAMPLE CPC CODES:")
        session = patstat.connection_manager.db_session
        TLS224_APPLN_CPC = models['TLS224_APPLN_CPC']
        
        sample_cpc = session.query(TLS224_APPLN_CPC.cpc_class_symbol).distinct().limit(10).all()
        for i, (cpc_code,) in enumerate(sample_cpc):
            print(f"   {i+1}. {cpc_code}")
        
        print(f"\n💡 RECOMMENDATIONS:")
        if desc_cols:
            print("✅ CPC descriptions available in PATSTAT - use them!")
        else:
            print("⚠️ No CPC descriptions in PATSTAT - need external CPC database")
            print("💡 Consider EPO CPC database or static CPC description files")
        
    except Exception as e:
        print(f"❌ Error checking CPC tables: {e}")

if __name__ == "__main__":
    check_cpc_tables()