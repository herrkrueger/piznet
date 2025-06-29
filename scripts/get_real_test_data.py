#!/usr/bin/env python3
"""
Get Real Test Data from PATSTAT
Queries PATSTAT to find actual application numbers for testing
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_access import PatstatClient

def get_real_test_data():
    """Query PATSTAT for real application numbers to use in tests."""
    
    print("🔍 Querying PATSTAT for real test data...")
    
    try:
        # Initialize PATSTAT client
        client = PatstatClient(environment='PROD')
        
        if not client.is_connected:
            print("❌ Could not connect to PATSTAT")
            return None
            
        print("✅ Connected to PATSTAT")
        
        # Get database session and models
        session = client.connection_manager.get_session()
        models = client.connection_manager.get_models()
        
        print("📊 Querying for family data...")
        
        import pandas as pd
        
        # Use SQLAlchemy models to query
        TLS201_APPLN = models['TLS201_APPLN']
        TLS202_APPLN_TITLE = models['TLS202_APPLN_TITLE']
        
        # Get distinct families first - more selective approach
        from sqlalchemy import func
        
        distinct_families = session.query(
            TLS201_APPLN.docdb_family_id,
            func.min(TLS201_APPLN.appln_id).label('min_appln_id'),
            func.max(TLS201_APPLN.docdb_family_size).label('family_size')
        ).filter(
            TLS201_APPLN.earliest_filing_year.between(2018, 2020),
            TLS201_APPLN.docdb_family_size.between(3, 15),  # Medium-sized families
            TLS201_APPLN.docdb_family_id.isnot(None)
        ).group_by(
            TLS201_APPLN.docdb_family_id
        ).order_by(
            func.max(TLS201_APPLN.docdb_family_size).desc()
        ).limit(5).all()
        
        # Extract just the data we need
        family_data = []
        for fam_id, min_appln_id, family_size in distinct_families:
            # Get one representative application from each family
            app_query = session.query(
                TLS201_APPLN.appln_id,
                TLS201_APPLN.docdb_family_id,
                TLS201_APPLN.earliest_filing_year,
                TLS201_APPLN.docdb_family_size,
                TLS202_APPLN_TITLE.appln_title
            ).outerjoin(
                TLS202_APPLN_TITLE, TLS201_APPLN.appln_id == TLS202_APPLN_TITLE.appln_id
            ).filter(
                TLS201_APPLN.docdb_family_id == fam_id
            ).first()
            
            if app_query:
                family_data.append({
                    'appln_id': app_query.appln_id,
                    'docdb_family_id': app_query.docdb_family_id,
                    'earliest_filing_year': app_query.earliest_filing_year,
                    'docdb_family_size': app_query.docdb_family_size,
                    'appln_title': app_query.appln_title
                })
        
        # Convert to DataFrame
        results = pd.DataFrame(family_data)
        
        
        if results is not None and len(results) > 0:
            print(f"✅ Found {len(results)} patent families")
            print("\n🎯 Real Test Data Found:")
            print("=" * 60)
            
            for i, row in results.iterrows():
                title = str(row['appln_title'])[:50] + "..." if len(str(row['appln_title'])) > 50 else str(row['appln_title'])
                print(f"Family {i+1}:")
                print(f"  📋 Family ID: {row['docdb_family_id']}")
                print(f"  📋 Application ID: {row['appln_id']}")
                print(f"  📅 Filing Year: {row['earliest_filing_year']}")
                print(f"  👥 Family Size: {row['docdb_family_size']}")
                print(f"  📝 Title: {title}")
                print()
            
            # Also check for applicant data
            print("🏢 Checking applicant data availability...")
            family_ids = results['docdb_family_id'].tolist()[:3]  # Check first 3
            
            TLS207_PERS_APPLN = models['TLS207_PERS_APPLN']
            TLS206_PERSON = models['TLS206_PERSON']
            
            applicant_query = session.query(
                TLS201_APPLN.docdb_family_id,
                TLS206_PERSON.person_name,
                TLS207_PERS_APPLN.applt_seq_nr
            ).join(
                TLS207_PERS_APPLN, TLS201_APPLN.appln_id == TLS207_PERS_APPLN.appln_id
            ).join(
                TLS206_PERSON, TLS207_PERS_APPLN.person_id == TLS206_PERSON.person_id
            ).filter(
                TLS201_APPLN.docdb_family_id.in_(family_ids),
                TLS207_PERS_APPLN.applt_seq_nr > 0
            ).limit(20)
            
            applicant_raw = applicant_query.all()
            applicant_results = pd.DataFrame([{
                'docdb_family_id': r.docdb_family_id,
                'person_name': r.person_name,
                'applt_seq_nr': r.applt_seq_nr
            } for r in applicant_raw])
            
            if applicant_results is not None and len(applicant_results) > 0:
                print(f"✅ Found applicant data for {len(applicant_results)} records")
                for _, row in applicant_results.head().iterrows():
                    print(f"  👥 Family {row['docdb_family_id']}: {row['person_name']}")
            else:
                print("⚠️ No applicant data found for these families")
            
            return results
            
        else:
            print("❌ No results found")
            return None
            
    except Exception as e:
        print(f"❌ Error querying PATSTAT: {e}")
        return None

def generate_config_section(results):
    """Generate YAML config section with real data."""
    if results is None or len(results) == 0:
        return None
        
    print("\n📝 Generated Config Section:")
    print("=" * 40)
    print("# Real PATSTAT Test Data")
    print("demo_parameters:")
    print("  test_families:")
    
    for i, row in results.head(5).iterrows():
        print(f"    - {row['docdb_family_id']}  # {str(row['appln_title'])[:30]}...")
    
    print("  ")
    print("  test_applications:")
    for i, row in results.head(5).iterrows():
        print(f"    - {row['appln_id']}  # Family {row['docdb_family_id']}")

if __name__ == "__main__":
    results = get_real_test_data()
    if results is not None:
        generate_config_section(results)
    else:
        print("❌ Failed to get real test data")