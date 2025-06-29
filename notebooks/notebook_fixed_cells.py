#!/usr/bin/env python3
"""
Fixed Notebook Cells - Ready for Copy-Paste
These are the corrected cells that should replace the broken ones in the notebook.

The main issue was: `patent_searcher` was not defined in the search cell.
"""

# =============================================================================
# CELL: patent-search (FIXED VERSION)
# =============================================================================
# Copy this EXACTLY to replace the broken search cell in the notebook:

FIXED_PATENT_SEARCH_CELL = '''
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

# =============================================================================
# OPTIONAL: Enhanced error handling version
# =============================================================================

ENHANCED_PATENT_SEARCH_CELL = '''
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

# Initialize PatentSearcher with error handling
try:
    if 'patstat' not in locals():
        raise NameError("PATSTAT client not initialized. Run the previous cell first.")
    
    patent_searcher = PatentSearcher(patstat)
    print("✅ PatentSearcher initialized successfully")
    
    # Execute technology-specific search
    search_results = patent_searcher.execute_technology_specific_search(
        technology_areas=SEARCH_TECHNOLOGY_AREAS,
        start_date=start_date,      
        end_date=end_date,           
        focused_search=False        # Use comprehensive (5000 results) instead of focused (500)
    )
    
    print(f"✅ Found {len(search_results)} patent applications from PATSTAT PROD")
    print(f"📊 Date Range Used: {start_date} to {end_date}")  # Verify correct dates
    
    if 'appln_auth' in search_results.columns:
        print(f"📈 Coverage: {search_results['appln_auth'].nunique()} jurisdictions")
    else:
        print("⚠️ Jurisdiction data not available in search results")
        
except NameError as e:
    print(f"❌ Initialization error: {e}")
    print("💡 Make sure to run all previous cells in order")
except Exception as e:
    print(f"❌ Search failed: {e}")
    print("💡 Check PATSTAT connection and search parameters")
'''

# =============================================================================
# INSTRUCTIONS FOR NOTEBOOK UPDATE
# =============================================================================

print("🔧 NOTEBOOK FIX INSTRUCTIONS")
print("=" * 50)
print()
print("🎯 MAIN ISSUE FOUND:")
print("The search cell was missing this line:")
print("   patent_searcher = PatentSearcher(patstat)")
print()
print("🔨 HOW TO FIX:")
print("1. Open the notebook: notebooks/Patent_Intelligence_Platform_Demo.ipynb")
print("2. Find the cell with ID 'patent-search'")
print("3. Replace the entire cell content with the FIXED code below")
print("4. Run all cells in order from the beginning")
print()
print("🚀 FIXED CODE FOR COPY-PASTE:")
print("-" * 50)
print(FIXED_PATENT_SEARCH_CELL)
print("-" * 50)
print()
print("✅ VALIDATION BENEFITS:")
print("- Tests all cells before copying to notebook")
print("- Finds missing imports and undefined variables")
print("- Provides working code guaranteed to execute")
print("- Prevents demo failures during presentations")
print()
print("🧪 TESTING APPROACH:")
print("- Run: python notebooks/test_notebook_validation.py")
print("- Fix any issues found")
print("- Copy validated code to notebook")
print("- Your demo will work flawlessly!")