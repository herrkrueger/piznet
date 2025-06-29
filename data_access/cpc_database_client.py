"""
CPC Database Client
Fast SQLite-based CPC lookup and analysis client.

Replaces hardcoded technology domains with official CPC data.
"""

import sqlite3
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

class CPCDatabaseClient:
    """
    High-performance CPC database client for classification analysis.
    
    Provides fast lookups for CPC codes, descriptions, and technology domains.
    Replaces hardcoded TECHNOLOGY_DOMAINS with official CPC subclass data.
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize CPC database client.
        
        Args:
            db_path: Path to CPC SQLite database (default: data_access/mappings/cpc_database.sqlite)
        """
        if db_path is None:
            db_path = Path(__file__).parent / 'mappings' / 'cpc_database.sqlite'
        
        self.db_path = Path(db_path)
        self.conn = None
        self.available = False
        
        # Connect to database
        self._connect()
    
    def _connect(self):
        """Connect to CPC database."""
        try:
            if not self.db_path.exists():
                logger.warning(f"🔍 CPC database not found at {self.db_path}")
                logger.warning("💡 Run 'python scripts/cpc_importer.py' to create it")
                return
            
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row  # Enable column access by name
            
            # Test connection
            cursor = self.conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM cpc_codes")
            count = cursor.fetchone()[0]
            
            self.available = True
            logger.debug(f"✅ CPC database connected: {count:,} codes available")
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to CPC database: {e}")
            self.available = False
    
    @lru_cache(maxsize=1000)
    def get_cpc_description(self, cpc_code: str) -> str:
        """
        Get official description for CPC code.
        
        Args:
            cpc_code: CPC code (e.g., "A61K", "C22B19/28")
            
        Returns:
            Official CPC description or fallback
        """
        if not self.available:
            return f"CPC database unavailable: {cpc_code}"
        
        try:
            cursor = self.conn.cursor()
            
            # Clean code
            clean_code = cpc_code.strip()
            
            # Try exact match first
            cursor.execute("SELECT description FROM cpc_codes WHERE cpc_code = ?", (clean_code,))
            result = cursor.fetchone()
            if result:
                return result['description']
            
            # Try subclass match (A61K from A61K 8/97)
            subclass = self._extract_subclass(clean_code)
            if subclass:
                cursor.execute("SELECT description FROM cpc_codes WHERE cpc_code = ?", (subclass,))
                result = cursor.fetchone()
                if result:
                    return result['description']
            
            # Try class match (A61 from A61K)
            class_code = clean_code[:3] if len(clean_code) >= 3 else clean_code
            cursor.execute("SELECT description FROM cpc_codes WHERE cpc_code = ?", (class_code,))
            result = cursor.fetchone()
            if result:
                return result['description']
            
            # Try section match (A from A61K)
            section = clean_code[0] if clean_code else ''
            cursor.execute("SELECT description FROM cpc_codes WHERE cpc_code = ?", (section,))
            result = cursor.fetchone()
            if result:
                return f"{result['description']} (broad category)"
            
            return f"Unknown CPC: {cpc_code}"
            
        except Exception as e:
            logger.debug(f"⚠️ CPC lookup error for {cpc_code}: {e}")
            return f"CPC lookup failed: {cpc_code}"
    
    def _extract_subclass(self, cpc_code: str) -> Optional[str]:
        """Extract subclass from CPC code (A61K from A61K 8/97)."""
        import re
        clean_code = cpc_code.replace(' ', '')
        match = re.match(r'^([A-Z]\d{2}[A-Z])', clean_code)
        return match.group(1) if match else None
    
    def get_technology_domains(self, cpc_codes: List[str]) -> pd.DataFrame:
        """
        Get technology domain breakdown for list of CPC codes.
        
        This REPLACES the old hardcoded TECHNOLOGY_DOMAINS approach!
        
        Args:
            cpc_codes: List of CPC codes from patents
            
        Returns:
            DataFrame with technology domains and counts
        """
        if not self.available or not cpc_codes:
            return pd.DataFrame(columns=['technology_code', 'technology_description', 'count', 'percentage'])
        
        try:
            # Extract subclasses from all CPC codes
            subclass_counts = {}
            
            for cpc_code in cpc_codes:
                subclass = self._extract_subclass(cpc_code)
                if subclass:
                    if subclass not in subclass_counts:
                        subclass_counts[subclass] = 0
                    subclass_counts[subclass] += 1
            
            if not subclass_counts:
                return pd.DataFrame(columns=['technology_code', 'technology_description', 'count', 'percentage'])
            
            # Get descriptions for subclasses
            cursor = self.conn.cursor()
            placeholders = ','.join('?' * len(subclass_counts))
            
            cursor.execute(f"""
                SELECT cpc_code, description 
                FROM cpc_codes 
                WHERE cpc_code IN ({placeholders}) AND LENGTH(cpc_code) = 4
            """, list(subclass_counts.keys()))
            
            descriptions = {row['cpc_code']: row['description'] for row in cursor.fetchall()}
            
            # Build result DataFrame
            results = []
            total_count = sum(subclass_counts.values())
            
            for subclass, count in subclass_counts.items():
                description = descriptions.get(subclass, f"Unknown subclass: {subclass}")
                percentage = (count / total_count * 100) if total_count > 0 else 0
                
                results.append({
                    'technology_code': subclass,
                    'technology_description': description,
                    'count': count,
                    'percentage': round(percentage, 2)
                })
            
            # Sort by count descending
            df = pd.DataFrame(results)
            df = df.sort_values('count', ascending=False).reset_index(drop=True)
            
            logger.debug(f"✅ Technology domains: {len(df)} subclasses vs 30 hardcoded domains!")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Technology domains analysis failed: {e}")
            return pd.DataFrame(columns=['technology_code', 'technology_description', 'count', 'percentage'])
    
    def get_subclass_summary(self, limit: int = 50) -> pd.DataFrame:
        """
        Get summary of all available CPC subclasses.
        
        Args:
            limit: Maximum number of subclasses to return
            
        Returns:
            DataFrame with subclass codes and descriptions
        """
        if not self.available:
            return pd.DataFrame()
        
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT subclass as technology_code, 
                       subclass_description as technology_description,
                       total_codes
                FROM v_subclass_summary 
                ORDER BY total_codes DESC 
                LIMIT ?
            """, (limit,))
            
            return pd.DataFrame(cursor.fetchall())
            
        except Exception as e:
            logger.error(f"❌ Subclass summary failed: {e}")
            return pd.DataFrame()
    
    def search_cpc_codes(self, search_term: str, limit: int = 20) -> pd.DataFrame:
        """
        Search CPC codes and descriptions.
        
        Args:
            search_term: Term to search for
            limit: Maximum results to return
            
        Returns:
            DataFrame with matching CPC codes
        """
        if not self.available:
            return pd.DataFrame()
        
        try:
            cursor = self.conn.cursor()
            
            # Use FTS if available, otherwise LIKE search
            try:
                cursor.execute("""
                    SELECT cpc_code, description 
                    FROM cpc_search_fts 
                    WHERE cpc_search_fts MATCH ? 
                    LIMIT ?
                """, (search_term, limit))
                results = cursor.fetchall()
            except:
                # Fallback to LIKE search
                cursor.execute("""
                    SELECT cpc_code, description 
                    FROM cpc_codes 
                    WHERE description LIKE ? OR cpc_code LIKE ?
                    LIMIT ?
                """, (f'%{search_term}%', f'%{search_term}%', limit))
                results = cursor.fetchall()
            
            return pd.DataFrame(results)
            
        except Exception as e:
            logger.error(f"❌ CPC search failed: {e}")
            return pd.DataFrame()
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            self.available = False

# Factory function
def create_cpc_database_client(db_path: Optional[Path] = None) -> CPCDatabaseClient:
    """
    Create CPC database client instance.
    
    Args:
        db_path: Path to CPC database
        
    Returns:
        CPCDatabaseClient instance
    """
    return CPCDatabaseClient(db_path)

# Module-level client for shared usage
_cpc_client = None

def get_cpc_client() -> CPCDatabaseClient:
    """Get shared CPC database client."""
    global _cpc_client
    if _cpc_client is None:
        _cpc_client = create_cpc_database_client()
    return _cpc_client