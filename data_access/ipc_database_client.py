"""
IPC Database Client
Fast SQLite-based IPC lookup and analysis client.

Provides IPC classification analysis parallel to CPC database client.
Compatible API for easy switching between IPC and CPC systems.
"""

import sqlite3
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

class IPCDatabaseClient:
    """
    High-performance IPC database client for classification analysis.
    
    Provides fast lookups for IPC codes, descriptions, and technology domains.
    Compatible with CPC database client API for easy switching.
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize IPC database client.
        
        Args:
            db_path: Path to IPC SQLite database (default: data_access/mappings/ipc_database.sqlite)
        """
        if db_path is None:
            db_path = Path(__file__).parent / 'mappings' / 'ipc_database.sqlite'
        
        self.db_path = Path(db_path)
        self.conn = None
        self.available = False
        
        # Connect to database
        self._connect()
    
    def _connect(self):
        """Connect to IPC database."""
        try:
            if not self.db_path.exists():
                logger.warning(f"🔍 IPC database not found at {self.db_path}")
                logger.warning("💡 Run 'python scripts/ipc_importer.py' to create it")
                return
            
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row  # Enable column access by name
            
            # Test connection
            cursor = self.conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM ipc_codes")
            count = cursor.fetchone()[0]
            
            self.available = True
            logger.debug(f"✅ IPC database connected: {count:,} codes available")
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to IPC database: {e}")
            self.available = False
    
    @lru_cache(maxsize=1000)
    def get_ipc_description(self, ipc_code: str) -> str:
        """
        Get official description for IPC code.
        
        Args:
            ipc_code: IPC code (e.g., "A61K", "A01B0001000000")
            
        Returns:
            Official IPC description or fallback
        """
        if not self.available:
            return f"IPC database unavailable: {ipc_code}"
        
        try:
            cursor = self.conn.cursor()
            
            # Clean code
            clean_code = ipc_code.strip()
            
            # Try exact match first
            cursor.execute("SELECT description FROM ipc_codes WHERE ipc_code = ?", (clean_code,))
            result = cursor.fetchone()
            if result:
                return result['description']
            
            # Try subclass match (A01B from A01B0001000000)
            subclass = self._extract_subclass(clean_code)
            if subclass:
                cursor.execute("SELECT description FROM ipc_codes WHERE ipc_code = ?", (subclass,))
                result = cursor.fetchone()
                if result:
                    return result['description']
            
            # Try class match (A01 from A01B)
            class_code = clean_code[:3] if len(clean_code) >= 3 else clean_code
            cursor.execute("SELECT description FROM ipc_codes WHERE ipc_code = ?", (class_code,))
            result = cursor.fetchone()
            if result:
                return result['description']
            
            # Try section match (A from A01B)
            section = clean_code[0] if clean_code else ''
            cursor.execute("SELECT description FROM ipc_codes WHERE ipc_code = ?", (section,))
            result = cursor.fetchone()
            if result:
                return f"{result['description']} (broad category)"
            
            return f"Unknown IPC: {ipc_code}"
            
        except Exception as e:
            logger.debug(f"⚠️ IPC lookup error for {ipc_code}: {e}")
            return f"IPC lookup failed: {ipc_code}"
    
    def _extract_subclass(self, ipc_code: str) -> Optional[str]:
        """Extract subclass from IPC code (A01B from A01B0001000000)."""
        import re
        clean_code = ipc_code.replace(' ', '')
        
        # IPC subclass is first 4 characters (like CPC)
        if len(clean_code) >= 4:
            match = re.match(r'^([A-Z]\d{2}[A-Z])', clean_code)
            return match.group(1) if match else None
        
        return None
    
    def get_technology_domains(self, ipc_codes: List[str]) -> pd.DataFrame:
        """
        Get technology domain breakdown for list of IPC codes.
        
        This provides IPC-based technology domains parallel to CPC approach!
        Compatible API with CPC database client.
        
        Args:
            ipc_codes: List of IPC codes from patents
            
        Returns:
            DataFrame with technology domains and counts
        """
        if not self.available or not ipc_codes:
            return pd.DataFrame(columns=['technology_code', 'technology_description', 'count', 'percentage'])
        
        try:
            # Extract subclasses from all IPC codes
            subclass_counts = {}
            
            for ipc_code in ipc_codes:
                subclass = self._extract_subclass(ipc_code)
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
                SELECT ipc_code, description 
                FROM ipc_codes 
                WHERE ipc_code IN ({placeholders}) AND LENGTH(ipc_code) = 4
            """, list(subclass_counts.keys()))
            
            descriptions = {row['ipc_code']: row['description'] for row in cursor.fetchall()}
            
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
            
            logger.debug(f"✅ IPC technology domains: {len(df)} subclasses")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ IPC technology domains analysis failed: {e}")
            return pd.DataFrame(columns=['technology_code', 'technology_description', 'count', 'percentage'])
    
    def get_subclass_summary(self, limit: int = 50) -> pd.DataFrame:
        """
        Get summary of all available IPC subclasses.
        
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
                SELECT technology_code, 
                       technology_description,
                       total_codes
                FROM v_ipc_subclass_summary 
                ORDER BY total_codes DESC 
                LIMIT ?
            """, (limit,))
            
            return pd.DataFrame(cursor.fetchall())
            
        except Exception as e:
            logger.error(f"❌ IPC subclass summary failed: {e}")
            return pd.DataFrame()
    
    def search_ipc_codes(self, search_term: str, limit: int = 20) -> pd.DataFrame:
        """
        Search IPC codes and descriptions.
        
        Args:
            search_term: Term to search for
            limit: Maximum results to return
            
        Returns:
            DataFrame with matching IPC codes
        """
        if not self.available:
            return pd.DataFrame()
        
        try:
            cursor = self.conn.cursor()
            
            # Use FTS if available, otherwise LIKE search
            try:
                cursor.execute("""
                    SELECT ipc_code, description 
                    FROM ipc_search_fts 
                    WHERE ipc_search_fts MATCH ? 
                    LIMIT ?
                """, (search_term, limit))
                results = cursor.fetchall()
            except:
                # Fallback to LIKE search
                cursor.execute("""
                    SELECT ipc_code, description 
                    FROM ipc_codes 
                    WHERE description LIKE ? OR ipc_code LIKE ?
                    LIMIT ?
                """, (f'%{search_term}%', f'%{search_term}%', limit))
                results = cursor.fetchall()
            
            return pd.DataFrame(results)
            
        except Exception as e:
            logger.error(f"❌ IPC search failed: {e}")
            return pd.DataFrame()
    
    def get_illustrations(self, ipc_code: str) -> pd.DataFrame:
        """
        Get illustration references for IPC code.
        
        Args:
            ipc_code: IPC code to search for
            
        Returns:
            DataFrame with illustration references
        """
        if not self.available:
            return pd.DataFrame()
        
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT illustration_label, src_filename, format
                FROM ipc_illustrations 
                WHERE ipc_code = ?
                ORDER BY illustration_label
            """, (ipc_code,))
            
            return pd.DataFrame(cursor.fetchall())
            
        except Exception as e:
            logger.error(f"❌ IPC illustrations lookup failed: {e}")
            return pd.DataFrame()
    
    def get_section_overview(self) -> pd.DataFrame:
        """Get overview of all IPC sections."""
        if not self.available:
            return pd.DataFrame()
        
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM v_ipc_section_overview")
            return pd.DataFrame(cursor.fetchall())
            
        except Exception as e:
            logger.error(f"❌ IPC section overview failed: {e}")
            return pd.DataFrame()
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            self.available = False

# Factory function
def create_ipc_database_client(db_path: Optional[Path] = None) -> IPCDatabaseClient:
    """
    Create IPC database client instance.
    
    Args:
        db_path: Path to IPC database
        
    Returns:
        IPCDatabaseClient instance
    """
    return IPCDatabaseClient(db_path)

# Module-level client for shared usage
_ipc_client = None

def get_ipc_client() -> IPCDatabaseClient:
    """Get shared IPC database client."""
    global _ipc_client
    if _ipc_client is None:
        _ipc_client = create_ipc_database_client()
    return _ipc_client

# Compatibility aliases for easy switching between IPC/CPC
def get_classification_client(classification_system: str = 'ipc') -> IPCDatabaseClient:
    """
    Get classification client for specified system.
    
    Args:
        classification_system: 'ipc' or 'cpc'
        
    Returns:
        Classification database client
    """
    if classification_system.lower() == 'ipc':
        return get_ipc_client()
    elif classification_system.lower() == 'cpc':
        # Import CPC client
        from .cpc_database_client import get_cpc_client
        return get_cpc_client()
    else:
        raise ValueError(f"Unknown classification system: {classification_system}")