"""
CPC Description Data Access Module
Provides access to official Cooperative Patent Classification descriptions and hierarchies.

Data Source: https://www.cooperativepatentclassification.org/cpcSchemeAndDefinitions/bulk
Directory Structure Required:
- CPCTitleList202505/: TXT files by section (cpc-section-A_20250501.txt, etc.)
- CPCSchemeXML202505/: Hundreds of XML files (cpc-scheme-A61K.xml, etc.)
- FullCPCDefinitionXML202505/: Hundreds of XML files (cpc-definition-A61B.xml, etc.)
"""

import xml.etree.ElementTree as ET
import pandas as pd
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from functools import lru_cache
import glob

logger = logging.getLogger(__name__)

class CPCDescriptionManager:
    """
    Manager for CPC (Cooperative Patent Classification) descriptions and hierarchies.
    
    Handles the complex multi-directory structure with hundreds of XML files and TXT files.
    Provides fast lookup for CPC descriptions and hierarchical information.
    """
    
    def __init__(self, mappings_dir: Optional[Path] = None):
        """
        Initialize CPC description manager.
        
        Args:
            mappings_dir: Directory containing CPC directories (default: data_access/mappings)
        """
        if mappings_dir is None:
            mappings_dir = Path(__file__).parent / 'mappings'
        
        self.mappings_dir = Path(mappings_dir)
        
        # Data storage
        self.cpc_titles = {}           # Fast lookup: code -> description
        self.cpc_hierarchy = {}        # Hierarchical structure
        self.cpc_index = {}           # Quick index for searching
        self.loaded = False
        
        # Directory paths
        self.titles_dir = self.mappings_dir / 'CPCTitleList202505'
        self.scheme_dir = self.mappings_dir / 'CPCSchemeXML202505'
        self.definitions_dir = self.mappings_dir / 'FullCPCDefinitionXML202505'
        
        # Load CPC data if directories exist
        self._load_cpc_data()
    
    def _load_cpc_data(self):
        """Load CPC data from multi-directory structure."""
        try:
            # Check if CPC directories exist
            dirs_exist = [d.exists() for d in [self.titles_dir, self.scheme_dir, self.definitions_dir]]
            
            if not any(dirs_exist):
                logger.warning("🔍 CPC directories not found. Download from:")
                logger.warning("   https://www.cooperativepatentclassification.org/cpcSchemeAndDefinitions/bulk")
                logger.warning("   Required directories: CPCTitleList202505, CPCSchemeXML202505, FullCPCDefinitionXML202505")
                return
            
            logger.info("📚 Loading CPC data from bulk download structure...")
            
            # Load titles first (fastest for basic lookups)
            if self.titles_dir.exists():
                self._load_title_files()
                logger.debug(f"✅ CPC titles loaded: {len(self.cpc_titles)} entries")
            
            # Load scheme files (hierarchical structure - selective loading)
            if self.scheme_dir.exists():
                self._load_scheme_files_selective()
                logger.debug(f"✅ CPC scheme data loaded")
            
            # Definitions can be loaded on-demand to save memory
            self._check_definitions_availability()
            
            self.loaded = True
            logger.info(f"✅ CPC data ready: {len(self.cpc_titles)} codes indexed")
            
        except Exception as e:
            logger.error(f"❌ Failed to load CPC data: {e}")
    
    def _load_title_files(self):
        """Load CPC titles from TXT files by section."""
        for txt_file in self.titles_dir.glob('cpc-section-*.txt'):
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and '\t' in line:
                            parts = line.split('\t')
                            if len(parts) >= 3:
                                cpc_code = parts[0].strip()
                                level = parts[1].strip() if parts[1].isdigit() else '0'
                                description = parts[2].strip()
                                
                                # Store in titles dictionary
                                self.cpc_titles[cpc_code] = description
                                
                                # Store hierarchical info
                                self.cpc_hierarchy[cpc_code] = {
                                    'level': int(level),
                                    'description': description,
                                    'section': cpc_code[0] if cpc_code else ''
                                }
                                
            except Exception as e:
                logger.warning(f"⚠️ Error loading {txt_file.name}: {e}")
    
    def _load_scheme_files_selective(self):
        """Load scheme files selectively to avoid memory overload."""
        # Load only a subset of scheme files for detailed hierarchy
        # Priority: common subclasses and user's search areas
        priority_patterns = ['A61*', 'C22*', 'H01*', 'Y02*']  # Common + REE-related
        
        scheme_files = list(self.scheme_dir.glob('cpc-scheme-*.xml'))
        total_files = len(scheme_files)
        
        logger.debug(f"📋 Found {total_files} scheme files, loading selectively...")
        
        loaded_count = 0
        for xml_file in scheme_files[:50]:  # Load first 50 files to avoid memory issues
            try:
                self._parse_scheme_xml(xml_file)
                loaded_count += 1
            except Exception as e:
                logger.debug(f"⚠️ Error loading {xml_file.name}: {e}")
        
        logger.debug(f"📊 Loaded {loaded_count} scheme files")
    
    def _parse_scheme_xml(self, xml_file: Path):
        """Parse individual CPC scheme XML file."""
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Extract classification items
            for item in root.findall('.//classification-item'):
                symbol_elem = item.find('classification-symbol')
                title_elem = item.find('.//class-title//text')
                level_attr = item.get('level', '0')
                
                if symbol_elem is not None and title_elem is not None:
                    cpc_code = symbol_elem.text.strip()
                    description = title_elem.text.strip()
                    level = int(level_attr)
                    
                    # Update our data structures
                    if cpc_code not in self.cpc_titles:  # Don't overwrite TXT data
                        self.cpc_titles[cpc_code] = description
                    
                    # Enhanced hierarchy info from XML
                    if cpc_code not in self.cpc_hierarchy:
                        self.cpc_hierarchy[cpc_code] = {}
                    
                    self.cpc_hierarchy[cpc_code].update({
                        'level': level,
                        'description': description,
                        'section': cpc_code[0] if cpc_code else '',
                        'source': 'scheme_xml'
                    })
                    
        except ET.ParseError as e:
            logger.debug(f"⚠️ XML parse error in {xml_file.name}: {e}")
        except Exception as e:
            logger.debug(f"⚠️ Error parsing {xml_file.name}: {e}")
    
    def _check_definitions_availability(self):
        """Check if definition files are available (for on-demand loading)."""
        if self.definitions_dir.exists():
            def_files = list(self.definitions_dir.glob('cpc-definition-*.xml'))
            logger.debug(f"📚 {len(def_files)} definition files available for on-demand loading")
    
    def _load_specific_scheme_file(self, cpc_subclass: str) -> bool:
        """Load specific scheme file on-demand for a CPC subclass."""
        try:
            scheme_file = self.scheme_dir / f'cpc-scheme-{cpc_subclass}.xml'
            if scheme_file.exists():
                self._parse_scheme_xml(scheme_file)
                return True
        except Exception as e:
            logger.debug(f"⚠️ Could not load scheme for {cpc_subclass}: {e}")
        return False
    
    @lru_cache(maxsize=1000)
    def get_cpc_description(self, cpc_code: str) -> str:
        """
        Get description for a CPC code with intelligent fallback.
        
        Args:
            cpc_code: CPC code (e.g., "A61K", "A61K 8/97", "A61K   8/9789")
            
        Returns:
            Description string or fallback description
        """
        if not self.loaded:
            return f"CPC data not loaded: {cpc_code}"
        
        # Clean CPC code (remove extra spaces)
        clean_code = self._clean_cpc_code(cpc_code)
        
        # Try exact match first
        if clean_code in self.cpc_titles:
            return self.cpc_titles[clean_code]
        
        # Try subclass level (A61K from A61K 8/97)
        subclass = self._extract_subclass(clean_code)
        if subclass and subclass in self.cpc_titles:
            return self.cpc_titles[subclass]
        
        # Try loading specific scheme file on-demand
        if subclass and subclass not in self.cpc_titles:
            if self._load_specific_scheme_file(subclass):
                if subclass in self.cpc_titles:
                    return self.cpc_titles[subclass]
        
        # Try class level (A61 from A61K)
        class_code = self._extract_class(clean_code)
        if class_code and class_code in self.cpc_titles:
            return self.cpc_titles[class_code]
        
        # Try section level (A from A61K)
        section = clean_code[0] if clean_code else ''
        if section in self.cpc_titles:
            return f"{self.cpc_titles[section]} (broad category)"
        
        # Fallback: extract technology domain from code structure
        return self._generate_fallback_description(clean_code)
    
    @lru_cache(maxsize=1000)
    def get_cpc_hierarchy(self, cpc_code: str) -> Dict[str, str]:
        """
        Get hierarchical breakdown of CPC code.
        
        Args:
            cpc_code: CPC code
            
        Returns:
            Dictionary with section, class, subclass, group, etc.
        """
        clean_code = self._clean_cpc_code(cpc_code)
        hierarchy = {}
        
        # Parse CPC hierarchy: A61K 8/97 -> section A, class A61, subclass A61K, etc.
        if len(clean_code) >= 1:
            section = clean_code[0]
            hierarchy['section'] = section
            hierarchy['section_desc'] = self.get_cpc_description(section)
        
        if len(clean_code) >= 3:
            class_code = clean_code[:3]
            hierarchy['class'] = class_code
            hierarchy['class_desc'] = self.get_cpc_description(class_code)
        
        if len(clean_code) >= 4:
            subclass = self._extract_subclass(clean_code)
            hierarchy['subclass'] = subclass
            hierarchy['subclass_desc'] = self.get_cpc_description(subclass)
        
        # Add full code info
        hierarchy['full_code'] = clean_code
        hierarchy['full_desc'] = self.get_cpc_description(clean_code)
        
        # Add hierarchy info if available
        if clean_code in self.cpc_hierarchy:
            hierarchy.update(self.cpc_hierarchy[clean_code])
        
        return hierarchy
    
    def _generate_fallback_description(self, cpc_code: str) -> str:
        """Generate fallback description for unknown CPC codes."""
        if not cpc_code:
            return "Unknown CPC code"
        
        # Basic section mapping
        section_map = {
            'A': 'Human Necessities',
            'B': 'Performing Operations; Transporting',
            'C': 'Chemistry; Metallurgy',
            'D': 'Textiles; Paper',
            'E': 'Fixed Constructions',
            'F': 'Mechanical Engineering; Lighting; Heating',
            'G': 'Physics',
            'H': 'Electricity',
            'Y': 'General Tagging'
        }
        
        section = cpc_code[0] if cpc_code else ''
        base_desc = section_map.get(section, 'Technology')
        
        return f"{base_desc} - {cpc_code}"
    
    def _clean_cpc_code(self, cpc_code: str) -> str:
        """Clean CPC code by removing extra spaces."""
        return re.sub(r'\s+', ' ', cpc_code.strip())
    
    def _extract_subclass(self, cpc_code: str) -> str:
        """Extract subclass from CPC code (A61K from A61K 8/97)."""
        # Match pattern like "A61K" at start
        match = re.match(r'^([A-Z]\d{2}[A-Z])', cpc_code)
        return match.group(1) if match else cpc_code[:4] if len(cpc_code) >= 4 else cpc_code
    
    def _extract_class(self, cpc_code: str) -> str:
        """Extract class from CPC code (A61 from A61K)."""
        return cpc_code[:3] if len(cpc_code) >= 3 else cpc_code
    
    def get_technology_summary(self, cpc_codes: List[str]) -> pd.DataFrame:
        """
        Get technology summary for list of CPC codes.
        
        Args:
            cpc_codes: List of CPC codes
            
        Returns:
            DataFrame with technology breakdown by subclass
        """
        if not cpc_codes:
            return pd.DataFrame()
        
        # Group by subclass level for technology analysis
        subclass_counts = {}
        
        for cpc_code in cpc_codes:
            subclass = self._extract_subclass(cpc_code)
            description = self.get_cpc_description(subclass)
            
            if subclass not in subclass_counts:
                subclass_counts[subclass] = {
                    'cpc_subclass': subclass,
                    'description': description,
                    'count': 0,
                    'percentage': 0.0
                }
            subclass_counts[subclass]['count'] += 1
        
        # Convert to DataFrame and calculate percentages
        df = pd.DataFrame(list(subclass_counts.values()))
        if not df.empty:
            total = df['count'].sum()
            df['percentage'] = (df['count'] / total * 100).round(2)
            df = df.sort_values('count', ascending=False)
        
        return df
    
    def is_available(self) -> bool:
        """Check if CPC data is loaded and available."""
        return self.loaded and len(self.cpc_titles) > 0

# Factory function
def create_cpc_description_manager(mappings_dir: Optional[Path] = None) -> CPCDescriptionManager:
    """
    Create CPC description manager instance.
    
    Args:
        mappings_dir: Directory containing CPC XML files
        
    Returns:
        CPCDescriptionManager instance
    """
    return CPCDescriptionManager(mappings_dir)

# Module-level instance for shared usage
_cpc_manager = None

def get_cpc_description_manager() -> CPCDescriptionManager:
    """Get shared CPC description manager instance."""
    global _cpc_manager
    if _cpc_manager is None:
        _cpc_manager = create_cpc_description_manager()
    return _cpc_manager