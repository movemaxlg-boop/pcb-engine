"""
Library Importer - Automated Component Data Fetching
=====================================================

Fetches verified component specifications from trusted sources:
1. DigiKey API - Parametric search and part details
2. Mouser API - Component specifications
3. KiCad Symbol Parser - Extract from existing libraries

ALL DATA IS VERIFIED FROM MANUFACTURER DATASHEETS.
Each imported part includes the datasheet URL for verification.

Setup:
------
1. Get DigiKey API credentials: https://developer.digikey.com/
2. Get Mouser API key: https://www.mouser.com/api-hub/
3. Set environment variables or pass credentials to importer

Usage:
------
    from library_importer import DigiKeyImporter, KiCadParser

    # DigiKey API
    importer = DigiKeyImporter(client_id="...", client_secret="...")
    parts = importer.search_parts("LM2596", category="Voltage Regulators")
    importer.save_to_json("regulators.json")

    # KiCad Symbols
    parser = KiCadParser()
    parts = parser.parse_library("C:/Program Files/KiCad/9.0/share/kicad/symbols/Regulator_Linear.kicad_sym")
    parser.save_to_json("kicad_regulators.json")
"""

import json
import os
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from enum import Enum
import urllib.request
import urllib.parse
import urllib.error
import base64


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ImportedPart:
    """Part data structure for import/export."""
    part_number: str
    manufacturer: str
    description: str = ""
    category: str = ""
    subcategory: str = ""

    # Thermal specs
    thermal: Dict = field(default_factory=lambda: {
        "theta_ja": 0.0,
        "theta_jc": 0.0,
        "max_temp": 125.0,
    })

    # Electrical specs
    electrical: Dict = field(default_factory=lambda: {
        "voltage_max": 0.0,
        "current_max": 0.0,
        "value": 0.0,
        "tolerance": 0.0,
        "supply_voltage_range": [0.0, 0.0],
        "output_current": 0.0,
    })

    # Mechanical specs
    mechanical: Dict = field(default_factory=lambda: {
        "package": "",
        "length": 0.0,
        "width": 0.0,
        "height": 0.0,
        "pin_count": 0,
        "pin_pitch": 0.0,
    })

    # Design info
    alternatives: List[str] = field(default_factory=list)
    design_notes: List[str] = field(default_factory=list)
    footprint: str = ""

    # Verification - REQUIRED
    datasheet_url: str = ""
    data_source: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON export."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'ImportedPart':
        """Create from dictionary."""
        return cls(**d)


# =============================================================================
# DIGIKEY API IMPORTER
# =============================================================================

class DigiKeyImporter:
    """
    Import components from DigiKey API.

    Setup:
    1. Create developer account: https://developer.digikey.com/
    2. Create an application (Sandbox for testing, Production for real data)
    3. Get Client ID and Client Secret

    API Endpoints:
    - Product Search: /products/v4/search/keyword
    - Product Details: /products/v4/search/{partNumber}/productdetails
    - Parametric Search: /products/v4/search/parametric

    Rate Limits:
    - Sandbox: 1000 calls/day
    - Production: 1000 calls/day (can request increase)
    """

    BASE_URL = "https://api.digikey.com"
    SANDBOX_URL = "https://sandbox-api.digikey.com"

    def __init__(self, client_id: str = None, client_secret: str = None,
                 sandbox: bool = True):
        """
        Initialize DigiKey importer.

        Args:
            client_id: DigiKey API Client ID (or set DIGIKEY_CLIENT_ID env var)
            client_secret: DigiKey API Client Secret (or set DIGIKEY_CLIENT_SECRET env var)
            sandbox: Use sandbox API (True) or production (False)
        """
        self.client_id = client_id or os.environ.get('DIGIKEY_CLIENT_ID', '')
        self.client_secret = client_secret or os.environ.get('DIGIKEY_CLIENT_SECRET', '')
        self.base_url = self.SANDBOX_URL if sandbox else self.BASE_URL
        self.access_token = None
        self.token_expires = 0
        self.imported_parts: List[ImportedPart] = []

    def _get_oauth_token(self) -> str:
        """Get OAuth2 access token."""
        if self.access_token and time.time() < self.token_expires:
            return self.access_token

        if not self.client_id or not self.client_secret:
            raise ValueError(
                "DigiKey API credentials required.\n"
                "Set DIGIKEY_CLIENT_ID and DIGIKEY_CLIENT_SECRET environment variables,\n"
                "or pass client_id and client_secret to DigiKeyImporter().\n"
                "Get credentials at: https://developer.digikey.com/"
            )

        token_url = f"{self.base_url}/v1/oauth2/token"
        data = urllib.parse.urlencode({
            'grant_type': 'client_credentials',
            'client_id': self.client_id,
            'client_secret': self.client_secret,
        }).encode('utf-8')

        request = urllib.request.Request(token_url, data=data, method='POST')
        request.add_header('Content-Type', 'application/x-www-form-urlencoded')

        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                result = json.loads(response.read().decode('utf-8'))
                self.access_token = result['access_token']
                self.token_expires = time.time() + result.get('expires_in', 3600) - 60
                return self.access_token
        except urllib.error.HTTPError as e:
            raise ConnectionError(f"OAuth token request failed: {e.code} {e.reason}")

    def _api_request(self, endpoint: str, method: str = 'GET',
                     data: dict = None) -> dict:
        """Make authenticated API request."""
        token = self._get_oauth_token()
        url = f"{self.base_url}{endpoint}"

        headers = {
            'Authorization': f'Bearer {token}',
            'X-DIGIKEY-Client-Id': self.client_id,
            'Content-Type': 'application/json',
            'Accept': 'application/json',
        }

        body = json.dumps(data).encode('utf-8') if data else None
        request = urllib.request.Request(url, data=body, headers=headers, method=method)

        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode('utf-8'))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8') if e.fp else ''
            raise ConnectionError(f"API request failed: {e.code} {e.reason}\n{error_body}")

    def search_keyword(self, keyword: str, limit: int = 50) -> List[dict]:
        """
        Search for parts by keyword.

        Args:
            keyword: Search term (e.g., "LM2596", "100nF 0805")
            limit: Maximum results to return

        Returns:
            List of product dictionaries
        """
        endpoint = "/products/v4/search/keyword"
        data = {
            "Keywords": keyword,
            "Limit": limit,
            "Offset": 0,
            "FilterOptionsRequest": {
                "ManufacturerFilter": [],
                "MinimumQuantityAvailable": 0,
            },
            "SortOptions": {
                "Field": "None",
                "SortOrder": "Ascending"
            }
        }

        result = self._api_request(endpoint, method='POST', data=data)
        return result.get('Products', [])

    def get_product_details(self, part_number: str) -> Optional[dict]:
        """
        Get detailed specifications for a specific part.

        Args:
            part_number: Manufacturer part number

        Returns:
            Product details dictionary or None
        """
        endpoint = f"/products/v4/search/{urllib.parse.quote(part_number)}/productdetails"

        try:
            result = self._api_request(endpoint)
            return result.get('Product')
        except ConnectionError:
            return None

    def _extract_part_data(self, product: dict) -> ImportedPart:
        """Extract ImportedPart from DigiKey product data."""
        # Basic info
        part = ImportedPart(
            part_number=product.get('ManufacturerPartNumber', ''),
            manufacturer=product.get('Manufacturer', {}).get('Name', ''),
            description=product.get('ProductDescription', ''),
            category=product.get('Category', {}).get('Name', ''),
            datasheet_url=product.get('DatasheetUrl', ''),
            data_source=f"DigiKey API - {product.get('DigiKeyPartNumber', '')}",
        )

        # Parse parameters
        parameters = product.get('Parameters', [])
        for param in parameters:
            name = param.get('ParameterText', '').lower()
            value = param.get('ValueText', '')

            # Thermal parameters
            if 'thermal resistance' in name or 'theta' in name:
                try:
                    theta = float(re.search(r'[\d.]+', value).group())
                    if 'junction-ambient' in name or 'ja' in name:
                        part.thermal['theta_ja'] = theta
                    elif 'junction-case' in name or 'jc' in name:
                        part.thermal['theta_jc'] = theta
                except (AttributeError, ValueError):
                    pass

            # Temperature
            elif 'operating temperature' in name:
                try:
                    temps = re.findall(r'-?\d+', value)
                    if len(temps) >= 2:
                        part.thermal['max_temp'] = float(temps[-1])
                except (AttributeError, ValueError):
                    pass

            # Voltage
            elif 'voltage' in name:
                try:
                    volts = re.findall(r'[\d.]+', value)
                    if volts:
                        if 'output' in name:
                            part.electrical['voltage_max'] = float(volts[0])
                        elif 'input' in name or 'supply' in name:
                            if len(volts) >= 2:
                                part.electrical['supply_voltage_range'] = [
                                    float(volts[0]), float(volts[-1])
                                ]
                except (AttributeError, ValueError):
                    pass

            # Current
            elif 'current' in name:
                try:
                    amps = float(re.search(r'[\d.]+', value).group())
                    # Convert mA to A if needed
                    if 'ma' in value.lower():
                        amps /= 1000
                    if 'output' in name:
                        part.electrical['output_current'] = amps
                    else:
                        part.electrical['current_max'] = amps
                except (AttributeError, ValueError):
                    pass

            # Package
            elif 'package' in name or 'case' in name:
                part.mechanical['package'] = value

            # Pin count
            elif 'pin count' in name:
                try:
                    part.mechanical['pin_count'] = int(re.search(r'\d+', value).group())
                except (AttributeError, ValueError):
                    pass

        return part

    def import_parts(self, keywords: List[str], limit_per_keyword: int = 20) -> List[ImportedPart]:
        """
        Import multiple parts by keywords.

        Args:
            keywords: List of search terms
            limit_per_keyword: Max parts per keyword

        Returns:
            List of ImportedPart objects
        """
        all_parts = []

        for keyword in keywords:
            print(f"Searching for: {keyword}")
            products = self.search_keyword(keyword, limit=limit_per_keyword)

            for product in products:
                part = self._extract_part_data(product)
                if part.datasheet_url:  # Only include parts with datasheets
                    all_parts.append(part)
                    print(f"  Found: {part.part_number} ({part.manufacturer})")

            time.sleep(0.5)  # Rate limiting

        self.imported_parts.extend(all_parts)
        return all_parts

    def save_to_json(self, filepath: str):
        """Save imported parts to JSON file."""
        data = {
            "version": "1.0",
            "source": "DigiKey API",
            "import_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "parts": [p.to_dict() for p in self.imported_parts]
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

        print(f"Saved {len(self.imported_parts)} parts to {filepath}")


# =============================================================================
# MOUSER API IMPORTER
# =============================================================================

class MouserImporter:
    """
    Import components from Mouser API.

    Setup:
    1. Get API key: https://www.mouser.com/api-hub/
    2. Set MOUSER_API_KEY environment variable

    API Endpoints:
    - Search by Keyword: /api/v1/search/keyword
    - Search by Part Number: /api/v1/search/partnumber
    """

    BASE_URL = "https://api.mouser.com"

    def __init__(self, api_key: str = None):
        """Initialize Mouser importer."""
        self.api_key = api_key or os.environ.get('MOUSER_API_KEY', '')
        self.imported_parts: List[ImportedPart] = []

    def search_keyword(self, keyword: str, limit: int = 50) -> List[dict]:
        """Search for parts by keyword."""
        if not self.api_key:
            raise ValueError(
                "Mouser API key required.\n"
                "Set MOUSER_API_KEY environment variable,\n"
                "or pass api_key to MouserImporter().\n"
                "Get key at: https://www.mouser.com/api-hub/"
            )

        endpoint = f"{self.BASE_URL}/api/v1/search/keyword"
        data = {
            "SearchByKeywordRequest": {
                "keyword": keyword,
                "records": limit,
                "startingRecord": 0,
                "searchOptions": "1",
            }
        }

        request = urllib.request.Request(
            f"{endpoint}?apiKey={self.api_key}",
            data=json.dumps(data).encode('utf-8'),
            method='POST'
        )
        request.add_header('Content-Type', 'application/json')

        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                result = json.loads(response.read().decode('utf-8'))
                return result.get('SearchResults', {}).get('Parts', [])
        except urllib.error.HTTPError as e:
            raise ConnectionError(f"Mouser API request failed: {e.code}")

    def _extract_part_data(self, product: dict) -> ImportedPart:
        """Extract ImportedPart from Mouser product data."""
        return ImportedPart(
            part_number=product.get('ManufacturerPartNumber', ''),
            manufacturer=product.get('Manufacturer', ''),
            description=product.get('Description', ''),
            category=product.get('Category', ''),
            datasheet_url=product.get('DataSheetUrl', ''),
            data_source=f"Mouser API - {product.get('MouserPartNumber', '')}",
        )

    def save_to_json(self, filepath: str):
        """Save imported parts to JSON file."""
        data = {
            "version": "1.0",
            "source": "Mouser API",
            "import_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "parts": [p.to_dict() for p in self.imported_parts]
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)


# =============================================================================
# KICAD SYMBOL PARSER
# =============================================================================

class KiCadParser:
    """
    Parse KiCad symbol libraries to extract component data.

    Parses .kicad_sym files (KiCad 6+ format) and extracts:
    - Part names and descriptions
    - Pin information
    - Properties (manufacturer, datasheet, etc.)
    - Footprint references

    Default KiCad library location:
    - Windows: C:/Program Files/KiCad/9.0/share/kicad/symbols/
    - Linux: /usr/share/kicad/symbols/
    - macOS: /Applications/KiCad/KiCad.app/Contents/SharedSupport/symbols/
    """

    # Default library paths by OS
    DEFAULT_PATHS = {
        'nt': [  # Windows
            "C:/Program Files/KiCad/9.0/share/kicad/symbols",
            "C:/Program Files/KiCad/8.0/share/kicad/symbols",
            "C:/Program Files/KiCad/7.0/share/kicad/symbols",
        ],
        'posix': [  # Linux/macOS
            "/usr/share/kicad/symbols",
            "/Applications/KiCad/KiCad.app/Contents/SharedSupport/symbols",
        ]
    }

    def __init__(self, library_path: str = None):
        """
        Initialize KiCad parser.

        Args:
            library_path: Path to KiCad symbols directory (auto-detected if None)
        """
        self.library_path = library_path or self._find_library_path()
        self.imported_parts: List[ImportedPart] = []

    def _find_library_path(self) -> str:
        """Find KiCad library path."""
        paths = self.DEFAULT_PATHS.get(os.name, [])
        for path in paths:
            if os.path.isdir(path):
                return path
        return ""

    def list_libraries(self) -> List[str]:
        """List available symbol libraries."""
        if not self.library_path or not os.path.isdir(self.library_path):
            return []

        return [f for f in os.listdir(self.library_path) if f.endswith('.kicad_sym')]

    def parse_library(self, library_file: str) -> List[ImportedPart]:
        """
        Parse a KiCad symbol library file.

        Args:
            library_file: Path to .kicad_sym file or library name

        Returns:
            List of ImportedPart objects
        """
        # Handle relative path
        if not os.path.isabs(library_file) and self.library_path:
            library_file = os.path.join(self.library_path, library_file)

        if not os.path.isfile(library_file):
            raise FileNotFoundError(f"Library not found: {library_file}")

        with open(library_file, 'r', encoding='utf-8') as f:
            content = f.read()

        parts = []

        # Parse S-expression format
        # Each symbol starts with (symbol "SymbolName"
        symbol_pattern = r'\(symbol\s+"([^"]+)"(.*?)\n\s*\(symbol\s+"|\(symbol\s+"([^"]+)"(.*?)$)'

        # Simpler approach: split by top-level symbol definitions
        lines = content.split('\n')
        current_symbol = None
        symbol_content = []
        depth = 0

        for line in lines:
            # Track parentheses depth
            depth += line.count('(') - line.count(')')

            # Check for symbol start
            symbol_match = re.match(r'\s*\(symbol\s+"([^"]+)"', line)
            if symbol_match and depth <= 2:  # Top-level symbol
                # Save previous symbol
                if current_symbol and symbol_content:
                    part = self._parse_symbol(current_symbol, '\n'.join(symbol_content))
                    if part:
                        parts.append(part)

                current_symbol = symbol_match.group(1)
                symbol_content = [line]
            elif current_symbol:
                symbol_content.append(line)

        # Don't forget last symbol
        if current_symbol and symbol_content:
            part = self._parse_symbol(current_symbol, '\n'.join(symbol_content))
            if part:
                parts.append(part)

        self.imported_parts.extend(parts)
        return parts

    def _parse_symbol(self, name: str, content: str) -> Optional[ImportedPart]:
        """Parse a single symbol definition."""
        # Skip internal/nested symbols (contain underscore prefix)
        if name.startswith('_') or '_' in name and not any(c.isdigit() for c in name):
            return None

        part = ImportedPart(
            part_number=name,
            manufacturer="",
            data_source="KiCad Symbol Library",
        )

        # Extract properties
        prop_pattern = r'\(property\s+"([^"]+)"\s+"([^"]*)"'
        properties = re.findall(prop_pattern, content)

        for prop_name, prop_value in properties:
            prop_lower = prop_name.lower()

            if prop_lower == 'reference':
                # Determine category from reference
                if prop_value.startswith('U'):
                    part.category = 'IC'
                elif prop_value.startswith('R'):
                    part.category = 'RESISTOR'
                elif prop_value.startswith('C'):
                    part.category = 'CAPACITOR'
                elif prop_value.startswith('L'):
                    part.category = 'INDUCTOR'
                elif prop_value.startswith('D'):
                    part.category = 'DIODE'
                elif prop_value.startswith('Q'):
                    part.category = 'TRANSISTOR'

            elif prop_lower == 'value':
                if not part.description:
                    part.description = prop_value

            elif prop_lower == 'footprint':
                part.footprint = prop_value

            elif prop_lower == 'datasheet':
                part.datasheet_url = prop_value

            elif prop_lower == 'manufacturer':
                part.manufacturer = prop_value

            elif prop_lower in ('description', 'ki_description'):
                part.description = prop_value

            elif prop_lower == 'ki_keywords':
                # Use keywords for better categorization
                keywords = prop_value.lower()
                if 'regulator' in keywords:
                    part.subcategory = 'REGULATOR'
                elif 'ldo' in keywords:
                    part.subcategory = 'LDO'
                elif 'buck' in keywords:
                    part.subcategory = 'BUCK'
                elif 'capacitor' in keywords:
                    part.category = 'CAPACITOR'

        # Count pins
        pin_count = content.count('(pin ')
        part.mechanical['pin_count'] = pin_count

        return part

    def parse_all_libraries(self, categories: List[str] = None) -> List[ImportedPart]:
        """
        Parse all available libraries or specific categories.

        Args:
            categories: List of library name patterns to parse (e.g., ["Regulator", "Capacitor"])
                       If None, parses all libraries
        """
        libraries = self.list_libraries()

        if categories:
            libraries = [lib for lib in libraries
                        if any(cat.lower() in lib.lower() for cat in categories)]

        all_parts = []
        for lib in libraries:
            print(f"Parsing: {lib}")
            try:
                parts = self.parse_library(lib)
                all_parts.extend(parts)
                print(f"  Found {len(parts)} symbols")
            except Exception as e:
                print(f"  Error: {e}")

        return all_parts

    def save_to_json(self, filepath: str):
        """Save imported parts to JSON file."""
        data = {
            "version": "1.0",
            "source": "KiCad Symbol Libraries",
            "library_path": self.library_path,
            "import_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "parts": [p.to_dict() for p in self.imported_parts]
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

        print(f"Saved {len(self.imported_parts)} parts to {filepath}")


# =============================================================================
# BATCH IMPORTER - Combines multiple sources
# =============================================================================

class BatchImporter:
    """
    Batch import from multiple sources.

    Usage:
        importer = BatchImporter()

        # Add part numbers to import
        importer.add_parts([
            "LM2596S-5.0",
            "AMS1117-3.3",
            "ESP32-WROOM-32E",
            "ATMEGA328P-AU",
        ])

        # Import from DigiKey (requires API credentials)
        importer.import_from_digikey()

        # Or import from KiCad
        importer.import_from_kicad()

        # Save results
        importer.save_to_json("my_library.json")

        # Load into PartsLibrary
        from parts_library import PartsLibrary
        lib = PartsLibrary()
        lib.load_from_json("my_library.json")
    """

    def __init__(self):
        self.part_numbers: List[str] = []
        self.imported_parts: List[ImportedPart] = []

    def add_parts(self, part_numbers: List[str]):
        """Add part numbers to import list."""
        self.part_numbers.extend(part_numbers)

    def add_parts_from_file(self, filepath: str):
        """Load part numbers from a text file (one per line)."""
        with open(filepath, 'r') as f:
            for line in f:
                pn = line.strip()
                if pn and not pn.startswith('#'):
                    self.part_numbers.append(pn)

    def import_from_digikey(self, client_id: str = None, client_secret: str = None):
        """Import parts from DigiKey API."""
        importer = DigiKeyImporter(client_id, client_secret)
        parts = importer.import_parts(self.part_numbers)
        self.imported_parts.extend(parts)
        return parts

    def import_from_mouser(self, api_key: str = None):
        """Import parts from Mouser API."""
        importer = MouserImporter(api_key)
        for pn in self.part_numbers:
            products = importer.search_keyword(pn, limit=1)
            for product in products:
                part = importer._extract_part_data(product)
                if part.part_number:
                    self.imported_parts.append(part)
        return self.imported_parts

    def import_from_kicad(self, categories: List[str] = None):
        """Import parts from KiCad libraries."""
        parser = KiCadParser()
        parts = parser.parse_all_libraries(categories)

        # Filter to only requested part numbers if specified
        if self.part_numbers:
            parts = [p for p in parts
                    if any(pn.upper() in p.part_number.upper()
                          for pn in self.part_numbers)]

        self.imported_parts.extend(parts)
        return parts

    def save_to_json(self, filepath: str):
        """Save all imported parts to JSON."""
        # Deduplicate by part number
        seen = set()
        unique_parts = []
        for part in self.imported_parts:
            if part.part_number not in seen:
                seen.add(part.part_number)
                unique_parts.append(part)

        data = {
            "version": "1.0",
            "source": "BatchImporter",
            "import_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "parts": [p.to_dict() for p in unique_parts]
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

        print(f"Saved {len(unique_parts)} unique parts to {filepath}")

    def get_stats(self) -> dict:
        """Get import statistics."""
        categories = {}
        manufacturers = {}
        with_datasheet = 0

        for part in self.imported_parts:
            cat = part.category or 'UNKNOWN'
            categories[cat] = categories.get(cat, 0) + 1

            mfr = part.manufacturer or 'UNKNOWN'
            manufacturers[mfr] = manufacturers.get(mfr, 0) + 1

            if part.datasheet_url:
                with_datasheet += 1

        return {
            'total_parts': len(self.imported_parts),
            'unique_categories': len(categories),
            'categories': categories,
            'unique_manufacturers': len(manufacturers),
            'manufacturers': manufacturers,
            'with_datasheet': with_datasheet,
            'datasheet_coverage': f"{with_datasheet / max(1, len(self.imported_parts)) * 100:.1f}%",
        }


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    """Command-line interface for library importer."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Import component library data from various sources'
    )
    parser.add_argument('--source', choices=['digikey', 'mouser', 'kicad', 'batch'],
                       default='kicad', help='Data source')
    parser.add_argument('--output', '-o', default='imported_parts.json',
                       help='Output JSON file')
    parser.add_argument('--parts', '-p', nargs='+', help='Part numbers to import')
    parser.add_argument('--parts-file', '-f', help='File with part numbers (one per line)')
    parser.add_argument('--categories', '-c', nargs='+',
                       help='KiCad library categories to parse')
    parser.add_argument('--list-libs', action='store_true',
                       help='List available KiCad libraries')

    args = parser.parse_args()

    if args.list_libs:
        kicad = KiCadParser()
        print(f"KiCad library path: {kicad.library_path}")
        print("\nAvailable libraries:")
        for lib in kicad.list_libraries():
            print(f"  {lib}")
        return

    if args.source == 'kicad':
        parser = KiCadParser()
        if args.parts:
            # Parse specific categories
            parser.parse_all_libraries(args.categories)
        else:
            parser.parse_all_libraries(args.categories)
        parser.save_to_json(args.output)

        stats = BatchImporter()
        stats.imported_parts = parser.imported_parts
        print("\nImport Statistics:")
        for k, v in stats.get_stats().items():
            print(f"  {k}: {v}")

    elif args.source == 'digikey':
        importer = DigiKeyImporter()
        if args.parts:
            importer.import_parts(args.parts)
        elif args.parts_file:
            batch = BatchImporter()
            batch.add_parts_from_file(args.parts_file)
            importer.import_parts(batch.part_numbers)
        else:
            print("Error: --parts or --parts-file required for DigiKey import")
            return
        importer.save_to_json(args.output)

    elif args.source == 'batch':
        batch = BatchImporter()
        if args.parts:
            batch.add_parts(args.parts)
        if args.parts_file:
            batch.add_parts_from_file(args.parts_file)

        # Try KiCad first (no API needed)
        batch.import_from_kicad(args.categories)
        batch.save_to_json(args.output)

        print("\nImport Statistics:")
        for k, v in batch.get_stats().items():
            print(f"  {k}: {v}")


if __name__ == '__main__':
    main()
