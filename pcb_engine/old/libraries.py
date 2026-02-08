"""
PCB Engine - Libraries Module
=============================

This module provides access to all external libraries that the engine CAN use.
It handles optional dependencies gracefully and provides fallbacks.

LIBRARY CATEGORIES:
==================
1. CORE (Built-in Python) - Always available
2. MATH/GEOMETRY - For advanced calculations
3. GRAPH/OPTIMIZATION - For placement and routing algorithms
4. VISUALIZATION - For debug output and previews
5. KICAD - For direct integration with KiCad

Usage:
    from pcb_engine.libraries import libs

    # Check if a library is available
    if libs.has_numpy:
        import numpy as np

    # Use recommended imports
    plt = libs.get_matplotlib()
    nx = libs.get_networkx()
"""

import sys
from typing import Optional, Any, Dict, List
from dataclasses import dataclass


# =============================================================================
# LIBRARY STATUS TRACKING
# =============================================================================

@dataclass
class LibraryInfo:
    """Information about a library"""
    name: str
    package: str
    available: bool
    version: str
    purpose: str
    installation: str
    fallback: str


class LibraryManager:
    """
    Manages all external libraries for the PCB Engine.

    Provides:
    - Availability checking
    - Lazy loading
    - Fallback implementations
    - Installation guidance
    """

    def __init__(self):
        self._cache = {}
        self._info = {}
        self._initialize_libraries()

    def _initialize_libraries(self):
        """Check availability of all libraries"""

        # =====================================================================
        # NUMPY - Numerical Computing
        # =====================================================================
        try:
            import numpy
            self._cache['numpy'] = numpy
            self._info['numpy'] = LibraryInfo(
                name='NumPy',
                package='numpy',
                available=True,
                version=numpy.__version__,
                purpose='Optimized numerical operations, array math, matrix operations',
                installation='pip install numpy',
                fallback='Python built-in math module (slower but functional)'
            )
        except ImportError:
            self._info['numpy'] = LibraryInfo(
                name='NumPy',
                package='numpy',
                available=False,
                version='',
                purpose='Optimized numerical operations, array math, matrix operations',
                installation='pip install numpy',
                fallback='Python built-in math module (slower but functional)'
            )

        # =====================================================================
        # MATPLOTLIB - Visualization
        # =====================================================================
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            self._cache['matplotlib'] = matplotlib
            self._cache['pyplot'] = plt
            self._info['matplotlib'] = LibraryInfo(
                name='Matplotlib',
                package='matplotlib',
                available=True,
                version=matplotlib.__version__,
                purpose='Debug visualization, placement preview, routing diagrams',
                installation='pip install matplotlib',
                fallback='Text-based output only (no graphics)'
            )
        except ImportError:
            self._info['matplotlib'] = LibraryInfo(
                name='Matplotlib',
                package='matplotlib',
                available=False,
                version='',
                purpose='Debug visualization, placement preview, routing diagrams',
                installation='pip install matplotlib',
                fallback='Text-based output only (no graphics)'
            )

        # =====================================================================
        # NETWORKX - Graph Algorithms
        # =====================================================================
        try:
            import networkx as nx
            self._cache['networkx'] = nx
            self._info['networkx'] = LibraryInfo(
                name='NetworkX',
                package='networkx',
                available=True,
                version=nx.__version__,
                purpose='Graph algorithms, connectivity analysis, hub detection, clustering',
                installation='pip install networkx',
                fallback='Built-in graph implementation (basic functionality)'
            )
        except ImportError:
            self._info['networkx'] = LibraryInfo(
                name='NetworkX',
                package='networkx',
                available=False,
                version='',
                purpose='Graph algorithms, connectivity analysis, hub detection, clustering',
                installation='pip install networkx',
                fallback='Built-in graph implementation (basic functionality)'
            )

        # =====================================================================
        # SCIPY - Scientific Computing / Optimization
        # =====================================================================
        try:
            import scipy
            from scipy import optimize
            from scipy import spatial
            self._cache['scipy'] = scipy
            self._cache['scipy.optimize'] = optimize
            self._cache['scipy.spatial'] = spatial
            self._info['scipy'] = LibraryInfo(
                name='SciPy',
                package='scipy',
                available=True,
                version=scipy.__version__,
                purpose='Simulated annealing, spatial trees, optimization algorithms',
                installation='pip install scipy',
                fallback='Built-in optimization (basic simulated annealing)'
            )
        except ImportError:
            self._info['scipy'] = LibraryInfo(
                name='SciPy',
                package='scipy',
                available=False,
                version='',
                purpose='Simulated annealing, spatial trees, optimization algorithms',
                installation='pip install scipy',
                fallback='Built-in optimization (basic simulated annealing)'
            )

        # =====================================================================
        # SHAPELY - Geometric Operations
        # =====================================================================
        try:
            import shapely
            from shapely.geometry import Point, Polygon, LineString
            from shapely.ops import unary_union
            self._cache['shapely'] = shapely
            self._info['shapely'] = LibraryInfo(
                name='Shapely',
                package='shapely',
                available=True,
                version=shapely.__version__,
                purpose='Polygon operations, zone handling, courtyard collision detection',
                installation='pip install shapely',
                fallback='Basic rectangle collision only'
            )
        except ImportError:
            self._info['shapely'] = LibraryInfo(
                name='Shapely',
                package='shapely',
                available=False,
                version='',
                purpose='Polygon operations, zone handling, courtyard collision detection',
                installation='pip install shapely',
                fallback='Basic rectangle collision only'
            )

        # =====================================================================
        # RTREE - Spatial Indexing
        # =====================================================================
        try:
            from rtree import index
            self._cache['rtree'] = index
            self._info['rtree'] = LibraryInfo(
                name='Rtree',
                package='rtree',
                available=True,
                version='unknown',  # rtree doesn't expose version easily
                purpose='Fast spatial queries for collision detection, nearest neighbor',
                installation='pip install rtree',
                fallback='Brute-force distance checking (O(n²) vs O(log n))'
            )
        except ImportError:
            self._info['rtree'] = LibraryInfo(
                name='Rtree',
                package='rtree',
                available=False,
                version='',
                purpose='Fast spatial queries for collision detection, nearest neighbor',
                installation='pip install rtree',
                fallback='Brute-force distance checking (O(n²) vs O(log n))'
            )

        # =====================================================================
        # KICAD (pcbnew) - Direct KiCad Integration
        # =====================================================================
        try:
            import pcbnew
            self._cache['pcbnew'] = pcbnew
            self._info['pcbnew'] = LibraryInfo(
                name='KiCad pcbnew',
                package='pcbnew',
                available=True,
                version=pcbnew.Version(),
                purpose='Direct KiCad board manipulation without scripts',
                installation='Installed with KiCad (not pip installable)',
                fallback='Generate Python scripts for KiCad instead'
            )
        except ImportError:
            self._info['pcbnew'] = LibraryInfo(
                name='KiCad pcbnew',
                package='pcbnew',
                available=False,
                version='',
                purpose='Direct KiCad board manipulation without scripts',
                installation='Installed with KiCad (not pip installable)',
                fallback='Generate Python scripts for KiCad instead'
            )

    # =========================================================================
    # AVAILABILITY PROPERTIES
    # =========================================================================

    @property
    def has_numpy(self) -> bool:
        return self._info.get('numpy', LibraryInfo('', '', False, '', '', '', '')).available

    @property
    def has_matplotlib(self) -> bool:
        return self._info.get('matplotlib', LibraryInfo('', '', False, '', '', '', '')).available

    @property
    def has_networkx(self) -> bool:
        return self._info.get('networkx', LibraryInfo('', '', False, '', '', '', '')).available

    @property
    def has_scipy(self) -> bool:
        return self._info.get('scipy', LibraryInfo('', '', False, '', '', '', '')).available

    @property
    def has_shapely(self) -> bool:
        return self._info.get('shapely', LibraryInfo('', '', False, '', '', '', '')).available

    @property
    def has_rtree(self) -> bool:
        return self._info.get('rtree', LibraryInfo('', '', False, '', '', '', '')).available

    @property
    def has_kicad(self) -> bool:
        return self._info.get('pcbnew', LibraryInfo('', '', False, '', '', '', '')).available

    # =========================================================================
    # LIBRARY GETTERS
    # =========================================================================

    def get_numpy(self):
        """Get numpy module or None"""
        return self._cache.get('numpy')

    def get_matplotlib(self):
        """Get matplotlib.pyplot or None"""
        return self._cache.get('pyplot')

    def get_networkx(self):
        """Get networkx module or None"""
        return self._cache.get('networkx')

    def get_scipy(self):
        """Get scipy module or None"""
        return self._cache.get('scipy')

    def get_scipy_optimize(self):
        """Get scipy.optimize module or None"""
        return self._cache.get('scipy.optimize')

    def get_scipy_spatial(self):
        """Get scipy.spatial module or None"""
        return self._cache.get('scipy.spatial')

    def get_shapely(self):
        """Get shapely module or None"""
        return self._cache.get('shapely')

    def get_rtree(self):
        """Get rtree.index module or None"""
        return self._cache.get('rtree')

    def get_kicad(self):
        """Get pcbnew module or None"""
        return self._cache.get('pcbnew')

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def get_status(self) -> str:
        """Get a formatted status report of all libraries"""
        lines = [
            "=" * 70,
            "PCB ENGINE LIBRARY STATUS",
            "=" * 70,
            "",
            f"{'Library':<20} {'Status':<12} {'Version':<15} {'Purpose'}",
            "-" * 70,
        ]

        for name, info in self._info.items():
            status = "✓ Available" if info.available else "✗ Missing"
            version = info.version if info.version else "N/A"
            lines.append(f"{info.name:<20} {status:<12} {version:<15} {info.purpose[:30]}...")

        lines.extend([
            "",
            "=" * 70,
            "INSTALLATION COMMANDS:",
            "=" * 70,
            "",
            "# Minimal (basic functionality):",
            "pip install numpy matplotlib",
            "",
            "# Recommended (full optimization):",
            "pip install numpy matplotlib networkx scipy shapely rtree",
            "",
            "# KiCad integration:",
            "# Install KiCad 6+ from https://www.kicad.org/",
            "# pcbnew is included with KiCad installation",
            "",
            "=" * 70,
        ])

        return "\n".join(lines)

    def get_missing(self) -> List[str]:
        """Get list of missing library installation commands"""
        missing = []
        for name, info in self._info.items():
            if not info.available and info.installation.startswith('pip'):
                missing.append(info.installation)
        return missing

    def install_command(self) -> str:
        """Get pip command to install all missing libraries"""
        missing_packages = []
        for name, info in self._info.items():
            if not info.available and info.installation.startswith('pip'):
                # Extract package name from "pip install <package>"
                pkg = info.installation.replace('pip install ', '')
                missing_packages.append(pkg)

        if missing_packages:
            return f"pip install {' '.join(missing_packages)}"
        return "# All libraries are installed!"

    def check_for_algorithm(self, algorithm: str) -> Dict[str, Any]:
        """Check library requirements for a specific algorithm"""
        requirements = {
            'placement': {
                'required': [],
                'recommended': ['numpy', 'networkx', 'scipy'],
                'description': 'Force-directed placement with simulated annealing'
            },
            'routing': {
                'required': [],
                'recommended': ['numpy', 'rtree'],
                'description': 'A* pathfinding with spatial collision detection'
            },
            'validation': {
                'required': [],
                'recommended': ['shapely'],
                'description': 'DRC with polygon intersection detection'
            },
            'visualization': {
                'required': ['matplotlib'],
                'recommended': ['numpy'],
                'description': 'Debug visualization and design preview'
            },
            'training': {
                'required': [],
                'recommended': ['numpy', 'networkx'],
                'description': 'Pattern analysis from existing designs'
            },
        }

        if algorithm not in requirements:
            return {'error': f'Unknown algorithm: {algorithm}'}

        req = requirements[algorithm]
        result = {
            'algorithm': algorithm,
            'description': req['description'],
            'status': 'ready',
            'missing_required': [],
            'missing_recommended': [],
        }

        for lib in req['required']:
            if not self._info.get(lib, LibraryInfo('', '', False, '', '', '', '')).available:
                result['missing_required'].append(lib)
                result['status'] = 'blocked'

        for lib in req['recommended']:
            if not self._info.get(lib, LibraryInfo('', '', False, '', '', '', '')).available:
                result['missing_recommended'].append(lib)
                if result['status'] == 'ready':
                    result['status'] = 'degraded'

        return result


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

# Global library manager instance
libs = LibraryManager()


# =============================================================================
# FALLBACK IMPLEMENTATIONS
# =============================================================================

class FallbackNumpy:
    """Basic fallback for numpy operations using pure Python"""

    @staticmethod
    def array(data):
        """Create a list (fallback for np.array)"""
        if isinstance(data, (list, tuple)):
            return list(data)
        return [data]

    @staticmethod
    def zeros(shape):
        """Create zero-filled list"""
        if isinstance(shape, int):
            return [0.0] * shape
        elif len(shape) == 1:
            return [0.0] * shape[0]
        elif len(shape) == 2:
            return [[0.0] * shape[1] for _ in range(shape[0])]
        raise ValueError("Only 1D and 2D arrays supported in fallback")

    @staticmethod
    def sqrt(x):
        """Square root"""
        import math
        if isinstance(x, (list, tuple)):
            return [math.sqrt(v) for v in x]
        return math.sqrt(x)

    @staticmethod
    def mean(data):
        """Calculate mean"""
        if not data:
            return 0
        return sum(data) / len(data)

    @staticmethod
    def min(data):
        """Get minimum"""
        return min(data)

    @staticmethod
    def max(data):
        """Get maximum"""
        return max(data)


class FallbackGraph:
    """Basic fallback for networkx graph operations"""

    def __init__(self):
        self.nodes = {}
        self.edges = []

    def add_node(self, node, **attrs):
        self.nodes[node] = attrs

    def add_edge(self, u, v, **attrs):
        self.edges.append((u, v, attrs))

    def neighbors(self, node):
        result = []
        for u, v, _ in self.edges:
            if u == node:
                result.append(v)
            elif v == node:
                result.append(u)
        return result

    def degree(self, node):
        count = 0
        for u, v, _ in self.edges:
            if u == node or v == node:
                count += 1
        return count


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_numpy_or_fallback():
    """Get numpy or fallback implementation"""
    np = libs.get_numpy()
    return np if np else FallbackNumpy()


def get_graph_or_fallback():
    """Get networkx Graph or fallback implementation"""
    nx = libs.get_networkx()
    if nx:
        return nx.Graph
    return FallbackGraph


def print_library_status():
    """Print library status to console"""
    print(libs.get_status())


def ensure_libraries(*required):
    """Ensure required libraries are available, raise if not"""
    missing = []
    for lib in required:
        if not getattr(libs, f'has_{lib}', False):
            missing.append(lib)

    if missing:
        raise ImportError(
            f"Required libraries missing: {', '.join(missing)}\n"
            f"Install with: {libs.install_command()}"
        )


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == '__main__':
    print_library_status()
    print("\nInstall missing libraries:")
    print(libs.install_command())

    print("\nAlgorithm checks:")
    for algo in ['placement', 'routing', 'validation', 'visualization', 'training']:
        result = libs.check_for_algorithm(algo)
        status_icon = {'ready': '✓', 'degraded': '⚠', 'blocked': '✗'}[result['status']]
        print(f"  {status_icon} {algo}: {result['status']}")
        if result['missing_recommended']:
            print(f"      Recommended: {', '.join(result['missing_recommended'])}")
