"""
PCB Engine Path Configuration - SINGLE SOURCE OF TRUTH
=======================================================

All paths used by the PCB Engine are defined here.
Import from this module instead of hardcoding paths.

Usage:
    from pcb_engine.paths import OUTPUT_BASE, get_output_dir

    # Get the base output folder
    output = OUTPUT_BASE  # D:\Anas\tmp\output

    # Get a specific board output folder
    board_dir = get_output_dir('my_board')  # D:\Anas\tmp\output\my_board_20260208_1234_PASS
"""

import os
from datetime import datetime
from pathlib import Path

# ============================================================================
# CORE PATHS - These are the ONLY paths that should be modified
# ============================================================================

# Base output directory for all PCB designs (production)
OUTPUT_BASE = Path(r'D:\Anas\tmp\output')

# Project root (where pcb_engine package lives)
PROJECT_ROOT = Path(__file__).parent.parent

# PCB Engine package directory
ENGINE_ROOT = Path(__file__).parent

# ============================================================================
# DERIVED PATHS - Computed from core paths
# ============================================================================

# Old designs archive folder
OLD_ARCHIVE = OUTPUT_BASE / 'old'

# Learning database
LEARNING_DB = ENGINE_ROOT / 'drc_learning_db.json'

# BBL history
BBL_HISTORY = ENGINE_ROOT / 'bbl_history.json'

# ============================================================================
# PATH UTILITIES
# ============================================================================

def ensure_output_base():
    """Ensure the output base directory exists."""
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    return OUTPUT_BASE


def get_output_dir(board_name: str, status: str = 'PENDING') -> Path:
    """
    Get the output directory for a specific board.

    Args:
        board_name: Name of the PCB board
        status: PASS, FAIL, or PENDING

    Returns:
        Path to the board's output directory
    """
    ensure_output_base()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    folder_name = f"{board_name}_{timestamp}_{status}"
    return OUTPUT_BASE / folder_name


def get_old_archive() -> Path:
    """Get the old archive directory, creating it if needed."""
    OLD_ARCHIVE.mkdir(parents=True, exist_ok=True)
    return OLD_ARCHIVE


def is_valid_output_path(path: str | Path) -> bool:
    """Check if a path is within the valid output structure."""
    path = Path(path).resolve()
    return str(path).startswith(str(OUTPUT_BASE.resolve()))


# ============================================================================
# LEGACY COMPATIBILITY
# ============================================================================

# For backwards compatibility with existing code
DEFAULT_OUTPUT_BASE = str(OUTPUT_BASE)


def migrate_relative_path(relative_path: str) -> Path:
    """
    Convert a relative path like './output' to the proper absolute path.

    This is used during migration to fix old code that used relative paths.
    """
    if relative_path.startswith('./') or relative_path.startswith('.\\'):
        # Old relative path - convert to proper output base
        return OUTPUT_BASE
    return Path(relative_path)
