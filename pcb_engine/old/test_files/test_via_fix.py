"""Test via connectivity fix"""
import sys
sys.path.insert(0, '..')

from pcb_engine import PCBEngine, EngineConfig

# LDO voltage regulator circuit - tests via layer transitions
parts_db = {
    'parts': {
        'U1': {
            'footprint': 'SOT-223',
            'pins': [
                {'number': '1', 'net': 'VIN', 'offset': (-2.3, 0), 'size': (1.0, 1.8)},
                {'number': '2', 'net': 'GND', 'offset': (0, 0), 'size': (1.0, 1.8)},
                {'number': '3', 'net': 'VOUT', 'offset': (2.3, 0), 'size': (1.0, 1.8)},
                {'number': '4', 'net': 'VOUT', 'offset': (0, 3.25), 'size': (3.0, 1.5)},
            ]
        },
        'C1': {
            'footprint': '0805',
            'pins': [
                {'number': '1', 'net': 'VIN', 'offset': (-0.95, 0), 'size': (0.9, 1.25)},
                {'number': '2', 'net': 'GND', 'offset': (0.95, 0), 'size': (0.9, 1.25)},
            ]
        },
        'C2': {
            'footprint': '0805',
            'pins': [
                {'number': '1', 'net': 'VOUT', 'offset': (-0.95, 0), 'size': (0.9, 1.25)},
                {'number': '2', 'net': 'GND', 'offset': (0.95, 0), 'size': (0.9, 1.25)},
            ]
        },
    },
    'nets': {
        'VIN': {'pins': [('U1', '1'), ('C1', '1')]},
        'VOUT': {'pins': [('U1', '3'), ('U1', '4'), ('C2', '1')]},
        'GND': {'pins': [('U1', '2'), ('C1', '2'), ('C2', '2')]},
    }
}

def test_ldo_circuit():
    """Test LDO circuit routing with layer transitions"""
    print("=" * 60)
    print("TESTING VIA CONNECTIVITY FIX")
    print("=" * 60)

    config = EngineConfig(
        board_name='ldo_test',
        board_width=30.0,
        board_height=25.0
    )
    engine = PCBEngine(config)

    print("\nRunning BBL for LDO circuit...")
    result = engine.run_bbl(parts_db)

    print(f"\nResult: success={result.success}")
    print(f"Routing: {result.routed_count}/{result.total_nets}")
    print(f"Internal DRC: {'PASS' if result.drc_passed else 'FAIL'}")
    print(f"KiCad DRC: {'PASS' if result.kicad_drc_passed else 'FAIL'}")

    if result.output_file:
        print(f"Output: {result.output_file}")

    # Check for specific error types
    if hasattr(result, 'kicad_drc_report') and result.kicad_drc_report:
        report = result.kicad_drc_report
        unconnected = len(report.get('unconnected_items', []))
        violations = report.get('violations', [])

        dangling_vias = sum(1 for v in violations if v.get('type') == 'via_dangling')
        dangling_tracks = sum(1 for v in violations if v.get('type') == 'track_dangling')
        hole_to_hole = sum(1 for v in violations if v.get('type') == 'hole_to_hole')

        print(f"\nDRC Breakdown:")
        print(f"  - Unconnected items: {unconnected}")
        print(f"  - Dangling vias: {dangling_vias}")
        print(f"  - Dangling tracks: {dangling_tracks}")
        print(f"  - Hole-to-hole violations: {hole_to_hole}")

    print("=" * 60)
    return result.kicad_drc_passed

if __name__ == '__main__':
    success = test_ldo_circuit()
    sys.exit(0 if success else 1)
