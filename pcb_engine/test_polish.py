"""Test the Polish Piston on the LDO circuit"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# LDO voltage regulator circuit
complex_parts_db = {
    'parts': {
        'U1': {  # LDO Regulator
            'footprint': 'SOT-223',
            'pins': [
                {'number': '1', 'net': 'VIN', 'offset': (-2.3, 0), 'size': (1.0, 1.8)},
                {'number': '2', 'net': 'GND', 'offset': (0, 0), 'size': (1.0, 1.8)},
                {'number': '3', 'net': 'VOUT', 'offset': (2.3, 0), 'size': (1.0, 1.8)},
                {'number': '4', 'net': 'VOUT', 'offset': (0, 3.25), 'size': (3.0, 1.5)},
            ]
        },
        'C1': {  # Input cap
            'footprint': '0805',
            'pins': [
                {'number': '1', 'net': 'VIN', 'offset': (-0.95, 0), 'size': (0.9, 1.25)},
                {'number': '2', 'net': 'GND', 'offset': (0.95, 0), 'size': (0.9, 1.25)},
            ]
        },
        'C2': {  # Output cap
            'footprint': '0805',
            'pins': [
                {'number': '1', 'net': 'VOUT', 'offset': (-0.95, 0), 'size': (0.9, 1.25)},
                {'number': '2', 'net': 'GND', 'offset': (0.95, 0), 'size': (0.9, 1.25)},
            ]
        },
        'R1': {  # LED resistor
            'footprint': '0603',
            'pins': [
                {'number': '1', 'net': 'VOUT', 'offset': (-0.75, 0), 'size': (0.6, 0.9)},
                {'number': '2', 'net': 'LED_A', 'offset': (0.75, 0), 'size': (0.6, 0.9)},
            ]
        },
        'D1': {  # LED
            'footprint': '0805',
            'pins': [
                {'number': '1', 'net': 'LED_A', 'offset': (-0.95, 0), 'size': (0.9, 1.25)},
                {'number': '2', 'net': 'GND', 'offset': (0.95, 0), 'size': (0.9, 1.25)},
            ]
        },
    },
    'nets': {
        'VIN': {'pins': [('U1', '1'), ('C1', '1')]},
        'VOUT': {'pins': [('U1', '3'), ('U1', '4'), ('C2', '1'), ('R1', '1')]},
        'GND': {'pins': [('U1', '2'), ('C1', '2'), ('C2', '2'), ('D1', '2')]},
        'LED_A': {'pins': [('R1', '2'), ('D1', '1')]},
    }
}

print("=" * 70)
print("POLISH PISTON TEST: LDO + LED Circuit")
print("=" * 70)

# Step 1: Placement
print("\n1. PLACEMENT")
from pcb_engine.placement_engine import PlacementEngine, PlacementConfig

pe = PlacementEngine(PlacementConfig(
    board_width=40.0,
    board_height=30.0,
    algorithm='hybrid'
))
placement_result = pe.place(complex_parts_db, {})
print(f"   Placed {len(placement_result.positions)} components")

# Step 2: Routing
print("\n2. ROUTING")
from pcb_engine.routing_piston import RoutingPiston, RoutingConfig

routeable_nets = ['VIN', 'VOUT', 'GND', 'LED_A']
rp = RoutingPiston(RoutingConfig(
    board_width=40.0,
    board_height=30.0,
    algorithm='hybrid',
    trace_width=0.25,
    clearance=0.2,
    via_cost=5  # Low cost for routability
))

routing_result = rp.route(
    parts_db=complex_parts_db,
    escapes={},
    placement=placement_result.positions,
    net_order=routeable_nets
)

print(f"   Routed: {routing_result.routed_count}/{routing_result.total_count}")
print(f"   Via count: {routing_result.via_count}")
print(f"   Total length: {routing_result.total_wirelength:.1f}mm")

# Count segments
total_segments = sum(len(r.segments) for r in routing_result.routes.values())
print(f"   Total segments: {total_segments}")

# Step 3: POLISH
print("\n3. POLISH (Post-routing optimization)")
from pcb_engine.polish_piston import PolishPiston, PolishConfig, PolishLevel

pp = PolishPiston(PolishConfig(
    level=PolishLevel.STANDARD,
    reduce_vias=False,  # Disabled - creates crossing traces without collision detection
    simplify_traces=True,  # Safe - just merges collinear segments
    shrink_board=False,  # Disabled - would need component repositioning
    use_arcs=False,  # Disabled for now
    verbose=True
))

polish_result = pp.polish(
    routes=routing_result.routes,
    parts_db=complex_parts_db,
    placement=placement_result.positions,
    board_width=40.0,
    board_height=30.0
)

print(f"\n   POLISH RESULTS:")
print(f"   - Vias: {polish_result.original_via_count} -> {polish_result.new_via_count} ({polish_result.vias_removed} removed)")
print(f"   - Segments: {polish_result.original_segment_count} -> {polish_result.new_segment_count} ({polish_result.segments_merged} merged)")
print(f"   - Board: {polish_result.original_board[0]:.1f}x{polish_result.original_board[1]:.1f} -> {polish_result.new_board[0]:.1f}x{polish_result.new_board[1]:.1f}")
print(f"   - Board reduction: {polish_result.board_reduction_percent:.1f}%")

# Step 4: Generate output with polished routes
print("\n4. OUTPUT GENERATION (with polished routes)")
from pcb_engine.output_piston import OutputPiston, OutputConfig

op = OutputPiston(OutputConfig(
    board_width=polish_result.new_board[0],
    board_height=polish_result.new_board[1],
    board_name='ldo_led_polished'
))
output_result = op.generate(
    parts_db=complex_parts_db,
    placement=placement_result.positions,
    routes=polish_result.routes
)

# Get PCB file
output_file = None
if output_result.success and output_result.files_generated:
    for f in output_result.files_generated:
        if f.endswith('.kicad_pcb'):
            output_file = f
            break

if output_file:
    print(f"   Generated: {output_file}")
else:
    print(f"   ERROR: No output generated")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Original: {routing_result.via_count} vias, {total_segments} segments")
print(f"Polished: {polish_result.new_via_count} vias, {polish_result.new_segment_count} segments")
print(f"Board shrunk by {polish_result.board_reduction_percent:.1f}%")
print("=" * 70)
