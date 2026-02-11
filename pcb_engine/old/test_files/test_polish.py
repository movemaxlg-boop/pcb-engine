"""Test the Polish Piston on the LDO circuit"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# LDO voltage regulator circuit with LED indicator
# COURTYARDS: Pre-calculated by Parts Piston using IPC-7351B standard
# Format: {'width': W, 'height': H} centered on component origin
# IPC-7351B Level B (Nominal) = 0.25mm courtyard excess on all sides
complex_parts_db = {
    'parts': {
        'U1': {  # LDO Regulator - SOT-223
            'footprint': 'SOT-223',
            'pins': [
                {'number': '1', 'net': 'VIN', 'offset': (-2.3, 0), 'size': (1.0, 1.8)},
                {'number': '2', 'net': 'GND', 'offset': (0, 0), 'size': (1.0, 1.8)},
                {'number': '3', 'net': 'VOUT', 'offset': (2.3, 0), 'size': (1.0, 1.8)},
                {'number': '4', 'net': 'VOUT', 'offset': (0, 3.25), 'size': (3.0, 1.5)},
            ],
            # IPC-7351B courtyard: pad_bbox + 0.25mm margin
            'courtyard': {'width': 6.10, 'height': 5.40}
        },
        'C1': {  # Input cap - 0805
            'footprint': '0805',
            'pins': [
                {'number': '1', 'net': 'VIN', 'offset': (-0.95, 0), 'size': (0.9, 1.25)},
                {'number': '2', 'net': 'GND', 'offset': (0.95, 0), 'size': (0.9, 1.25)},
            ],
            'courtyard': {'width': 3.30, 'height': 1.75}
        },
        'C2': {  # Output cap - 0805
            'footprint': '0805',
            'pins': [
                {'number': '1', 'net': 'VOUT', 'offset': (-0.95, 0), 'size': (0.9, 1.25)},
                {'number': '2', 'net': 'GND', 'offset': (0.95, 0), 'size': (0.9, 1.25)},
            ],
            'courtyard': {'width': 3.30, 'height': 1.75}
        },
        'R1': {  # LED resistor - 0603
            'footprint': '0603',
            'pins': [
                # Use correct 0603 pad offset from common_types.py FootprintDefinition
                {'number': '1', 'net': 'VOUT', 'offset': (-0.775, 0), 'size': (0.75, 0.9)},
                {'number': '2', 'net': 'LED_A', 'offset': (0.775, 0), 'size': (0.75, 0.9)},
            ],
            'courtyard': {'width': 2.60, 'height': 1.40}
        },
        'D1': {  # LED - 0805
            'footprint': '0805',
            'pins': [
                {'number': '1', 'net': 'LED_A', 'offset': (-0.95, 0), 'size': (0.9, 1.25)},
                {'number': '2', 'net': 'GND', 'offset': (0.95, 0), 'size': (0.9, 1.25)},
            ],
            'courtyard': {'width': 3.30, 'height': 1.75}
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

# Step 1: Placement (with stronger net attraction for aligned traces)
print("\n1. PLACEMENT")
from pcb_engine.placement_engine import PlacementEngine, PlacementConfig

pe = PlacementEngine(PlacementConfig(
    board_width=40.0,
    board_height=30.0,
    algorithm='hybrid',
    min_spacing=1.0,  # Increase spacing for routing channels
))
placement_result = pe.place(complex_parts_db, {})
print(f"   Placed {len(placement_result.positions)} components")
print(f"   Overlap area: {placement_result.overlap_area:.2f} mm²")
for ref, pos in placement_result.positions.items():
    print(f"   {ref}: ({pos[0]:.2f}, {pos[1]:.2f})")

# Step 2: GND POUR (eliminates GND traces - they connect via ground plane)
print("\n2. GND POUR")
from pcb_engine.pour_piston import PourPiston, PourConfig

pour_piston = PourPiston(PourConfig(
    net="GND",
    layer="B.Cu",
    clearance=0.3,
    add_stitching_vias=True,
))
pour_result = pour_piston.generate(
    parts_db=complex_parts_db,
    placement=placement_result.positions,
    board_width=40.0,
    board_height=30.0
)
print(f"   GND pour: {pour_result.success}")
print(f"   GND pins connected via pour (no traces needed)")

# Step 3: Routing (ALL nets including GND)
print("\n3. ROUTING (all nets)")
from pcb_engine.routing_piston import RoutingPiston, RoutingConfig

# Route ALL nets - GND pour provides solid ground plane but SMD pads
# still need routes/vias to connect to the pour
routeable_nets = ['VIN', 'VOUT', 'GND', 'LED_A']
rp = RoutingPiston(RoutingConfig(
    board_width=40.0,
    board_height=30.0,
    algorithm='hybrid',  # Use Hybrid (A* + Steiner + Ripup) for better completion
    trace_width=0.20,    # Thinner traces for tighter spacing
    clearance=0.20,      # 0.2mm clearance matches KiCad's default
    via_cost=3,          # Lower via cost to allow more layer changes
    allow_45_degree=True,  # Enable diagonal routing
    grid_size=0.05,      # Finer grid reduces pad-to-grid snap gaps
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

# Step 4: POLISH
print("\n4. POLISH (Post-routing optimization)")
from pcb_engine.polish_piston import PolishPiston, PolishConfig, PolishLevel

pp = PolishPiston(PolishConfig(
    level=PolishLevel.STANDARD,
    reduce_vias=False,  # Disabled - creates crossing traces without collision detection
    simplify_traces=True,  # Safe - just merges collinear segments
    eliminate_staircases=False,  # DISABLED - breaks via connectivity, needs more work
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

# Step 5: Generate output with polished routes + GND pour
print("\n5. OUTPUT GENERATION (with polished routes + GND pour)")
from pcb_engine.output_piston import OutputPiston, OutputConfig

op = OutputPiston(OutputConfig(
    board_width=polish_result.new_board[0],
    board_height=polish_result.new_board[1],
    board_name='ldo_led_polished',
    generate_gnd_pour=True,  # OutputPiston generates GND pour automatically
))
output_result = op.generate(
    parts_db=complex_parts_db,
    placement=placement_result.positions,
    routes=polish_result.routes,
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

# Step 6: SAFETY CHECK (Manufacturing Quality Gate)
print("\n6. SAFETY CHECK (Manufacturing Quality Gate)")
from pcb_engine.safety_piston import SafetyPiston, SafetyConfig

# Collect vias from all routes
all_vias = []
for route in polish_result.routes.values():
    all_vias.extend(route.vias)

safety_piston = SafetyPiston(SafetyConfig())
safety_result = safety_piston.check(
    parts_db=complex_parts_db,
    placement=placement_result.positions,
    routes=polish_result.routes,
    board_width=40.0,
    board_height=30.0,
    vias=all_vias
)
safety_piston.print_report(safety_result)

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Original: {routing_result.via_count} vias, {total_segments} segments")
print(f"Polished: {polish_result.new_via_count} vias, {polish_result.new_segment_count} segments")
print(f"Board shrunk by {polish_result.board_reduction_percent:.1f}%")
print(f"Safety Score: {safety_result.overall_score:.0f}/100")
print("=" * 70)
