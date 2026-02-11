"""Debug routing piston issue"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Simple 2-resistor circuit
parts_db = {
    'parts': {
        'R1': {
            'footprint': '0805',
            'pins': [
                {'number': '1', 'net': 'VCC', 'offset': (-0.95, 0), 'size': (0.9, 1.25)},
                {'number': '2', 'net': 'SIG', 'offset': (0.95, 0), 'size': (0.9, 1.25)},
            ]
        },
        'R2': {
            'footprint': '0805',
            'pins': [
                {'number': '1', 'net': 'SIG', 'offset': (-0.95, 0), 'size': (0.9, 1.25)},
                {'number': '2', 'net': 'GND', 'offset': (0.95, 0), 'size': (0.9, 1.25)},
            ]
        },
    },
    'nets': {
        'VCC': {'pins': [('R1', '1')]},
        'SIG': {'pins': [('R1', '2'), ('R2', '1')]},  # This net has 2 pins - should be routed
        'GND': {'pins': [('R2', '2')]},
    }
}

print("=" * 60)
print("DEBUG: Testing RoutingPiston directly")
print("=" * 60)

# First, do placement
print("\n1. Running placement...")
from pcb_engine.placement_engine import PlacementEngine, PlacementConfig as PlacementEngineConfig

placement_config = PlacementEngineConfig(
    board_width=30.0,
    board_height=25.0,
    algorithm='hybrid'
)
placement_engine = PlacementEngine(placement_config)
placement_result = placement_engine.place(parts_db, {})

print(f"   Placed {len(placement_result.positions)} components:")
for ref, pos in placement_result.positions.items():
    print(f"     {ref}: {pos}")

# Now test routing
print("\n2. Setting up routing piston...")
from pcb_engine.routing_piston import RoutingPiston, RoutingConfig

routing_config = RoutingConfig(
    board_width=30.0,
    board_height=25.0,
    algorithm='lee',
    trace_width=0.25,
    clearance=0.2
)
routing_piston = RoutingPiston(routing_config)

print("\n3. Calling route()...")

# Determine which nets to route (nets with 2+ pins)
net_order = []
for net_name, net_info in parts_db['nets'].items():
    pins = net_info.get('pins', [])
    if len(pins) >= 2:
        net_order.append(net_name)

print(f"   Nets to route: {net_order}")
print(f"   SIG net pins: {parts_db['nets']['SIG']['pins']}")

# Manually test _get_escape_endpoints before routing
routing_piston._placement = placement_result.positions
routing_piston._parts_db = parts_db

sig_pins = parts_db['nets']['SIG']['pins']
endpoints = routing_piston._get_escape_endpoints(sig_pins, {})
print(f"   Endpoints found: {endpoints}")

# Debug the grid before routing
routing_piston._initialize_grids()
routing_piston._register_components(placement_result.positions, parts_db)

# Check if endpoints are blocked
start = endpoints[0]
end = endpoints[1]
start_col = int(start[0] / routing_piston.config.grid_size)
start_row = int(start[1] / routing_piston.config.grid_size)
end_col = int(end[0] / routing_piston.config.grid_size)
end_row = int(end[1] / routing_piston.config.grid_size)

print(f"   Start grid: ({start_row}, {start_col}) = {routing_piston.fcu_grid[start_row][start_col] if start_row < len(routing_piston.fcu_grid) and start_col < len(routing_piston.fcu_grid[0]) else 'OOB'}")
print(f"   End grid: ({end_row}, {end_col}) = {routing_piston.fcu_grid[end_row][end_col] if end_row < len(routing_piston.fcu_grid) and end_col < len(routing_piston.fcu_grid[0]) else 'OOB'}")
print(f"   Grid size: {routing_piston.grid_rows} x {routing_piston.grid_cols}")

routing_result = routing_piston.route(
    parts_db=parts_db,
    escapes={},  # Empty escapes for simple design
    placement=placement_result.positions,
    net_order=net_order
)

print(f"\n4. Routing Results:")
print(f"   routed_count: {routing_result.routed_count}")
print(f"   total_count: {routing_result.total_count}")
print(f"   success: {routing_result.success}")
print(f"   routes: {list(routing_result.routes.keys())}")

for net_name, route in routing_result.routes.items():
    print(f"\n   Route '{net_name}':")
    print(f"     segments: {len(route.segments)}")
    if route.segments:
        for i, seg in enumerate(route.segments[:5]):  # First 5 segments
            print(f"       [{i}] {seg}")

print("=" * 60)
