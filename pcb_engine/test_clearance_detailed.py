"""Debug clearance check in detail"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Simple voltage divider - 3 components
parts_db = {
    'parts': {
        'R1': {
            'footprint': '0805',
            'pins': [
                {'number': '1', 'net': 'VIN', 'offset': (-0.95, 0), 'size': (0.9, 1.25)},
                {'number': '2', 'net': 'VDIV', 'offset': (0.95, 0), 'size': (0.9, 1.25)},
            ]
        },
        'R2': {
            'footprint': '0805',
            'pins': [
                {'number': '1', 'net': 'VDIV', 'offset': (-0.95, 0), 'size': (0.9, 1.25)},
                {'number': '2', 'net': 'GND', 'offset': (0.95, 0), 'size': (0.9, 1.25)},
            ]
        },
        'C1': {
            'footprint': '0805',
            'pins': [
                {'number': '1', 'net': 'VDIV', 'offset': (-0.95, 0), 'size': (0.9, 1.25)},
                {'number': '2', 'net': 'GND', 'offset': (0.95, 0), 'size': (0.9, 1.25)},
            ]
        },
    },
    'nets': {
        'VIN': {'pins': [('R1', '1')]},
        'VDIV': {'pins': [('R1', '2'), ('R2', '1'), ('C1', '1')]},
        'GND': {'pins': [('R2', '2'), ('C1', '2')]},
    }
}

print("=" * 70)
print("DEBUG: Clearance Zone Analysis")
print("=" * 70)

from pcb_engine.placement_engine import PlacementEngine, PlacementConfig
from pcb_engine.routing_piston import RoutingPiston, RoutingConfig

# Placement
pe = PlacementEngine(PlacementConfig(board_width=30, board_height=25, algorithm='sa'))
result = pe.place(parts_db, {})
placement = result.positions

print(f"\nPlacement:")
for ref, pos in placement.items():
    print(f"  {ref}: ({pos[0]:.2f}, {pos[1]:.2f})")

# Routing setup
rp = RoutingPiston(RoutingConfig(
    board_width=30, board_height=25,
    algorithm='lee',
    trace_width=0.25,
    clearance=0.2,
    grid_size=0.1
))
rp._initialize_grids()
rp._placement = placement
rp._parts_db = parts_db
rp._register_components(placement, parts_db)

print(f"\nGrid config: {rp.grid_rows}x{rp.grid_cols}, clearance_cells={rp.clearance_cells}")

# Check end point for VDIV (R2.1)
# R2 at (20.0, 14.5), pin 1 at offset (-0.95, 0) = (19.05, 14.50)
end_pos = (19.05, 14.50)
end_col = int(end_pos[0] / 0.1)
end_row = int(end_pos[1] / 0.1)

print(f"\nAnalyzing end point at ({end_pos[0]}, {end_pos[1]}) = grid[{end_row}][{end_col}]")
print(f"Center cell value: {rp.fcu_grid[end_row][end_col]!r}")

# Check ALL cells in clearance zone
cc = rp.clearance_cells
print(f"\nClearance zone ({2*cc+1}x{2*cc+1} = {(2*cc+1)**2} cells):")
blocking_cells = []
for dr in range(-cc, cc + 1):
    for dc in range(-cc, cc + 1):
        r, c = end_row + dr, end_col + dc
        if rp._in_bounds(r, c):
            occ = rp.fcu_grid[r][c]
            if occ is not None and occ != 'VDIV':
                blocking_cells.append((r, c, occ, dr, dc))

print(f"Found {len(blocking_cells)} cells that are NOT 'VDIV':")
for r, c, occ, dr, dc in blocking_cells:
    world_x = c * 0.1
    world_y = r * 0.1
    dist = (dr**2 + dc**2)**0.5
    print(f"  [{r}][{c}] ({world_x:.1f}, {world_y:.1f}) = {occ!r} (offset {dr},{dc}, dist={dist:.1f})")

# Also check what's at R2.2 (GND)
gnd_pos = (20.95, 14.50)  # R2 pin 2
gnd_col = int(gnd_pos[0] / 0.1)
gnd_row = int(gnd_pos[1] / 0.1)
print(f"\nR2.2 GND pad at ({gnd_pos[0]}, {gnd_pos[1]}) = grid[{gnd_row}][{gnd_col}]")
print(f"  Cell value: {rp.fcu_grid[gnd_row][gnd_col]!r}")

# Check distance between R2.1 (VDIV) and R2.2 (GND)
dist_pins = abs(gnd_col - end_col)
print(f"\nDistance between R2.1 and R2.2: {dist_pins} cells ({dist_pins * 0.1:.2f} mm)")
print(f"Clearance zone radius: {cc} cells")

if dist_pins <= 2 * cc:
    print(f"\n*** PROBLEM: Pads are closer than 2x clearance zone! ***")
    print(f"    Routing cannot pass through one pad without violating clearance of the other.")

# What's the solution?
print("\n" + "=" * 70)
print("ANALYSIS")
print("=" * 70)
print("""
The issue is that the clearance check `_is_cell_clear_for_net()` checks ALL cells
in the clearance zone. If any cell in that zone belongs to a DIFFERENT net,
the cell is considered blocked.

For 0805 components:
- Pin offset: 0.95mm from center
- Pin-to-pin distance: 1.9mm
- clearance_cells=4 means 0.4mm clearance radius
- But 0805 pads are so close that their clearance zones overlap with DIFFERENT nets

Solutions:
1. REDUCE clearance_cells for 0805 (fine pitch) components
2. Allow routing THROUGH pad cells if they're the target net
3. Exempt target endpoint from clearance check
4. Use pad-aware clearance (check pad vs pad differently than pad vs track)
""")

print("=" * 70)
