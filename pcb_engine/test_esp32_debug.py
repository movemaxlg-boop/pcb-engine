#!/usr/bin/env python3
"""Debug ESP32 routing - find exactly what's blocking."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pcb_engine.routing_piston import RoutingPiston
from pcb_engine.routing_types import RoutingConfig
from pcb_engine.test_complex_esp32 import create_esp32_sensor_parts_db

def debug_routing():
    """Debug which nets fail and why."""

    parts_db = create_esp32_sensor_parts_db()

    # Get placements from the hybrid algorithm
    # For debugging, let's use a simple manual placement with more spacing
    placement = {
        'U1': (15.0, 20.0),   # ESP32 in center-left
        'U2': (35.0, 25.0),   # LDO on right
        'U3': (35.0, 15.0),   # BME280 below LDO
        'C1': (42.0, 30.0),   # Input cap near LDO
        'C2': (42.0, 20.0),   # Output cap near LDO
        'C3': (8.0, 20.0),    # ESP32 decoupling
        'R1': (25.0, 12.0),   # I2C pull-ups
        'R2': (25.0, 8.0),
        'R3': (8.0, 28.0),    # LED resistors
        'D1': (5.0, 28.0),
        'R4': (8.0, 32.0),
        'D2': (5.0, 32.0),
    }

    config = RoutingConfig(
        board_width=50.0,
        board_height=40.0,
        origin_x=0.0,
        origin_y=0.0,
        grid_size=0.25,  # Larger grid for faster routing
        trace_width=0.25,
        clearance=0.15,
        via_diameter=0.6,
        via_drill=0.3,
        algorithm='lee'  # Use simple Lee first
    )

    print(f"Board: {config.board_width}x{config.board_height}mm")
    print(f"Grid: {int(config.board_width/config.grid_size)}x{int(config.board_height/config.grid_size)} cells")
    print()

    # Get nets
    nets = list(parts_db['nets'].keys())
    print(f"Nets to route: {nets}")
    print()

    # Route each net individually to find which ones fail
    for net_name in nets:
        net_info = parts_db['nets'][net_name]
        pins = net_info.get('pins', [])
        if len(pins) < 2:
            print(f"{net_name}: Skipped (only {len(pins)} pin)")
            continue

        # Create fresh piston for each net
        piston = RoutingPiston(config)
        result = piston.route(parts_db, {}, placement, [net_name])

        route = result.routes.get(net_name)
        if route and route.success:
            print(f"{net_name}: OK ({len(route.segments)} segments, {len(pins)} pins)")
        else:
            error = route.error if route else "No route object"
            print(f"{net_name}: FAILED - {error} ({len(pins)} pins: {pins})")

if __name__ == '__main__':
    debug_routing()
