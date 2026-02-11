#!/usr/bin/env python3
"""Debug test for routing piston."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from routing_piston import RoutingPiston
from routing_types import RoutingConfig


def test_simple_routing():
    print('=' * 60)
    print('ROUTING DEBUG TEST - Simple 2-component, 1-net')
    print('=' * 60)

    parts_db = {
        'parts': {
            'R1': {
                'value': '330R',
                'footprint': '0805',
                'pins': [
                    {'number': '1', 'net': 'NET1', 'offset': (-0.95, 0), 'size': (0.9, 1.25)},
                    {'number': '2', 'net': '', 'offset': (0.95, 0), 'size': (0.9, 1.25)}
                ]
            },
            'R2': {
                'value': '330R',
                'footprint': '0805',
                'pins': [
                    {'number': '1', 'net': 'NET1', 'offset': (-0.95, 0), 'size': (0.9, 1.25)},
                    {'number': '2', 'net': '', 'offset': (0.95, 0), 'size': (0.9, 1.25)}
                ]
            },
        },
        'nets': {
            'NET1': {'pins': [('R1', '1'), ('R2', '1')]},
        }
    }

    placement = {
        'R1': (10.0, 15.0),
        'R2': (20.0, 15.0),
    }

    config = RoutingConfig(
        board_width=30.0,
        board_height=30.0,
        origin_x=0.0,
        origin_y=0.0,
        grid_size=0.1,
        trace_width=0.25,
        clearance=0.15,
        algorithm='lee'
    )

    print(f'Board: {config.board_width}x{config.board_height}mm')

    piston = RoutingPiston(config)
    result = piston.route(parts_db, {}, placement, ['NET1'])

    print(f'Success: {result.success}')
    print(f'Routed: {result.routed_count}/{result.total_count}')

    if result.success:
        route = result.routes.get('NET1')
        if route:
            print(f'Segments: {len(route.segments)}')
            for seg in route.segments:
                print(f'  {seg.start} -> {seg.end}')
    else:
        route = result.routes.get('NET1')
        if route:
            print(f'Error: {route.error}')

    return result.success


if __name__ == '__main__':
    test_simple_routing()
