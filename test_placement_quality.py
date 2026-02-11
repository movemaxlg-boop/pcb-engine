"""
PLACEMENT QUALITY MEASUREMENT TEST
====================================

Runs placement ONLY (no routing) and measures quality with precise metrics:

1. OVERLAP TEST - Do any component courtyards overlap?
2. SPACING TEST - Minimum gap between any two components (mm)
3. DISTRIBUTION TEST - How well spread across the board?
4. BOUNDARY TEST - Any components out of bounds?
5. UTILIZATION TEST - What % of board area is used?
6. WIRELENGTH TEST - Total estimated wirelength (HPWL)
7. COURTYARD COMPARISON - Before/after courtyard fix comparison
8. ROUTING CHANNEL TEST - Are there routing channels between components?

Outputs a placement-only .kicad_pcb file for visual inspection in KiCad.
"""
import sys
import os
import math
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pcb_engine.placement_engine import PlacementEngine as PlacementPiston, PlacementConfig
from pcb_engine.common_types import calculate_courtyard, get_pins


def create_parts_db():
    """Same 18-component ESP32 sensor board from the real test."""
    parts_db = {
        'parts': {
            'U1': {
                'name': 'ESP32', 'footprint': 'QFN-32', 'value': 'ESP32-WROOM-32',
                'size': (18.0, 25.5),
                'pins': [
                    {'number': '1', 'name': 'GND', 'type': 'power_in', 'net': 'GND',
                     'physical': {'offset_x': -8.5, 'offset_y': -10.0}},
                    {'number': '2', 'name': 'VCC', 'type': 'power_in', 'net': '3V3',
                     'physical': {'offset_x': -8.5, 'offset_y': -7.5}},
                    {'number': '3', 'name': 'EN', 'type': 'input', 'net': 'EN',
                     'physical': {'offset_x': -8.5, 'offset_y': -5.0}},
                    {'number': '4', 'name': 'IO0', 'type': 'bidirectional', 'net': 'BOOT',
                     'physical': {'offset_x': -8.5, 'offset_y': -2.5}},
                    {'number': '5', 'name': 'IO2', 'type': 'bidirectional', 'net': 'LED1_CTRL',
                     'physical': {'offset_x': -8.5, 'offset_y': 0.0}},
                    {'number': '6', 'name': 'IO4', 'type': 'bidirectional', 'net': 'LED2_CTRL',
                     'physical': {'offset_x': -8.5, 'offset_y': 2.5}},
                    {'number': '7', 'name': 'IO18', 'type': 'bidirectional', 'net': 'USB_DN',
                     'physical': {'offset_x': -8.5, 'offset_y': 5.0}},
                    {'number': '8', 'name': 'IO19', 'type': 'bidirectional', 'net': 'USB_DP',
                     'physical': {'offset_x': -8.5, 'offset_y': 7.5}},
                    {'number': '9', 'name': 'IO21', 'type': 'bidirectional', 'net': 'I2C_SDA',
                     'physical': {'offset_x': 8.5, 'offset_y': 7.5}},
                    {'number': '10', 'name': 'IO22', 'type': 'bidirectional', 'net': 'I2C_SCL',
                     'physical': {'offset_x': 8.5, 'offset_y': 5.0}},
                    {'number': '11', 'name': 'RXD0', 'type': 'input', 'net': 'UART_RX',
                     'physical': {'offset_x': 8.5, 'offset_y': 2.5}},
                    {'number': '12', 'name': 'TXD0', 'type': 'output', 'net': 'UART_TX',
                     'physical': {'offset_x': 8.5, 'offset_y': 0.0}},
                    {'number': '13', 'name': 'GND2', 'type': 'power_in', 'net': 'GND',
                     'physical': {'offset_x': 8.5, 'offset_y': -2.5}},
                    {'number': '14', 'name': 'IO25', 'type': 'bidirectional', 'net': 'NC1',
                     'physical': {'offset_x': 8.5, 'offset_y': -5.0}},
                    {'number': '15', 'name': 'IO26', 'type': 'bidirectional', 'net': 'NC2',
                     'physical': {'offset_x': 8.5, 'offset_y': -7.5}},
                    {'number': '16', 'name': 'IO27', 'type': 'bidirectional', 'net': 'NC3',
                     'physical': {'offset_x': 8.5, 'offset_y': -10.0}},
                ],
            },
            'U2': {
                'name': 'LDO', 'footprint': 'SOT-223', 'value': 'AMS1117-3.3',
                'size': (6.5, 3.5),
                'pins': [
                    {'number': '1', 'name': 'GND', 'net': 'GND', 'physical': {'offset_x': -2.3, 'offset_y': 1.6}, 'size': (0.7, 1.5)},
                    {'number': '2', 'name': 'VOUT', 'net': '3V3', 'physical': {'offset_x': 0.0, 'offset_y': 1.6}, 'size': (0.7, 1.5)},
                    {'number': '3', 'name': 'VIN', 'net': 'VBUS', 'physical': {'offset_x': 2.3, 'offset_y': 1.6}, 'size': (0.7, 1.5)},
                    {'number': '4', 'name': 'TAB', 'net': '3V3', 'physical': {'offset_x': 0.0, 'offset_y': -1.6}, 'size': (3.5, 1.5)},
                ],
            },
            'U3': {
                'name': 'USBLC6', 'footprint': 'SOT-23-6', 'value': 'USBLC6-2SC6',
                'size': (2.9, 1.6),
                'pins': [
                    {'number': '1', 'name': 'IO1', 'net': 'USB_DP', 'physical': {'offset_x': -0.95, 'offset_y': 0.7}, 'size': (0.6, 0.7)},
                    {'number': '2', 'name': 'GND', 'net': 'GND', 'physical': {'offset_x': 0.0, 'offset_y': 0.7}, 'size': (0.6, 0.7)},
                    {'number': '3', 'name': 'IO2', 'net': 'USB_DN', 'physical': {'offset_x': 0.95, 'offset_y': 0.7}, 'size': (0.6, 0.7)},
                    {'number': '4', 'name': 'IO3', 'net': 'USB_DN', 'physical': {'offset_x': 0.95, 'offset_y': -0.7}, 'size': (0.6, 0.7)},
                    {'number': '5', 'name': 'VCC', 'net': '3V3', 'physical': {'offset_x': 0.0, 'offset_y': -0.7}, 'size': (0.6, 0.7)},
                    {'number': '6', 'name': 'IO4', 'net': 'USB_DP', 'physical': {'offset_x': -0.95, 'offset_y': -0.7}, 'size': (0.6, 0.7)},
                ],
            },
            'U4': {
                'name': 'BME280', 'footprint': 'LGA-8', 'value': 'BME280',
                'size': (2.5, 2.5),
                'pins': [
                    {'number': '1', 'name': 'GND', 'net': 'GND', 'physical': {'offset_x': -0.975, 'offset_y': -0.65}},
                    {'number': '2', 'name': 'CSB', 'net': 'BME_CS', 'physical': {'offset_x': -0.325, 'offset_y': -0.65}},
                    {'number': '3', 'name': 'SDI', 'net': 'I2C_SDA', 'physical': {'offset_x': 0.325, 'offset_y': -0.65}},
                    {'number': '4', 'name': 'SCK', 'net': 'I2C_SCL', 'physical': {'offset_x': 0.975, 'offset_y': -0.65}},
                    {'number': '5', 'name': 'SDO', 'net': 'BME_SDO', 'physical': {'offset_x': 0.975, 'offset_y': 0.65}},
                    {'number': '6', 'name': 'VDDIO', 'net': '3V3', 'physical': {'offset_x': 0.325, 'offset_y': 0.65}},
                    {'number': '7', 'name': 'GND2', 'net': 'GND', 'physical': {'offset_x': -0.325, 'offset_y': 0.65}},
                    {'number': '8', 'name': 'VDD', 'net': '3V3', 'physical': {'offset_x': -0.975, 'offset_y': 0.65}},
                ],
            },
            'J1': {
                'name': 'USB-C', 'footprint': 'USB-C-16P', 'value': 'USB-C',
                'size': (9.0, 7.5),
                'pins': [
                    {'number': '1', 'name': 'GND1', 'net': 'GND', 'physical': {'offset_x': -3.5, 'offset_y': -3.0}},
                    {'number': '2', 'name': 'VBUS1', 'net': 'VBUS', 'physical': {'offset_x': -2.5, 'offset_y': -3.0}},
                    {'number': '3', 'name': 'CC1', 'net': 'CC1', 'physical': {'offset_x': -1.5, 'offset_y': -3.0}},
                    {'number': '4', 'name': 'DP1', 'net': 'USB_DP', 'physical': {'offset_x': -0.5, 'offset_y': -3.0}},
                    {'number': '5', 'name': 'DN1', 'net': 'USB_DN', 'physical': {'offset_x': 0.5, 'offset_y': -3.0}},
                    {'number': '6', 'name': 'GND2', 'net': 'GND', 'physical': {'offset_x': 1.5, 'offset_y': -3.0}},
                    {'number': '7', 'name': 'SHIELD1', 'net': 'GND', 'physical': {'offset_x': 3.5, 'offset_y': 0.0}},
                    {'number': '8', 'name': 'SHIELD2', 'net': 'GND', 'physical': {'offset_x': -3.5, 'offset_y': 0.0}},
                ],
            },
        },
        'nets': {
            'GND': {'pins': ['U1.1', 'U1.13', 'U2.1', 'U3.2', 'U4.1', 'U4.7', 'J1.1', 'J1.6', 'J1.7', 'J1.8',
                              'C1.2', 'C2.2', 'C3.2', 'C4.2', 'C5.2', 'C6.2', 'LED1.2', 'LED2.2', 'R3.2']},
            '3V3': {'pins': ['U1.2', 'U2.2', 'U2.4', 'U3.5', 'U4.6', 'U4.8', 'C1.1', 'C2.1', 'C3.1', 'R1.1', 'R2.1']},
            'VBUS': {'pins': ['U2.3', 'J1.2', 'C5.1', 'C6.1']},
            'USB_DP': {'pins': ['U1.8', 'U3.1', 'U3.6']},
            'USB_DN': {'pins': ['U1.7', 'U3.3', 'U3.4']},
            'I2C_SDA': {'pins': ['U1.9', 'U4.3', 'R1.2']},
            'I2C_SCL': {'pins': ['U1.10', 'U4.4', 'R2.2']},
            'BME_CS': {'pins': ['U4.2']},
            'BME_SDO': {'pins': ['U4.5', 'R3.1']},
            'CC1': {'pins': ['J1.3', 'R4.1']},
            'LED1_CTRL': {'pins': ['U1.5', 'R5.1']},
            'LED2_CTRL': {'pins': ['U1.6', 'R6.1']},
            'EN': {'pins': ['U1.3', 'R4.2']},
            'BOOT': {'pins': ['U1.4', 'C4.1']},
        }
    }

    # Add passives (MUST be before return!)
    for i, ref in enumerate(['C1', 'C2', 'C3', 'C4', 'C5', 'C6']):
        nets = {
            'C1': ('3V3', 'GND'), 'C2': ('3V3', 'GND'), 'C3': ('3V3', 'GND'),
            'C4': ('BOOT', 'GND'), 'C5': ('VBUS', 'GND'), 'C6': ('VBUS', 'GND')
        }
        n1, n2 = nets[ref]
        parts_db['parts'][ref] = {
            'name': ref, 'footprint': '0402', 'value': '100nF',
            'size': (1.0, 0.5),
            'pins': [
                {'number': '1', 'net': n1, 'physical': {'offset_x': -0.4, 'offset_y': 0.0}},
                {'number': '2', 'net': n2, 'physical': {'offset_x': 0.4, 'offset_y': 0.0}},
            ]
        }

    for i, ref in enumerate(['R1', 'R2', 'R3', 'R4', 'R5', 'R6']):
        nets = {
            'R1': ('3V3', 'I2C_SDA'), 'R2': ('3V3', 'I2C_SCL'),
            'R3': ('BME_SDO', 'GND'), 'R4': ('CC1', 'EN'),
            'R5': ('LED1_CTRL', 'LED_A1'), 'R6': ('LED2_CTRL', 'LED_A2')
        }
        n1, n2 = nets[ref]
        parts_db['parts'][ref] = {
            'name': ref, 'footprint': '0402', 'value': '10K',
            'size': (1.0, 0.5),
            'pins': [
                {'number': '1', 'net': n1, 'physical': {'offset_x': -0.4, 'offset_y': 0.0}},
                {'number': '2', 'net': n2, 'physical': {'offset_x': 0.4, 'offset_y': 0.0}},
            ]
        }

    for ref in ['LED1', 'LED2']:
        nets = {'LED1': ('LED_A1', 'GND'), 'LED2': ('LED_A2', 'GND')}
        n1, n2 = nets[ref]
        parts_db['parts'][ref] = {
            'name': ref, 'footprint': '0402', 'value': 'LED',
            'size': (1.0, 0.5),
            'pins': [
                {'number': '1', 'net': n1, 'physical': {'offset_x': -0.4, 'offset_y': 0.0}},
                {'number': '2', 'net': n2, 'physical': {'offset_x': 0.4, 'offset_y': 0.0}},
            ]
        }

    return parts_db


def measure_placement_quality(parts_db, result, config):
    """Measure placement quality with 8 precise metrics."""
    parts = parts_db.get('parts', {})
    positions = result.positions
    board_w = config.board_width
    board_h = config.board_height

    print("\n" + "=" * 70)
    print("PLACEMENT QUALITY REPORT")
    print("=" * 70)

    # Calculate courtyards for all components
    courtyards = {}
    for ref, part in parts.items():
        footprint = part.get('footprint', '')
        cy = calculate_courtyard(part, footprint_name=footprint)
        courtyards[ref] = cy

    # =====================================================================
    # TEST 1: OVERLAP - Do any component courtyards overlap?
    # =====================================================================
    refs = list(positions.keys())
    overlaps = []
    for i in range(len(refs)):
        for j in range(i + 1, len(refs)):
            a, b = refs[i], refs[j]
            if a not in courtyards or b not in courtyards:
                continue
            ax, ay = positions[a]
            bx, by = positions[b]
            ca, cb = courtyards[a], courtyards[b]

            # Check courtyard overlap
            a_left = ax + ca.min_x
            a_right = ax + ca.max_x
            a_top = ay + ca.min_y
            a_bottom = ay + ca.max_y
            b_left = bx + cb.min_x
            b_right = bx + cb.max_x
            b_top = by + cb.min_y
            b_bottom = by + cb.max_y

            overlap_x = max(0, min(a_right, b_right) - max(a_left, b_left))
            overlap_y = max(0, min(a_bottom, b_bottom) - max(a_top, b_top))

            if overlap_x > 0.01 and overlap_y > 0.01:
                area = overlap_x * overlap_y
                overlaps.append((a, b, area))

    if overlaps:
        print(f"\n  [FAIL] OVERLAP TEST: {len(overlaps)} courtyard overlaps found!")
        for a, b, area in sorted(overlaps, key=lambda x: -x[2])[:10]:
            print(f"         {a} <-> {b}: {area:.2f} mm^2 overlap")
    else:
        print(f"\n  [PASS] OVERLAP TEST: 0 courtyard overlaps")

    # =====================================================================
    # TEST 2: SPACING - Minimum edge-to-edge gap between components
    # =====================================================================
    min_gap = float('inf')
    min_gap_pair = None
    all_gaps = []

    for i in range(len(refs)):
        for j in range(i + 1, len(refs)):
            a, b = refs[i], refs[j]
            if a not in courtyards or b not in courtyards:
                continue
            ax, ay = positions[a]
            bx, by = positions[b]
            ca, cb = courtyards[a], courtyards[b]

            # Edge-to-edge gap (negative = overlap)
            gap_x = abs(ax - bx) - (ca.width / 2 + cb.width / 2)
            gap_y = abs(ay - by) - (ca.height / 2 + cb.height / 2)

            # The gap is the maximum of X and Y (if both are negative, they overlap)
            # If one axis overlaps but other doesn't, the gap is the non-overlapping one
            if gap_x >= 0 or gap_y >= 0:
                gap = max(gap_x, gap_y)
            else:
                gap = max(gap_x, gap_y)  # Both negative = overlap

            all_gaps.append((a, b, gap))
            if gap < min_gap:
                min_gap = gap
                min_gap_pair = (a, b)

    if min_gap >= 0:
        status = "PASS" if min_gap >= 0.5 else "WARN"
        print(f"\n  [{status}] SPACING TEST: Min gap = {min_gap:.2f}mm ({min_gap_pair[0]}<->{min_gap_pair[1]})")
    else:
        print(f"\n  [FAIL] SPACING TEST: Components overlap by {-min_gap:.2f}mm ({min_gap_pair[0]}<->{min_gap_pair[1]})")

    # Show 5 closest pairs
    all_gaps.sort(key=lambda x: x[2])
    print("         5 closest pairs:")
    for a, b, gap in all_gaps[:5]:
        print(f"           {a:>5s} <-> {b:<5s}: {gap:+.2f}mm")

    # =====================================================================
    # TEST 3: DISTRIBUTION - How well spread across the board?
    # Metric: coefficient of variation of x and y positions
    # =====================================================================
    xs = [positions[r][0] for r in refs]
    ys = [positions[r][1] for r in refs]

    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    std_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs) / len(xs))
    std_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys) / len(ys))

    # Ideal distribution: components should span 60-90% of board
    x_span = max(xs) - min(xs)
    y_span = max(ys) - min(ys)
    x_spread = x_span / board_w * 100
    y_spread = y_span / board_h * 100

    status = "PASS" if x_spread > 50 and y_spread > 50 else "WARN" if x_spread > 30 and y_spread > 30 else "FAIL"
    print(f"\n  [{status}] DISTRIBUTION TEST:")
    print(f"         X spread: {x_span:.1f}mm / {board_w:.0f}mm = {x_spread:.0f}%")
    print(f"         Y spread: {y_span:.1f}mm / {board_h:.0f}mm = {y_spread:.0f}%")
    print(f"         Center: ({mean_x:.1f}, {mean_y:.1f}) vs board center ({board_w/2:.1f}, {board_h/2:.1f})")
    print(f"         Std dev: X={std_x:.1f}mm, Y={std_y:.1f}mm")

    # =====================================================================
    # TEST 4: BOUNDARY - Any components out of bounds?
    # =====================================================================
    oob_count = 0
    margin = config.edge_margin
    for ref in refs:
        if ref not in courtyards:
            continue
        x, y = positions[ref]
        cy = courtyards[ref]
        left = x + cy.min_x
        right = x + cy.max_x
        top = y + cy.min_y
        bottom = y + cy.max_y

        violations = []
        if left < margin:
            violations.append(f"left edge {left:.1f} < margin {margin}")
        if right > board_w - margin:
            violations.append(f"right edge {right:.1f} > {board_w - margin}")
        if top < margin:
            violations.append(f"top edge {top:.1f} < margin {margin}")
        if bottom > board_h - margin:
            violations.append(f"bottom edge {bottom:.1f} > {board_h - margin}")

        if violations:
            oob_count += 1
            print(f"\n  [FAIL] BOUNDARY: {ref} out of bounds:")
            for v in violations:
                print(f"           {v}")

    if oob_count == 0:
        print(f"\n  [PASS] BOUNDARY TEST: All {len(refs)} components within board")

    # =====================================================================
    # TEST 5: UTILIZATION - Board area usage
    # =====================================================================
    total_courtyard_area = sum(cy.width * cy.height for cy in courtyards.values())
    board_area = board_w * board_h
    utilization = total_courtyard_area / board_area * 100

    status = "PASS" if 15 < utilization < 80 else "WARN"
    print(f"\n  [{status}] UTILIZATION TEST:")
    print(f"         Total courtyard area: {total_courtyard_area:.1f} mm^2")
    print(f"         Board area: {board_area:.0f} mm^2")
    print(f"         Utilization: {utilization:.1f}%")

    # =====================================================================
    # TEST 6: WIRELENGTH - Half-Perimeter Wirelength (HPWL) estimate
    # =====================================================================
    nets = parts_db.get('nets', {})
    total_hpwl = 0
    net_wls = {}
    for net_name, net_info in nets.items():
        pin_refs = net_info.get('pins', [])
        net_xs, net_ys = [], []

        for pin_ref in pin_refs:
            if isinstance(pin_ref, str):
                p = pin_ref.split('.')
                comp = p[0]
            else:
                continue

            if comp in positions:
                net_xs.append(positions[comp][0])
                net_ys.append(positions[comp][1])

        if len(net_xs) >= 2:
            hpwl = (max(net_xs) - min(net_xs)) + (max(net_ys) - min(net_ys))
            total_hpwl += hpwl
            net_wls[net_name] = hpwl

    print(f"\n  [INFO] WIRELENGTH TEST:")
    print(f"         Total HPWL: {total_hpwl:.1f}mm")
    print(f"         Top 5 longest nets:")
    for net, wl in sorted(net_wls.items(), key=lambda x: -x[1])[:5]:
        print(f"           {net:>15s}: {wl:.1f}mm")

    # =====================================================================
    # TEST 7: COURTYARD COMPARISON - Old vs New dimensions
    # =====================================================================
    print(f"\n  [INFO] COURTYARD COMPARISON (Old body vs New courtyard):")
    print(f"         {'Ref':>6s}  {'Footprint':<12s}  {'Old W×H':<12s}  {'New W×H':<12s}  {'Change':>8s}")
    for ref in sorted(parts.keys()):
        part = parts[ref]
        size = part.get('size', (2.0, 2.0))
        if isinstance(size, (list, tuple)):
            old_w, old_h = size[0], size[1]
        else:
            old_w = old_h = float(size)

        cy = courtyards[ref]
        change = ((cy.width * cy.height) / (old_w * old_h) - 1) * 100

        fp = part.get('footprint', '?')
        print(f"         {ref:>6s}  {fp:<12s}  {old_w:.1f}×{old_h:.1f}{'mm':<5s}  {cy.width:.1f}×{cy.height:.1f}{'mm':<5s}  {change:+.0f}%")

    # =====================================================================
    # TEST 8: ROUTING CHANNEL - Are there clear routing channels?
    # Measures the average gap between components
    # =====================================================================
    positive_gaps = [g for _, _, g in all_gaps if g > 0]
    if positive_gaps:
        avg_gap = sum(positive_gaps) / len(positive_gaps)
        median_gap = sorted(positive_gaps)[len(positive_gaps) // 2]
        status = "PASS" if avg_gap >= 1.0 else "WARN" if avg_gap >= 0.5 else "FAIL"
        print(f"\n  [{status}] ROUTING CHANNEL TEST:")
        print(f"         Average gap: {avg_gap:.2f}mm")
        print(f"         Median gap: {median_gap:.2f}mm")
        print(f"         Min gap: {min(positive_gaps):.2f}mm")
        print(f"         Gaps > 1mm: {sum(1 for g in positive_gaps if g > 1.0)}/{len(positive_gaps)}")
    else:
        print(f"\n  [FAIL] ROUTING CHANNEL TEST: No positive gaps!")

    # =====================================================================
    # COMPONENT POSITIONS TABLE
    # =====================================================================
    print(f"\n  COMPONENT POSITIONS:")
    print(f"         {'Ref':>6s}  {'X':>8s}  {'Y':>8s}  {'W':>6s}  {'H':>6s}  {'Footprint'}")
    for ref in sorted(refs):
        x, y = positions[ref]
        cy = courtyards.get(ref)
        w = cy.width if cy else 0
        h = cy.height if cy else 0
        fp = parts.get(ref, {}).get('footprint', '?')
        print(f"         {ref:>6s}  {x:8.2f}  {y:8.2f}  {w:6.2f}  {h:6.2f}  {fp}")

    # =====================================================================
    # OVERALL SCORE
    # =====================================================================
    score = 0
    score += 30 if len(overlaps) == 0 else 0
    score += 20 if min_gap >= 0.5 else (10 if min_gap >= 0 else 0)
    score += 15 if x_spread > 50 and y_spread > 50 else (8 if x_spread > 30 and y_spread > 30 else 0)
    score += 10 if oob_count == 0 else 0
    score += 10 if 15 < utilization < 80 else 5
    score += 15 if avg_gap >= 1.0 else (8 if avg_gap >= 0.5 else 0) if positive_gaps else 0

    print(f"\n{'=' * 70}")
    print(f"PLACEMENT QUALITY SCORE: {score}/100")
    if score >= 80:
        print("VERDICT: EXCELLENT - Ready for routing")
    elif score >= 60:
        print("VERDICT: GOOD - Minor issues, routing may succeed")
    elif score >= 40:
        print("VERDICT: FAIR - Routing will struggle")
    else:
        print("VERDICT: POOR - Placement needs rework")
    print(f"{'=' * 70}")

    return score, overlaps, min_gap


def generate_placement_only_kicad(parts_db, positions, courtyards, config, output_path):
    """Generate a .kicad_pcb with ONLY placement (no routing) for visual inspection."""
    import uuid

    board_w = config.board_width
    board_h = config.board_height

    lines = []
    lines.append('(kicad_pcb')
    lines.append('  (version 20240108)')
    lines.append('  (generator "pcb_engine_placement_test")')
    lines.append('  (generator_version "1.0")')
    lines.append(f'  (general (thickness 1.6) (legacy_teardrops no))')
    lines.append('  (paper "A4")')

    # Layers
    lines.append('  (layers')
    lines.append('    (0 "F.Cu" signal)')
    lines.append('    (31 "B.Cu" signal)')
    lines.append('    (36 "B.SilkS" user "B.Silkscreen")')
    lines.append('    (37 "F.SilkS" user "F.Silkscreen")')
    lines.append('    (44 "Edge.Cuts" user "Edge.Cuts")')
    lines.append('    (46 "B.CrtYd" user "B.Courtyard")')
    lines.append('    (47 "F.CrtYd" user "F.Courtyard")')
    lines.append('    (48 "B.Fab" user "B.Fabrication")')
    lines.append('    (49 "F.Fab" user "F.Fabrication")')
    lines.append('  )')

    # Setup
    lines.append('  (setup')
    lines.append('    (pad_to_mask_clearance 0.1)')
    lines.append('    (allow_soldermask_bridges_in_footprints no)')
    lines.append('    (pcbplotparams')
    lines.append('      (layerselection 0x00010fc_ffffffff)')
    lines.append('      (plot_on_all_layers_selection 0x0000000_00000000)')
    lines.append('    )')
    lines.append('  )')

    # Board outline
    corners = [(0, 0), (board_w, 0), (board_w, board_h), (0, board_h)]
    for i in range(4):
        x1, y1 = corners[i]
        x2, y2 = corners[(i + 1) % 4]
        lines.append(f'  (gr_line (start {x1:.4f} {y1:.4f}) (end {x2:.4f} {y2:.4f})'
                     f' (stroke (width 0.1) (type solid)) (layer "Edge.Cuts") (uuid "{uuid.uuid4()}"))')

    # Net declarations
    net_ids = {'': 0}
    all_nets = set()
    for ref, part in parts_db['parts'].items():
        for pin in get_pins(part):
            net = pin.get('net', '')
            if net:
                all_nets.add(net)
    for i, net in enumerate(sorted(all_nets), 1):
        net_ids[net] = i
        lines.append(f'  (net {i} "{net}")')

    # Footprints with courtyards
    parts = parts_db['parts']
    for ref in sorted(positions.keys()):
        x, y = positions[ref]
        part = parts.get(ref, {})
        cy = courtyards.get(ref)
        if not cy:
            continue

        fp_name = part.get('footprint', 'Unknown')
        value = part.get('value', '')
        pins = get_pins(part)

        lines.append(f'  (footprint "{fp_name}"')
        lines.append(f'    (layer "F.Cu")')
        lines.append(f'    (uuid "{uuid.uuid4()}")')
        lines.append(f'    (at {x:.4f} {y:.4f})')

        # Reference
        ref_y_offset = -(cy.height / 2 + 0.8)
        lines.append(f'    (fp_text reference "{ref}"')
        lines.append(f'      (at 0 {ref_y_offset:.2f})')
        lines.append(f'      (layer "F.SilkS")')
        lines.append(f'      (uuid "{uuid.uuid4()}")')
        lines.append(f'      (effects (font (size 0.8 0.8) (thickness 0.12)))')
        lines.append(f'    )')

        # Value
        lines.append(f'    (fp_text value "{value}"')
        lines.append(f'      (at 0 {-ref_y_offset:.2f})')
        lines.append(f'      (layer "F.Fab")')
        lines.append(f'      (uuid "{uuid.uuid4()}")')
        lines.append(f'      (effects (font (size 0.8 0.8) (thickness 0.12)))')
        lines.append(f'    )')

        # Courtyard rectangle
        lines.append(f'    (fp_rect (start {cy.min_x:.4f} {cy.min_y:.4f}) (end {cy.max_x:.4f} {cy.max_y:.4f})')
        lines.append(f'      (stroke (width 0.05) (type solid)) (fill none) (layer "F.CrtYd") (uuid "{uuid.uuid4()}"))')

        # Fab rectangle (body)
        body_size = part.get('size', (1.0, 1.0))
        if isinstance(body_size, (list, tuple)):
            bw, bh = body_size[0], body_size[1]
        else:
            bw = bh = float(body_size)
        lines.append(f'    (fp_rect (start {-bw/2:.4f} {-bh/2:.4f}) (end {bw/2:.4f} {bh/2:.4f})')
        lines.append(f'      (stroke (width 0.1) (type solid)) (fill none) (layer "F.Fab") (uuid "{uuid.uuid4()}"))')

        # Pads
        for pin in pins:
            pin_num = pin.get('number', '1')
            net = pin.get('net', '')
            net_id = net_ids.get(net, 0)

            offset = pin.get('offset', None)
            if offset and isinstance(offset, (list, tuple)) and len(offset) >= 2:
                ox, oy = float(offset[0]), float(offset[1])
            else:
                physical = pin.get('physical', {})
                ox = float(physical.get('offset_x', 0))
                oy = float(physical.get('offset_y', 0))

            pad_size = pin.get('size', (1.0, 0.6))
            if isinstance(pad_size, (list, tuple)):
                pw, ph = pad_size[0], pad_size[1]
            else:
                pw = ph = 1.0

            lines.append(f'    (pad "{pin_num}" smd roundrect')
            lines.append(f'      (at {ox:.4f} {oy:.4f})')
            lines.append(f'      (size {pw:.4f} {ph:.4f})')
            lines.append(f'      (layers "F.Cu" "F.Paste" "F.Mask")')
            lines.append(f'      (roundrect_rratio 0.25)')
            if net:
                lines.append(f'      (net {net_id} "{net}")')
            lines.append(f'      (uuid "{uuid.uuid4()}")')
            lines.append(f'    )')

        lines.append(f'  )')

    lines.append(')')

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"\n  Placement-only KiCad file written to: {output_path}")
    return output_path


def main():
    print("=" * 70)
    print("PLACEMENT-ONLY QUALITY TEST")
    print("=" * 70)

    parts_db = create_parts_db()
    print(f"Components: {len(parts_db['parts'])}")
    print(f"Nets: {len(parts_db.get('nets', {}))}")

    # Configure placement
    config = PlacementConfig(
        board_width=50.0,
        board_height=40.0,
        algorithm='hybrid',
        seed=42,
        min_spacing=2.0,
        edge_margin=2.0,
    )

    # Build adjacency graph from nets
    adjacency = {}
    for net_name, net_info in parts_db.get('nets', {}).items():
        pin_refs = net_info.get('pins', [])
        comps = set()
        for pr in pin_refs:
            if isinstance(pr, str):
                comps.add(pr.split('.')[0])
        for a in comps:
            for b in comps:
                if a != b:
                    if a not in adjacency:
                        adjacency[a] = {}
                    adjacency[a][b] = adjacency[a].get(b, 0) + 1

    graph = {'adjacency': adjacency}

    # Run placement
    print(f"\nRunning placement (algorithm: {config.algorithm})...")
    t0 = time.time()
    piston = PlacementPiston(config)
    result = piston.place(parts_db, graph)
    elapsed = time.time() - t0
    print(f"Placement complete in {elapsed:.1f}s")
    print(f"Algorithm used: {result.algorithm_used}")
    print(f"Converged: {result.converged}")
    print(f"Success: {result.success}")

    # Calculate courtyards
    courtyards = {}
    for ref, part in parts_db['parts'].items():
        fp = part.get('footprint', '')
        courtyards[ref] = calculate_courtyard(part, footprint_name=fp)

    # Measure quality
    score, overlaps, min_gap = measure_placement_quality(parts_db, result, config)

    # Generate placement-only KiCad file
    output_dir = r"D:\Anas\tmp\output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "placement_test.kicad_pcb")
    generate_placement_only_kicad(parts_db, result.positions, courtyards, config, output_path)

    return score


if __name__ == '__main__':
    score = main()
    sys.exit(0 if score >= 60 else 1)
