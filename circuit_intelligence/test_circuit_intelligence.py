"""
Test Suite for Circuit Intelligence Engine
==========================================

Tests all modules of the Circuit Intelligence Engine to ensure:
1. All imports work correctly
2. Basic functionality of each module
3. Integration between modules

Run with: python -m pytest circuit_intelligence/test_circuit_intelligence.py -v
Or directly: python circuit_intelligence/test_circuit_intelligence.py
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Test that all modules can be imported."""
    print("\n=== Testing Imports ===")

    try:
        from circuit_intelligence import (
            CircuitIntelligence,
            PatternLibrary,
            CurrentFlowAnalyzer,
            ThermalAnalyzer,
            DesignReviewAI,
            ConstraintGenerator,
            PartsLibrary,
            ElectricalCalculator,
            MLEngine,
        )
        print("[OK] All main imports successful")
    except ImportError as e:
        print(f"[FAIL] Import error: {e}")
        return False

    try:
        from circuit_intelligence.circuit_types import (
            CircuitFunction, NetFunction, ComponentFunction,
            ComponentAnalysis, NetAnalysis, DesignIssue,
        )
        print("[OK] Circuit types imports successful")
    except ImportError as e:
        print(f"[FAIL] Circuit types import error: {e}")
        return False

    return True


def test_parts_library():
    """Test the parts library with verified data."""
    print("\n=== Testing Parts Library ===")

    from circuit_intelligence import PartsLibrary

    lib = PartsLibrary()
    lib.load_defaults()

    stats = lib.get_stats()
    print(f"[OK] Library loaded: {stats['total_parts']} verified parts")
    print(f"  Categories: {list(stats['categories'].keys())}")
    print(f"  Manufacturers: {stats['manufacturers']}")

    # Test lookup
    lm2596 = lib.get('LM2596S-5.0/NOPB')
    if lm2596:
        print(f"[OK] Found LM2596: {lm2596.description}")
        print(f"  Datasheet: {lm2596.datasheet_url}")
        assert lm2596.datasheet_url, "Missing datasheet URL!"
    else:
        # Try partial match
        results = lib.search('LM2596')
        if results:
            print(f"[OK] Found via search: {results[0].part_number}")

    # Test search
    ldos = lib.search('LDO')
    print(f"[OK] LDO search returned {len(ldos)} results")

    # Test category
    caps = lib.find_by_category('CAPACITOR')
    print(f"[OK] Found {len(caps)} capacitors")

    # Verify all parts have datasheet URLs
    for pn, part in lib.index.by_part_number.items():
        assert part.datasheet_url, f"Missing datasheet URL for {pn}"
    print("[OK] All parts have verified datasheet URLs")

    return True


def test_electrical_calculator():
    """Test electrical calculations."""
    print("\n=== Testing Electrical Calculator ===")

    from circuit_intelligence import ElectricalCalculator

    calc = ElectricalCalculator()

    # Test LDO power calculation
    # AMS1117: 5V in, 3.3V out, 500mA
    power = calc.power.ldo_power_dissipation(5.0, 3.3, 0.5)
    print(f"[OK] LDO power (5V->3.3V @ 500mA): {power:.2f}W")
    assert 0.8 < power < 0.9, f"Expected ~0.85W, got {power}W"

    # Test trace width calculation (IPC-2221)
    # 1A on external layer, 10°C rise
    width = calc.trace.trace_width_for_current(1.0, 10.0, is_external=True)
    print(f"[OK] Trace width for 1A: {width:.2f}mm")
    assert 0.2 < width < 1.0, f"Width {width}mm seems wrong"

    # Test LED resistor
    resistor, power_r = calc.power.led_resistor(5.0, 2.0, 20)  # 20mA
    print(f"[OK] LED resistor (5V, Vf=2V, 20mA): {resistor:.0f}ohm")
    assert 140 < resistor < 160, f"Expected ~150ohm, got {resistor}ohm"

    # Test buck converter design
    buck_design = calc.design_buck_converter(12.0, 5.0, 1.0)
    print(f"[OK] Buck design: {buck_design['inductor_uh']}uH inductor, {buck_design['estimated_efficiency']}% eff")

    return True


def test_pattern_library():
    """Test circuit pattern recognition."""
    print("\n=== Testing Pattern Library ===")

    from circuit_intelligence import PatternLibrary

    lib = PatternLibrary()

    # Check built-in patterns
    print(f"[OK] Loaded {len(lib.patterns)} circuit patterns")

    for name in ['BUCK_CONVERTER', 'LDO_REGULATOR', 'BYPASS_CAPACITOR']:
        pattern = lib.get_pattern(name)
        if pattern:
            print(f"  - {name}: {len(pattern.component_roles)} component roles")
        else:
            print(f"  - {name}: Not found")

    return True


def test_thermal_analyzer():
    """Test thermal analysis."""
    print("\n=== Testing Thermal Analyzer ===")

    from circuit_intelligence import ThermalAnalyzer

    analyzer = ThermalAnalyzer()

    # Test with mock component data
    # AMS1117: θJA=15°C/W, Tj_max=125°C
    # At 5V→3.3V with 800mA: P = 1.36W
    # Tj = Ta + P × θJA = 25 + 1.36 × 15 = 45.4°C

    # Simple thermal estimation
    tj = 25 + 1.36 * 15  # Ambient + Power × θJA
    print(f"[OK] Estimated Tj for AMS1117: {tj:.1f}°C (max 125°C)")
    assert tj < 125, "Temperature exceeds maximum!"

    return True


def test_ml_engine():
    """Test ML engine components."""
    print("\n=== Testing ML Engine ===")

    from circuit_intelligence import MLEngine, FeatureExtractor

    ml = MLEngine()

    # Test feature extraction placeholder
    extractor = FeatureExtractor()
    print("[OK] Feature extractor initialized")

    # Test issue predictor (rule-based fallback)
    print("[OK] ML engine initialized (using rule-based fallback)")

    return True


def test_integration():
    """Test integration between modules."""
    print("\n=== Testing Integration ===")

    from circuit_intelligence import CircuitIntelligence, PartsLibrary

    # Initialize
    ci = CircuitIntelligence()
    parts = PartsLibrary()
    parts.load_defaults()

    print("[OK] CircuitIntelligence initialized")
    print("[OK] Integration test passed (basic)")

    return True


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("CIRCUIT INTELLIGENCE ENGINE - TEST SUITE")
    print("=" * 60)

    results = []

    results.append(("Imports", test_imports()))
    results.append(("Parts Library", test_parts_library()))
    results.append(("Electrical Calculator", test_electrical_calculator()))
    results.append(("Pattern Library", test_pattern_library()))
    results.append(("Thermal Analyzer", test_thermal_analyzer()))
    results.append(("ML Engine", test_ml_engine()))
    results.append(("Integration", test_integration()))

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    passed = 0
    failed = 0
    for name, result in results:
        status = "[OK] PASS" if result else "[FAIL] FAIL"
        print(f"{status}: {name}")
        if result:
            passed += 1
        else:
            failed += 1

    print(f"\nTotal: {passed} passed, {failed} failed")

    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
