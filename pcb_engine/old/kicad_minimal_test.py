"""
Minimal KiCad Test Script
=========================
Run this in KiCad to diagnose API compatibility issues.

Usage: exec(open(r'path/to/kicad_minimal_test.py').read())
"""

import pcbnew

print("=" * 60)
print("KICAD API DIAGNOSTIC TEST")
print("=" * 60)

# Step 1: Get board
print("\n1. Getting board...")
try:
    board = pcbnew.GetBoard()
    if board is None:
        print("   ERROR: No board loaded")
    else:
        print("   OK: Board found")
except Exception as e:
    print(f"   ERROR: {e}")

# Step 2: Check KiCad version
print("\n2. KiCad Version Info...")
try:
    version = pcbnew.Version()
    print(f"   Version: {version}")
except:
    print("   Could not get version")

# Step 3: Test basic imports
print("\n3. Testing imports...")
try:
    from pcbnew import VECTOR2I, FromMM, F_Cu, B_Cu, Edge_Cuts
    print("   OK: Basic imports work")
except Exception as e:
    print(f"   ERROR: {e}")

# Step 4: Test mm conversion
print("\n4. Testing unit conversion...")
try:
    val = FromMM(1.0)
    print(f"   OK: FromMM(1.0) = {val}")
except Exception as e:
    print(f"   ERROR: {e}")

# Step 5: Test VECTOR2I
print("\n5. Testing VECTOR2I...")
try:
    pt = VECTOR2I(FromMM(10), FromMM(10))
    print(f"   OK: VECTOR2I created: {pt}")
except Exception as e:
    print(f"   ERROR: {e}")

# Step 6: Test PCB_SHAPE (board outline)
print("\n6. Testing PCB_SHAPE...")
try:
    shape = pcbnew.PCB_SHAPE(board)
    print("   OK: PCB_SHAPE created")

    # Test SetShape
    try:
        shape.SetShape(pcbnew.SHAPE_T_SEGMENT)
        print("   OK: SetShape(SHAPE_T_SEGMENT)")
    except AttributeError:
        try:
            shape.SetShape(pcbnew.S_SEGMENT)
            print("   OK: SetShape(S_SEGMENT) - old API")
        except Exception as e2:
            print(f"   ERROR SetShape: {e2}")
except Exception as e:
    print(f"   ERROR: {e}")

# Step 7: Test PCB_TRACK
print("\n7. Testing PCB_TRACK...")
try:
    track = pcbnew.PCB_TRACK(board)
    print("   OK: PCB_TRACK created")
except Exception as e:
    print(f"   ERROR: {e}")

# Step 8: Test PCB_VIA
print("\n8. Testing PCB_VIA...")
try:
    via = pcbnew.PCB_VIA(board)
    print("   OK: PCB_VIA created")

    # Test via type constants
    print("   Checking via type constants...")
    if hasattr(pcbnew, 'VIATYPE_THROUGH'):
        print("   OK: VIATYPE_THROUGH exists")
    elif hasattr(pcbnew, 'VIA_THROUGH'):
        print("   OK: VIA_THROUGH exists (KiCad 7+)")
    else:
        print("   WARN: No via type constant found")
except Exception as e:
    print(f"   ERROR: {e}")

# Step 9: Test ZONE
print("\n9. Testing ZONE...")
try:
    zone = pcbnew.ZONE(board)
    print("   OK: ZONE created")

    # Test zone outline methods
    print("   Checking zone outline methods...")
    if hasattr(zone, 'Outline'):
        print("   OK: zone.Outline() exists")
        try:
            outline = zone.Outline()
            print(f"   OK: Outline type: {type(outline)}")
            if hasattr(outline, 'Append'):
                print("   OK: outline.Append() exists")
            else:
                print("   WARN: outline.Append() not found")
        except Exception as e:
            print(f"   ERROR getting outline: {e}")
    else:
        print("   WARN: zone.Outline() not found")

    # Test zone settings
    if hasattr(zone, 'GetZoneSettings'):
        print("   OK: GetZoneSettings exists (KiCad 7+)")
    if hasattr(zone, 'SetZoneClearance'):
        print("   OK: SetZoneClearance exists (old API)")
    if hasattr(zone, 'SetLocalClearance'):
        print("   OK: SetLocalClearance exists")
except Exception as e:
    print(f"   ERROR: {e}")

# Step 10: Test NETINFO_ITEM
print("\n10. Testing NETINFO_ITEM...")
try:
    net = pcbnew.NETINFO_ITEM(board, "TEST_NET")
    print("   OK: NETINFO_ITEM created")
except Exception as e:
    print(f"   ERROR: {e}")

# Step 11: Test PCB_TEXT
print("\n11. Testing PCB_TEXT...")
try:
    text = pcbnew.PCB_TEXT(board)
    print("   OK: PCB_TEXT created")

    # Test text justification constants
    if hasattr(pcbnew, 'GR_TEXT_H_ALIGN_CENTER'):
        print("   OK: GR_TEXT_H_ALIGN_CENTER exists")
    elif hasattr(pcbnew, 'TEXT_H_ALIGN_CENTER'):
        print("   OK: TEXT_H_ALIGN_CENTER exists")
    else:
        print("   WARN: No text align constant found")
except Exception as e:
    print(f"   ERROR: {e}")

# Step 12: Test Design Settings
print("\n12. Testing Design Settings...")
try:
    ds = board.GetDesignSettings()
    print("   OK: GetDesignSettings()")

    # Check available attributes
    attrs = ['m_TrackMinWidth', 'm_MinClearance', 'm_ViasMinSize', 'm_ViasMinDrill']
    for attr in attrs:
        if hasattr(ds, attr):
            print(f"   OK: ds.{attr} exists")
        else:
            print(f"   WARN: ds.{attr} not found")
except Exception as e:
    print(f"   ERROR: {e}")

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)
print("\nCopy the output above and share it to identify the issue.")
