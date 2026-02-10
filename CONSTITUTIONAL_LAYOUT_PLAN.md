# Constitutional Layout (c_layout) Implementation Plan

## Executive Summary

The Constitutional Layout system is the **missing link** between human intent and machine execution. It provides:
1. **User → AI dialogue** to convert natural language into machine-readable specs
2. **Rule hierarchy** (Inviolable/Recommended/Optional) for prioritized enforcement
3. **AI override authority** with justification for engineering flexibility
4. **Validation gate** before BBL execution
5. **Feedback loop** for AI to iterate on failures

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER                                            │
│                    "I want a sensor board with ESP32,                        │
│                     USB-C, and 3 analog inputs"                              │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CIRCUIT AI AGENT                                     │
│                    (circuit_intelligence/circuit_ai.py)                      │
│                                                                              │
│  1. Dialogue with user → clarify requirements                                │
│  2. Select parts from database + infer supporting components                 │
│  3. Query Rules API → get applicable rules for this design                   │
│  4. Classify rules into hierarchy (Inviolable/Recommended/Optional)          │
│  5. Generate placement & routing hints                                       │
│  6. Create CONSTITUTIONAL LAYOUT (c_layout)                                  │
│  7. Apply overrides with justification if needed                             │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    c_layout VALIDATION GATE                                  │
│               (circuit_intelligence/clayout_validator.py)                    │
│                                                                              │
│  ✓ All parts exist in parts_db                                               │
│  ✓ All nets have valid endpoints                                             │
│  ✓ No conflicting rules                                                      │
│  ✓ Board can physically fit all parts (rough check)                          │
│  ✓ Overrides have valid justifications                                       │
│  ✓ Rule hierarchy is consistent                                              │
│                                                                              │
│  FAIL → Return to AI Agent for correction                                    │
│  PASS → Proceed to BBL                                                       │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      CONSTITUTIONAL LAYOUT (c_layout)                        │
│                   (circuit_intelligence/clayout_types.py)                    │
│                                                                              │
│  {                                                                           │
│    "design_name": "ESP32_Sensor_v1",                                         │
│    "board": {...},           // Size, layers, stackup constraints            │
│    "components": [...],      // From parts database                          │
│    "nets": [...],            // Connectivity                                 │
│    "rules": {                                                                │
│      "inviolable": [...],    // MUST pass - abort if violated                │
│      "recommended": [...],   // SHOULD pass - warn if violated               │
│      "optional": [...]       // MAY pass - note if violated                  │
│    },                                                                        │
│    "overrides": [...],       // AI-justified rule modifications              │
│    "placement_hints": {...}, // Proximity groups, zones, edge components     │
│    "routing_hints": {...},   // Priority nets, impedance requirements        │
│    "user_requirements": [...],  // Original user statements                  │
│    "ai_assumptions": [...]   // What AI inferred                             │
│  }                                                                           │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PCB ENGINE (BBL)                                   │
│                         (pcb_engine/bbl_engine.py)                           │
│                                                                              │
│  Executes c_layout respecting rule hierarchy:                                │
│  - INVIOLABLE violation → ABORT + escalate to AI Agent                       │
│  - RECOMMENDED violation → WARN + continue + report to AI Agent              │
│  - OPTIONAL violation → LOG + continue                                       │
│                                                                              │
│  Returns: BBLResult with success/failure + escalation report if needed       │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
         ┌────────────────────────┼────────────────────────┐
         │                        │                        │
         ▼                        ▼                        ▼
      SUCCESS                  PARTIAL                  FAILED
         │                        │                        │
         ▼                        ▼                        ▼
    ┌─────────┐           ┌─────────────┐         ┌─────────────────┐
    │ OUTPUT  │           │  WARN USER  │         │ ESCALATION RPT  │
    │.kicad   │           │  Continue   │         │ Back to AI Agent│
    └─────────┘           └─────────────┘         └────────┬────────┘
                                                           │
                                                           ▼
                                                  ┌─────────────────┐
                                                  │   AI AGENT      │
                                                  │  Adjusts and    │
                                                  │  resubmits      │
                                                  │  c_layout       │
                                                  └─────────────────┘
```

---

## Phase 1: Core Data Types (clayout_types.py)

### Files to Create
- `circuit_intelligence/clayout_types.py`

### Data Structures

```python
@dataclass
class ConstitutionalLayout:
    """Machine-readable design specification with rule hierarchy."""

    # DESIGN IDENTITY
    design_name: str
    version: str = "1.0"
    created_by: str = "CircuitAI"
    created_at: str = ""  # ISO timestamp

    # BOARD CONSTRAINTS
    board: BoardConstraints

    # COMPONENTS (from parts_db)
    components: List[ComponentDefinition]

    # CONNECTIVITY
    nets: List[NetDefinition]

    # RULE HIERARCHY (THE KEY INNOVATION)
    rules: RuleHierarchy

    # AI OVERRIDES (with justification)
    overrides: List[RuleOverride]

    # PLACEMENT INTENT
    placement_hints: PlacementHints

    # ROUTING INTENT
    routing_hints: RoutingHints

    # METADATA
    user_requirements: List[str]
    ai_assumptions: List[str]
    validation_status: str = "pending"


@dataclass
class BoardConstraints:
    """Physical board constraints."""
    width_mm: float
    height_mm: float
    layer_count: int
    stackup: Optional[str] = None  # "standard_4layer", "rf_6layer", etc.
    min_trace_mm: float = 0.15
    min_space_mm: float = 0.15
    min_via_drill_mm: float = 0.3
    copper_weight_oz: float = 1.0
    board_thickness_mm: float = 1.6


@dataclass
class ComponentDefinition:
    """Component specification for c_layout."""
    ref_des: str                    # U1, C1, R1
    part_number: str                # From parts_db
    footprint: str                  # Package type
    value: Optional[str] = None     # 10uF, 10k, etc.

    # Electrical properties
    voltage_rating: float = 0.0
    current_rating: float = 0.0
    power_dissipation: float = 0.0

    # AI-inferred
    inferred: bool = False          # True if AI added this
    inference_reason: str = ""      # "Decoupling for U1"


@dataclass
class NetDefinition:
    """Net connectivity specification."""
    name: str
    net_type: str                   # POWER, GND, SIGNAL, DIFF_PAIR, HIGH_SPEED
    pins: List[str]                 # ["U1.VCC", "C1.1", "C2.1"]

    # Electrical properties
    voltage: float = 0.0
    current_max: float = 0.0
    frequency: float = 0.0

    # Constraints
    impedance_ohm: Optional[float] = None
    max_length_mm: Optional[float] = None
    matched_with: Optional[str] = None  # For diff pairs


@dataclass
class RuleHierarchy:
    """Three-tier rule classification - THE KEY INNOVATION."""

    # Rules that MUST pass - abort if violated
    inviolable: List[RuleBinding]

    # Rules that SHOULD pass - warn if violated
    recommended: List[RuleBinding]

    # Rules that MAY pass - note if violated
    optional: List[RuleBinding]

    # Default category for rules not explicitly listed
    default_category: str = "recommended"


@dataclass
class RuleBinding:
    """Binding of a rule to this design with specific parameters."""
    rule_id: str                    # "USB2_LENGTH_MATCHING"
    parameters: Dict[str, Any]      # {"max_mismatch_mm": 1.25}
    applies_to: List[str]           # ["USB_DP", "USB_DM"] or ["*"]
    reason: str                     # Why this category


@dataclass
class RuleOverride:
    """AI-justified modification to a rule."""
    rule_id: str
    original_value: Any
    new_value: Any
    justification: str              # Engineering reason
    evidence: str                   # Source of justification
    approved_by: str = "AI"         # "AI" or "user"
    timestamp: str = ""


@dataclass
class PlacementHints:
    """High-level placement guidance for the engine."""

    # Components that must be close together
    proximity_groups: List[ProximityGroup]

    # Components on board edge
    edge_components: List[str]

    # Components that must be apart
    keep_apart: List[KeepApart]

    # Functional zones
    zones: Dict[str, List[str]]     # {"analog": ["R1", "R2"], "power": ["U2"]}

    # Fixed positions (if any)
    fixed_positions: Dict[str, Tuple[float, float]]


@dataclass
class ProximityGroup:
    """Components that should be placed close together."""
    components: List[str]
    max_distance_mm: float
    reason: str


@dataclass
class KeepApart:
    """Components that should be separated."""
    component_a: str
    component_b: str
    min_distance_mm: float
    reason: str


@dataclass
class RoutingHints:
    """High-level routing guidance for the engine."""

    # Nets to route first (in order)
    priority_nets: List[str]

    # Differential pairs
    diff_pairs: List[DiffPairSpec]

    # Length matching groups
    length_match_groups: List[LengthMatchGroup]

    # Nets that need specific layers
    layer_assignments: Dict[str, List[int]]

    # Nets to route last
    deprioritized_nets: List[str]


@dataclass
class DiffPairSpec:
    """Differential pair specification."""
    positive_net: str
    negative_net: str
    impedance_ohm: float
    max_mismatch_mm: float
    max_length_mm: Optional[float] = None


@dataclass
class LengthMatchGroup:
    """Group of nets that must be length-matched."""
    name: str
    nets: List[str]
    max_mismatch_mm: float
    reference_net: Optional[str] = None
```

### Estimated Effort: 2 hours

---

## Phase 2: Rule Hierarchy Engine (rule_hierarchy.py)

### Files to Create
- `circuit_intelligence/rule_hierarchy.py`

### Purpose
Automatically classify rules into Inviolable/Recommended/Optional based on:
1. Design characteristics (has USB? has DDR? voltage levels?)
2. Safety implications
3. Industry standards requirements

### Key Functions

```python
class RuleHierarchyEngine:
    """Classifies rules into hierarchy based on design context."""

    def __init__(self):
        self.rules_api = RulesAPI()

        # Default classifications
        self.default_inviolable = [
            # Safety-critical
            "CONDUCTOR_SPACING",      # Fire hazard if violated
            "CREEPAGE_DISTANCE",      # High voltage safety
            "THERMAL_MAX_TJ",         # Component damage
            "CURRENT_CAPACITY",       # Trace melting

            # Functionality-critical
            "DDR*_IMPEDANCE",         # Won't work if wrong
            "PCIE*_IMPEDANCE",
            "USB*_IMPEDANCE",
        ]

        self.default_recommended = [
            "DECOUPLING_DISTANCE",
            "CRYSTAL_DISTANCE",
            "USB*_LENGTH_MATCHING",
            "DDR*_LENGTH_MATCHING",
            "EMI_LOOP_AREA",
            "THERMAL_VIA_COUNT",
        ]

        self.default_optional = [
            "SILKSCREEN_*",
            "COURTYARD_SPACING",
            "TESTPOINT_*",
        ]

    def classify_rules(self, design_context: Dict) -> RuleHierarchy:
        """Generate rule hierarchy for a specific design."""

    def get_applicable_rules(self, design_context: Dict) -> List[str]:
        """Get list of rule IDs that apply to this design."""

    def promote_rule(self, rule_id: str, to_category: str, reason: str):
        """Move a rule to a stricter category."""

    def demote_rule(self, rule_id: str, to_category: str, reason: str):
        """Move a rule to a less strict category."""
```

### Classification Logic

| Design Feature | Rules Promoted to INVIOLABLE |
|----------------|------------------------------|
| Has USB High-Speed | USB2_LENGTH_MATCHING, USB2_IMPEDANCE |
| Has DDR3/DDR4 | DDR*_IMPEDANCE, DDR*_LENGTH_MATCHING |
| Has PCIe | PCIE*_IMPEDANCE |
| Voltage > 50V | CREEPAGE_DISTANCE, HI_POT_CLEARANCE |
| Power > 5W | THERMAL_* rules |
| Medical device | ALL safety rules → INVIOLABLE |

### Estimated Effort: 2 hours

---

## Phase 3: Parts Inference Engine (parts_inference.py)

### Files to Create
- `circuit_intelligence/parts_inference.py`

### Purpose
Given user-specified components, infer required supporting components.

### Inference Rules

```python
class PartsInferenceEngine:
    """Infers required supporting components."""

    INFERENCE_RULES = {
        # MCU requires decoupling
        "MCU": {
            "requires": [
                {"type": "capacitor", "value": "100nF", "quantity": "per_vcc_pin"},
                {"type": "capacitor", "value": "10uF", "quantity": 1},
            ],
            "reason": "Decoupling for stable power"
        },

        # USB connector requires ESD protection
        "USB_CONNECTOR": {
            "requires": [
                {"type": "esd_protection", "part": "USBLC6-2SC6", "quantity": 1},
            ],
            "reason": "ESD protection per USB spec"
        },

        # Crystal requires load capacitors
        "CRYSTAL": {
            "requires": [
                {"type": "capacitor", "quantity": 2, "value": "from_crystal_spec"},
            ],
            "reason": "Load capacitors for oscillation"
        },

        # LDO requires input/output caps
        "LDO_REGULATOR": {
            "requires": [
                {"type": "capacitor", "value": "10uF", "quantity": 1, "position": "input"},
                {"type": "capacitor", "value": "10uF", "quantity": 1, "position": "output"},
            ],
            "reason": "Stability capacitors per datasheet"
        },

        # Power input requires bulk capacitor
        "POWER_INPUT": {
            "requires": [
                {"type": "capacitor", "value": "100uF", "quantity": 1},
            ],
            "reason": "Bulk capacitance for power supply"
        },
    }

    def infer_components(self, user_components: List[ComponentDefinition]) -> List[ComponentDefinition]:
        """Return list of inferred components to add."""

    def explain_inference(self, inferred_component: ComponentDefinition) -> str:
        """Explain why a component was inferred."""
```

### Estimated Effort: 1.5 hours

---

## Phase 4: c_layout Validator (clayout_validator.py)

### Files to Create
- `circuit_intelligence/clayout_validator.py`

### Validation Checks

```python
class CLayoutValidator:
    """Validates c_layout before passing to BBL."""

    def validate(self, clayout: ConstitutionalLayout) -> ValidationResult:
        """Run all validation checks."""

    def check_parts_exist(self, clayout: ConstitutionalLayout) -> List[str]:
        """Verify all parts exist in parts_db."""

    def check_nets_valid(self, clayout: ConstitutionalLayout) -> List[str]:
        """Verify all net endpoints exist."""

    def check_no_rule_conflicts(self, clayout: ConstitutionalLayout) -> List[str]:
        """Check for conflicting rules (e.g., spacing < trace width)."""

    def check_board_fits(self, clayout: ConstitutionalLayout) -> List[str]:
        """Rough check: can all components fit on board?"""

    def check_overrides_valid(self, clayout: ConstitutionalLayout) -> List[str]:
        """Verify all overrides have justifications."""

    def check_hierarchy_consistent(self, clayout: ConstitutionalLayout) -> List[str]:
        """No rule in multiple categories, etc."""

    def estimate_routability(self, clayout: ConstitutionalLayout) -> float:
        """0.0-1.0 estimate of routing success probability."""


@dataclass
class ValidationResult:
    """Result of c_layout validation."""
    valid: bool
    errors: List[str]       # Must fix before proceeding
    warnings: List[str]     # Should review
    suggestions: List[str]  # Nice to fix
    routability_estimate: float
```

### Estimated Effort: 2 hours

---

## Phase 5: Circuit AI Agent (circuit_ai.py) - ENHANCED

### Files to Modify
- `circuit_intelligence/circuit_ai.py` (major enhancement)

### New Capabilities

```python
class CircuitAI:
    """AI agent that converts user intent to c_layout."""

    def __init__(self):
        self.rules_api = RulesAPI()
        self.hierarchy_engine = RuleHierarchyEngine()
        self.parts_inference = PartsInferenceEngine()
        self.validator = CLayoutValidator()
        self.feedback_processor = AIFeedbackProcessor()

    # =========================================================================
    # DIALOGUE SYSTEM
    # =========================================================================

    def parse_user_intent(self, user_input: str) -> DesignIntent:
        """Extract design requirements from natural language."""

    def ask_clarifying_questions(self, intent: DesignIntent) -> List[str]:
        """Generate questions to clarify ambiguous requirements."""

    def process_user_answers(self, intent: DesignIntent, answers: Dict) -> DesignIntent:
        """Update intent based on user's answers."""

    # =========================================================================
    # c_layout GENERATION
    # =========================================================================

    def generate_clayout(self, intent: DesignIntent) -> ConstitutionalLayout:
        """Generate complete c_layout from design intent."""

        # Step 1: Select components from parts_db
        components = self._select_components(intent)

        # Step 2: Infer supporting components
        inferred = self.parts_inference.infer_components(components)
        all_components = components + inferred

        # Step 3: Define nets
        nets = self._define_nets(all_components, intent)

        # Step 4: Get applicable rules
        design_context = self._build_design_context(all_components, nets, intent)

        # Step 5: Classify rules into hierarchy
        rule_hierarchy = self.hierarchy_engine.classify_rules(design_context)

        # Step 6: Generate placement hints
        placement_hints = self._generate_placement_hints(all_components, nets)

        # Step 7: Generate routing hints
        routing_hints = self._generate_routing_hints(nets, design_context)

        # Step 8: Apply any overrides
        overrides = self._determine_overrides(intent, design_context)

        # Step 9: Build c_layout
        clayout = ConstitutionalLayout(
            design_name=intent.design_name,
            board=self._determine_board_constraints(intent, all_components),
            components=all_components,
            nets=nets,
            rules=rule_hierarchy,
            overrides=overrides,
            placement_hints=placement_hints,
            routing_hints=routing_hints,
            user_requirements=intent.original_statements,
            ai_assumptions=self._document_assumptions(),
        )

        # Step 10: Validate
        validation = self.validator.validate(clayout)
        if not validation.valid:
            clayout = self._fix_validation_errors(clayout, validation)

        return clayout

    # =========================================================================
    # OVERRIDE AUTHORITY
    # =========================================================================

    def create_override(
        self,
        rule_id: str,
        new_value: Any,
        justification: str,
        evidence: str
    ) -> RuleOverride:
        """Create a justified rule override."""

    def review_override(self, override: RuleOverride) -> bool:
        """Validate that an override is justified."""

    # =========================================================================
    # ESCALATION HANDLING
    # =========================================================================

    def handle_bbl_failure(
        self,
        clayout: ConstitutionalLayout,
        failure_report: Dict
    ) -> ConstitutionalLayout:
        """Adjust c_layout based on BBL failure report."""

    def escalate_to_user(self, issue: str, options: List[str]) -> str:
        """Ask user to make a decision AI can't make."""
```

### Estimated Effort: 4 hours

---

## Phase 6: BBL Integration (bbl_engine.py) - ENHANCED

### Files to Modify
- `pcb_engine/bbl_engine.py`

### Changes Required

```python
class BBLEngine:
    """Enhanced BBL with c_layout support."""

    def run_bbl_with_clayout(
        self,
        clayout: ConstitutionalLayout,
        progress_callback: Optional[Callable] = None,
        escalation_callback: Optional[Callable] = None,
    ) -> BBLResult:
        """Execute BBL respecting c_layout rule hierarchy."""

        # Apply overrides from c_layout
        self._apply_overrides(clayout.overrides)

        # Use placement hints
        self._configure_placement(clayout.placement_hints)

        # Use routing hints
        self._configure_routing(clayout.routing_hints)

        # Execute with rule hierarchy enforcement
        result = self._execute_with_hierarchy(clayout.rules)

        return result

    def _execute_with_hierarchy(self, rules: RuleHierarchy) -> BBLResult:
        """Execute BBL respecting rule hierarchy."""

        # After each piston, check rules:
        # - INVIOLABLE violation → ABORT immediately
        # - RECOMMENDED violation → WARN, continue, add to report
        # - OPTIONAL violation → LOG, continue

    def _check_inviolable_rules(self, state: Dict) -> List[str]:
        """Check if any inviolable rules are violated."""

    def _generate_escalation_report(
        self,
        failure_type: str,
        details: Dict
    ) -> EscalationReport:
        """Generate report for AI agent to process."""


@dataclass
class EscalationReport:
    """Report sent to AI agent when BBL needs help."""
    failure_type: str           # "routing_failed", "placement_impossible", etc.
    phase: str                  # Which BBL phase failed
    violated_rules: List[str]   # Which rules couldn't be met
    suggestions: List[str]      # What might help
    metrics: Dict               # Routing %, attempts, etc.
```

### Estimated Effort: 3 hours

---

## Phase 7: End-to-End Demo (demo_clayout.py)

### Files to Create
- `circuit_intelligence/demo_clayout.py`

### Demo Flow

```python
def demo_clayout_system():
    """Demonstrate complete c_layout workflow."""

    # 1. User provides natural language request
    user_input = """
    I need a sensor board with:
    - ESP32-WROOM-32 module
    - USB-C for power and programming
    - 3 analog inputs with 0-10V range
    - Board size around 50x40mm
    - 2 layers is fine
    """

    # 2. Circuit AI parses intent
    ai = CircuitAI()
    intent = ai.parse_user_intent(user_input)

    # 3. AI asks clarifying questions
    questions = ai.ask_clarifying_questions(intent)
    # User answers...

    # 4. AI generates c_layout
    clayout = ai.generate_clayout(intent)

    # 5. Show what AI inferred
    print("=== AI INFERRED COMPONENTS ===")
    for comp in clayout.components:
        if comp.inferred:
            print(f"  + {comp.ref_des}: {comp.part_number} ({comp.inference_reason})")

    # 6. Show rule hierarchy
    print("\n=== RULE HIERARCHY ===")
    print(f"INVIOLABLE: {len(clayout.rules.inviolable)} rules")
    print(f"RECOMMENDED: {len(clayout.rules.recommended)} rules")
    print(f"OPTIONAL: {len(clayout.rules.optional)} rules")

    # 7. Validate c_layout
    validator = CLayoutValidator()
    result = validator.validate(clayout)
    print(f"\nValidation: {'PASS' if result.valid else 'FAIL'}")
    print(f"Routability estimate: {result.routability_estimate:.0%}")

    # 8. Run BBL with c_layout
    engine = BBLEngine()
    bbl_result = engine.run_bbl_with_clayout(clayout)

    # 9. Show results
    print(f"\nBBL Result: {'SUCCESS' if bbl_result.success else 'FAILED'}")
    if not bbl_result.success:
        print("Escalation report generated for AI review")
```

### Estimated Effort: 1.5 hours

---

## Implementation Order

| Phase | Task | Files | Effort | Dependencies |
|-------|------|-------|--------|--------------|
| 1 | Core Data Types | clayout_types.py | 2h | None |
| 2 | Rule Hierarchy Engine | rule_hierarchy.py | 2h | Phase 1, rules_api.py |
| 3 | Parts Inference Engine | parts_inference.py | 1.5h | Phase 1 |
| 4 | c_layout Validator | clayout_validator.py | 2h | Phase 1, 2, 3 |
| 5 | Circuit AI Agent | circuit_ai.py | 4h | Phase 1, 2, 3, 4 |
| 6 | BBL Integration | bbl_engine.py | 3h | Phase 1, 5 |
| 7 | End-to-End Demo | demo_clayout.py | 1.5h | All phases |
| **Total** | | | **16h** | |

---

## Success Criteria

1. **c_layout Generation**: AI can convert natural language to valid c_layout
2. **Parts Inference**: AI adds required supporting components automatically
3. **Rule Hierarchy**: All 631 rules classified into 3 tiers
4. **Validation Gate**: Invalid c_layouts caught before BBL
5. **Override Authority**: AI can modify rules with justification
6. **BBL Integration**: Engine respects rule hierarchy
7. **Escalation Loop**: Failed BBL returns actionable report to AI
8. **Demo Works**: End-to-end demo runs successfully

---

## File Summary

| File | Status | Purpose |
|------|--------|---------|
| `circuit_intelligence/clayout_types.py` | NEW | Core data structures |
| `circuit_intelligence/rule_hierarchy.py` | NEW | Rule classification engine |
| `circuit_intelligence/parts_inference.py` | NEW | Component inference |
| `circuit_intelligence/clayout_validator.py` | NEW | Validation gate |
| `circuit_intelligence/circuit_ai.py` | MODIFY | Enhanced AI agent |
| `pcb_engine/bbl_engine.py` | MODIFY | c_layout support |
| `circuit_intelligence/demo_clayout.py` | NEW | End-to-end demo |

---

## Next Steps

After approval, implementation begins with **Phase 1: Core Data Types**.
