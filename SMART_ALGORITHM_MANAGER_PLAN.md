# SMART ALGORITHM MANAGER - Complete Design Plan

## CURRENT STATE ANALYSIS

### Two Engines Working Together (Currently)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER (BOSS)                                     │
│   "I want an ESP32 temperature logger with WiFi and OLED display"           │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CIRCUIT AI (ENGINEER)                                │
│   circuit_ai.py - 2500+ lines                                                │
│                                                                              │
│   CURRENT CAPABILITIES:                                                      │
│   ✅ Parse natural language requirements                                     │
│   ✅ Select components (MCU, sensors, passives)                              │
│   ✅ Generate parts_db with nets                                             │
│   ✅ Create topology and power tree                                          │
│   ✅ Suggest improvements (decoupling caps, pull-ups)                        │
│   ✅ Handle escalations from Engine                                          │
│                                                                              │
│   MISSING (Algorithm Intelligence):                                          │
│   ❌ Analyze design BEFORE routing to recommend algorithms                   │
│   ❌ Per-net algorithm selection based on net class                          │
│   ❌ Learning from past routing successes/failures                           │
│   ❌ Real-time algorithm switching during routing                            │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                          create_handoff() - EngineHandoff
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          PCB ENGINE (FOREMAN)                                │
│   pcb_engine.py + bbl_engine.py - Combined ~2000 lines                       │
│                                                                              │
│   CURRENT CAPABILITIES:                                                      │
│   ✅ Coordinate 18 pistons in sequence                                       │
│   ✅ BBL (Big Beautiful Loop) with 6 improvements                            │
│   ✅ Checkpoints, rollback, timeouts                                         │
│   ✅ Progress reporting                                                      │
│   ✅ Escalation to Circuit AI                                                │
│                                                                              │
│   CURRENT ALGORITHM MANAGEMENT (The Problem):                                │
│   ❌ Blind CASCADE - Fixed order regardless of design                        │
│   ❌ All-or-nothing - One algorithm for ALL nets                             │
│   ❌ No learning - Repeats same mistakes                                     │
│   ❌ No specialization - Algorithms not matched to net type                  │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                              Piston Work Orders
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             18 PISTONS (WORKERS)                             │
│                                                                              │
│   CORE: Parts → Order → Placement → Escape → Routing → DRC → Output         │
│   ADVANCED: Stackup, Thermal, PDN, Signal Integrity, Topological            │
│   LEARNING: LearningPiston (exists but underutilized)                        │
│                                                                              │
│   ROUTING PISTON specifically has:                                           │
│   ✅ 12 routing algorithms implemented                                       │
│   ✅ Push-and-Shove (new)                                                    │
│   ✅ Net Class constraints (new)                                             │
│   ✅ Trunk chain detection (new)                                             │
│   ✅ Return path awareness (new)                                             │
│   ❌ But NO SMART SELECTION - uses whatever Engine tells it                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## THE SOLUTION: SMART ALGORITHM MANAGER

### New Component: RoutingPlanner

**Location**: `pcb_engine/routing_planner.py` (NEW FILE)
**Role**: Bridge between Circuit AI (Engineer) and Routing Piston
**Purpose**: Analyze design and create per-net routing strategies

```
                    CIRCUIT AI (Engineer)
                           │
                           │ parts_db + requirements
                           ▼
              ┌─────────────────────────────┐
              │     ROUTING PLANNER         │  ← NEW COMPONENT
              │                             │
              │  1. Analyze Design Profile  │
              │  2. Classify All Nets       │
              │  3. Select Algorithms       │
              │  4. Create Routing Plan     │
              │  5. Query Learning DB       │
              └─────────────┬───────────────┘
                            │
                            │ RoutingPlan (per-net strategies)
                            ▼
                      PCB ENGINE (Foreman)
                            │
                            │ Executes plan
                            ▼
                      ROUTING PISTON
```

---

## DETAILED DESIGN

### 1. RoutingPlanner Class

```python
@dataclass
class NetRoutingStrategy:
    """Strategy for routing a single net"""
    net_name: str
    net_class: NetClass                    # POWER, SIGNAL, DIFFERENTIAL, etc.
    priority: int                          # 1=highest (route first)

    # Algorithm selection
    primary_algorithm: RoutingAlgorithm    # First algorithm to try
    fallback_algorithms: List[RoutingAlgorithm]  # Backup algorithms

    # Constraints from net class
    trace_width: float
    clearance: float
    via_size: float

    # Special handling flags
    use_push_and_shove: bool = False       # Enable push-and-shove for this net
    trunk_chain_id: Optional[int] = None   # Part of trunk chain (bus routing)
    check_return_path: bool = False        # High-frequency, check ground reference
    length_match_group: Optional[str] = None  # Differential pair group

    # Learning database reference
    learned_algorithm: Optional[str] = None  # From past success


@dataclass
class RoutingPlan:
    """Complete routing plan for a design"""
    design_profile: DesignProfile          # Board characteristics
    net_strategies: Dict[str, NetRoutingStrategy]  # Per-net strategies
    routing_order: List[str]               # Net order (by priority)

    # Global settings
    enable_trunk_chains: bool
    enable_return_path_check: bool
    ground_pour_recommended: bool

    # Learning database queries
    similar_designs: List[str]             # Similar past designs
    success_patterns: Dict[str, float]     # Algorithm success rates


class RoutingPlanner:
    """
    Analyzes design and creates optimal routing plan.

    Acts as intelligence layer between Circuit AI and Routing Piston.
    """

    def __init__(self, learning_db: 'LearningDatabase' = None):
        self.learning_db = learning_db or LearningDatabase()

    def analyze_design(self, parts_db: Dict, board_config: Dict) -> DesignProfile:
        """
        STEP 1: Analyze design characteristics

        Examines:
        - Number and types of components
        - Net count and connectivity
        - Board size and layer count
        - Package types (BGA, QFN, etc.)
        - Special requirements (high-speed, power, etc.)
        """
        pass

    def classify_nets(self, parts_db: Dict) -> Dict[str, NetClass]:
        """
        STEP 2: Classify every net

        Uses:
        - Net naming conventions (VCC, GND, CLK, etc.)
        - Connected component types
        - Pin count and spread
        - User-provided net classes
        """
        pass

    def select_algorithms(self, net_class: NetClass,
                         design_profile: DesignProfile) -> Tuple[RoutingAlgorithm, List[RoutingAlgorithm]]:
        """
        STEP 3: Select best algorithm for each net class

        Decision matrix:
        ┌──────────────────┬─────────────────┬────────────────────────────┐
        │ Net Class        │ Primary Algo    │ Fallback Algorithms        │
        ├──────────────────┼─────────────────┼────────────────────────────┤
        │ POWER            │ LEE             │ PATHFINDER, A*             │
        │ GROUND           │ (use pour)      │ LEE                        │
        │ HIGH_SPEED       │ STEINER         │ LEE, HADLOCK               │
        │ DIFFERENTIAL     │ STEINER         │ LEE (with length match)    │
        │ BUS (parallel)   │ PATHFINDER      │ CHANNEL, HYBRID            │
        │ ANALOG           │ LEE             │ HADLOCK (avoid crossings)  │
        │ SIGNAL (default) │ HYBRID          │ A*, PATHFINDER, LEE        │
        │ RF               │ LEE             │ (minimal vias)             │
        │ HIGH_CURRENT     │ LEE             │ (wide traces only)         │
        └──────────────────┴─────────────────┴────────────────────────────┘

        Also queries learning database for past successes.
        """
        pass

    def create_routing_plan(self, parts_db: Dict,
                           board_config: Dict) -> RoutingPlan:
        """
        STEP 4: Create complete routing plan

        Combines:
        - Design profile analysis
        - Net classification
        - Algorithm selection
        - Priority ordering
        - Special feature flags
        """
        pass
```

---

### 2. LearningDatabase Class

```python
@dataclass
class RoutingOutcome:
    """Record of a routing attempt"""
    net_name: str
    net_class: NetClass
    algorithm_used: RoutingAlgorithm
    success: bool
    time_ms: float
    via_count: int
    wire_length: float
    drc_violations: int
    timestamp: str
    design_hash: str  # Fingerprint of similar designs


class LearningDatabase:
    """
    Persistent learning from routing successes and failures.

    Stores:
    - Algorithm success rates per net class
    - Design fingerprints for similarity matching
    - Time/quality tradeoffs
    - Failure patterns to avoid
    """

    def __init__(self, db_path: str = None):
        self.db_path = db_path or 'routing_learning.json'
        self.outcomes: List[RoutingOutcome] = []
        self.algorithm_stats: Dict[str, Dict] = {}  # net_class -> {algo -> stats}
        self._load()

    def record_outcome(self, outcome: RoutingOutcome):
        """Record a routing attempt result"""
        pass

    def get_best_algorithm(self, net_class: NetClass,
                          design_profile: DesignProfile) -> Optional[RoutingAlgorithm]:
        """Get historically best algorithm for this net class"""
        pass

    def get_success_rate(self, algorithm: RoutingAlgorithm,
                        net_class: NetClass) -> float:
        """Get success rate for algorithm on net class"""
        pass

    def find_similar_designs(self, design_hash: str) -> List[str]:
        """Find similar past designs for learning"""
        pass
```

---

### 3. Enhanced PCB Engine Integration

```python
# In pcb_engine.py - modify run_bbl() method

class PCBEngine:
    def run_bbl(self, parts_db: Dict, ...) -> BBLResult:
        """
        The Big Beautiful Loop with SMART algorithm management
        """

        # NEW: Create routing planner
        planner = RoutingPlanner(self.learning_db)

        # NEW: Phase 1.5 - Create routing plan BEFORE routing
        routing_plan = planner.create_routing_plan(
            parts_db,
            self.board_config
        )

        # Pass plan to routing phase
        result = self._execute_routing_phase(routing_plan, parts_db)

        # NEW: Record outcomes to learning database
        for net, outcome in result.outcomes.items():
            self.learning_db.record_outcome(outcome)
```

---

### 4. Enhanced Routing Piston Integration

```python
# In routing_piston.py - modify route() method

class RoutingPiston:
    def route_with_plan(self, routing_plan: RoutingPlan,
                       parts_db: Dict, ...) -> RoutingResult:
        """
        Route using the smart routing plan
        """
        results = {}

        for net_name in routing_plan.routing_order:
            strategy = routing_plan.net_strategies[net_name]

            # Try primary algorithm first
            result = self._route_net_with_algorithm(
                net_name,
                strategy.primary_algorithm,
                strategy.trace_width,
                strategy.clearance
            )

            if not result.success:
                # Try fallback algorithms
                for fallback in strategy.fallback_algorithms:
                    result = self._route_net_with_algorithm(
                        net_name, fallback, ...
                    )
                    if result.success:
                        break

            # Use push-and-shove if enabled and still failing
            if not result.success and strategy.use_push_and_shove:
                result = self._route_with_push_and_shove(net_name, ...)

            results[net_name] = result

        return RoutingResult(routes=results, ...)
```

---

## COMPLETE FLOW DIAGRAM

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    SMART ALGORITHM MANAGEMENT FLOW                             ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                                ║
║  USER (Boss)                                                                   ║
║  "ESP32 board with USB, sensors, WiFi"                                        ║
║       │                                                                        ║
║       ▼                                                                        ║
║  ┌─────────────────────────────────────────────────────────────────────────┐  ║
║  │  CIRCUIT AI (Engineer) - circuit_ai.py                                   │  ║
║  │                                                                          │  ║
║  │  1. Parse requirements                                                   │  ║
║  │  2. Select components                                                    │  ║
║  │  3. Generate parts_db with nets                                          │  ║
║  │  4. Create power tree and topology                                       │  ║
║  │                                                                          │  ║
║  │  Output: EngineHandoff with parts_db                                     │  ║
║  └──────────────────────────────────────────────────┬──────────────────────┘  ║
║                                                     │                          ║
║                                                     ▼                          ║
║  ┌─────────────────────────────────────────────────────────────────────────┐  ║
║  │  ROUTING PLANNER (NEW) - routing_planner.py                              │  ║
║  │                                                                          │  ║
║  │  STEP 1: Analyze Design Profile                                          │  ║
║  │    • Board: 50x40mm, 2-layer                                             │  ║
║  │    • Components: 25 (1 QFN, 2 SOT-23, 22 passives)                       │  ║
║  │    • Nets: 67 total                                                      │  ║
║  │    • Density: MEDIUM                                                     │  ║
║  │                                                                          │  ║
║  │  STEP 2: Classify Nets                                                   │  ║
║  │    ┌──────────────┬───────┬────────────────────────────────────────┐     │  ║
║  │    │ Net Class    │ Count │ Examples                               │     │  ║
║  │    ├──────────────┼───────┼────────────────────────────────────────┤     │  ║
║  │    │ POWER        │ 3     │ VCC, 3V3, VBAT                         │     │  ║
║  │    │ GROUND       │ 1     │ GND (→ use pour)                       │     │  ║
║  │    │ HIGH_SPEED   │ 2     │ USB_D+, USB_D-                         │     │  ║
║  │    │ BUS          │ 8     │ DATA[0:7]                              │     │  ║
║  │    │ I2C          │ 2     │ SDA, SCL                               │     │  ║
║  │    │ SPI          │ 4     │ MOSI, MISO, SCK, CS                    │     │  ║
║  │    │ SIGNAL       │ 47    │ Everything else                        │     │  ║
║  │    └──────────────┴───────┴────────────────────────────────────────┘     │  ║
║  │                                                                          │  ║
║  │  STEP 3: Select Algorithms (Query Learning DB)                           │  ║
║  │    ┌──────────────┬────────────────┬────────────────────────────┐        │  ║
║  │    │ Net Class    │ Primary Algo   │ Fallback Chain             │        │  ║
║  │    ├──────────────┼────────────────┼────────────────────────────┤        │  ║
║  │    │ POWER        │ LEE            │ PATHFINDER, A*             │        │  ║
║  │    │ GROUND       │ POUR           │ (no routing needed)        │        │  ║
║  │    │ HIGH_SPEED   │ STEINER*       │ LEE (learned: 94% success) │        │  ║
║  │    │ BUS          │ PATHFINDER     │ CHANNEL, HYBRID            │        │  ║
║  │    │ I2C          │ LEE            │ HADLOCK                    │        │  ║
║  │    │ SPI          │ A*             │ LEE, HADLOCK               │        │  ║
║  │    │ SIGNAL       │ HYBRID         │ A*, PATHFINDER             │        │  ║
║  │    └──────────────┴────────────────┴────────────────────────────┘        │  ║
║  │    * Learning DB says: STEINER has 94% success on HIGH_SPEED nets        │  ║
║  │                                                                          │  ║
║  │  STEP 4: Create Routing Plan                                             │  ║
║  │    • Priority Order: POWER → HIGH_SPEED → BUS → I2C → SPI → SIGNAL      │  ║
║  │    • Enable trunk chain for BUS nets                                     │  ║
║  │    • Enable return path check for HIGH_SPEED                             │  ║
║  │    • Enable push-and-shove as last resort                                │  ║
║  │    • Recommend GND pour (2-layer board)                                  │  ║
║  │                                                                          │  ║
║  │  Output: RoutingPlan with 67 NetRoutingStrategies                        │  ║
║  └──────────────────────────────────────────────────┬──────────────────────┘  ║
║                                                     │                          ║
║                                                     ▼                          ║
║  ┌─────────────────────────────────────────────────────────────────────────┐  ║
║  │  PCB ENGINE (Foreman) - pcb_engine.py + bbl_engine.py                    │  ║
║  │                                                                          │  ║
║  │  BBL Phase 1: ORDER RECEIVED                                             │  ║
║  │    • Receives RoutingPlan from Planner                                   │  ║
║  │    • Validates constraints                                               │  ║
║  │                                                                          │  ║
║  │  BBL Phase 2: PISTON EXECUTION                                           │  ║
║  │    • Parts Piston: builds component database                             │  ║
║  │    • Order Piston: confirms net order from plan                          │  ║
║  │    • Placement Piston: places components                                 │  ║
║  │    • Escape Piston: handles QFN escape routing                           │  ║
║  │    • Pour Piston: creates GND pour (plan recommended)                    │  ║
║  │    • Routing Piston: USES THE ROUTING PLAN (see below)                   │  ║
║  │    • DRC Piston: validates                                               │  ║
║  │    • Output Piston: generates .kicad_pcb                                 │  ║
║  │                                                                          │  ║
║  │  BBL Phase 5: KICAD DRC                                                  │  ║
║  │    • Final validation                                                    │  ║
║  │                                                                          │  ║
║  │  BBL Phase 6: LEARNING                                                   │  ║
║  │    • Record outcomes to LearningDatabase                                 │  ║
║  │    • Update algorithm success rates                                      │  ║
║  └──────────────────────────────────────────────────┬──────────────────────┘  ║
║                                                     │                          ║
║                                                     ▼                          ║
║  ┌─────────────────────────────────────────────────────────────────────────┐  ║
║  │  ROUTING PISTON - routing_piston.py                                      │  ║
║  │                                                                          │  ║
║  │  SMART ROUTING (Using RoutingPlan):                                      │  ║
║  │                                                                          │  ║
║  │  FOR net in routing_plan.routing_order:                                  │  ║
║  │      strategy = routing_plan.net_strategies[net]                         │  ║
║  │                                                                          │  ║
║  │      ┌─────────────────────────────────────────────────────────────┐     │  ║
║  │      │  Net: VCC (POWER class, priority 1)                         │     │  ║
║  │      │  Primary: LEE (guaranteed shortest)                         │     │  ║
║  │      │  Trace width: 0.5mm                                         │     │  ║
║  │      │  Result: SUCCESS in 45ms                                    │     │  ║
║  │      └─────────────────────────────────────────────────────────────┘     │  ║
║  │                                                                          │  ║
║  │      ┌─────────────────────────────────────────────────────────────┐     │  ║
║  │      │  Net: USB_D+ (HIGH_SPEED class, priority 2)                 │     │  ║
║  │      │  Primary: STEINER (learning DB: 94% success)                │     │  ║
║  │      │  Check: return path awareness                               │     │  ║
║  │      │  Result: SUCCESS in 78ms                                    │     │  ║
║  │      └─────────────────────────────────────────────────────────────┘     │  ║
║  │                                                                          │  ║
║  │      ┌─────────────────────────────────────────────────────────────┐     │  ║
║  │      │  Net: DATA[0:7] (BUS class, priority 3)                     │     │  ║
║  │      │  Primary: PATHFINDER (congestion-aware)                     │     │  ║
║  │      │  Apply: trunk chain (parallel routing)                      │     │  ║
║  │      │  Result: SUCCESS in 210ms (8 nets routed together)          │     │  ║
║  │      └─────────────────────────────────────────────────────────────┘     │  ║
║  │                                                                          │  ║
║  │      ┌─────────────────────────────────────────────────────────────┐     │  ║
║  │      │  Net: GPIO_5 (SIGNAL class, priority 6)                     │     │  ║
║  │      │  Primary: HYBRID → FAIL (blocked by other traces)           │     │  ║
║  │      │  Fallback 1: A* → FAIL                                      │     │  ║
║  │      │  Push-and-Shove: ENABLED → SUCCESS                          │     │  ║
║  │      │  Result: SUCCESS after push-and-shove (150ms)               │     │  ║
║  │      └─────────────────────────────────────────────────────────────┘     │  ║
║  │                                                                          │  ║
║  │  Final: 67/67 nets routed                                                │  ║
║  │  Time: 3.2 seconds (vs 12+ seconds with blind cascade)                   │  ║
║  └──────────────────────────────────────────────────────────────────────────┘  ║
║                                                                                ║
╚════════════════════════════════════════════════════════════════════════════════╝
```

---

## FILE CHANGES SUMMARY

| File | Action | Lines | Description |
|------|--------|-------|-------------|
| `routing_planner.py` | NEW | ~400 | RoutingPlanner, NetRoutingStrategy, RoutingPlan |
| `learning_database.py` | NEW | ~250 | LearningDatabase, RoutingOutcome |
| `routing_piston.py` | MODIFY | ~100 | Add `route_with_plan()` method |
| `pcb_engine.py` | MODIFY | ~50 | Integrate RoutingPlanner in BBL |
| `bbl_engine.py` | MODIFY | ~30 | Add learning phase |
| `circuit_ai.py` | MODIFY | ~20 | Pass design hints to planner |

**Total new code**: ~850 lines
**Total modifications**: ~200 lines

---

## ALGORITHM SELECTION MATRIX

### Decision Table: Which Algorithm for Which Net Class

| Net Class | Characteristics | Primary | Why | Fallback 1 | Fallback 2 |
|-----------|-----------------|---------|-----|------------|------------|
| **POWER** | Wide traces, low impedance | LEE | Guaranteed shortest path | PATHFINDER | A* |
| **GROUND** | Many connections | POUR | Plane is better than traces | LEE | - |
| **HIGH_SPEED** | >50MHz, USB, DDR | STEINER | Optimal multi-terminal | LEE | HADLOCK |
| **DIFFERENTIAL** | USB, HDMI, Ethernet | STEINER | Length matching | LEE | - |
| **BUS** | Parallel signals | PATHFINDER | Congestion-aware | CHANNEL | HYBRID |
| **I2C** | 2 wires, pull-ups | LEE | Short paths critical | HADLOCK | - |
| **SPI** | 4 wires, medium speed | A* | Fast, good enough | LEE | HADLOCK |
| **ANALOG** | Sensitive signals | LEE | No crosstalk | HADLOCK | - |
| **RF** | High frequency | LEE | Minimal vias | - | - |
| **HIGH_CURRENT** | Motor, LED power | LEE | Thick traces only | - | - |
| **SIGNAL** | General GPIO | HYBRID | Best overall | A* | PATHFINDER |

### Special Feature Activation

| Feature | When Activated | Nets Affected |
|---------|----------------|---------------|
| **Push-and-Shove** | Last resort after fallbacks | All (as needed) |
| **Trunk Chain** | Detected parallel nets | BUS, DATA buses |
| **Return Path Check** | HIGH_SPEED, RF, DIFFERENTIAL | Frequency-sensitive |
| **Length Matching** | DIFFERENTIAL pairs | USB, HDMI, DDR |

---

## LEARNING DATABASE SCHEMA

```json
{
  "version": "1.0",
  "created": "2026-02-09",
  "algorithm_stats": {
    "POWER": {
      "LEE": {"attempts": 145, "successes": 143, "avg_time_ms": 52},
      "PATHFINDER": {"attempts": 12, "successes": 11, "avg_time_ms": 78},
      "A*": {"attempts": 8, "successes": 6, "avg_time_ms": 45}
    },
    "HIGH_SPEED": {
      "STEINER": {"attempts": 89, "successes": 84, "avg_time_ms": 95},
      "LEE": {"attempts": 23, "successes": 20, "avg_time_ms": 67}
    },
    "SIGNAL": {
      "HYBRID": {"attempts": 512, "successes": 487, "avg_time_ms": 38},
      "A*": {"attempts": 156, "successes": 142, "avg_time_ms": 29},
      "PATHFINDER": {"attempts": 98, "successes": 91, "avg_time_ms": 56}
    }
  },
  "design_patterns": [
    {
      "hash": "esp32_sensor_2layer",
      "description": "ESP32 with sensors on 2-layer board",
      "best_config": {
        "placement": "force_directed",
        "routing": "per_net_class",
        "ground_pour": true
      },
      "success_rate": 0.94
    }
  ],
  "failure_patterns": [
    {
      "pattern": "bga_escape_dense",
      "description": "BGA with >100 pins on 2-layer",
      "failure_reason": "insufficient escape routes",
      "solution": "add_layers_or_reduce_density"
    }
  ]
}
```

---

## IMPLEMENTATION ORDER

### Phase 1: Core Infrastructure (Day 1)
1. Create `routing_planner.py` with RoutingPlanner class
2. Create `learning_database.py` with LearningDatabase class
3. Define all data structures (NetRoutingStrategy, RoutingPlan, etc.)

### Phase 2: Net Classification (Day 1)
4. Implement `classify_nets()` - detect net class from name/connections
5. Implement `analyze_design()` - design profile analysis
6. Add net class detection rules

### Phase 3: Algorithm Selection (Day 2)
7. Implement `select_algorithms()` with decision matrix
8. Integrate learning database queries
9. Add fallback chain logic

### Phase 4: Integration (Day 2)
10. Add `route_with_plan()` to RoutingPiston
11. Modify PCBEngine to use RoutingPlanner
12. Add learning recording to BBL Phase 6

### Phase 5: Testing & Validation (Day 3)
13. Test with various board types
14. Validate learning database updates
15. Measure improvement vs blind cascade

---

## SUCCESS METRICS

| Metric | Current (Blind) | Target (Smart) | How to Measure |
|--------|-----------------|----------------|----------------|
| **First-try success rate** | ~60% | >85% | Routes complete on first algorithm |
| **Average routing time** | 12s | 4s | Wall clock time for 50-net board |
| **Algorithm switches** | 4-5 per board | 1-2 per board | Fallback chain usage |
| **Learning utilization** | 0% | >50% | Nets using learned algorithms |
| **DRC pass rate** | 70% | >90% | First-time KiCad DRC pass |

---

## APPROVAL REQUEST

This plan requires:
1. Creating 2 new files (~650 lines)
2. Modifying 4 existing files (~200 lines)
3. Estimated 2-3 days of implementation

**Benefits**:
- 40-70% faster routing
- 25%+ higher first-try success rate
- Learning system improves over time
- New features (Push-and-Shove, Trunk Chain, Return Path) used intelligently

**Risks**:
- Minimal - all changes are additive
- Existing cascade still works as fallback
- Learning DB is optional (works without it)

Ready to implement when you approve.
