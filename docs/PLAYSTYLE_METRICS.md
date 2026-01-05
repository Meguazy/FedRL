# Chess Playstyle Metrics Documentation

This document provides a comprehensive explanation of the playstyle metrics system used to analyze and classify chess playing styles in the federated learning chess framework.

## Table of Contents

1. [Overview](#overview)
2. [Core Metrics (Novachess.ai Methodology)](#core-metrics-novachessai-methodology)
3. [Enhanced Metrics (Research Paper Based)](#enhanced-metrics-research-paper-based)
4. [Computed Metrics and Classification](#computed-metrics-and-classification)
5. [Implementation Details](#implementation-details)
6. [Configuration](#configuration)
7. [Storage Structure](#storage-structure)

---

## Overview

The playstyle metrics system analyzes chess games to classify playing styles on a spectrum from **"Very Positional"** to **"Very Tactical"**. The system combines:

1. **Core metrics** from novachess.ai methodology (attacked material, legal moves, captures, center control)
2. **Enhanced metrics** from research paper (legal moves per phase, delta/tipping points, pawn structure, move diversity)

### Classification Scale

| Score Range | Classification | Playing Style |
|-------------|---------------|---------------|
| > 0.70 | Very Tactical | Aggressive, forcing moves, complications |
| 0.65 - 0.70 | Tactical | Prefers tactics over strategy |
| 0.60 - 0.65 | Balanced | Mix of tactics and positional play |
| 0.50 - 0.60 | Positional | Strategic maneuvering, long-term plans |
| < 0.50 | Very Positional | Highly strategic, solid structures |

---

## Core Metrics (Novachess.ai Methodology)

### 1. Attacked Material

**Definition**: Total value of opponent pieces that can be captured in the current position.

**Formula**:
```
AttackedMaterial = Σ(PIECE_VALUES[captured_piece])
                   for all legal capture moves
```

**Piece Values**:
- Pawn = 1
- Knight = 3
- Bishop = 3
- Rook = 5
- Queen = 9
- King = 0 (not capturable)

**Analysis Range**: Plies 12-50 (moves 6-25)

**Interpretation**:
- High attacked material → Tactical pressure, creating threats
- Low attacked material → Solid position, few immediate tactics

**Implementation**: See `_calculate_attacked_material()` in `playstyle_metrics.py:507-525`

---

### 2. Legal Moves

**Definition**: Number of legal moves available to the player in the current position.

**Formula**:
```
LegalMoves = |{m : m ∈ board.legal_moves}|
```

**Analysis Range**:
- **Core tracking**: Plies 12-50 (for tactical score calculation)
- **Phase tracking**: Entire game (see Enhanced Metrics)

**Interpretation**:
- More legal moves → Flexibility, control, space advantage
- Fewer legal moves → Restricted position, defensive

**Normalization** (for tactical score):
```
MovesMetric = min(1.0, TotalLegalMoves / 40.0)
```

**Implementation**: See `_analyze_position()` in `playstyle_metrics.py:314-355`

---

### 3. Material Captured

**Definition**: Total value of pieces actually captured during the game (not just threatened).

**Formula**:
```
MaterialCaptured = Σ(PIECE_VALUES[captured_piece])
                   for all captures made by player
```

**Analysis Range**: Plies 1-50 (moves 1-25)

**Interpretation**:
- High captures → Tactical exchanges, dynamic play
- Low captures → Maneuvering game, few exchanges

**Normalization** (for tactical score):
```
MaterialMetric = min(1.0, TotalCaptures / 20.0)
```

**Implementation**: See `analyze_game()` capture tracking in `playstyle_metrics.py:264-274`

---

### 4. Center Control

**Definition**: Number of pieces attacking or controlling the four central squares (d4, d5, e4, e5).

**Formula**:
```
CenterControl = Σ(|attackers(square, color)|)
                for square in {d4, d5, e4, e5}
```

**Analysis Range**: Plies 12-50

**Interpretation**:
- High center control → Classical positional play, controlling the board
- Low center control → Hypermodern or flank play

**Implementation**: See `_calculate_center_control()` in `playstyle_metrics.py:351-375`

---

### 5. Tactical Score (Core)

**Definition**: Weighted combination of normalized metrics to produce a single tactical vs. positional score.

**Formula**:

**With captures** (material_metric > 0):
```
TacticalScore = (AttacksMetric + MovesMetric + MaterialMetric) / 3
```

**Without captures** (material_metric = 0):
```
TacticalScore = (AttacksMetric + MovesMetric) / 2
```

Where:
- `AttacksMetric = TotalAttackedMaterial / 39.0`
- `MovesMetric = min(1.0, TotalLegalMoves / 40.0)`
- `MaterialMetric = min(1.0, TotalCaptures / 20.0)`

**Normalization Constants**:
- 39 = Maximum reasonable attacked material per position
- 40 = Typical total legal moves in analyzed positions
- 20 = Typical total material captures in a game

**Implementation**: See `_compute_normalized_metrics()` in `playstyle_metrics.py:778-828`

---

## Enhanced Metrics (Research Paper Based)

### 6. Legal Moves Per Game Phase

**Definition**: Track legal moves separately for opening, middlegame, and endgame to understand phase-specific playing style.

**Game Phases**:
- **Opening**: Plies 1-12 (moves 1-6)
- **Middlegame**: Plies 13-40 (moves 7-20)
- **Endgame**: Plies 41+ (moves 21+)

**Formulas**:
```
AvgLegalMovesOpening = LegalMovesOpening / OpeningPositionsCount
AvgLegalMovesMiddlegame = LegalMovesMiddlegame / MiddlegamePositionsCount
AvgLegalMovesEndgame = LegalMovesEndgame / EndgamePositionsCount
```

**Interpretation**:
- **Tactical players**: High legal moves in all phases (maintaining options)
- **Positional players**: May restrict opponent's moves while maintaining own flexibility
- **Phase comparison**: Compare opening vs. middlegame vs. endgame to see if style changes

**Research Basis**: Paper showed legal moves is a key differentiator between human and engine play, and between tactical and positional styles.

**Implementation**: See phase tracking in `compute_game_metrics()` in `playstyle_metrics.py:453-495`

---

### 7. Delta Metric (Tipping Points)

**Definition**: Difference between the evaluation of the best move and the second-best move at critical positions. Indicates decision clarity and forcing nature of positions.

**Formula**:
```
Delta = |Eval(BestMove) - Eval(SecondBestMove)| / 100
```
(converted to pawns from centipawns)

**Analysis Method**:
- **Sparse sampling**: Every 3rd position (configurable)
- **Analysis range**: Middlegame only (plies 15-40)
- **Engine depth**: 12 (configurable, balance speed vs. accuracy)

**Computed Statistics**:
```
AvgDelta = Σ(delta_i) / N
MaxDelta = max(delta_i)
MinDelta = min(delta_i)
```

**Interpretation**:
- **High Delta** (> 1.5 pawns): Forcing positions, tactical play, one clear best move
- **Low Delta** (< 0.5 pawns): Flexible positions, multiple good continuations, positional play
- **MaxDelta**: Identifies critical turning points
- **AvgDelta**: Overall decision clarity through the game

**Mate Score Handling**:
```
if score.is_mate():
    cp = 10000 if mate_in_moves > 0 else -10000
```

**Performance Optimization**:
- Without optimization: ~30-60 seconds per game
- With sparse sampling (every 3rd) + depth 12: ~2-5 seconds per game
- **80-90% time reduction** with minimal accuracy loss

**Research Basis**: Paper identified this as the "Delta" metric showing tipping points in games where advantage swings occur.

**Implementation**: See `_analyze_delta()` in `playstyle_metrics.py:377-437`

---

### 8. Pawn Structure Metrics

**Definition**: Metrics describing pawn formation and advancement, indicating aggressive vs. conservative pawn play.

**Sampling**: Every 5 plies after opening (to reduce computational cost)

**Metrics**:

#### 8.1 Average Pawn Rank
```
AvgPawnRank = Σ(rank(pawn)) / PawnCount
```
where rank ∈ [1, 8] (1 = back rank, 8 = promotion rank)

**Interpretation**:
- **White**: Higher rank → More advanced pawns, aggressive
- **Black**: Lower rank → More advanced pawns (from black's perspective), aggressive
- **Typical**: White ~2.5-3.5, Black ~5.5-6.5

#### 8.2 Isolated Pawns
```
IsolatedPawns = |{p : p has no friendly pawns on adjacent files}|
```

**Interpretation**:
- High isolated pawns → Structural weaknesses (positional)
- Low isolated pawns → Connected pawn structure
- **Aggressive players**: May sacrifice structure for activity
- **Positional players**: Maintain solid pawn chains

#### 8.3 Doubled Pawns
```
DoubledPawns = Σ(max(0, pawns_on_file - 1))
                for each file
```

**Interpretation**:
- High doubled pawns → Compromised structure
- May indicate tactical exchanges or sacrifices

**Implementation**: See `_analyze_pawn_structure()` in `playstyle_metrics.py:439-492`

---

### 9. Move Diversity

**Definition**: Measures how many unique destination squares are targeted by moves throughout the game.

**Formula**:
```
UniqueMoveDestinations = |{move.to_square : move ∈ all_moves}|

MoveDiversityRatio = UniqueMoveDestinations / TotalMoves
```

**Interpretation**:
- **High diversity** (~0.7-0.9): Exploratory play, maneuvering, positional probing
- **Low diversity** (~0.3-0.5): Repetitive targeting, focused pressure, tactical themes
- **Very low** (< 0.3): Move repetition, forcing sequences

**Typical Values**:
- Positional players: Higher diversity (exploring multiple strategic ideas)
- Tactical players: Lower diversity (focused on specific targets)

**Example**:
- 40 total moves, 30 unique destinations → 0.75 ratio (high diversity)
- 40 total moves, 15 unique destinations → 0.375 ratio (low diversity, repetitive)

**Implementation**: See move diversity tracking in `analyze_game()` in `playstyle_metrics.py:258-262`

---

## Computed Metrics and Classification

### Aggregation Process

For each game, metrics are:
1. **Collected** per position (raw data)
2. **Aggregated** per player (summed/averaged)
3. **Normalized** (scaled to [0, 1] range)
4. **Combined** into tactical score
5. **Classified** into category

### Per-Player Aggregation

From `PlayerComputedMetrics`:

```python
# Raw aggregates
total_attacked_material: float      # Sum of all attacked material
total_legal_moves: int             # Sum of all legal moves
total_captures: int                # Sum of all captured material
avg_center_control: float          # Average center control
positions_analyzed: int            # Number of positions analyzed

# Phase-specific
legal_moves_opening: int           # Total legal moves in opening
legal_moves_middlegame: int        # Total legal moves in middlegame
legal_moves_endgame: int           # Total legal moves in endgame
avg_legal_moves_opening: float     # Average per opening position
avg_legal_moves_middlegame: float  # Average per middlegame position
avg_legal_moves_endgame: float     # Average per endgame position

# Delta statistics
avg_delta: float                   # Average delta across sampled positions
max_delta: float                   # Maximum delta (biggest tipping point)
min_delta: float                   # Minimum delta (most flexible position)
delta_samples: int                 # Number of positions analyzed

# Pawn structure
avg_pawn_rank: float              # Average pawn advancement
avg_isolated_pawns: float         # Average isolated pawns
avg_doubled_pawns: float          # Average doubled pawns

# Move diversity
unique_move_destinations: int     # Total unique squares targeted
move_diversity_ratio: float       # unique / total moves

# Normalized (for tactical score)
attacks_metric: float             # [0, 1+]
moves_metric: float               # [0, 1]
material_metric: float            # [0, 1]

# Final classification
tactical_score: float             # [0, 1+]
classification: str               # Category name
```

---

## Implementation Details

### Code Structure

**Main Files**:
- `server/evaluation/playstyle_metrics.py` (580 lines) - Core metrics implementation
- `server/evaluation/model_evaluator.py` (790 lines) - Game orchestration and ELO estimation
- `server/evaluation/opening_classifier.py` (160 lines) - Opening classification
- `server/main.py` - Integration into training pipeline

### Data Flow

```
PGN String
    ↓
GameAnalyzer.analyze_game()
    ↓
GameMetrics (raw data)
    ├─ position_metrics: List[PositionMetrics]
    ├─ captures: List[CaptureEvent]
    ├─ delta_metrics: List[DeltaMetric]
    ├─ pawn_structure_metrics: List[PawnStructureMetrics]
    └─ unique_destinations: Set[Square]
    ↓
PlaystyleMetricsCalculator.compute_game_metrics()
    ↓
ComputedGameMetrics
    ├─ white_metrics: PlayerComputedMetrics
    └─ black_metrics: PlayerComputedMetrics
    ↓
Storage (4 levels)
    ├─ Per-game raw metrics
    ├─ Per-game computed metrics
    ├─ Per-cluster aggregated metrics
    └─ Global evaluation summary
```

### Analysis Windows

| Metric | Analysis Range | Reason |
|--------|---------------|--------|
| Core metrics (attacked material, legal moves, center control) | Plies 12-50 | Avoid opening book and endgame simplifications |
| Captures | Plies 1-50 | Track material exchanges through critical phases |
| Legal moves (phase tracking) | Entire game | Understand phase-specific playing style |
| Delta (tipping points) | Plies 15-40 (every 3rd) | Middlegame is most tactically rich, sparse sampling for speed |
| Pawn structure | Plies 13+ (every 5) | After opening, sample to reduce cost |
| Move diversity | Entire game | Full picture of targeting patterns |

### Stockfish Integration

**Two Separate Instances**:
1. **Game playing**: StockfishPlayer with ELO limiting
2. **Delta analysis**: GameAnalyzer's stockfish engine at fixed depth

**Delta Analysis Engine**:
```python
self.stockfish = chess.engine.SimpleEngine.popen_uci(path)
info = self.stockfish.analyse(
    board,
    chess.engine.Limit(depth=12),  # Configurable
    multipv=2                       # Get top 2 moves
)
```

**Configuration**:
- `enable_delta_analysis`: Enable/disable delta computation
- `delta_sampling_rate`: Analyze every Nth position (3 recommended)
- `stockfish_depth`: Search depth (12-15 recommended for speed/accuracy balance)

---

## Configuration

### YAML Configuration

In `config/server_config.yaml`:

```yaml
evaluation_config:
  enabled: true                      # Enable playstyle evaluation
  interval_rounds: 10                # Run evaluation every N rounds
  games_per_elo_level: 10            # Number of games per Stockfish ELO level
  stockfish_elo_levels: [1000, 1200, 1400]  # ELO levels to test against
  time_per_move: 0.1                 # Time per move in seconds
  skip_check_positions: true         # Skip positions in check for analysis
  stockfish_path: null               # Path to Stockfish (null = auto-detect)

  # Enhanced metrics configuration
  enable_delta_analysis: true        # Enable delta (tipping point) metric
  delta_sampling_rate: 3             # Analyze every Nth position
  stockfish_depth: 12                # Search depth (12-15 recommended)
```

### Tuning Recommendations

**For Speed** (reduce evaluation time):
- Set `enable_delta_analysis: false` (saves 2-5 seconds per game)
- Increase `delta_sampling_rate` to 5 or 7
- Reduce `stockfish_depth` to 10
- Reduce `games_per_elo_level`

**For Accuracy** (more detailed analysis):
- Keep `enable_delta_analysis: true`
- Set `delta_sampling_rate` to 2 or 1 (every 2nd or every position)
- Increase `stockfish_depth` to 15-18
- Increase `games_per_elo_level`

**Balanced** (default):
- `enable_delta_analysis: true`
- `delta_sampling_rate: 3`
- `stockfish_depth: 12`
- `games_per_elo_level: 10`

---

## Storage Structure

### 4-Level Storage Hierarchy

#### Level 1: Per-Game Raw Metrics
**Entity Type**: `EntityType.CUSTOM`
**Entity ID**: `playstyle_eval_game_{i}`

Contains complete `GameMetrics.to_dict()`:
- All position metrics
- All captures
- All delta samples
- All pawn structure samples
- Move diversity counts
- Opening information (ECO, name)

**Use Case**: Deep analysis of individual games, debugging, position-by-position study

---

#### Level 2: Per-Game Computed Metrics
**Entity Type**: `EntityType.CUSTOM`
**Entity ID**: `playstyle_eval_game_{i}_computed`

Contains `ComputedGameMetrics.to_dict()`:
- White player computed metrics
- Black player computed metrics
- Game characteristics (avg tactical score, imbalance)

**Use Case**: Compare playing styles between players, analyze specific games

---

#### Level 3: Per-Cluster Aggregated Metrics
**Entity Type**: `EntityType.CLUSTER`
**Entity ID**: `{cluster_id}`

Contains `ClusterEvaluationMetrics.to_dict()`:
- Average metrics across all games
- ELO estimation
- Win/draw/loss rates
- Match results by opponent
- Opening frequency and top openings

**Use Case**: Track cluster performance over rounds, compare clusters

---

#### Level 4: Global Evaluation Summary
**Entity Type**: `EntityType.SERVER`
**Entity ID**: `playstyle_evaluator`

Contains:
- ELO rankings
- Tactical score rankings
- Playstyle divergence (how different clusters are)
- ELO spread
- Evaluation duration

**Use Case**: High-level overview, training progress tracking, quick comparisons

---

## Usage Examples

### Example 1: Identifying Tactical vs Positional Clusters

**Query**: Which cluster is more tactical?

**Analysis**:
1. Look at Level 4 summary → `tactical_rankings`
2. Compare `tactical_score`:
   - Cluster A: 0.72 (Very Tactical)
   - Cluster B: 0.48 (Very Positional)

**Supporting Evidence** (Level 3):
- **Cluster A**:
  - High `avg_attacked_material`
  - High `avg_legal_moves_middlegame`
  - High `avg_delta` (forcing positions)
  - Low `move_diversity_ratio` (focused attacks)
- **Cluster B**:
  - Lower attacked material
  - Moderate legal moves
  - Low `avg_delta` (flexible positions)
  - High `avg_pawn_rank` stability

---

### Example 2: Understanding Phase-Specific Style

**Query**: Does a cluster play differently in opening vs. endgame?

**Analysis** (Level 3):
```python
cluster_metrics = evaluation_results["cluster_metrics"]["cluster_tactical"]

opening_moves = cluster_metrics["avg_legal_moves_opening"]    # e.g., 24.5
middlegame_moves = cluster_metrics["avg_legal_moves_middlegame"]  # e.g., 32.8
endgame_moves = cluster_metrics["avg_legal_moves_endgame"]    # e.g., 18.2
```

**Interpretation**:
- High middlegame moves → Maintains complexity and options
- Drop in endgame → Focused on specific winning plans

---

### Example 3: Detecting Tipping Points in Specific Games

**Query**: Which game had the most critical decision points?

**Analysis** (Level 2):
```python
for i, game in enumerate(evaluation_results["computed_metrics"]):
    white_max_delta = game["white_metrics"]["max_delta"]
    black_max_delta = game["black_metrics"]["max_delta"]

    if white_max_delta > 3.0:
        print(f"Game {i}: White had critical decision (delta={white_max_delta})")
```

**Deep Dive** (Level 1):
```python
game_raw = evaluation_results["game_results"][i]
delta_metrics = game_raw["delta_metrics"]

# Find the ply with maximum delta
critical_positions = [dm for dm in delta_metrics
                     if dm["delta_white"] and dm["delta_white"] > 3.0]
```

---

### Example 4: Pawn Structure Analysis

**Query**: Does the tactical cluster sacrifice pawn structure for activity?

**Analysis** (Level 3):
```python
tactical_cluster = evaluation_results["cluster_metrics"]["cluster_tactical"]
positional_cluster = evaluation_results["cluster_metrics"]["cluster_positional"]

# Compare pawn structure
tactical_isolated = tactical_cluster["avg_isolated_pawns"]     # e.g., 2.3
positional_isolated = positional_cluster["avg_isolated_pawns"] # e.g., 0.8

tactical_pawn_rank = tactical_cluster["avg_pawn_rank"]     # e.g., 3.8 (advanced)
positional_pawn_rank = positional_cluster["avg_pawn_rank"] # e.g., 2.9 (solid)
```

**Interpretation**:
- Tactical cluster: More isolated pawns, more advanced → Sacrifices structure for activity
- Positional cluster: Fewer isolated pawns, less advanced → Maintains solid structure

---

## Formula Summary Table

| Metric | Formula | Range | Interpretation |
|--------|---------|-------|----------------|
| Attacked Material | Σ(piece_values of capturable pieces) | [0, ∞) | Higher = more tactical pressure |
| Legal Moves | Count of legal moves | [0, 218] | Higher = more flexibility |
| Material Captured | Σ(piece_values of captured pieces) | [0, 39] | Higher = more exchanges |
| Center Control | Count of attackers on {d4,d5,e4,e5} | [0, ∞) | Higher = central dominance |
| Attacks Metric | attacked_material / 39 | [0, ∞) | Normalized attack pressure |
| Moves Metric | min(1.0, legal_moves / 40) | [0, 1] | Normalized mobility |
| Material Metric | min(1.0, captures / 20) | [0, 1] | Normalized exchanges |
| Tactical Score | (attacks + moves + material) / 3 | [0, 1+] | Overall tactical tendency |
| Delta | \|eval(move1) - eval(move2)\| / 100 | [0, ∞) | Decision clarity |
| Avg Pawn Rank | Σ(pawn_ranks) / pawn_count | [1, 8] | Pawn advancement |
| Isolated Pawns | Count of pawns with no adjacent friendly pawns | [0, 8] | Structural weakness |
| Doubled Pawns | Σ(max(0, pawns_per_file - 1)) | [0, 7] | Structural compromise |
| Move Diversity | unique_destinations / total_moves | [0, 1] | Targeting variety |

---

## References

1. **Novachess.ai Methodology**: Original tactical vs. positional classification system
2. **Research Paper**: "Statistical analysis of chess games: space control and tipping points" (Barthelemy et al.)
   - Legal moves as style differentiator
   - Delta metric for tipping points
   - Pawn structure dynamics
3. **Implementation**: `server/evaluation/playstyle_metrics.py`

---

## Future Enhancements

Potential additions based on the research paper:

1. **Space Control Gradient**: Measure uniformity of piece placement (engines = uniform, humans = centralized)
2. **Wasserstein Distances**: Compare piece heatmaps between players (requires large datasets of same opening)
3. **Interaction Graphs**: Network analysis of piece-to-piece relationships
4. **Temporal Analysis**: Track metric evolution within a single game (already partially implemented via per-position storage)

These were not implemented due to:
- Computational complexity (Wasserstein)
- Data requirements (need many games with same opening)
- Unclear benefit for federated learning use case
- The current metrics already provide strong differentiation

---

## Glossary

- **Ply**: A single move by one player (half-move). 1 move = 2 plies (one for White, one for Black)
- **FEN**: Forsyth-Edwards Notation, a standard notation for describing a chess position
- **PGN**: Portable Game Notation, a standard format for recording chess games
- **ECO**: Encyclopedia of Chess Openings, a classification system for chess openings
- **Centipawn**: 1/100th of a pawn value, used in engine evaluations
- **Sparse Sampling**: Analyzing only a subset of positions (e.g., every 3rd) to reduce computational cost
- **Multi-PV**: Multi-Principal Variation, engine mode that shows top N moves instead of just one

---

*Document Version: 1.0*
*Last Updated: 2025-01-19*
*Author: Chess Federated Learning Team*
