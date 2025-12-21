# Federated Chess Learning - Metrics Visualization Summary

**Analysis Period:** Rounds 10-250
**Experiments Analyzed:** B1_full_sharing, B2_no_sharing, P1_share_early, P2_share_middle, P3_share_late, P4_share_backbone
**Total Plots Generated:** 104

---

## Cross-Experiment Comparison Plots (20 plots)

These plots compare all 6 experiments side-by-side:

### Model Divergence & Performance
1. **divergence_trajectories_all.png** - All layer groups' divergence evolution across experiments
2. **global_divergence_comparison.png** - Global mean divergence over training
3. **elo_comparison_all.png** - ELO progression for all experiments (tactical vs positional)
4. **final_divergence_comparison.png** - Layer-wise divergence at round 250
5. **elo_vs_divergence.png** - Scatter plot: ELO vs divergence relationship

### Behavioral Metrics
6. **move_type_differences_final.png** - Move type comparison (aggressive, captures, checks, quiet) at round 250
7. **behavioral_separation_all.png** - Cluster behavioral difference (aggressive move %) over time

### Cluster Averages (Average of Tactical + Positional)
8. **cluster_avg_elo_all.png** - Average ELO per cluster across all experiments
9. **cluster_avg_win_rate_all.png** - Average win rate per cluster across all experiments
10. **cluster_avg_legal_moves_all.png** - Average legal moves per cluster across all experiments
11. **cluster_avg_material_metrics_all.png** - Average attacked material and captures per cluster
12. **cluster_avg_center_control_all.png** - Average center control per cluster across all experiments
13. **cluster_avg_move_diversity_all.png** - Average move diversity ratio per cluster across all experiments

### Divergence Analysis (Detailed)
14. **divergence_per_round_all.png** - Global divergence per round (6 subplots, one per experiment)
15. **divergence_by_layer_group_all.png** - Layer group divergence per round (6 subplots)
16. **policy_head_divergence_over_rounds.png** - Policy head divergence evolution for all experiments
17. **value_head_divergence_over_rounds.png** - Value head divergence evolution for all experiments
18. **policy_vs_value_divergence_all.png** - Difference between policy and value head divergence
19. **early_vs_late_divergence_all.png** - Difference between late and early residual block divergence

---

## Per-Experiment Plots (83 plots = 6 experiments × ~14 plots each)

### Divergence & Model Analysis
- **divergence_heatmap_*.png** - Layer-wise divergence heatmap over rounds
- **weight_change_*.png** - Weight change magnitude by layer group
- **sparsity_evolution_*.png** - Model sparsity evolution
- **l2_norm_by_layer_*.png** - L2 norm by layer group

### Performance Metrics
- **elo_progression_*.png** - ELO improvement over training
- **win_rates_*.png** - Win rates vs Stockfish (1000, 1200, 1400)
- **win_draw_loss_*.png** - Win/Draw/Loss distribution over time

### Behavioral Metrics
- **move_types_evolution_*.png** - 4 subplots: aggressive%, captures%, checks%, quiet moves%
- **move_diversity_*.png** - Move diversity ratio evolution
- **unique_move_destinations_*.png** - Average unique squares targeted per game

### Game Phase Analysis
- **legal_moves_by_phase_*.png** - Legal moves in opening/middlegame/endgame
- **material_metrics_*.png** - Attacked material (threats) and captures
- **center_control_*.png** - Center control evolution

### Opening Analysis
- **opening_diversity_*.png** - Number of unique openings played

---

## Key Metrics Visualized

### Model Divergence Metrics
- Global mean divergence
- Per-layer-group divergence (input_block, early_residual, middle_residual, late_residual, policy_head, value_head)
- Cosine similarity
- L2 distance
- Divergence index

### Performance Metrics
- Estimated ELO
- Win rate
- Draw rate
- Loss rate
- Win rates vs specific Stockfish levels

### Behavioral Metrics
- Aggressive move percentage (captures + checks)
- Captures percentage
- Checks percentage
- Quiet moves percentage
- Pawn advances percentage
- Move diversity ratio
- Unique move destinations

### Game Analysis Metrics
- Legal moves (total, by phase: opening/middlegame/endgame)
- Attacked material (threat generation)
- Material captured (actual exchanges)
- Center control (d4, d5, e4, e5)

### Weight Statistics
- Mean relative weight change
- Global sparsity
- L2 norm by layer group

### Opening Metrics
- Number of unique openings (ECO codes)
- Opening frequency distribution

---

## How to Use These Plots

### 1. **Identify Best Experiment**
Start with:
- `elo_comparison_all.png` - Which experiment achieves highest ELO?
- `final_divergence_comparison.png` - Which shows proper divergence patterns?
- `behavioral_separation_all.png` - Which shows cluster differentiation?

### 2. **Understand Divergence Patterns**
For each experiment:
- `divergence_heatmap_*.png` - Are early layers staying similar? Are late layers diverging?
- Compare against hypothesis: early shared, late diverged

### 3. **Validate Learning**
- `elo_progression_*.png` - Is the model improving?
- `win_rates_*.png` - Performance against different strength levels
- `weight_change_*.png` - Are all layers learning?

### 4. **Analyze Behavioral Differences**
- `move_types_evolution_*.png` - Do clusters play differently?
- `move_diversity_*.png` - Targeting patterns
- `material_metrics_*.png` - Aggression vs solidity

### 5. **Cross-Experiment Insights**
- `elo_vs_divergence.png` - Is there a tradeoff between divergence and performance?
- `move_type_differences_final.png` - Which experiments show behavioral separation?

---

## Expected Patterns (Thesis Validation)

### ✅ Successful Experiment Should Show:
1. **Divergence**: Early layers stay similar (~0.1-0.3), late layers diverge (~0.4-0.7)
2. **ELO**: Steady improvement to 1400-1600+
3. **Behavioral**: 5-10% difference in aggressive move % between clusters
4. **Learning**: Consistent weight changes, no dead layers

### ❌ Failed Experiment Indicators:
1. **Divergence**: All layers diverge equally OR all stay similar
2. **ELO**: Plateaus below 1200 or doesn't improve
3. **Behavioral**: <2% difference between clusters
4. **Learning**: Some layers stop changing (dead layers)

---

## Plot File Naming Convention

```
{metric}_{experiment}.png         # Per-experiment plot
{metric}_all.png                  # Cross-experiment comparison
{metric}_final.png                # Final round (250) comparison
```

---

**Generated by:** `analyze_metrics.py`
**Date:** 2025-12-21
**Data Source:** `storage/metrics/`
