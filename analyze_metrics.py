#!/usr/bin/env python3
"""
Metrics Analysis and Visualization Script
Analyzes federated chess learning experiments and generates comprehensive plots.

Usage:
    python analyze_metrics.py

Output:
    - Creates plots/ directory with all visualizations
    - Uses data from rounds 10-250 only
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from collections import defaultdict

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Experiments to analyze
EXPERIMENTS = [
    'B1_full_sharing',
    'B2_no_sharing',
    'P1_share_early',
    'P2_share_middle',
    'P3_share_late',
    'P4_share_backbone'
]

# Cluster names
CLUSTERS = ['tactical', 'positional']

# Round range
MIN_ROUND = 10
MAX_ROUND = 350

# Layer groups for analysis
LAYER_GROUPS = ['input_block', 'early_residual', 'middle_residual', 'late_residual', 'policy_head', 'value_head']


class MetricsLoader:
    """Loads metrics data from storage directory."""

    def __init__(self, base_path: str = "storage/metrics"):
        self.base_path = Path(base_path)

    def load_evaluation_metrics(self, experiment: str, cluster: str, round_num: int) -> Dict:
        """Load evaluation metrics for a specific round."""
        file_path = self.base_path / experiment / cluster / f"evaluation_round_{round_num}.json"
        if not file_path.exists():
            return None
        with open(file_path) as f:
            return json.load(f)

    def load_move_types(self, experiment: str, cluster: str, round_num: int) -> Dict:
        """Load move type metrics for a specific round."""
        file_path = self.base_path / experiment / cluster / f"move_types_round_{round_num}.json"
        if not file_path.exists():
            return None
        with open(file_path) as f:
            return json.load(f)

    def load_weight_stats(self, experiment: str, cluster: str, round_num: int) -> Dict:
        """Load weight statistics for a specific round."""
        file_path = self.base_path / experiment / cluster / f"weight_stats_round_{round_num}.json"
        if not file_path.exists():
            return None
        with open(file_path) as f:
            return json.load(f)

    def load_divergence(self, experiment: str, round_num: int) -> Dict:
        """Load model divergence metrics for a specific round."""
        file_path = self.base_path / experiment / "model_divergence" / f"round_{round_num}.json"
        if not file_path.exists():
            return None
        with open(file_path) as f:
            return json.load(f)

    def get_available_rounds(self, experiment: str, cluster: str) -> List[int]:
        """Get list of available rounds for an experiment."""
        eval_dir = self.base_path / experiment / cluster
        if not eval_dir.exists():
            return []

        rounds = []
        for file in eval_dir.glob("evaluation_round_*.json"):
            try:
                round_num = int(file.stem.split('_')[-1])
                if MIN_ROUND <= round_num <= MAX_ROUND:
                    rounds.append(round_num)
            except ValueError:
                continue

        return sorted(rounds)


class MetricsAnalyzer:
    """Analyzes and visualizes metrics data."""

    def __init__(self, loader: MetricsLoader, output_dir: str = "plots"):
        self.loader = loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def save_csv(self, filename: str, data: pd.DataFrame):
        """Save data to CSV file."""
        csv_path = self.output_dir / f"{filename}.csv"
        data.to_csv(csv_path, index=False)
        print(f"  ✓ Saved CSV: {filename}.csv")

    def plot_divergence_heatmap(self, experiment: str):
        """Plot layer-wise divergence heatmap over time."""
        rounds = self.loader.get_available_rounds(experiment, 'tactical')
        if not rounds:
            print(f"No data for {experiment}")
            return

        # Collect divergence data
        divergence_data = {group: [] for group in LAYER_GROUPS}
        valid_rounds = []

        for round_num in rounds:
            div_data = self.loader.load_divergence(experiment, round_num)
            if div_data and 'per_group' in div_data:
                valid_rounds.append(round_num)
                for group in LAYER_GROUPS:
                    if group in div_data['per_group']:
                        divergence_data[group].append(
                            div_data['per_group'][group].get('mean_divergence_index', 0)
                        )
                    else:
                        divergence_data[group].append(0)

        if not valid_rounds:
            return

        # Save CSV
        csv_data = {'round': valid_rounds}
        csv_data.update(divergence_data)
        df = pd.DataFrame(csv_data)
        self.save_csv(f'divergence_heatmap_{experiment}', df)

        # Create heatmap
        fig, ax = plt.subplots(figsize=(14, 6))

        # Prepare data matrix
        matrix = np.array([divergence_data[group] for group in LAYER_GROUPS])

        # Plot
        im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')

        # Set ticks
        ax.set_yticks(range(len(LAYER_GROUPS)))
        ax.set_yticklabels(LAYER_GROUPS)

        # Set x-axis to show every 10th round
        x_ticks = range(0, len(valid_rounds), max(1, len(valid_rounds) // 10))
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([valid_rounds[i] for i in x_ticks])

        ax.set_xlabel('Training Round')
        ax.set_ylabel('Layer Group')
        ax.set_title(f'Layer-wise Divergence Evolution: {experiment}')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Divergence Index', rotation=270, labelpad=20)

        plt.tight_layout()
        plt.savefig(self.output_dir / f'divergence_heatmap_{experiment}.png', dpi=150)
        plt.close()

    def plot_divergence_trajectories(self):
        """Plot divergence trajectories for all experiments."""
        # Collect all data for CSV
        all_data = []

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        for idx, experiment in enumerate(EXPERIMENTS):
            ax = axes[idx]
            rounds = self.loader.get_available_rounds(experiment, 'tactical')

            if not rounds:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(experiment)
                continue

            # Collect data for each layer group
            for group in LAYER_GROUPS:
                group_divs = []
                valid_rounds = []

                for round_num in rounds:
                    div_data = self.loader.load_divergence(experiment, round_num)
                    if div_data and 'per_group' in div_data and group in div_data['per_group']:
                        div_val = div_data['per_group'][group].get('mean_divergence_index', 0)
                        group_divs.append(div_val)
                        valid_rounds.append(round_num)
                        all_data.append({'experiment': experiment, 'layer_group': group, 'round': round_num, 'divergence': div_val})

                if group_divs:
                    ax.plot(valid_rounds, group_divs, label=group, linewidth=2, alpha=0.8)

            ax.set_xlabel('Training Round')
            ax.set_ylabel('Divergence Index')
            ax.set_title(experiment)
            ax.legend(fontsize=8, loc='best')
            ax.grid(True, alpha=0.3)

        # Save CSV
        if all_data:
            df = pd.DataFrame(all_data)
            self.save_csv('divergence_trajectories_all', df)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'divergence_trajectories_all.png', dpi=150)
        plt.close()

    def plot_global_divergence_comparison(self):
        """Compare global divergence across all experiments."""
        all_data = []
        fig, ax = plt.subplots(figsize=(12, 6))

        for experiment in EXPERIMENTS:
            rounds = self.loader.get_available_rounds(experiment, 'tactical')
            if not rounds:
                continue

            global_divs = []
            valid_rounds = []

            for round_num in rounds:
                div_data = self.loader.load_divergence(experiment, round_num)
                if div_data and 'global' in div_data:
                    div_val = div_data['global'].get('mean_divergence', 0)
                    global_divs.append(div_val)
                    valid_rounds.append(round_num)
                    all_data.append({'experiment': experiment, 'round': round_num, 'global_divergence': div_val})

            if global_divs:
                ax.plot(valid_rounds, global_divs, label=experiment, linewidth=2.5, marker='o',
                       markersize=3, alpha=0.8)

        # Save CSV
        if all_data:
            df = pd.DataFrame(all_data)
            self.save_csv('global_divergence_comparison', df)

        ax.set_xlabel('Training Round', fontsize=12)
        ax.set_ylabel('Global Mean Divergence', fontsize=12)
        ax.set_title('Global Divergence Evolution: All Experiments', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'global_divergence_comparison.png', dpi=150)
        plt.close()

    def plot_elo_progression(self, experiment: str):
        """Plot ELO progression for both clusters."""
        all_data = []
        fig, ax = plt.subplots(figsize=(12, 6))

        for cluster in CLUSTERS:
            rounds = self.loader.get_available_rounds(experiment, cluster)
            if not rounds:
                continue

            elos = []
            valid_rounds = []

            for round_num in rounds:
                eval_data = self.loader.load_evaluation_metrics(experiment, cluster, round_num)
                if eval_data and 'estimated_elo' in eval_data:
                    elo = eval_data['estimated_elo']
                    elos.append(elo)
                    valid_rounds.append(round_num)
                    all_data.append({'cluster': cluster, 'round': round_num, 'elo': elo})

            if elos:
                ax.plot(valid_rounds, elos, label=f'{cluster.capitalize()} Cluster',
                       linewidth=2.5, marker='o', markersize=4, alpha=0.8)

        # Save CSV
        if all_data:
            df = pd.DataFrame(all_data)
            self.save_csv(f'elo_progression_{experiment}', df)

        ax.set_xlabel('Training Round', fontsize=12)
        ax.set_ylabel('Estimated ELO', fontsize=12)
        ax.set_title(f'ELO Progression: {experiment}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / f'elo_progression_{experiment}.png', dpi=150)
        plt.close()

    def plot_elo_comparison_all(self):
        """Plot ELO progression for all experiments."""
        all_data = []
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        for idx, experiment in enumerate(EXPERIMENTS):
            ax = axes[idx]

            for cluster in CLUSTERS:
                rounds = self.loader.get_available_rounds(experiment, cluster)
                if not rounds:
                    continue

                elos = []
                valid_rounds = []

                for round_num in rounds:
                    eval_data = self.loader.load_evaluation_metrics(experiment, cluster, round_num)
                    if eval_data and 'estimated_elo' in eval_data:
                        elo = eval_data['estimated_elo']
                        elos.append(elo)
                        valid_rounds.append(round_num)
                        all_data.append({'experiment': experiment, 'cluster': cluster, 'round': round_num, 'elo': elo})

                if elos:
                    ax.plot(valid_rounds, elos, label=cluster.capitalize(),
                           linewidth=2, marker='o', markersize=3, alpha=0.8)

            ax.set_xlabel('Round')
            ax.set_ylabel('ELO')
            ax.set_title(experiment)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        # Save CSV
        if all_data:
            df = pd.DataFrame(all_data)
            self.save_csv('elo_comparison_all', df)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'elo_comparison_all.png', dpi=150)
        plt.close()

    def plot_win_rates(self, experiment: str):
        """Plot win rates against different Stockfish levels."""
        all_data = []
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        for cluster_idx, cluster in enumerate(CLUSTERS):
            ax = ax1 if cluster_idx == 0 else ax2
            rounds = self.loader.get_available_rounds(experiment, cluster)

            if not rounds:
                continue

            # Track win rates per opponent
            opponent_win_rates = defaultdict(lambda: {'rounds': [], 'win_rates': []})

            for round_num in rounds:
                eval_data = self.loader.load_evaluation_metrics(experiment, cluster, round_num)
                if eval_data and 'match_results' in eval_data:
                    for match in eval_data['match_results']:
                        opponent = match['opponent']
                        win_rate = match.get('win_rate', 0)
                        opponent_win_rates[opponent]['rounds'].append(round_num)
                        opponent_win_rates[opponent]['win_rates'].append(win_rate)
                        all_data.append({'cluster': cluster, 'opponent': opponent, 'round': round_num, 'win_rate': win_rate})

            # Plot each opponent
            for opponent, data in sorted(opponent_win_rates.items()):
                ax.plot(data['rounds'], data['win_rates'],
                       label=opponent, linewidth=2, marker='o', markersize=3, alpha=0.8)

            ax.set_xlabel('Training Round', fontsize=11)
            ax.set_ylabel('Win Rate (%)', fontsize=11)
            ax.set_title(f'{cluster.capitalize()} Cluster', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9, loc='best')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-5, 105)

        if all_data:
            df = pd.DataFrame(all_data)
            self.save_csv(f'win_rates_{experiment}', df)

        fig.suptitle(f'Win Rates vs Stockfish: {experiment}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / f'win_rates_{experiment}.png', dpi=150)
        plt.close()

    def plot_win_draw_loss_rates(self, experiment: str):
        """Plot win/draw/loss rates evolution."""
        all_data = []
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        for cluster_idx, cluster in enumerate(CLUSTERS):
            ax = ax1 if cluster_idx == 0 else ax2
            rounds = self.loader.get_available_rounds(experiment, cluster)

            if not rounds:
                continue

            win_rates = []
            draw_rates = []
            loss_rates = []
            valid_rounds = []

            for round_num in rounds:
                eval_data = self.loader.load_evaluation_metrics(experiment, cluster, round_num)
                if eval_data:
                    wr = eval_data.get('win_rate', 0) * 100
                    dr = eval_data.get('draw_rate', 0) * 100
                    lr = eval_data.get('loss_rate', 0) * 100
                    win_rates.append(wr)
                    draw_rates.append(dr)
                    loss_rates.append(lr)
                    valid_rounds.append(round_num)
                    all_data.append({'cluster': cluster, 'round': round_num, 'win_rate': wr, 'draw_rate': dr, 'loss_rate': lr})

            if valid_rounds:
                ax.fill_between(valid_rounds, 0, win_rates, label='Win', alpha=0.7, color='green')
                ax.fill_between(valid_rounds, win_rates,
                               [w + d for w, d in zip(win_rates, draw_rates)],
                               label='Draw', alpha=0.7, color='gray')
                ax.fill_between(valid_rounds,
                               [w + d for w, d in zip(win_rates, draw_rates)],
                               100, label='Loss', alpha=0.7, color='red')

            ax.set_xlabel('Training Round', fontsize=11)
            ax.set_ylabel('Percentage (%)', fontsize=11)
            ax.set_title(f'{cluster.capitalize()} Cluster', fontsize=12, fontweight='bold')
            ax.legend(fontsize=10, loc='best')
            ax.grid(True, alpha=0.3, axis='x')
            ax.set_ylim(0, 100)

        if all_data:
            df = pd.DataFrame(all_data)
            self.save_csv(f'win_draw_loss_{experiment}', df)

        fig.suptitle(f'Win/Draw/Loss Distribution: {experiment}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / f'win_draw_loss_{experiment}.png', dpi=150)
        plt.close()

    def plot_move_types_evolution(self, experiment: str):
        """Plot move type percentages over time."""
        all_data = []
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        move_type_keys = [
            ('aggressive_pct', 'Aggressive Moves %'),
            ('captures_pct', 'Captures %'),
            ('checks_pct', 'Checks %'),
            ('quiet_moves_pct', 'Quiet Moves %')
        ]

        for idx, (key, title) in enumerate(move_type_keys):
            ax = axes[idx // 2, idx % 2]

            for cluster in CLUSTERS:
                rounds = self.loader.get_available_rounds(experiment, cluster)
                if not rounds:
                    continue

                values = []
                valid_rounds = []

                for round_num in rounds:
                    move_data = self.loader.load_move_types(experiment, cluster, round_num)
                    if move_data and 'percentages' in move_data and key in move_data['percentages']:
                        val = move_data['percentages'][key]
                        values.append(val)
                        valid_rounds.append(round_num)
                        all_data.append({'cluster': cluster, 'move_type': key, 'round': round_num, 'percentage': val})

                if values:
                    ax.plot(valid_rounds, values, label=cluster.capitalize(),
                           linewidth=2.5, marker='o', markersize=4, alpha=0.8)

            ax.set_xlabel('Training Round', fontsize=11)
            ax.set_ylabel('Percentage (%)', fontsize=11)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

        if all_data:
            df = pd.DataFrame(all_data)
            self.save_csv(f'move_types_evolution_{experiment}', df)

        fig.suptitle(f'Move Type Evolution: {experiment}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / f'move_types_evolution_{experiment}.png', dpi=150)
        plt.close()

    def plot_move_diversity(self, experiment: str):
        """Plot move diversity ratio over time."""
        all_data = []
        fig, ax = plt.subplots(figsize=(12, 6))

        for cluster in CLUSTERS:
            rounds = self.loader.get_available_rounds(experiment, cluster)
            if not rounds:
                continue

            diversity = []
            valid_rounds = []

            for round_num in rounds:
                eval_data = self.loader.load_evaluation_metrics(experiment, cluster, round_num)
                if eval_data and 'avg_move_diversity_ratio' in eval_data:
                    div = eval_data['avg_move_diversity_ratio']
                    diversity.append(div)
                    valid_rounds.append(round_num)
                    all_data.append({'cluster': cluster, 'round': round_num, 'diversity_ratio': div})

            if diversity:
                ax.plot(valid_rounds, diversity, label=cluster.capitalize(),
                       linewidth=2.5, marker='o', markersize=4, alpha=0.8)

        if all_data:
            df = pd.DataFrame(all_data)
            self.save_csv(f'move_diversity_{experiment}', df)

        ax.set_xlabel('Training Round', fontsize=12)
        ax.set_ylabel('Move Diversity Ratio', fontsize=12)
        ax.set_title(f'Move Diversity Evolution: {experiment}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / f'move_diversity_{experiment}.png', dpi=150)
        plt.close()

    def plot_weight_change_magnitude(self, experiment: str):
        """Plot weight change magnitude per layer group."""
        all_data = []
        fig, ax = plt.subplots(figsize=(14, 6))

        # We'll average across both clusters
        cluster = CLUSTERS[0]  # Use tactical as representative
        rounds = self.loader.get_available_rounds(experiment, cluster)

        if not rounds:
            return

        # Collect weight change data
        for group in LAYER_GROUPS:
            changes = []
            valid_rounds = []

            for round_num in rounds[1:]:  # Skip first round (no previous to compare)
                weight_data = self.loader.load_weight_stats(experiment, cluster, round_num)
                if weight_data and 'per_group' in weight_data and group in weight_data['per_group']:
                    if 'mean_relative_change' in weight_data['per_group'][group]:
                        change = weight_data['per_group'][group]['mean_relative_change']
                        changes.append(change)
                        valid_rounds.append(round_num)
                        all_data.append({'layer_group': group, 'round': round_num, 'mean_relative_change': change})

            if changes:
                ax.plot(valid_rounds, changes, label=group, linewidth=2, marker='o',
                       markersize=3, alpha=0.8)

        if all_data:
            df = pd.DataFrame(all_data)
            self.save_csv(f'weight_change_{experiment}', df)

        ax.set_xlabel('Training Round', fontsize=12)
        ax.set_ylabel('Mean Relative Weight Change', fontsize=12)
        ax.set_title(f'Weight Change Magnitude by Layer Group: {experiment}',
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        plt.tight_layout()
        plt.savefig(self.output_dir / f'weight_change_{experiment}.png', dpi=150)
        plt.close()

    def plot_sparsity_evolution(self, experiment: str):
        """Plot model sparsity evolution."""
        all_data = []
        fig, ax = plt.subplots(figsize=(12, 6))

        for cluster in CLUSTERS:
            rounds = self.loader.get_available_rounds(experiment, cluster)
            if not rounds:
                continue

            sparsity = []
            valid_rounds = []

            for round_num in rounds:
                weight_data = self.loader.load_weight_stats(experiment, cluster, round_num)
                if weight_data and 'summary' in weight_data and 'global_sparsity' in weight_data['summary']:
                    sp = weight_data['summary']['global_sparsity'] * 100
                    sparsity.append(sp)
                    valid_rounds.append(round_num)
                    all_data.append({'cluster': cluster, 'round': round_num, 'sparsity': sp})

            if sparsity:
                ax.plot(valid_rounds, sparsity, label=cluster.capitalize(),
                       linewidth=2.5, marker='o', markersize=4, alpha=0.8)

        if all_data:
            df = pd.DataFrame(all_data)
            self.save_csv(f'sparsity_evolution_{experiment}', df)

        ax.set_xlabel('Training Round', fontsize=12)
        ax.set_ylabel('Global Sparsity (%)', fontsize=12)
        ax.set_title(f'Model Sparsity Evolution: {experiment}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / f'sparsity_evolution_{experiment}.png', dpi=150)
        plt.close()

    def plot_final_divergence_comparison(self):
        """Compare final divergence across all experiments."""
        final_round = MAX_ROUND

        experiments = []
        global_divs = []
        layer_divs = {group: [] for group in LAYER_GROUPS}
        all_data = []

        for experiment in EXPERIMENTS:
            div_data = self.loader.load_divergence(experiment, final_round)
            if div_data and 'global' in div_data:
                experiments.append(experiment)
                global_divs.append(div_data['global'].get('mean_divergence', 0))

                for group in LAYER_GROUPS:
                    if 'per_group' in div_data and group in div_data['per_group']:
                        div_val = div_data['per_group'][group].get('mean_divergence_index', 0)
                        layer_divs[group].append(div_val)
                        all_data.append({'experiment': experiment, 'layer_group': group, 'divergence_index': div_val})
                    else:
                        layer_divs[group].append(0)
                        all_data.append({'experiment': experiment, 'layer_group': group, 'divergence_index': 0})

        if not experiments:
            return

        if all_data:
            df = pd.DataFrame(all_data)
            self.save_csv('final_divergence_comparison', df)

        # Create grouped bar chart
        fig, ax = plt.subplots(figsize=(14, 7))

        x = np.arange(len(experiments))
        width = 0.12

        for idx, group in enumerate(LAYER_GROUPS):
            offset = (idx - len(LAYER_GROUPS) / 2) * width + width / 2
            ax.bar(x + offset, layer_divs[group], width, label=group, alpha=0.8)

        ax.set_xlabel('Experiment', fontsize=12)
        ax.set_ylabel('Divergence Index', fontsize=12)
        ax.set_title(f'Final Divergence Comparison (Round {final_round})',
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(experiments, rotation=45, ha='right')
        ax.legend(fontsize=9, loc='best', ncol=2)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'final_divergence_comparison.png', dpi=150)
        plt.close()

    def plot_elo_vs_divergence(self):
        """Scatter plot of ELO vs Divergence for all experiments."""
        final_round = MAX_ROUND

        data_points = []

        for experiment in EXPERIMENTS:
            # Get divergence
            div_data = self.loader.load_divergence(experiment, final_round)
            if not div_data or 'global' not in div_data:
                continue

            divergence = div_data['global'].get('mean_divergence', 0)

            # Get ELO for both clusters
            for cluster in CLUSTERS:
                eval_data = self.loader.load_evaluation_metrics(experiment, cluster, final_round)
                if eval_data and 'estimated_elo' in eval_data:
                    elo = eval_data['estimated_elo']
                    data_points.append({
                        'experiment': experiment,
                        'cluster': cluster,
                        'divergence': divergence,
                        'elo': elo
                    })

        if not data_points:
            return

        # Save CSV
        df = pd.DataFrame(data_points)
        self.save_csv('elo_vs_divergence', df)

        # Create scatter plot
        fig, ax = plt.subplots(figsize=(12, 8))

        # Group by experiment
        exp_colors = plt.cm.tab10(np.linspace(0, 1, len(EXPERIMENTS)))
        exp_to_color = {exp: exp_colors[i] for i, exp in enumerate(EXPERIMENTS)}

        for point in data_points:
            marker = 'o' if point['cluster'] == 'tactical' else 's'
            ax.scatter(point['divergence'], point['elo'],
                      c=[exp_to_color[point['experiment']]],
                      marker=marker, s=150, alpha=0.7, edgecolors='black', linewidth=1.5)

        # Add labels for each point
        for point in data_points:
            ax.annotate(f"{point['experiment']}\n{point['cluster'][:3]}",
                       (point['divergence'], point['elo']),
                       fontsize=7, ha='center', va='bottom')

        ax.set_xlabel('Global Mean Divergence', fontsize=12)
        ax.set_ylabel('Estimated ELO', fontsize=12)
        ax.set_title(f'ELO vs Divergence (Round {final_round})', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Create legend
        legend_elements = [
            mpatches.Patch(color=exp_to_color[exp], label=exp, alpha=0.7)
            for exp in EXPERIMENTS if exp in exp_to_color
        ]
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                         markerfacecolor='gray', markersize=8,
                                         label='Tactical', linestyle=''))
        legend_elements.append(plt.Line2D([0], [0], marker='s', color='w',
                                         markerfacecolor='gray', markersize=8,
                                         label='Positional', linestyle=''))
        ax.legend(handles=legend_elements, fontsize=9, loc='best')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'elo_vs_divergence.png', dpi=150)
        plt.close()

    def plot_opening_diversity(self, experiment: str):
        """Plot opening diversity over time."""
        all_data = []
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        for cluster_idx, cluster in enumerate(CLUSTERS):
            ax = ax1 if cluster_idx == 0 else ax2
            rounds = self.loader.get_available_rounds(experiment, cluster)

            if not rounds:
                continue

            num_openings = []
            valid_rounds = []

            for round_num in rounds:
                eval_data = self.loader.load_evaluation_metrics(experiment, cluster, round_num)
                if eval_data and 'opening_frequency' in eval_data:
                    num_open = len(eval_data['opening_frequency'])
                    num_openings.append(num_open)
                    valid_rounds.append(round_num)
                    all_data.append({'cluster': cluster, 'round': round_num, 'num_openings': num_open})

            if num_openings:
                ax.plot(valid_rounds, num_openings, linewidth=2.5, marker='o',
                       markersize=4, alpha=0.8, color='steelblue')

            ax.set_xlabel('Training Round', fontsize=11)
            ax.set_ylabel('Number of Unique Openings', fontsize=11)
            ax.set_title(f'{cluster.capitalize()} Cluster', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)

        if all_data:
            df = pd.DataFrame(all_data)
            self.save_csv(f'opening_diversity_{experiment}', df)

        fig.suptitle(f'Opening Diversity: {experiment}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / f'opening_diversity_{experiment}.png', dpi=150)
        plt.close()

    def plot_legal_moves_by_phase(self, experiment: str):
        """Plot legal moves by game phase (opening/middlegame/endgame)."""
        all_data = []
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        phases = [
            ('avg_legal_moves_opening', 'Opening', axes[0]),
            ('avg_legal_moves_middlegame', 'Middlegame', axes[1]),
            ('avg_legal_moves_endgame', 'Endgame', axes[2])
        ]

        for metric_key, phase_name, ax in phases:
            for cluster in CLUSTERS:
                rounds = self.loader.get_available_rounds(experiment, cluster)
                if not rounds:
                    continue

                values = []
                valid_rounds = []

                for round_num in rounds:
                    eval_data = self.loader.load_evaluation_metrics(experiment, cluster, round_num)
                    if eval_data and metric_key in eval_data:
                        val = eval_data[metric_key]
                        values.append(val)
                        valid_rounds.append(round_num)
                        all_data.append({'cluster': cluster, 'phase': phase_name, 'round': round_num, 'avg_legal_moves': val})

                if values:
                    ax.plot(valid_rounds, values, label=cluster.capitalize(),
                           linewidth=2.5, marker='o', markersize=4, alpha=0.8)

            ax.set_xlabel('Training Round', fontsize=11)
            ax.set_ylabel('Avg Legal Moves', fontsize=11)
            ax.set_title(phase_name, fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

        if all_data:
            df = pd.DataFrame(all_data)
            self.save_csv(f'legal_moves_by_phase_{experiment}', df)

        fig.suptitle(f'Legal Moves by Game Phase: {experiment}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / f'legal_moves_by_phase_{experiment}.png', dpi=150)
        plt.close()

    def plot_attacked_material_and_captures(self, experiment: str):
        """Plot attacked material and actual captures over time."""
        all_data = []
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        for cluster in CLUSTERS:
            rounds = self.loader.get_available_rounds(experiment, cluster)
            if not rounds:
                continue

            attacked_mat = []
            captures = []
            valid_rounds = []

            for round_num in rounds:
                eval_data = self.loader.load_evaluation_metrics(experiment, cluster, round_num)
                if eval_data:
                    atk_mat = eval_data.get('avg_attacked_material', 0)
                    cap = eval_data.get('avg_captures', 0)
                    if 'avg_attacked_material' in eval_data:
                        attacked_mat.append(atk_mat)
                    if 'avg_captures' in eval_data:
                        captures.append(cap)
                    valid_rounds.append(round_num)
                    all_data.append({'cluster': cluster, 'round': round_num, 'attacked_material': atk_mat, 'captures': cap})

            if attacked_mat:
                ax1.plot(valid_rounds, attacked_mat, label=cluster.capitalize(),
                        linewidth=2.5, marker='o', markersize=4, alpha=0.8)
            if captures:
                ax2.plot(valid_rounds, captures, label=cluster.capitalize(),
                        linewidth=2.5, marker='o', markersize=4, alpha=0.8)

        if all_data:
            df = pd.DataFrame(all_data)
            self.save_csv(f'material_metrics_{experiment}', df)

        ax1.set_xlabel('Training Round', fontsize=11)
        ax1.set_ylabel('Avg Attacked Material', fontsize=11)
        ax1.set_title('Attacked Material (Threats)', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        ax2.set_xlabel('Training Round', fontsize=11)
        ax2.set_ylabel('Avg Captures', fontsize=11)
        ax2.set_title('Material Captured', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        fig.suptitle(f'Material Metrics: {experiment}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / f'material_metrics_{experiment}.png', dpi=150)
        plt.close()

    def plot_center_control(self, experiment: str):
        """Plot center control evolution."""
        all_data = []
        fig, ax = plt.subplots(figsize=(12, 6))

        for cluster in CLUSTERS:
            rounds = self.loader.get_available_rounds(experiment, cluster)
            if not rounds:
                continue

            center_control = []
            valid_rounds = []

            for round_num in rounds:
                eval_data = self.loader.load_evaluation_metrics(experiment, cluster, round_num)
                if eval_data and 'avg_center_control' in eval_data:
                    cc = eval_data['avg_center_control']
                    center_control.append(cc)
                    valid_rounds.append(round_num)
                    all_data.append({'cluster': cluster, 'round': round_num, 'center_control': cc})

            if center_control:
                ax.plot(valid_rounds, center_control, label=cluster.capitalize(),
                       linewidth=2.5, marker='o', markersize=4, alpha=0.8)

        if all_data:
            df = pd.DataFrame(all_data)
            self.save_csv(f'center_control_{experiment}', df)

        ax.set_xlabel('Training Round', fontsize=12)
        ax.set_ylabel('Avg Center Control', fontsize=12)
        ax.set_title(f'Center Control Evolution: {experiment}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / f'center_control_{experiment}.png', dpi=150)
        plt.close()

    def plot_unique_move_destinations(self, experiment: str):
        """Plot unique move destinations (targeting variety)."""
        all_data = []
        fig, ax = plt.subplots(figsize=(12, 6))

        for cluster in CLUSTERS:
            rounds = self.loader.get_available_rounds(experiment, cluster)
            if not rounds:
                continue

            unique_dests = []
            valid_rounds = []

            for round_num in rounds:
                eval_data = self.loader.load_evaluation_metrics(experiment, cluster, round_num)
                if eval_data and 'avg_unique_move_destinations' in eval_data:
                    ud = eval_data['avg_unique_move_destinations']
                    unique_dests.append(ud)
                    valid_rounds.append(round_num)
                    all_data.append({'cluster': cluster, 'round': round_num, 'unique_destinations': ud})

            if unique_dests:
                ax.plot(valid_rounds, unique_dests, label=cluster.capitalize(),
                       linewidth=2.5, marker='o', markersize=4, alpha=0.8)

        if all_data:
            df = pd.DataFrame(all_data)
            self.save_csv(f'unique_move_destinations_{experiment}', df)

        ax.set_xlabel('Training Round', fontsize=12)
        ax.set_ylabel('Avg Unique Move Destinations', fontsize=12)
        ax.set_title(f'Move Targeting Variety: {experiment}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / f'unique_move_destinations_{experiment}.png', dpi=150)
        plt.close()

    def plot_l2_norm_by_layer(self, experiment: str):
        """Plot L2 norm evolution by layer group."""
        all_data = []
        fig, ax = plt.subplots(figsize=(14, 6))

        cluster = CLUSTERS[0]  # Use tactical as representative
        rounds = self.loader.get_available_rounds(experiment, cluster)

        if not rounds:
            return

        for group in LAYER_GROUPS:
            norms = []
            valid_rounds = []

            for round_num in rounds:
                weight_data = self.loader.load_weight_stats(experiment, cluster, round_num)
                if weight_data and 'per_group' in weight_data and group in weight_data['per_group']:
                    if 'mean_l2_norm' in weight_data['per_group'][group]:
                        l2 = weight_data['per_group'][group]['mean_l2_norm']
                        norms.append(l2)
                        valid_rounds.append(round_num)
                        all_data.append({'layer_group': group, 'round': round_num, 'l2_norm': l2})

            if norms:
                ax.plot(valid_rounds, norms, label=group, linewidth=2, marker='o',
                       markersize=3, alpha=0.8)

        if all_data:
            df = pd.DataFrame(all_data)
            self.save_csv(f'l2_norm_by_layer_{experiment}', df)

        ax.set_xlabel('Training Round', fontsize=12)
        ax.set_ylabel('Mean L2 Norm', fontsize=12)
        ax.set_title(f'Weight L2 Norm by Layer Group: {experiment}',
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / f'l2_norm_by_layer_{experiment}.png', dpi=150)
        plt.close()

    def plot_move_type_differences(self):
        """Plot move type differences between clusters at round 250."""
        final_round = MAX_ROUND
        all_data = []

        move_types = ['aggressive_pct', 'captures_pct', 'checks_pct', 'quiet_moves_pct']

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        for idx, move_type in enumerate(move_types):
            ax = axes[idx]

            experiments = []
            tactical_vals = []
            positional_vals = []
            differences = []

            for experiment in EXPERIMENTS:
                tac_data = self.loader.load_move_types(experiment, 'tactical', final_round)
                pos_data = self.loader.load_move_types(experiment, 'positional', final_round)

                if tac_data and pos_data:
                    if 'percentages' in tac_data and move_type in tac_data['percentages']:
                        if 'percentages' in pos_data and move_type in pos_data['percentages']:
                            experiments.append(experiment)
                            tac_val = tac_data['percentages'][move_type]
                            pos_val = pos_data['percentages'][move_type]
                            tactical_vals.append(tac_val)
                            positional_vals.append(pos_val)
                            differences.append(tac_val - pos_val)
                            all_data.append({'experiment': experiment, 'move_type': move_type, 'tactical': tac_val, 'positional': pos_val, 'difference': tac_val - pos_val})

            if experiments:
                x = np.arange(len(experiments))
                width = 0.35

                ax.bar(x - width/2, tactical_vals, width, label='Tactical', alpha=0.8)
                ax.bar(x + width/2, positional_vals, width, label='Positional', alpha=0.8)

                ax.set_xlabel('Experiment', fontsize=11)
                ax.set_ylabel('Percentage (%)', fontsize=11)
                ax.set_title(move_type.replace('_', ' ').title(), fontsize=12, fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(experiments, rotation=45, ha='right', fontsize=9)
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3, axis='y')

        if all_data:
            df = pd.DataFrame(all_data)
            self.save_csv('move_type_differences_final', df)

        fig.suptitle(f'Move Type Comparison at Round {final_round}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'move_type_differences_final.png', dpi=150)
        plt.close()

    def plot_cluster_behavioral_difference(self):
        """Plot behavioral difference between clusters over time for all experiments."""
        all_data = []
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        for idx, experiment in enumerate(EXPERIMENTS):
            ax = axes[idx]
            rounds = self.loader.get_available_rounds(experiment, 'tactical')

            if not rounds:
                continue

            aggressive_diff = []
            valid_rounds = []

            for round_num in rounds:
                tac_data = self.loader.load_move_types(experiment, 'tactical', round_num)
                pos_data = self.loader.load_move_types(experiment, 'positional', round_num)

                if tac_data and pos_data:
                    if 'percentages' in tac_data and 'percentages' in pos_data:
                        tac_agg = tac_data['percentages'].get('aggressive_pct', 0)
                        pos_agg = pos_data['percentages'].get('aggressive_pct', 0)
                        diff = abs(tac_agg - pos_agg)
                        aggressive_diff.append(diff)
                        valid_rounds.append(round_num)
                        all_data.append({'experiment': experiment, 'round': round_num, 'difference': diff})

            if aggressive_diff:
                ax.plot(valid_rounds, aggressive_diff, linewidth=2.5, marker='o',
                       markersize=4, alpha=0.8, color='darkviolet')
                ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='No difference')

            ax.set_xlabel('Training Round', fontsize=10)
            ax.set_ylabel('|Tactical - Positional| (%)', fontsize=10)
            ax.set_title(experiment, fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

        if all_data:
            df = pd.DataFrame(all_data)
            self.save_csv('behavioral_separation_all', df)

        fig.suptitle('Behavioral Separation: Aggressive Move % Difference', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'behavioral_separation_all.png', dpi=150)
        plt.close()

    def plot_cluster_averages_legal_moves(self):
        """Plot average legal moves per cluster across all experiments."""
        all_data = []
        fig, ax = plt.subplots(figsize=(14, 8))

        for experiment in EXPERIMENTS:
            # Calculate average across both clusters
            all_legal_moves = []
            valid_rounds = []

            for cluster in CLUSTERS:
                rounds = self.loader.get_available_rounds(experiment, cluster)
                if not rounds:
                    continue

                for round_num in rounds:
                    eval_data = self.loader.load_evaluation_metrics(experiment, cluster, round_num)
                    if eval_data and 'avg_legal_moves' in eval_data:
                        all_legal_moves.append(eval_data['avg_legal_moves'])
                        valid_rounds.append(round_num)

            if all_legal_moves:
                # Group by round and average
                round_to_moves = defaultdict(list)
                for round_num, moves in zip(valid_rounds, all_legal_moves):
                    round_to_moves[round_num].append(moves)

                rounds_sorted = sorted(round_to_moves.keys())
                avg_moves = [np.mean(round_to_moves[r]) for r in rounds_sorted]

                for r, avg in zip(rounds_sorted, avg_moves):
                    all_data.append({'experiment': experiment, 'round': r, 'avg_legal_moves': avg})

                ax.plot(rounds_sorted, avg_moves, label=experiment,
                       linewidth=2.5, marker='o', markersize=4, alpha=0.8)

        if all_data:
            df = pd.DataFrame(all_data)
            self.save_csv('cluster_avg_legal_moves_all', df)

        ax.set_xlabel('Training Round', fontsize=12)
        ax.set_ylabel('Average Legal Moves (Both Clusters)', fontsize=12)
        ax.set_title('Average Legal Moves per Cluster: All Experiments', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'cluster_avg_legal_moves_all.png', dpi=150)
        plt.close()

    def plot_cluster_averages_material_metrics(self):
        """Plot average material metrics per cluster across all experiments."""
        all_data = []
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        for experiment in EXPERIMENTS:
            # Attacked material
            attacked_data = defaultdict(list)
            captures_data = defaultdict(list)

            for cluster in CLUSTERS:
                rounds = self.loader.get_available_rounds(experiment, cluster)
                if not rounds:
                    continue

                for round_num in rounds:
                    eval_data = self.loader.load_evaluation_metrics(experiment, cluster, round_num)
                    if eval_data:
                        if 'avg_attacked_material' in eval_data:
                            attacked_data[round_num].append(eval_data['avg_attacked_material'])
                        if 'avg_captures' in eval_data:
                            captures_data[round_num].append(eval_data['avg_captures'])

            # Plot attacked material
            if attacked_data:
                rounds_sorted = sorted(attacked_data.keys())
                avg_attacked = [np.mean(attacked_data[r]) for r in rounds_sorted]
                ax1.plot(rounds_sorted, avg_attacked, label=experiment,
                        linewidth=2.5, marker='o', markersize=4, alpha=0.8)
                for r, avg in zip(rounds_sorted, avg_attacked):
                    all_data.append({'experiment': experiment, 'round': r, 'avg_attacked_material': avg, 'avg_captures': np.mean(captures_data[r]) if r in captures_data else 0})

            # Plot captures
            if captures_data:
                rounds_sorted = sorted(captures_data.keys())
                avg_captures = [np.mean(captures_data[r]) for r in rounds_sorted]
                ax2.plot(rounds_sorted, avg_captures, label=experiment,
                        linewidth=2.5, marker='o', markersize=4, alpha=0.8)

        if all_data:
            df = pd.DataFrame(all_data)
            self.save_csv('cluster_avg_material_metrics_all', df)

        ax1.set_xlabel('Training Round', fontsize=11)
        ax1.set_ylabel('Avg Attacked Material', fontsize=11)
        ax1.set_title('Average Attacked Material per Cluster', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=9, loc='best')
        ax1.grid(True, alpha=0.3)

        ax2.set_xlabel('Training Round', fontsize=11)
        ax2.set_ylabel('Avg Captures', fontsize=11)
        ax2.set_title('Average Captures per Cluster', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=9, loc='best')
        ax2.grid(True, alpha=0.3)

        fig.suptitle('Material Metrics (Cluster Averages): All Experiments', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'cluster_avg_material_metrics_all.png', dpi=150)
        plt.close()

    def plot_cluster_averages_center_control(self):
        """Plot average center control per cluster across all experiments."""
        all_data = []
        fig, ax = plt.subplots(figsize=(14, 8))

        for experiment in EXPERIMENTS:
            center_data = defaultdict(list)

            for cluster in CLUSTERS:
                rounds = self.loader.get_available_rounds(experiment, cluster)
                if not rounds:
                    continue

                for round_num in rounds:
                    eval_data = self.loader.load_evaluation_metrics(experiment, cluster, round_num)
                    if eval_data and 'avg_center_control' in eval_data:
                        center_data[round_num].append(eval_data['avg_center_control'])

            if center_data:
                rounds_sorted = sorted(center_data.keys())
                avg_center = [np.mean(center_data[r]) for r in rounds_sorted]
                for r, avg in zip(rounds_sorted, avg_center):
                    all_data.append({'experiment': experiment, 'round': r, 'avg_center_control': avg})
                ax.plot(rounds_sorted, avg_center, label=experiment,
                       linewidth=2.5, marker='o', markersize=4, alpha=0.8)

        if all_data:
            df = pd.DataFrame(all_data)
            self.save_csv('cluster_avg_center_control_all', df)

        ax.set_xlabel('Training Round', fontsize=12)
        ax.set_ylabel('Average Center Control', fontsize=12)
        ax.set_title('Average Center Control per Cluster: All Experiments', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'cluster_avg_center_control_all.png', dpi=150)
        plt.close()

    def plot_cluster_averages_move_diversity(self):
        """Plot average move diversity per cluster across all experiments."""
        all_data = []
        fig, ax = plt.subplots(figsize=(14, 8))

        for experiment in EXPERIMENTS:
            diversity_data = defaultdict(list)

            for cluster in CLUSTERS:
                rounds = self.loader.get_available_rounds(experiment, cluster)
                if not rounds:
                    continue

                for round_num in rounds:
                    eval_data = self.loader.load_evaluation_metrics(experiment, cluster, round_num)
                    if eval_data and 'avg_move_diversity_ratio' in eval_data:
                        diversity_data[round_num].append(eval_data['avg_move_diversity_ratio'])

            if diversity_data:
                rounds_sorted = sorted(diversity_data.keys())
                avg_diversity = [np.mean(diversity_data[r]) for r in rounds_sorted]
                for r, avg in zip(rounds_sorted, avg_diversity):
                    all_data.append({'experiment': experiment, 'round': r, 'avg_move_diversity': avg})
                ax.plot(rounds_sorted, avg_diversity, label=experiment,
                       linewidth=2.5, marker='o', markersize=4, alpha=0.8)

        if all_data:
            df = pd.DataFrame(all_data)
            self.save_csv('cluster_avg_move_diversity_all', df)

        ax.set_xlabel('Training Round', fontsize=12)
        ax.set_ylabel('Average Move Diversity Ratio', fontsize=12)
        ax.set_title('Average Move Diversity per Cluster: All Experiments', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'cluster_avg_move_diversity_all.png', dpi=150)
        plt.close()

    def plot_cluster_averages_elo(self):
        """Plot average ELO per cluster across all experiments."""
        all_data = []
        fig, ax = plt.subplots(figsize=(14, 8))

        for experiment in EXPERIMENTS:
            elo_data = defaultdict(list)

            for cluster in CLUSTERS:
                rounds = self.loader.get_available_rounds(experiment, cluster)
                if not rounds:
                    continue

                for round_num in rounds:
                    eval_data = self.loader.load_evaluation_metrics(experiment, cluster, round_num)
                    if eval_data and 'estimated_elo' in eval_data:
                        elo_data[round_num].append(eval_data['estimated_elo'])

            if elo_data:
                rounds_sorted = sorted(elo_data.keys())
                avg_elo = [np.mean(elo_data[r]) for r in rounds_sorted]
                for r, avg in zip(rounds_sorted, avg_elo):
                    all_data.append({'experiment': experiment, 'round': r, 'avg_elo': avg})
                ax.plot(rounds_sorted, avg_elo, label=experiment,
                       linewidth=2.5, marker='o', markersize=4, alpha=0.8)

        if all_data:
            df = pd.DataFrame(all_data)
            self.save_csv('cluster_avg_elo_all', df)

        ax.set_xlabel('Training Round', fontsize=12)
        ax.set_ylabel('Average ELO (Both Clusters)', fontsize=12)
        ax.set_title('Average ELO per Cluster: All Experiments', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'cluster_avg_elo_all.png', dpi=150)
        plt.close()

    def plot_cluster_averages_win_rate(self):
        """Plot average win rate per cluster across all experiments."""
        all_data = []
        fig, ax = plt.subplots(figsize=(14, 8))

        for experiment in EXPERIMENTS:
            winrate_data = defaultdict(list)

            for cluster in CLUSTERS:
                rounds = self.loader.get_available_rounds(experiment, cluster)
                if not rounds:
                    continue

                for round_num in rounds:
                    eval_data = self.loader.load_evaluation_metrics(experiment, cluster, round_num)
                    if eval_data and 'win_rate' in eval_data:
                        winrate_data[round_num].append(eval_data['win_rate'] * 100)

            if winrate_data:
                rounds_sorted = sorted(winrate_data.keys())
                avg_winrate = [np.mean(winrate_data[r]) for r in rounds_sorted]
                for r, avg in zip(rounds_sorted, avg_winrate):
                    all_data.append({'experiment': experiment, 'round': r, 'avg_win_rate': avg})
                ax.plot(rounds_sorted, avg_winrate, label=experiment,
                       linewidth=2.5, marker='o', markersize=4, alpha=0.8)

        if all_data:
            df = pd.DataFrame(all_data)
            self.save_csv('cluster_avg_win_rate_all', df)

        ax.set_xlabel('Training Round', fontsize=12)
        ax.set_ylabel('Average Win Rate (%)', fontsize=12)
        ax.set_title('Average Win Rate per Cluster: All Experiments', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'cluster_avg_win_rate_all.png', dpi=150)
        plt.close()

    def plot_divergence_per_cluster(self):
        """Plot divergence metrics per cluster (tactical vs positional) across all experiments."""
        all_data = []
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        for idx, experiment in enumerate(EXPERIMENTS):
            ax = axes[idx]
            rounds = self.loader.get_available_rounds(experiment, 'tactical')

            if not rounds:
                continue

            # Get global divergence for each round
            divergences = []
            valid_rounds = []

            for round_num in rounds:
                div_data = self.loader.load_divergence(experiment, round_num)
                if div_data and 'global' in div_data:
                    div = div_data['global'].get('mean_divergence', 0)
                    divergences.append(div)
                    valid_rounds.append(round_num)
                    all_data.append({'experiment': experiment, 'round': round_num, 'global_divergence': div})

            if divergences:
                ax.plot(valid_rounds, divergences, linewidth=2.5, marker='o',
                       markersize=4, alpha=0.8, color='darkblue')

            ax.set_xlabel('Training Round', fontsize=10)
            ax.set_ylabel('Global Mean Divergence', fontsize=10)
            ax.set_title(experiment, fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)

        if all_data:
            df = pd.DataFrame(all_data)
            self.save_csv('divergence_per_round_all', df)

        fig.suptitle('Global Divergence Evolution: All Experiments', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'divergence_per_round_all.png', dpi=150)
        plt.close()

    def plot_divergence_by_layer_group_per_cluster(self):
        """Plot divergence by layer group, one subplot per experiment."""
        all_data = []
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        for idx, experiment in enumerate(EXPERIMENTS):
            ax = axes[idx]
            rounds = self.loader.get_available_rounds(experiment, 'tactical')

            if not rounds:
                continue

            # Collect divergence for each layer group
            for group in LAYER_GROUPS:
                group_divs = []
                valid_rounds = []

                for round_num in rounds:
                    div_data = self.loader.load_divergence(experiment, round_num)
                    if div_data and 'per_group' in div_data and group in div_data['per_group']:
                        div = div_data['per_group'][group].get('mean_divergence_index', 0)
                        group_divs.append(div)
                        valid_rounds.append(round_num)
                        all_data.append({'experiment': experiment, 'layer_group': group, 'round': round_num, 'divergence': div})

                if group_divs:
                    ax.plot(valid_rounds, group_divs, label=group, linewidth=2, alpha=0.8)

            ax.set_xlabel('Training Round', fontsize=10)
            ax.set_ylabel('Divergence Index', fontsize=10)
            ax.set_title(experiment, fontsize=11, fontweight='bold')
            ax.legend(fontsize=7, loc='best', ncol=2)
            ax.grid(True, alpha=0.3)

        if all_data:
            df = pd.DataFrame(all_data)
            self.save_csv('divergence_by_layer_group_all', df)

        fig.suptitle('Layer Group Divergence: All Experiments', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'divergence_by_layer_group_all.png', dpi=150)
        plt.close()

    def plot_policy_vs_value_divergence(self):
        """Compare policy head vs value head divergence across experiments."""
        all_data = []
        fig, ax = plt.subplots(figsize=(14, 8))

        for experiment in EXPERIMENTS:
            rounds = self.loader.get_available_rounds(experiment, 'tactical')
            if not rounds:
                continue

            policy_divs = []
            value_divs = []
            valid_rounds = []

            for round_num in rounds:
                div_data = self.loader.load_divergence(experiment, round_num)
                if div_data and 'per_group' in div_data:
                    if 'policy_head' in div_data['per_group'] and 'value_head' in div_data['per_group']:
                        pol = div_data['per_group']['policy_head'].get('mean_divergence_index', 0)
                        val = div_data['per_group']['value_head'].get('mean_divergence_index', 0)
                        policy_divs.append(pol)
                        value_divs.append(val)
                        valid_rounds.append(round_num)
                        all_data.append({'experiment': experiment, 'round': round_num, 'policy_divergence': pol, 'value_divergence': val, 'difference': pol - val})

            if policy_divs:
                # Plot difference between policy and value head divergence
                diff = [p - v for p, v in zip(policy_divs, value_divs)]
                ax.plot(valid_rounds, diff, label=experiment,
                       linewidth=2.5, marker='o', markersize=4, alpha=0.8)

        if all_data:
            df = pd.DataFrame(all_data)
            self.save_csv('policy_vs_value_divergence_all', df)

        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='No difference')
        ax.set_xlabel('Training Round', fontsize=12)
        ax.set_ylabel('Policy - Value Divergence', fontsize=12)
        ax.set_title('Policy Head vs Value Head Divergence Difference', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'policy_vs_value_divergence_all.png', dpi=150)
        plt.close()

    def plot_early_vs_late_divergence(self):
        """Compare early vs late residual block divergence."""
        all_data = []
        fig, ax = plt.subplots(figsize=(14, 8))

        for experiment in EXPERIMENTS:
            rounds = self.loader.get_available_rounds(experiment, 'tactical')
            if not rounds:
                continue

            early_divs = []
            late_divs = []
            valid_rounds = []

            for round_num in rounds:
                div_data = self.loader.load_divergence(experiment, round_num)
                if div_data and 'per_group' in div_data:
                    if 'early_residual' in div_data['per_group'] and 'late_residual' in div_data['per_group']:
                        early = div_data['per_group']['early_residual'].get('mean_divergence_index', 0)
                        late = div_data['per_group']['late_residual'].get('mean_divergence_index', 0)
                        early_divs.append(early)
                        late_divs.append(late)
                        valid_rounds.append(round_num)
                        all_data.append({'experiment': experiment, 'round': round_num, 'early_divergence': early, 'late_divergence': late, 'difference': late - early})

            if early_divs:
                # Plot difference between late and early
                diff = [l - e for l, e in zip(late_divs, early_divs)]
                ax.plot(valid_rounds, diff, label=experiment,
                       linewidth=2.5, marker='o', markersize=4, alpha=0.8)

        if all_data:
            df = pd.DataFrame(all_data)
            self.save_csv('early_vs_late_divergence_all', df)

        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='No difference')
        ax.set_xlabel('Training Round', fontsize=12)
        ax.set_ylabel('Late - Early Residual Divergence', fontsize=12)
        ax.set_title('Late vs Early Residual Block Divergence Difference', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'early_vs_late_divergence_all.png', dpi=150)
        plt.close()

    def plot_policy_head_divergence_over_rounds(self):
        """Plot policy head divergence evolution over rounds for all experiments."""
        all_data = []
        fig, ax = plt.subplots(figsize=(14, 8))

        for experiment in EXPERIMENTS:
            rounds = self.loader.get_available_rounds(experiment, 'tactical')
            if not rounds:
                continue

            policy_divs = []
            valid_rounds = []

            for round_num in rounds:
                div_data = self.loader.load_divergence(experiment, round_num)
                if div_data and 'per_group' in div_data and 'policy_head' in div_data['per_group']:
                    policy_div = div_data['per_group']['policy_head'].get('mean_divergence_index', 0)
                    policy_divs.append(policy_div)
                    valid_rounds.append(round_num)
                    all_data.append({'experiment': experiment, 'round': round_num, 'policy_head_divergence': policy_div})

            if policy_divs:
                ax.plot(valid_rounds, policy_divs, label=experiment,
                       linewidth=2.5, marker='o', markersize=4, alpha=0.8)

        if all_data:
            df = pd.DataFrame(all_data)
            self.save_csv('policy_head_divergence_over_rounds', df)

        ax.set_xlabel('Training Round', fontsize=12)
        ax.set_ylabel('Policy Head Divergence Index', fontsize=12)
        ax.set_title('Policy Head Divergence Evolution: All Experiments', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'policy_head_divergence_over_rounds.png', dpi=150)
        plt.close()

    def plot_value_head_divergence_over_rounds(self):
        """Plot value head divergence evolution over rounds for all experiments."""
        all_data = []
        fig, ax = plt.subplots(figsize=(14, 8))

        for experiment in EXPERIMENTS:
            rounds = self.loader.get_available_rounds(experiment, 'tactical')
            if not rounds:
                continue

            value_divs = []
            valid_rounds = []

            for round_num in rounds:
                div_data = self.loader.load_divergence(experiment, round_num)
                if div_data and 'per_group' in div_data and 'value_head' in div_data['per_group']:
                    value_div = div_data['per_group']['value_head'].get('mean_divergence_index', 0)
                    value_divs.append(value_div)
                    valid_rounds.append(round_num)
                    all_data.append({'experiment': experiment, 'round': round_num, 'value_head_divergence': value_div})

            if value_divs:
                ax.plot(valid_rounds, value_divs, label=experiment,
                       linewidth=2.5, marker='o', markersize=4, alpha=0.8)

        if all_data:
            df = pd.DataFrame(all_data)
            self.save_csv('value_head_divergence_over_rounds', df)

        ax.set_xlabel('Training Round', fontsize=12)
        ax.set_ylabel('Value Head Divergence Index', fontsize=12)
        ax.set_title('Value Head Divergence Evolution: All Experiments', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'value_head_divergence_over_rounds.png', dpi=150)
        plt.close()

    def generate_all_plots(self):
        """Generate all plots for all experiments."""
        print("Generating plots...")
        print("=" * 60)

        # Cross-experiment plots
        print("Generating cross-experiment comparison plots...")
        self.plot_divergence_trajectories()
        self.plot_global_divergence_comparison()
        self.plot_elo_comparison_all()
        self.plot_final_divergence_comparison()
        self.plot_elo_vs_divergence()
        self.plot_move_type_differences()
        self.plot_cluster_behavioral_difference()

        # Cluster average plots
        print("Generating cluster average plots...")
        self.plot_cluster_averages_legal_moves()
        self.plot_cluster_averages_material_metrics()
        self.plot_cluster_averages_center_control()
        self.plot_cluster_averages_move_diversity()
        self.plot_cluster_averages_elo()
        self.plot_cluster_averages_win_rate()

        # Divergence analysis plots
        print("Generating divergence analysis plots...")
        self.plot_divergence_per_cluster()
        self.plot_divergence_by_layer_group_per_cluster()
        self.plot_policy_head_divergence_over_rounds()
        self.plot_value_head_divergence_over_rounds()
        self.plot_policy_vs_value_divergence()
        self.plot_early_vs_late_divergence()

        # Per-experiment plots
        for experiment in EXPERIMENTS:
            print(f"\nGenerating plots for {experiment}...")

            # Check if experiment has data
            rounds = self.loader.get_available_rounds(experiment, 'tactical')
            if not rounds:
                print(f"  No data found for {experiment}, skipping...")
                continue

            # Divergence plots
            self.plot_divergence_heatmap(experiment)

            # Performance plots
            self.plot_elo_progression(experiment)
            self.plot_win_rates(experiment)
            self.plot_win_draw_loss_rates(experiment)

            # Move type plots
            self.plot_move_types_evolution(experiment)
            self.plot_move_diversity(experiment)
            self.plot_unique_move_destinations(experiment)

            # Game phase analysis
            self.plot_legal_moves_by_phase(experiment)
            self.plot_attacked_material_and_captures(experiment)
            self.plot_center_control(experiment)

            # Weight statistics plots
            self.plot_weight_change_magnitude(experiment)
            self.plot_sparsity_evolution(experiment)
            self.plot_l2_norm_by_layer(experiment)

            # Opening diversity
            self.plot_opening_diversity(experiment)

            print(f"  ✓ Completed {experiment}")

        print("\n" + "=" * 60)
        print(f"All plots saved to: {self.output_dir.absolute()}")
        print("=" * 60)


def main():
    """Main entry point."""
    print("\n" + "=" * 60)
    print("Federated Chess Learning - Metrics Analysis")
    print("=" * 60)
    print(f"Round range: {MIN_ROUND} - {MAX_ROUND}")
    print(f"Experiments: {', '.join(EXPERIMENTS)}")
    print("=" * 60 + "\n")

    # Initialize
    loader = MetricsLoader()
    analyzer = MetricsAnalyzer(loader)

    # Generate all plots
    analyzer.generate_all_plots()

    print("\n✓ Analysis complete!\n")


if __name__ == "__main__":
    main()
