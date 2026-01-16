"""Results export, visualization, and statistical analysis."""

import os
import json
import csv
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for publication-quality plots
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 12


class ResultsExporter:
    """Export evaluation results to various formats."""

    @staticmethod
    def to_csv(results: Dict[str, Any], output_path: str):
        """
        Export results to CSV format.
        
        Args:
            results: Results dict from evaluate_comprehensive
            output_path: Path to save CSV file
        """
        rows = []
        for opponent, metrics in results.items():
            row = {
                "opponent": opponent,
                "n_episodes": metrics["n_episodes"],
                "n_seeds": metrics["n_seeds"],
                "win_rate_mean": metrics["win_rate"]["mean"],
                "win_rate_ci_lower": metrics["win_rate"]["ci_lower"],
                "win_rate_ci_upper": metrics["win_rate"]["ci_upper"],
                "win_rate_std": metrics["win_rate"]["std"],
                "score_diff_mean": metrics["score_diff"]["mean"],
                "score_diff_ci_lower": metrics["score_diff"]["ci_lower"],
                "score_diff_ci_upper": metrics["score_diff"]["ci_upper"],
                "score_diff_std": metrics["score_diff"]["std"],
                "reward_mean": metrics["reward"]["mean"],
                "reward_std": metrics["reward"]["std"],
                "pistis_mean": metrics["pistis"]["mean"],
                "pistis_std": metrics["pistis"]["std"],
                "double_pistis_mean": metrics["double_pistis"]["mean"],
                "capture_efficiency": metrics["capture_efficiency"],
                "game_length_mean": metrics["game_length"]["mean"],
            }
            rows.append(row)
        
        with open(output_path, "w", newline="") as f:
            if rows:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

    @staticmethod
    def to_latex_table(results: Dict[str, Any], output_path: str, caption: str = "Evaluation Results"):
        """
        Export results to LaTeX table format.
        
        Args:
            results: Results dict from evaluate_comprehensive
            output_path: Path to save LaTeX file
            caption: Table caption
        """
        lines = [
            "\\begin{table}[h]",
            "\\centering",
            "\\begin{tabular}{lcccc}",
            "\\toprule",
            "Opponent & Win Rate & Score Diff & Pişti & Capture Eff. \\\\",
            "\\midrule",
        ]
        
        for opponent, metrics in results.items():
            win_rate = metrics["win_rate"]["mean"]
            win_rate_ci_l = metrics["win_rate"]["ci_lower"]
            win_rate_ci_u = metrics["win_rate"]["ci_upper"]
            score_diff = metrics["score_diff"]["mean"]
            score_diff_ci_l = metrics["score_diff"]["ci_lower"]
            score_diff_ci_u = metrics["score_diff"]["ci_upper"]
            pistis = metrics["pistis"]["mean"]
            capture_eff = metrics["capture_efficiency"]
            
            line = (
                f"{opponent.replace('_', ' ').title()} & "
                f"{win_rate:.2%} [{win_rate_ci_l:.2%}, {win_rate_ci_u:.2%}] & "
                f"{score_diff:.2f} [{score_diff_ci_l:.2f}, {score_diff_ci_u:.2f}] & "
                f"{pistis:.2f} & "
                f"{capture_eff:.2%} \\\\"
            )
            lines.append(line)
        
        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            f"\\caption{{{caption}}}",
            "\\label{tab:eval_results}",
            "\\end{table}",
        ])
        
        with open(output_path, "w") as f:
            f.write("\n".join(lines))

    @staticmethod
    def to_markdown(results: Dict[str, Any], output_path: str):
        """
        Export results to Markdown format.
        
        Args:
            results: Results dict from evaluate_comprehensive
            output_path: Path to save Markdown file
        """
        lines = [
            "# Evaluation Results",
            "",
            "| Opponent | Win Rate | Score Diff | Pişti | Capture Eff. |",
            "|----------|----------|------------|-------|--------------|",
        ]
        
        for opponent, metrics in results.items():
            win_rate = metrics["win_rate"]["mean"]
            win_rate_ci_l = metrics["win_rate"]["ci_lower"]
            win_rate_ci_u = metrics["win_rate"]["ci_upper"]
            score_diff = metrics["score_diff"]["mean"]
            score_diff_ci_l = metrics["score_diff"]["ci_lower"]
            score_diff_ci_u = metrics["score_diff"]["ci_upper"]
            pistis = metrics["pistis"]["mean"]
            capture_eff = metrics["capture_efficiency"]
            
            line = (
                f"| {opponent.replace('_', ' ').title()} | "
                f"{win_rate:.2%} [{win_rate_ci_l:.2%}, {win_rate_ci_u:.2%}] | "
                f"{score_diff:.2f} [{score_diff_ci_l:.2f}, {score_diff_ci_u:.2f}] | "
                f"{pistis:.2f} | "
                f"{capture_eff:.2%} |"
            )
            lines.append(line)
        
        with open(output_path, "w") as f:
            f.write("\n".join(lines))


class ResultsAnalyzer:
    """Statistical analysis of evaluation results."""

    @staticmethod
    def t_test(
        data1: np.ndarray, data2: np.ndarray, alternative: str = "two-sided"
    ) -> Dict[str, float]:
        """
        Perform t-test between two groups.
        
        Args:
            data1: First group of data
            data2: Second group of data
            alternative: "two-sided", "less", or "greater"
        
        Returns:
            Dict with test statistics
        """
        t_stat, p_value = stats.ttest_ind(data1, data2, alternative=alternative)
        
        return {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
        }

    @staticmethod
    def mann_whitney_u(
        data1: np.ndarray, data2: np.ndarray, alternative: str = "two-sided"
    ) -> Dict[str, float]:
        """
        Perform Mann-Whitney U test (non-parametric).
        
        Args:
            data1: First group of data
            data2: Second group of data
            alternative: "two-sided", "less", or "greater"
        
        Returns:
            Dict with test statistics
        """
        u_stat, p_value = stats.mannwhitneyu(data1, data2, alternative=alternative)
        
        return {
            "u_statistic": float(u_stat),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
        }

    @staticmethod
    def compare_opponents(
        results: Dict[str, Any], baseline: str = "random"
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare all opponents against a baseline using statistical tests.
        
        Args:
            results: Results dict from evaluate_comprehensive
            baseline: Baseline opponent name
        
        Returns:
            Dict with comparison results
        """
        if baseline not in results:
            raise ValueError(f"Baseline {baseline} not found in results")
        
        baseline_data = np.array(results[baseline]["raw_data"]["score_diffs"])
        comparisons = {}
        
        for opponent, metrics in results.items():
            if opponent == baseline:
                continue
            
            opponent_data = np.array(metrics["raw_data"]["score_diffs"])
            
            # Perform both tests
            t_test_result = ResultsAnalyzer.t_test(baseline_data, opponent_data)
            mw_test_result = ResultsAnalyzer.mann_whitney_u(baseline_data, opponent_data)
            
            comparisons[opponent] = {
                "vs_baseline": baseline,
                "t_test": t_test_result,
                "mann_whitney_u": mw_test_result,
                "effect_size": float(
                    (np.mean(opponent_data) - np.mean(baseline_data))
                    / np.std(np.concatenate([baseline_data, opponent_data]))
                ),
            }
        
        return comparisons


class ResultsVisualizer:
    """Generate visualizations from evaluation results."""

    @staticmethod
    def plot_win_rates(results: Dict[str, Any], output_path: str):
        """
        Plot win rates with confidence intervals.
        
        Args:
            results: Results dict from evaluate_comprehensive
            output_path: Path to save figure
        """
        opponents = list(results.keys())
        win_rates = [results[opp]["win_rate"]["mean"] for opp in opponents]
        ci_lowers = [
            results[opp]["win_rate"]["ci_lower"] for opp in opponents
        ]
        ci_uppers = [
            results[opp]["win_rate"]["ci_upper"] for opp in opponents
        ]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        x_pos = np.arange(len(opponents))
        
        ax.bar(x_pos, win_rates, yerr=[np.array(win_rates) - np.array(ci_lowers),
                                       np.array(ci_uppers) - np.array(win_rates)],
               capsize=5, alpha=0.7)
        ax.set_xlabel("Opponent")
        ax.set_ylabel("Win Rate")
        ax.set_title("Win Rate vs Different Opponents (95% CI)")
        ax.set_xticks(x_pos)
        ax.set_xticklabels([opp.replace("_", " ").title() for opp in opponents], rotation=45, ha="right")
        ax.set_ylim([0, 1])
        ax.grid(axis="y", alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    @staticmethod
    def plot_score_distributions(results: Dict[str, Any], output_path: str):
        """
        Plot score difference distributions.
        
        Args:
            results: Results dict from evaluate_comprehensive
            output_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for opponent, metrics in results.items():
            score_diffs = np.array(metrics["raw_data"]["score_diffs"])
            ax.hist(
                score_diffs,
                alpha=0.5,
                label=opponent.replace("_", " ").title(),
                bins=30,
            )
        
        ax.set_xlabel("Score Difference (Player 0 - Player 1)")
        ax.set_ylabel("Frequency")
        ax.set_title("Score Difference Distributions")
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    @staticmethod
    def plot_performance_comparison(results: Dict[str, Any], output_path: str):
        """
        Plot comprehensive performance comparison.
        
        Args:
            results: Results dict from evaluate_comprehensive
            output_path: Path to save figure
        """
        opponents = list(results.keys())
        metrics_to_plot = [
            ("win_rate", "Win Rate", "mean"),
            ("score_diff", "Score Diff", "mean"),
            ("pistis", "Pişti Count", "mean"),
            ("capture_efficiency", "Capture Efficiency", None),
        ]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, (metric_key, metric_label, subkey) in enumerate(metrics_to_plot):
            ax = axes[idx]
            values = []
            ci_lowers = []
            ci_uppers = []
            
            for opp in opponents:
                if subkey:
                    values.append(results[opp][metric_key][subkey])
                    if metric_key in ["win_rate", "score_diff"]:
                        ci_lowers.append(results[opp][metric_key]["ci_lower"])
                        ci_uppers.append(results[opp][metric_key]["ci_upper"])
                else:
                    values.append(results[opp][metric_key])
                    ci_lowers.append(None)
                    ci_uppers.append(None)
            
            x_pos = np.arange(len(opponents))
            if ci_lowers[0] is not None:
                errors = [
                    np.array(values) - np.array(ci_lowers),
                    np.array(ci_uppers) - np.array(values),
                ]
                ax.bar(x_pos, values, yerr=errors, capsize=5, alpha=0.7)
            else:
                ax.bar(x_pos, values, alpha=0.7)
            
            ax.set_xlabel("Opponent")
            ax.set_ylabel(metric_label)
            ax.set_title(metric_label)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(
                [opp.replace("_", " ").title() for opp in opponents],
                rotation=45,
                ha="right",
            )
            ax.grid(axis="y", alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    @staticmethod
    def plot_learning_curve(
        tensorboard_log_dir: str, output_path: str, metric: str = "rollout/ep_rew_mean"
    ):
        """
        Plot learning curve from TensorBoard logs.
        
        Args:
            tensorboard_log_dir: Directory with TensorBoard logs
            output_path: Path to save figure
            metric: Metric to plot
        """
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            
            ea = EventAccumulator(tensorboard_log_dir)
            ea.Reload()
            
            if metric not in ea.Tags()["scalars"]:
                print(f"Metric {metric} not found in logs")
                return
            
            scalar_events = ea.Scalars(metric)
            steps = [e.step for e in scalar_events]
            values = [e.value for e in scalar_events]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(steps, values)
            ax.set_xlabel("Training Steps")
            ax.set_ylabel(metric)
            ax.set_title("Learning Curve")
            ax.grid(alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()
        except ImportError:
            print("TensorBoard not available for learning curve plotting")
        except Exception as e:
            print(f"Error plotting learning curve: {e}")
