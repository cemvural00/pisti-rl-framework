"""Generate academic paper-ready reports from evaluation results."""

import argparse
import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime

from training.results import ResultsExporter, ResultsVisualizer, ResultsAnalyzer
from training.metadata import load_metadata


def generate_report(
    results_dir: str,
    output_path: str = None,
    formats: List[str] = None,
    checkpoint_path: Optional[str] = None,
):
    """
    Generate comprehensive academic report from evaluation results.
    
    Args:
        results_dir: Directory containing evaluation_results.json
        output_path: Path for output report (default: report.md in results_dir)
        formats: List of formats to generate (markdown, latex, html)
        checkpoint_path: Optional path to checkpoint for metadata
    """
    if formats is None:
        formats = ["markdown"]
    
    results_file = os.path.join(results_dir, "evaluation_results.json")
    
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    # Load results
    with open(results_file, "r") as f:
        results = json.load(f)
    
    # Load metadata if checkpoint provided
    metadata = None
    if checkpoint_path:
        checkpoint_dir = os.path.dirname(checkpoint_path)
        checkpoint_name = os.path.basename(checkpoint_path).replace(".zip", "")
        metadata = load_metadata(checkpoint_dir, checkpoint_name)
    
    # Generate visualizations
    print("Generating visualizations...")
    fig_dir = os.path.join(results_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    ResultsVisualizer.plot_win_rates(
        results, os.path.join(fig_dir, "win_rates.png")
    )
    ResultsVisualizer.plot_score_distributions(
        results, os.path.join(fig_dir, "score_distributions.png")
    )
    ResultsVisualizer.plot_performance_comparison(
        results, os.path.join(fig_dir, "performance_comparison.png")
    )
    
    # Statistical comparisons
    print("Performing statistical analysis...")
    analyzer = ResultsAnalyzer()
    comparisons = analyzer.compare_opponents(results, baseline="random")
    
    # Export to different formats
    if "csv" in formats:
        csv_path = os.path.join(results_dir, "results.csv")
        ResultsExporter.to_csv(results, csv_path)
        print(f"Exported CSV to {csv_path}")
    
    if "latex" in formats:
        latex_path = os.path.join(results_dir, "results_table.tex")
        ResultsExporter.to_latex_table(
            results, latex_path, caption="Evaluation Results: Win Rate and Performance Metrics"
        )
        print(f"Exported LaTeX table to {latex_path}")
    
    if "markdown" in formats:
        if output_path is None:
            output_path = os.path.join(results_dir, "report.md")
        
        generate_markdown_report(
            results, comparisons, metadata, output_path, fig_dir
        )
        print(f"Generated Markdown report to {output_path}")
    
    if "html" in formats:
        html_path = os.path.join(results_dir, "report.html")
        generate_html_report(results, comparisons, metadata, html_path, fig_dir)
        print(f"Generated HTML report to {html_path}")


def generate_markdown_report(
    results: Dict[str, Any],
    comparisons: Dict[str, Any],
    metadata: Optional[Any],
    output_path: str,
    fig_dir: str,
):
    """Generate Markdown report."""
    lines = [
        "# Pişti RL Evaluation Report",
        "",
        f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
        "",
        "## Executive Summary",
        "",
        "This report presents comprehensive evaluation results for the trained Pişti RL agent.",
        "",
        "## Results Overview",
        "",
    ]
    
    # Add results table
    lines.append("### Performance Metrics")
    lines.append("")
    lines.append("| Opponent | Win Rate (95% CI) | Score Diff (95% CI) | Pişti | Capture Eff. |")
    lines.append("|----------|-------------------|---------------------|-------|--------------|")
    
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
    
    lines.extend([
        "",
        "## Visualizations",
        "",
        "### Win Rates",
        "",
        f"![Win Rates](figures/win_rates.png)",
        "",
        "### Score Distributions",
        "",
        f"![Score Distributions](figures/score_distributions.png)",
        "",
        "### Performance Comparison",
        "",
        f"![Performance Comparison](figures/performance_comparison.png)",
        "",
        "## Statistical Analysis",
        "",
    ])
    
    # Add statistical comparisons
    if comparisons:
        lines.append("### Comparison vs Random Baseline")
        lines.append("")
        lines.append("| Opponent | t-test p-value | Mann-Whitney U p-value | Effect Size | Significant |")
        lines.append("|----------|----------------|----------------------|-------------|-------------|")
        
        for opponent, comp in comparisons.items():
            t_p = comp["t_test"]["p_value"]
            mw_p = comp["mann_whitney_u"]["p_value"]
            effect = comp["effect_size"]
            sig = "Yes" if (t_p < 0.05 or mw_p < 0.05) else "No"
            
            line = (
                f"| {opponent.replace('_', ' ').title()} | "
                f"{t_p:.4f} | {mw_p:.4f} | {effect:.3f} | {sig} |"
            )
            lines.append(line)
    
    # Add reproducibility section
    if metadata:
        lines.extend([
            "",
            "## Reproducibility",
            "",
            "### Training Configuration",
            "",
            f"- **Algorithm**: {metadata.algorithm}",
            f"- **Total Timesteps**: {metadata.total_timesteps:,}",
            f"- **Encoder Type**: {metadata.encoder_type}",
            f"- **Training Start**: {metadata.training_start_time}",
            f"- **Training End**: {metadata.training_end_time}",
            "",
            "### Hyperparameters",
            "",
        ])
        
        for key, value in metadata.hyperparameters.items():
            lines.append(f"- **{key}**: {value}")
        
        lines.extend([
            "",
            "### System Information",
            "",
            f"- **Python Version**: {metadata.python_version}",
            f"- **Platform**: {metadata.platform_info}",
        ])
        
        if metadata.git_commit_hash:
            lines.append(f"- **Git Commit**: {metadata.git_commit_hash}")
        
        if metadata.package_versions:
            lines.append("")
            lines.append("### Package Versions")
            lines.append("")
            for pkg, version in metadata.package_versions.items():
                lines.append(f"- **{pkg}**: {version}")
    
    lines.extend([
        "",
        "## Methodology",
        "",
        f"- **Total Episodes**: {results[list(results.keys())[0]]['n_episodes']}",
        f"- **Number of Seeds**: {results[list(results.keys())[0]]['n_seeds']}",
        "- **Confidence Intervals**: 95%",
        "- **Statistical Tests**: t-test and Mann-Whitney U test",
        "",
    ])
    
    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def generate_html_report(
    results: Dict[str, Any],
    comparisons: Dict[str, Any],
    metadata: Optional[Any],
    output_path: str,
    fig_dir: str,
):
    """Generate HTML report."""
    html = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "<title>Pişti RL Evaluation Report</title>",
        "<style>",
        "body { font-family: Arial, sans-serif; margin: 40px; }",
        "table { border-collapse: collapse; width: 100%; margin: 20px 0; }",
        "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
        "th { background-color: #4CAF50; color: white; }",
        "img { max-width: 100%; height: auto; margin: 20px 0; }",
        "</style>",
        "</head>",
        "<body>",
        f"<h1>Pişti RL Evaluation Report</h1>",
        f"<p><em>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>",
        "<h2>Results Overview</h2>",
        "<table>",
        "<tr><th>Opponent</th><th>Win Rate</th><th>Score Diff</th><th>Pişti</th><th>Capture Eff.</th></tr>",
    ]
    
    for opponent, metrics in results.items():
        win_rate = metrics["win_rate"]["mean"]
        score_diff = metrics["score_diff"]["mean"]
        pistis = metrics["pistis"]["mean"]
        capture_eff = metrics["capture_efficiency"]
        
        html.append(
            f"<tr>"
            f"<td>{opponent.replace('_', ' ').title()}</td>"
            f"<td>{win_rate:.2%}</td>"
            f"<td>{score_diff:.2f}</td>"
            f"<td>{pistis:.2f}</td>"
            f"<td>{capture_eff:.2%}</td>"
            f"</tr>"
        )
    
    html.extend([
        "</table>",
        "<h2>Visualizations</h2>",
        f'<img src="{os.path.join(fig_dir, "win_rates.png")}" alt="Win Rates">',
        f'<img src="{os.path.join(fig_dir, "score_distributions.png")}" alt="Score Distributions">',
        f'<img src="{os.path.join(fig_dir, "performance_comparison.png")}" alt="Performance Comparison">',
        "</body>",
        "</html>",
    ])
    
    with open(output_path, "w") as f:
        f.write("\n".join(html))


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate academic report from evaluation results")
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Directory containing evaluation_results.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for report (default: report.md in results-dir)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="markdown",
        help="Comma-separated formats: markdown,latex,html,csv (default: markdown)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint for metadata (optional)",
    )
    
    args = parser.parse_args()
    
    formats = [f.strip() for f in args.format.split(",")]
    
    generate_report(
        results_dir=args.results_dir,
        output_path=args.output,
        formats=formats,
        checkpoint_path=args.checkpoint,
    )


if __name__ == "__main__":
    main()
