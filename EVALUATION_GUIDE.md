# Academic Evaluation Guide

This guide explains how to use the comprehensive evaluation and reporting system for academic research.

## Quick Start

### 1. Train a Model (with automatic metadata saving)

```bash
python -m training.train_sb3 --config configs/default.yaml
```

This automatically saves:
- Model checkpoints: `checkpoints/pisti_model_*.zip`
- Metadata files: `checkpoints/pisti_model_*_metadata.json`
- TensorBoard logs: `logs/tensorboard/`

### 2. Comprehensive Evaluation

```bash
python -m training.evaluate_comprehensive \
    --checkpoint checkpoints/pisti_model_final \
    --opponents random,greedy,pisti_hunter,probabilistic \
    --n-episodes 1000 \
    --n-seeds 10 \
    --output-dir results/experiment_1
```

**Outputs:**
- `evaluation_results.json`: Full results with raw data
- Statistical analysis with 95% confidence intervals
- Results for all opponents

### 3. Generate Academic Report

```bash
python -m training.generate_report \
    --results-dir results/experiment_1 \
    --checkpoint checkpoints/pisti_model_final \
    --format markdown,latex,csv
```

**Outputs:**
- `report.md`: Markdown report with embedded figures
- `results_table.tex`: LaTeX table for papers
- `results.csv`: CSV for spreadsheet analysis
- `figures/`: Directory with publication-quality plots

## Evaluation Metrics

The comprehensive evaluation tracks:

1. **Win Rate**: Percentage of games won (with 95% CI)
2. **Score Differential**: Average score difference (with 95% CI)
3. **Pişti Frequency**: Average number of pişti bonuses per game
4. **Capture Efficiency**: Ratio of captures made
5. **Game Length**: Average number of steps per game

## Statistical Analysis

### Confidence Intervals
- 95% confidence intervals calculated using t-distribution
- Reported as [lower, upper] bounds

### Significance Tests
- **t-test**: Parametric test for score differences
- **Mann-Whitney U**: Non-parametric alternative
- Effect sizes calculated (Cohen's d)

### Multiple Seeds
- Recommended: 10+ seeds for robust statistics
- Episodes distributed evenly across seeds
- Results aggregated with proper statistical measures

## Report Contents

The generated report includes:

1. **Executive Summary**: High-level results
2. **Performance Metrics Table**: All opponents with CIs
3. **Visualizations**: 
   - Win rate bar charts with error bars
   - Score distribution histograms
   - Comprehensive performance comparison
4. **Statistical Analysis**: Significance tests vs baseline
5. **Reproducibility Section**: 
   - Full training configuration
   - Hyperparameters
   - System information
   - Git commit hash
   - Package versions

## LaTeX Table Format

The generated LaTeX table is ready for academic papers:

```latex
\begin{table}[h]
\centering
\begin{tabular}{lcccc}
\toprule
Opponent & Win Rate & Score Diff & Pişti & Capture Eff. \\
\midrule
Random & 0.65 [0.60, 0.70] & 2.3 [1.8, 2.8] & 0.5 & 52.3\% \\
...
\bottomrule
\end{tabular}
\caption{Evaluation Results}
\label{tab:eval_results}
\end{table}
```

## Best Practices

1. **Use Multiple Seeds**: At least 10 seeds for statistical robustness
2. **Sufficient Episodes**: 1000+ episodes per opponent for stable estimates
3. **Save Everything**: Keep all results directories for comparison
4. **Document Experiments**: Use descriptive output directory names
5. **Include Metadata**: Always provide checkpoint path when generating reports

## Example Workflow

```bash
# 1. Train model
python -m training.train_sb3 --config configs/default.yaml

# 2. Evaluate comprehensively
python -m training.evaluate_comprehensive \
    --checkpoint checkpoints/pisti_model_final \
    --n-episodes 2000 \
    --n-seeds 20 \
    --output-dir results/ppo_vs_all_baselines

# 3. Generate report
python -m training.generate_report \
    --results-dir results/ppo_vs_all_baselines \
    --checkpoint checkpoints/pisti_model_final \
    --format markdown,latex

# 4. Use LaTeX table in paper
# Copy results_table.tex into your LaTeX document
```

## Metadata Files

Each checkpoint saves a `*_metadata.json` file with:

```json
{
  "config_path": "configs/default.yaml",
  "algorithm": "PPO",
  "hyperparameters": {...},
  "encoder_type": "MultiHotEncoder",
  "total_timesteps": 1000000,
  "training_start_time": "2024-01-01T10:00:00",
  "training_end_time": "2024-01-01T12:00:00",
  "best_eval_score": 5.2,
  "git_commit_hash": "abc123...",
  "python_version": "3.11.9",
  "package_versions": {...}
}
```

This ensures full reproducibility of experiments.
