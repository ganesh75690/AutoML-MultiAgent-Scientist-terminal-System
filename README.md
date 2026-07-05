# AutoML Multi-Agent Scientist

A terminal-only enterprise AutoML platform that accepts a CSV file and orchestrates an autonomous multi-agent workflow for planning, cleaning, EDA, feature engineering, model debate, tournament selection, tuning, evaluation, explainability, critique, mentoring, reporting, and experiment tracking.

## Run

```bash
python main.py data/sample_dataset.csv
```

## What It Does

- Detects the target column, task type, business domain, dataset quality, risk level, and sensitive columns.
- Builds an execution plan before training starts.
- Creates temporary specialist agents when the dataset suggests a niche domain.
- Runs an AI debate between model specialist agents before selecting the champion.
- Uses self-healing retries for agent failures.
- Tracks every run as a unique experiment and stores it in `experiments/`.
- Produces terminal dashboards, colored logs, model artifacts, metrics, predictions, and a documentation bundle.
- Generates explainability artifacts with SHAP, permutation importance, partial dependence, correlation analysis, error analysis, and a LIME-style local surrogate.

## Output Locations

- `models/checkpoints/` for trained model checkpoints
- `reports/<dataset>_<experiment_id>/` for run artifacts and reports
- `docs/<experiment_id>/` for generated model cards, data cards, decision logs, and executive summaries
- `experiments/` for experiment metadata and the cross-run index
- `logs/` for `workflow.log`, `agent.log`, `error.log`, `performance.log`, and `audit.log`
- `memory/` for global and agent-specific memories

## Folder Structure

- `agents/`
- `configs/`
- `data/`
- `datasets/`
- `docs/`
- `experiments/`
- `logs/`
- `memory/`
- `models/`
- `plugins/`
- `reports/`
- `tests/`
- `utils/`
- `workflows/`

## Notes

- The project is intentionally terminal-only. No web UI is included.
- Optional libraries such as XGBoost, LightGBM, CatBoost, and SHAP are used when available.
- The workflow is resumable and will reuse prior artifacts when a matching dataset run has already completed.
- A sample dataset is available at `data/sample_dataset.csv` and `datasets/sample_dataset.csv`.
