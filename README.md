# Learning Consequence Fields on Protected Service Frontiers for Mixed Real-Time Scheduling

This repository contains the scheduler, experiment configurations, baseline adapters, and evaluation scripts used in the paper.

The package includes:

- protected-frontier consequence scheduling with uncertainty-reserved feasibility
- pretrained consequence-field and scalar-priority models
- six paper-based baseline adapters, three ablations, and internal controls
- fixed nominal-load and load-sweep configurations with root-level aggregation

Repository layout:

- `src/protected_frontier/`: scheduler, learned consequence field, and metrics
- `src/baselines/`: baseline adapters and replay utilities
- `configs/`: experiment matrix and method parameters
- `data/`: pretrained models and task-level evaluation inputs
- `manifests/`: task mappings used by the adapters
- `run.py`: experiment entry point

Install the dependencies and check the packaged inputs:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py check
```

Run the evaluation or a selected configuration:

```bash
python run.py all
python run.py run --root 4277551675 --load 1.00 --method protected_frontier
python run.py summarize
```

Results are written to `results/`. Use `--save-details` to retain per-job and per-event outputs.
