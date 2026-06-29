# Contributing

Thanks for your interest in the Face Recognition Attendance System. This guide
covers how to set up a development environment and the conventions the project
follows.

## Development setup

```bash
git clone https://github.com/yashtahlyani/Exploratory.git
cd Exploratory

# install runtime + dev tooling
pip install -r requirements-dev.txt

# verify your environment (libraries + webcam)
python check_env.py

# optional: enable the deep-learning engine (~37 MB)
python download_models.py
```

## Running the checks

```bash
pytest          # unit tests (tests/)
ruff check .    # lint
```

Both run automatically in CI on every push and pull request. Please make sure
they pass locally before opening a PR.

## Project layout

| Path | Responsibility |
|---|---|
| `core/` | Framework-agnostic CV engine (detectors, recognizers, pipeline, augmentation, dataset). **No Tkinter here.** |
| `app.py` | GUI controller — talks only to `core.pipeline`, never to OpenCV directly. |
| `config.py` | All tuneable constants. New magic numbers belong here, not inline. |
| `evaluate.py` | Reproducible model evaluation. |
| `tests/` | pytest suite — headless only (no GUI/webcam dependencies). |

## Conventions

- **Add a new detector or recognizer** by subclassing `BaseDetector` /
  `BaseRecognizer` and registering it in `core/pipeline.py`. The GUI needs no
  changes — that's the point of the abstraction.
- Keep tests headless: never open a camera or a Tkinter window in `tests/`.
- Follow the existing style (4-space indent, type hints, docstrings on public
  methods). `ruff` enforces the basics; see `pyproject.toml`.
- Update `CHANGELOG.md` under an *Unreleased* section for user-facing changes.

## Reporting issues

Use the issue templates (Bug report / Feature request). For bugs, include your
OS, Python version, and the relevant lines from `attendance.log`.
