# Multi-Agent Financial Analyst

A Python investment research agent that plans research for a stock symbol, calls financial and news tools, and optionally writes and evaluates an analysis. Saved analyses can be compared across runs.

## Team

- Sudhakaran Srinivasan
- April Bradick
- Jeffery Smith


## Overview

The application combines an LLM-driven research loop with Python workflows for news processing, evidence checks, reflection, evaluation, and saved history. Specialist roles use different prompts with the same configured model by default.

The command-line application lives in `src/finagent/`. Notebooks contain demonstrations and experiments; they are not executed by the command-line application.

## Agent Functions

The core Investment Research Agent demonstrates:

1. **Planning** — creates a research plan after checking the company profile.
2. **Tool use** — lets the model select fundamentals, earnings-history, and news tools using Yahoo Finance and Finnhub data.
3. **Reflection** — uses Python rules to retry eligible missing or failed tool calls once before synthesis.
4. **Analysis and evaluation** — optionally writes an evidence-based analysis, then scores and revises it against a rubric.
5. **Saved history** — stores analysis summaries and compares stance, confidence, signals, and data gaps with a recent run for the same ticker. Prior records are not currently passed into the synthesis prompt, and the model is not trained or fine-tuned by these records.

## Workflow Patterns

1. **Prompt Chaining** — Ingest News → Preprocess → Classify → Extract → Summarize.
2. **Routing** — Uses an article's category to select an earnings, market, or general-news specialist prompt.
3. **Evaluator–Optimizer** — Generates an analysis, evaluates its quality against a rubric, and refines it using that feedback in a loop.

## Tech Stack

- **Language/runtime**: Python 3.11 or newer. The code uses `datetime.UTC`, although `pyproject.toml` currently declares Python >=3.10.
- **Data sources**: Yahoo Finance through `yfinance` for company profiles and fundamentals; Finnhub for earnings history and company news.
- **Model access**: OpenRouter through the `openai` Python client. The default model is `openai/gpt-4o-mini`; the judge uses the same model by default.
- **Agent orchestration**: Python functions and workflows; LangChain and LangGraph are used in notebook experiments, not required by the command-line application.
- **Development tools**: `pytest` and `ruff`, installed with the `dev` dependency group.

Model names, the API base URL, and research-loop limits are configured in [config.py](src/finagent/config.py).

## Project Structure

```text
multi-agent-financial-analyst/
├── src/finagent/
│   ├── agent/         # Planning, tool loop, reflection, signals, synthesis
│   ├── tools/         # Financial-data and news tools
│   ├── workflows/     # News pipeline and evaluator-optimizer
│   ├── config.py      # Model settings and environment loading
│   ├── contracts.py   # Shared data structures
│   ├── memory.py      # Save analyses and compare runs
│   ├── pipeline.py    # Connect the research stages
│   └── run_agent.py   # Command-line entry point
├── notebooks/         # Demonstrations and separate experiments
├── tests/             # Automated tests with simulated external responses
├── data/              # Created as needed; memory.jsonl is ignored by Git
├── .env.example       # Example variable names; keep real keys in .env
├── .gitignore
├── pyproject.toml     # Package metadata and dependencies
├── requirements.txt   # Currently empty; install from pyproject.toml
└── README.md
```

## Setup

Install Python 3.11 or newer and Git. If you already cloned this repository, open a terminal in that existing folder and skip the clone command.

```powershell
git clone https://github.com/sudhakaran-srinivasan/multi-agent-financial-analyst.git
cd multi-agent-financial-analyst
```

Create and activate an isolated environment in Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

On macOS or Linux, use `python3 -m venv .venv` and `source .venv/bin/activate`, then run the same install command.

The editable install (`-e`) makes changes under `src/` available without reinstalling the project. `[dev]` also installs development tools. Dependencies are defined in `pyproject.toml`; `requirements.txt` is currently empty.

Create a local `.env` file if one does not already exist:

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

Edit `.env` and fill in these keys for live research:

```dotenv
OPENROUTER_API_KEY=your_openrouter_key
FINNHUB_API_KEY=your_finnhub_key
```

`OPENAI_API_KEY`, also listed in `.env.example`, is used by separate notebook experiments that call OpenAI directly. It is not required by the command-line application. Keep `.env` private; it is ignored by Git. OpenRouter model calls can consume account credit, and external services can impose usage limits.

Activate the environment again when opening a new terminal. To confirm the selected interpreter, run `python -c "import sys; print(sys.executable)"`; its path should point inside `.venv`. Run `deactivate` when finished.

## Usage

Run commands from the repository root with the environment activated. Display the available arguments without making API calls:

```powershell
python -m finagent.run_agent --help
```

Collect research data for a stock ticker, then optionally generate an analysis:

```powershell
python -m finagent.run_agent NVDA --no-memory
python -m finagent.run_agent NVDA --analyze --no-memory
```

Replace `NVDA` with another company stock ticker, such as `AAPL`. ETFs, indexes, and invalid tickers are rejected by the company-profile check. Both research commands use live APIs and LLM calls; omitting `--analyze` does not make a run offline.

| Argument | Behavior |
|---|---|
| `ticker` | Required company stock symbol. |
| `--analyze` | Write an analysis after gathering data and running reflection. |
| `--evaluate` | Judge and potentially revise the analysis; use together with `--analyze`. |
| `--no-memory` | Disable reading and writing saved analysis history. |
| `--json` | Print structured JSON instead of the text report. |
| `-h`, `--help` | Display command help and exit. |

Save a structured report to a file:

```powershell
python -m finagent.run_agent NVDA --analyze --no-memory --json > nvda-result.json
```

The text report shows the profile, plan, tool trace, collected results, and any analysis or evaluation requested. Check data gaps and stage errors even when the command completes: later-stage failures can leave partial results. `stop: done` means the data loop finished; `step_cap` means it reached the configured limit; `fatal_error` means it stopped on a fatal problem.

Without `--no-memory`, non-fallback analyses are appended to `data/memory.jsonl`, relative to the working directory. The newest eligible record for the same ticker within the 30-day age limit is used for comparisons. A saved record does not imply that evaluation passed; evaluation may be disabled or may fail its rubric.

For interactive examples, start with [demo_queries.ipynb](notebooks/demo_queries.ipynb). Use a kernel with the project installed. Notebook-specific libraries are not included in the CLI dependency list; [Jeff_News_Ingest.ipynb](notebooks/Jeff_News_Ingest.ipynb), for example, is a separate Colab-oriented experiment with its own dependencies, secrets, and news implementation.

## Testing

Run the automated tests without API keys or live model calls:

```powershell
python -m pytest -q
```

For a focused check of command-line behavior:

```powershell
python -m pytest tests/test_run_agent.py -v
```

Tests use simulated responses from external services. Passing tests verify the tested application behavior, not live API access or model accuracy. See the next section for live analysis evaluation.

## Evaluation & Iteration

Evaluation checks whether an analysis is supported by the data collected during the run. It is optional: `--analyze` writes an analysis, while adding `--evaluate` enables the evaluator-optimizer loop.

```bash
python -m finagent.run_agent NVDA --analyze --evaluate --no-memory
```

This command requires the OpenRouter and Finnhub keys configured above and makes live API calls. `--no-memory` disables reading and writing saved analyses; it does not disable model calls.

An LLM judge compares the draft with the raw tool results and Python-computed signals. It scores six criteria from **1 (weak) to 5 (strong)** using the anchors in [evaluation_rubric.py](src/finagent/workflows/evaluation_rubric.py):

| Criterion | What is checked |
|---|---|
| Evidence support | Claims are backed by specific collected evidence. |
| Numeric accuracy | Numbers match the tool results, allowing correct rounding. |
| Stance consistency | The stance follows the evidence and acknowledges conflicting signals. |
| Risk coverage | Risks are traceable to the data and important gaps are stated. |
| Confidence calibration | Confidence reflects the strength and completeness of the evidence. |
| Source quality | Sources are named for the evidence used; source authenticity is not verified. |

Python validates the judge's scores and decides the verdict. A draft passes only when its average score is **at least 4.0**, every criterion scores **at least 3**, and the judge reports **no critical issues**. These thresholds are defined in [evaluation_rules.py](src/finagent/workflows/evaluation_rules.py).

If the draft needs revision, the synthesis model receives the previous draft and feedback, then rewrites the analysis using the same collected evidence. Each revision goes through the same evidence and confidence checks as the initial draft. The loop stops on a pass or after **two revisions** (up to three evaluated drafts). It keeps the best evaluated draft: a passing draft first, then fewer critical issues, then the higher average score; ties keep the earlier draft. A run that never passes is explicitly reported as not passed. Judge or revision failures stop the loop and preserve the best available draft; if no judgment succeeded, the run is reported as not evaluated. Rule-based fallback analyses are not evaluated, and fallback revisions are not accepted.

The terminal report shows scores, revision count, and the stopping reason. Add `--json` for structured evaluation results. The implementation is in [evaluator_optimizer.py](src/finagent/workflows/evaluator_optimizer.py).

The automated tests described above cover evidence checks, confidence limits, judge-output validation, verdict thresholds, revision limits, and failure handling. For development iterations, compare the initial and revised analyses on the same collected data, record scores and critical issues, and manually review whether the revisions address the feedback before changing prompts or thresholds.

The writer and judge currently use the same model by default (`JUDGE_MODEL = MODEL` in [config.py](src/finagent/config.py)), which can introduce self-evaluation bias. The judge has not been calibrated against human ratings and can miss unsupported claims. Rubric scores measure support against the supplied data; they do not establish factual correctness or predict investment performance.

## Troubleshooting

| Symptom | What to check |
|---|---|
| `python` is not recognized | Install Python and open a new terminal, or use `py -m venv .venv` if the Windows Python launcher is available. |
| PowerShell blocks `Activate.ps1` | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate again. This applies only to the current terminal session. |
| `No module named finagent` | Activate `.venv`, change to the repository root, and run `python -m pip install -e ".[dev]"`. |
| Import error involving `datetime.UTC` | Use Python 3.11 or newer and recreate the environment with that interpreter. |
| A required key is missing | Check `.env` in the repository root. The CLI needs `OPENROUTER_API_KEY` and Finnhub tools need `FINNHUB_API_KEY`. Existing environment variables take precedence over `.env`. |
| Authentication, credit, or rate-limit errors | Check the relevant service account, key, available credit, and usage limits. |
| Missing news or financial fields | Inspect tool errors and data gaps; providers may have incomplete coverage or temporary failures. |

In PowerShell, `$LASTEXITCODE` reports the last native command's exit status. The CLI returns `0` for a nonfatal research run, `1` for a fatal research result, and `2` for a caught setup problem. Exit status `0` does not mean every stage succeeded or the analysis passed evaluation; inspect the report or JSON fields as well.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
