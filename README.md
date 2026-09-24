# multi-agent-financial-analyst
Autonomous multi-agent investment research system built with LLM agents: planning, tool use, self-reflection, and iterative learning via prompt chaining, routing, and evaluator-optimizer patterns

# Multi-Agent Financial Analyst

An autonomous multi-agent LLM system for investment research. Given a stock symbol, the system plans its own research steps, dynamically calls financial data tools, critiques and refines its own output, and retains brief memory notes across runs to improve future analyses.

## Team

- Sudhakaran Srinivasan
- April Bradick
- Jeffery Smith


## Overview

Investment firms increasingly deploy multi-agent LLM stacks to parse news, earnings, and market signals at scale. This project implements the same patterns at a project scale: an Investment Research Agent that reasons, plans, and acts across specialized sub-agents rather than following a fixed script.

## Agent Functions

The core Investment Research Agent demonstrates:

1. **Planning** — decomposes a stock symbol into a research plan (e.g., fetch financials → fetch news → fetch filings → synthesize).
2. **Tool use** — dynamically calls external APIs and datasets (Yahoo Finance, NewsAPI, FRED, SEC EDGAR, Alpha Vantage) as needed rather than following a fixed call sequence.
3. **Self-reflection** — evaluates its own draft analysis for quality/completeness before finalizing.
4. **Learning across runs** — persists brief memory notes (e.g., past analyses, lessons, recurring patterns) that inform future runs on the same or related symbols.

## Workflow Patterns

1. **Prompt Chaining** — Ingest News → Preprocess → Classify → Extract → Summarize.
2. **Routing** — Directs incoming content to the right specialist sub-agent (earnings analyzer, news analyzer, market/price analyzer).
3. **Evaluator–Optimizer** — Generates an analysis, evaluates its quality against a rubric, and refines it using that feedback in a loop.

## Tech Stack

- **Language/runtime**: Python 3.x
- **Data sources**: `yfinance` (prices/financials), NewsAPI / Kaggle financial news datasets, FRED API (macro data), SEC EDGAR (filings), Alpha Vantage (free tier)
- **LLM/agent framework**: TBD
- **Style**: PEP 8

## Project Structure

```
multi-agent-financial-analyst/
├── agents/            # Agent definitions (planner, researcher, evaluator, specialists)
├── workflows/         # Prompt chaining, routing, evaluator-optimizer implementations
├── notebooks/         # Exploration and the final submitted code notebook
├── data/              # Cached/sample data (raw API data excluded via .gitignore)
├── memory/            # Persisted run notes/learnings across sessions
├── tests/             # pytest —  5-6 tests 
├── .github/
│   └── workflows/
│       └── ci.yml     # runs pytest + lint on every push
├── .env.example       # documents required keys; real .env is gitignored
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/sudhakaran-srinivasan/multi-agent-financial-analyst.git
cd multi-agent-financial-analyst
pip install -r requirements.txt
```

Add required API keys to a local `.env` file (never committed — see `.gitignore`):

```
NEWSAPI_KEY=
FRED_API_KEY=
ALPHAVANTAGE_KEY=
```

## Usage

*(TBD once the agent entry point is implemented — e.g., `python run_research.py --symbol AAPL`.)*

## Evaluation & Iteration

*(TBD — document how output quality is measured and how the evaluator-optimizer loop improves results across iterations.)*

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
