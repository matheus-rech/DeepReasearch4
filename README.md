# Deep Research Systematic Review Screener

This repository contains a simplified yet fully functional rewrite of
the original `DeepResearch2` project.  The goal of this rewrite is to
retain the essential components needed to perform systematic review
screening with the OpenAI Deep Research API while removing unused or
duplicated files, merging disparate branches and modernising the
structure.  The new codebase is organised into a clear `backend` and
`frontend` layout and is ready to run locally or in a container.

## Overview

The application supports the following high‑level workflow:

1. **Load citations** – Upload a reference manager export (PubMed
   XML/MEDLINE, RIS, CSV or EndNote XML).  The parser detects the
   format automatically, normalises the fields and loads the data
   into a SQLite or PostgreSQL database.  A simple validator reports
   critical issues such as missing abstracts.

2. **Define criteria** – Specify PICOTT elements (population,
   intervention, comparator, outcome, timeframe and study type) and
   additional inclusion or exclusion rules.  These criteria will be
   passed to the Deep Research model.

3. **AI screening** – Launch a screening job through the OpenAI
   Deep Research API.  The model uses a local MCP server (the
   included `backend/mcp_server.py`) to search and retrieve
   citations.  Once the job completes, the response is parsed and
   stored.

4. **Review results** – View a summary of included and excluded
   citations, download the results as CSV and run a consistency
   analysis with the ICE critic to flag potential quality issues.

## Project structure

```
full_stack_app/
├── backend/
│   ├── __init__.py
│   ├── database.py          # SQLAlchemy models and persistence helpers
│   ├── parsers.py           # Citation file parsers (PubMed, RIS, CSV, etc.)
│   ├── data_validator.py    # Validate citation completeness and quality
│   ├── deep_research.py     # Wrapper for the OpenAI Deep Research API
│   ├── ice_critic.py        # Analyse screening results for inconsistencies
│   └── mcp_server.py        # FastAPI server exposing search/fetch endpoints
├── frontend/
│   ├── __init__.py
│   ├── app.py               # Streamlit UI for the workflow
│   └── main.py              # CLI to run the server and UI
├── tests/
│   ├── __init__.py
│   └── test_app.py          # Unified unit tests covering core functions
└── README.md                # This document
```

## Running the application

1. Install dependencies (ideally in a virtual environment):

```bash
pip install -r requirements.txt
```

2. Copy `.env.example` to `.env` and set your `OPENAI_API_KEY` and
   optionally `DATABASE_URL` and `MCP_URL`.  If you omit
   `DATABASE_URL`, a SQLite database will be created in the project
   directory.  `MCP_URL` defaults to `http://localhost:8001`.

3. Start both the MCP server and the Streamlit UI:

```bash
python -m full_stack_app.frontend.main both
```

The UI will be available at `http://localhost:8000` and the MCP
server at `http://localhost:8001`.

Alternatively, you can run the server and UI individually:

```bash
python -m full_stack_app.frontend.main server  # only the MCP server
python -m full_stack_app.frontend.main ui      # only the Streamlit UI
```

## Environment variables

The application reads several environment variables:

- `OPENAI_API_KEY` – API key for the OpenAI Deep Research API.  You must
  set this to run AI‑based screening.
- `DATABASE_URL` – SQLAlchemy URL for the database.  Defaults to a
  local SQLite file under `full_stack_app/citations.db`.
- `MCP_URL` – Base URL of the MCP server used by the Deep Research
  API.  Defaults to `http://localhost:8001`.

See `.env.example` for a template.

## Merging legacy components

The original repository contained numerous test files, UI analyses,
deployment scripts and other experimental components.  In this
rewrite, non‑functional or duplicative files have been omitted or
merged.  Only core functionality required for a full‑stack
implementation has been kept.  The unified `test_app.py` covers
parsers, database operations, validation and the ICE critic.  The
workflows directory, production readiness reports, GitHub actions and
similar auxiliary files from other branches have been intentionally
excluded to keep the codebase concise.  Should you wish to
reintroduce additional tooling such as CI workflows or extended
documentation, they can be added in a separate branch.

## Contributing

Feel free to open issues or submit pull requests to improve the
parsers, add new data sources or enhance the UI.  This project
serves as a minimal yet functional starting point for building more
complex systematic review applications on top of the OpenAI Deep
Research platform.
