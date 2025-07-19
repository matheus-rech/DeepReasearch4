"""
MCP server for the Deep Research full‑stack application.

This module defines a simple HTTP API using FastAPI that exposes
search, fetch, health and corpus statistics endpoints.  It wraps the
functions provided by the database module to perform all persistence
operations.  The server can be run directly via uvicorn or
programmatically by calling the ``run`` function defined below.

Endpoints:

* **GET /health** – Return a basic health status.  Intended for
  monitoring and readiness checks.

* **GET /corpus** – Return summary statistics about the loaded
  citations, including the total count and publication year
  distribution.

* **GET /search** – Search the citation corpus.  Accepts query
  parameters ``query`` for the search term, ``limit`` for the maximum
  number of results, and ``mode`` which may be ``fulltext`` (default)
  or ``semantic`` (currently a passthrough to full‑text search).  The
  response contains a ``results`` list of citations with minimal
  fields (id, title, snippet, url).

* **GET /fetch/{citation_id}** – Retrieve the full details for a
  citation given its identifier.  Returns a JSON object containing
  all stored fields.

The application configures CORS to allow requests from any origin so
that the Streamlit UI running on a different port can call the API
without restriction.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from . import database as db

logger = logging.getLogger(__name__)


# Create the FastAPI application
app = FastAPI(title="DeepResearch MCP Server", version="1.0.0")

# Configure CORS so that the Streamlit frontend can access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event() -> None:
    """Initialise the database when the application starts.

    This event handler runs once on server startup and ensures that the
    database schema exists before any requests are processed.
    """
    db.init_db()
    logger.info("MCP server startup complete")


@app.get("/health", response_model=Dict[str, Any])
async def health_check() -> Dict[str, Any]:
    """Return a basic health status.

    This endpoint returns a JSON object indicating that the server is
    healthy.  It can be used by load balancers or orchestration
    systems to verify liveness.
    """
    return {
        "status": "healthy",
        "server": "DeepResearchMCP",
    }


@app.get("/corpus", response_model=Dict[str, Any])
async def corpus_info() -> Dict[str, Any]:
    """Return summary statistics about the citation corpus.

    If an error occurs while computing statistics, a 500 HTTP error
    will be returned with a corresponding message.
    """
    try:
        stats = db.get_corpus_stats()
        return stats
    except Exception as e:  # pragma: no cover - just logs
        logger.error(f"Error computing corpus info: {e}")
        raise HTTPException(status_code=500, detail="Failed to compute corpus info")


@app.get("/search", response_model=Dict[str, Any])
async def search(
    query: str = "",
    limit: Optional[int] = None,
    mode: str = "fulltext",
) -> Dict[str, Any]:
    """Search the citation corpus.

    Args:
        query: The search string.  An empty string or ``*`` returns
            all citations up to ``limit``.
        limit: Optional maximum number of results to return.
        mode: Search mode.  ``fulltext`` performs a simple LIKE
            search on title and abstract.  ``semantic`` currently
            falls back to ``fulltext``.

    Returns:
        A dictionary with a single ``results`` key containing a list
        of citation objects.  Each object has ``id``, ``title``,
        ``text`` (snippet) and ``url`` fields.
    """
    try:
        # Return all citations if no query provided
        if not query or query.strip() == "*":
            citations = db.get_all_citations(limit)
            formatted: List[Dict[str, Any]] = []
            for c in citations:
                snippet_source = c["abstract"] or c["title"] or ""
                snippet = (snippet_source[:200] + "...") if len(snippet_source) > 200 else snippet_source
                url = ""
                cid = c["id"]
                if cid.startswith("PMID:"):
                    pmid = cid.replace("PMID:", "")
                    url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                elif c.get("doi"):
                    url = f"https://doi.org/{c['doi']}"
                formatted.append({
                    "id": cid,
                    "title": c["title"],
                    "text": snippet,
                    "url": url,
                })
            return {"results": formatted}
        # Dispatch based on search mode
        if mode == "semantic":
            results = db.semantic_search_citations(query, limit)
        else:
            results = db.search_citations(query, limit)
        return {"results": results}
    except Exception as e:  # pragma: no cover - just logs
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/fetch/{citation_id}", response_model=Dict[str, Any])
async def fetch(citation_id: str) -> Dict[str, Any]:
    """Retrieve the full details for a single citation.

    Args:
        citation_id: The identifier of the citation (e.g. ``PMID:12345``).

    Returns:
        A JSON object representing the citation.  If the citation is
        not found, a 404 error is raised.
    """
    try:
        citation = db.fetch_citation(citation_id)
        if not citation:
            raise HTTPException(status_code=404, detail="Citation not found")
        return citation
    except HTTPException:
        raise
    except Exception as e:  # pragma: no cover - just logs
        logger.error(f"Fetch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def run(host: str = "0.0.0.0", port: int = 8001) -> None:
    """Run the MCP server using uvicorn.

    This helper wraps uvicorn to start the application.  It is
    intended for CLI use and test convenience.  In production you may
    prefer to run uvicorn directly or under a process manager.
    """
    import uvicorn  # type: ignore

    logger.info(f"Starting MCP server on {host}:{port}")
    # Suppress uvicorn access logs to reduce noise
    uvicorn.run(
        "full_stack_app.backend.mcp_server:app",
        host=host,
        port=port,
        log_level="info",
        reload=False,
    )