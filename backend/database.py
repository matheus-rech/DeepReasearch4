"""
Database module for the Deep Research full‑stack application.

This module encapsulates all persistence logic for the app.  It uses
SQLAlchemy to manage a SQLite or PostgreSQL database that stores
citation metadata for systematic review screening.  The goal of this
implementation is to provide a small, dependency‑free persistence
layer that mirrors the core capabilities of the original project
without bringing along unused complexity.

Functions are exposed to initialise the database, insert or update
citation records, perform simple full‑text searches over the title
and abstract fields, fetch individual citations by ID and report
corpus statistics such as total citation count and publication year
distribution.
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import pandas as pd  # type: ignore
from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.pool import StaticPool

logger = logging.getLogger(__name__)

# SQLAlchemy base class used to declare models
Base = declarative_base()


class Citation(Base):
    """ORM model for a single citation entry.

    Each citation holds a minimal set of metadata required for the
    screening workflow: an ID, a title, an abstract, a publication
    year, authors, the journal name, a DOI, optional MeSH terms and
    keywords as comma‑separated strings, the raw source data in JSON
    form and a timestamp.
    """

    __tablename__ = 'citations'

    id = Column(String, primary_key=True)
    title = Column(Text, nullable=False)
    abstract = Column(Text, nullable=True)
    year = Column(Integer, nullable=True)
    authors = Column(Text, nullable=True)  # JSON list encoded as text
    journal = Column(String, nullable=True)
    doi = Column(String, nullable=True)
    mesh_terms = Column(Text, nullable=True)
    keywords = Column(Text, nullable=True)
    raw_data = Column(Text, nullable=True)  # JSON encoded string
    created_at = Column(DateTime, nullable=True)


def _get_database_url() -> str:
    """Resolve the database URL from environment variables.

    The application supports both SQLite (used by default) and
    PostgreSQL.  A DATABASE_URL environment variable can be provided to
    override the default.  When a PostgreSQL URL beginning with
    ``postgres://`` is supplied, it is rewritten to ``postgresql://``
    because SQLAlchemy does not recognise the former scheme.
    """
    url = os.getenv('DATABASE_URL')
    if url:
        if url.startswith('postgres://'):
            url = url.replace('postgres://', 'postgresql://', 1)
        logger.info(f"Using database URL from environment: {url}")
        return url
    # fallback to a local SQLite database in the project directory
    default_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'citations.db')
    logger.info(f"Using local SQLite database at {default_path}")
    return f"sqlite:///{default_path}"


# Create engine and session factory.  StaticPool is used for SQLite to allow
# sharing connections across threads when running tests or the UI.
DATABASE_URL = _get_database_url()
if DATABASE_URL.startswith('sqlite'):  # special handling for SQLite
    engine = create_engine(
        DATABASE_URL,
        connect_args={'check_same_thread': False},
        poolclass=StaticPool
    )
else:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def init_db() -> None:
    """Initialise the database schema.

    Creates all tables defined on the Base metadata.  If tables
    already exist this function is a no‑op.  Logging is used to
    indicate when the schema has been created.
    """
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialised (tables created if missing)")


@contextmanager
def get_db() -> Any:
    """Provide a transactional scope for database operations.

    This helper yields a SQLAlchemy session and ensures that it is
    properly committed or rolled back.  Sessions are always closed
    after use.
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def bulk_insert_citations(df: pd.DataFrame) -> Dict[str, int]:
    """Insert or update a list of citations from a Pandas DataFrame.

    This routine iterates over the rows in ``df`` and inserts a new
    citation if it does not already exist.  If the ID exists, the
    existing record is updated.  This behaviour makes repeated
    invocations idempotent.  The function returns a dictionary with
    counts of inserted, updated and skipped records.
    """
    stats = {"inserted": 0, "updated": 0, "skipped": 0, "total": len(df)}
    for _, row in df.iterrows():
        citation_id = str(row.get('id', '')).strip()
        if not citation_id:
            stats["skipped"] += 1
            continue
        with get_db() as session:
            existing = session.query(Citation).filter(Citation.id == citation_id).first()
            # prepare serialisable fields
            authors = row.get('authors') or []
            if isinstance(authors, list):
                authors_json = json.dumps(authors)
            else:
                authors_json = json.dumps([authors])
            mesh_terms = row.get('mesh_terms') or []
            mesh_terms_str = ', '.join(mesh_terms) if isinstance(mesh_terms, list) else str(mesh_terms)
            keywords = row.get('keywords') or []
            keywords_str = ', '.join(keywords) if isinstance(keywords, list) else str(keywords)
            raw_data = row.get('raw_data') or {}
            raw_data_str = json.dumps(raw_data)
            if existing:
                # update fields
                existing.title = row.get('title', existing.title)
                existing.abstract = row.get('abstract', existing.abstract)
                existing.year = row.get('year') or existing.year
                existing.authors = authors_json
                existing.journal = row.get('journal', existing.journal)
                existing.doi = row.get('doi', existing.doi)
                existing.mesh_terms = mesh_terms_str
                existing.keywords = keywords_str
                existing.raw_data = raw_data_str
                stats["updated"] += 1
            else:
                new_citation = Citation(
                    id=citation_id,
                    title=row.get('title', ''),
                    abstract=row.get('abstract', ''),
                    year=row.get('year'),
                    authors=authors_json,
                    journal=row.get('journal', ''),
                    doi=row.get('doi', ''),
                    mesh_terms=mesh_terms_str,
                    keywords=keywords_str,
                    raw_data=raw_data_str
                )
                session.add(new_citation)
                stats["inserted"] += 1
    return stats


def search_citations(query: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Perform a simple full‑text search over titles and abstracts.

    The search uses a case‑insensitive LIKE query to find citations
    where the query string appears in either the title or the abstract.
    Results are returned as a list of dictionaries containing the
    citation ID, title, a short snippet and a placeholder URL.

    Args:
        query: Search query string.
        limit: Optional maximum number of results to return.

    Returns:
        A list of dictionaries with keys ``id``, ``title``, ``text`` and ``url``.
    """
    if not query or not query.strip():
        return []
    pattern = f"%{query.strip()}%"
    with get_db() as session:
        results = session.query(Citation).filter(
            (Citation.title.ilike(pattern)) | (Citation.abstract.ilike(pattern))
        )
        if limit:
            results = results.limit(limit)
        formatted: List[Dict[str, Any]] = []
        for citation in results:
            snippet_source = citation.abstract or citation.title
            snippet = (snippet_source[:200] + '...') if snippet_source and len(snippet_source) > 200 else (snippet_source or '')
            url = ''
            # attempt to build a PubMed or DOI URL if possible
            if citation.id.startswith('PMID:'):
                pmid = citation.id.replace('PMID:', '')
                url = f'https://pubmed.ncbi.nlm.nih.gov/{pmid}/'
            elif citation.doi:
                url = f'https://doi.org/{citation.doi}'
            formatted.append({
                'id': citation.id,
                'title': citation.title,
                'text': snippet,
                'url': url,
            })
        return formatted


def get_all_citations(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Return all citations in the database.

    If a limit is provided, the result set is truncated accordingly.
    """
    with get_db() as session:
        query = session.query(Citation)
        if limit:
            query = query.limit(limit)
        citations: List[Dict[str, Any]] = []
        for c in query:
            citations.append({
                'id': c.id,
                'title': c.title,
                'abstract': c.abstract,
                'year': c.year,
                'authors': json.loads(c.authors) if c.authors else [],
                'journal': c.journal,
                'doi': c.doi,
                'mesh_terms': c.mesh_terms.split(', ') if c.mesh_terms else [],
                'keywords': c.keywords.split(', ') if c.keywords else [],
            })
        return citations


def fetch_citation(citation_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a single citation by its ID.

    Args:
        citation_id: The unique citation identifier.

    Returns:
        A dictionary representing the citation if found, otherwise ``None``.
    """
    with get_db() as session:
        citation = session.query(Citation).filter(Citation.id == citation_id).first()
        if not citation:
            return None
        return {
            'id': citation.id,
            'title': citation.title,
            'abstract': citation.abstract,
            'year': citation.year,
            'authors': json.loads(citation.authors) if citation.authors else [],
            'journal': citation.journal,
            'doi': citation.doi,
            'mesh_terms': citation.mesh_terms.split(', ') if citation.mesh_terms else [],
            'keywords': citation.keywords.split(', ') if citation.keywords else [],
            'raw_data': json.loads(citation.raw_data) if citation.raw_data else {},
        }


def get_corpus_stats() -> Dict[str, Any]:
    """Return simple statistics about the loaded citation corpus."""
    with get_db() as session:
        total = session.query(Citation).count()
        # compute year distribution
        year_counts: Dict[int, int] = {}
        for year, count in session.query(Citation.year, text('COUNT(*)')).group_by(Citation.year):
            if year is not None:
                year_counts[int(year)] = int(count)
        year_distribution = [
            {'year': year, 'count': count} for year, count in sorted(year_counts.items())
        ]
        return {
            'total_citations': total,
            'year_distribution': year_distribution,
        }


def semantic_search_citations(query: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Placeholder for semantic search.

    This function is included for API compatibility.  In this simplified
    implementation, semantic search falls back to standard full‑text
    search.  In a production deployment you could replace this with
    vector database queries or embeddings.
    """
    return search_citations(query, limit)
