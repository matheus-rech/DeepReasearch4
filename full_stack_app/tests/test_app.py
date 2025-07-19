"""
Unified test suite for the Deep Research full‑stack application.

These tests focus on the core backend functionality including
parsers, the persistence layer, validation, and analysis helpers.
External services such as the OpenAI API are not invoked.  The
database is configured to use an in‑memory SQLite instance for
isolation, and environment variables are manipulated as needed.
"""

from __future__ import annotations

import importlib
import os
import json
from io import StringIO

import pandas as pd  # type: ignore

from full_stack_app.backend import parsers, data_validator, ice_critic


def test_parse_csv_and_database_operations() -> None:
    """Verify CSV parsing and basic database CRUD operations."""
    # Prepare a simple CSV string
    csv_content = """PMID,Title,Abstract,Year,Authors,Journal,DOI,Keywords
PMID:12345,Test Title,This is an abstract.,2020,"Doe J; Smith A",Journal A,10.1000/test,"cancer; therapy"
PMID:67890,Another Title,Another abstract.,2019,"Brown B",Journal B,,"research"
"""
    csv_file = StringIO(csv_content)
    df = parsers.parse_csv(csv_file)
    assert len(df) == 2
    # Configure an in‑memory database for isolation
    os.environ['DATABASE_URL'] = 'sqlite:///:memory:'
    # Reload the database module to pick up the new environment
    from full_stack_app.backend import database as db  # type: ignore
    importlib.reload(db)
    db.init_db()
    stats = db.bulk_insert_citations(df)
    assert stats['inserted'] == 2
    # Search for a keyword present in the first abstract
    results = db.search_citations("abstract", limit=10)
    assert len(results) == 2
    # Fetch a citation by ID
    citation = db.fetch_citation('PMID:12345')
    assert citation is not None
    assert citation['title'] == 'Test Title'
    # Verify corpus statistics
    stats_info = db.get_corpus_stats()
    assert stats_info['total_citations'] == 2
    years = {entry['year'] for entry in stats_info['year_distribution']}
    assert years == {2019, 2020}


def test_validation_and_quality_scoring() -> None:
    """Ensure the citation validator computes quality metrics correctly."""
    # Construct a DataFrame with a missing abstract
    data = {
        'id': ['ID1', 'ID2'],
        'title': ['Title1', 'Title2'],
        'abstract': ['', 'Valid abstract'],
        'year': [2021, 2022],
        'authors': [['Author A'], ['Author B']],
        'journal': ['Journal1', 'Journal2'],
        'doi': ['', ''],
        'mesh_terms': [[], []],
        'keywords': [[], []],
    }
    df = pd.DataFrame(data)
    validator = data_validator.CitationValidator()
    _, report = validator.validate_citations(df)
    # One missing abstract should reduce the quality score
    assert report['critical_issues']['missing_abstracts'] == 1
    assert report['quality_score'] < 100


def test_ice_critic_analysis() -> None:
    """Test the ICE critic on synthetic screening results."""
    screening_results = [
        {
            'id': 'ID1',
            'decision': 'Include',
            'confidence': 'high',
            'picott': {
                'population': 'patients',
                'intervention': 'drug',
                'comparison': 'placebo',
                'outcome': 'improved',
                'timeframe': '6 months',
                'studyType': 'RCT',
            },
            'reason': '',
        },
        {
            'id': 'ID2',
            'decision': 'Exclude',
            'confidence': 'high',
            'picott': {
                'population': '',
                'intervention': '',
                'comparison': '',
                'outcome': '',
                'timeframe': '',
                'studyType': '',
            },
            'reason': 'not relevant population',
        },
    ]
    pico = {
        'population': 'patients',
        'intervention': 'drug',
        'comparator': 'placebo',
        'outcome': 'improved',
        'timeframe': '6 months',
        'study_type': 'RCT',
    }
    analysis = ice_critic.analyze_screening_consistency(screening_results, pico)
    # One high‑confidence exclusion should generate a low severity issue
    types = {issue['type'] for issue in analysis['issues']}
    assert 'high_confidence_exclusion' in types
    assert analysis['summary']['total_issues'] >= 1