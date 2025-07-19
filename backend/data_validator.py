"""
Data validation for citation records.

The goal of this module is to verify the integrity of citation
metadata before inserting it into the database.  Citations with
missing IDs, titles or abstracts are flagged and optional
enhancements (such as generating IDs from titles or keywords) are
performed.  A summary report describing the data quality is returned
alongside the validated DataFrame.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, List, Tuple

import pandas as pd  # type: ignore

try:
    # Optional function used to enrich citations with missing abstracts
    from .parsers import parse_pubmed_search, parse_arxiv_search  # type: ignore
    ENRICH_AVAILABLE = True
except Exception:
    ENRICH_AVAILABLE = False

logger = logging.getLogger(__name__)


class CitationValidator:
    """Validate and enhance a list of citations.

    Each instance of the validator tracks summary statistics about the
    citations it processes.  The primary entry point is
    ``validate_citations`` which accepts a Pandas DataFrame and
    returns a (validated_df, report) tuple.  The report contains
    counts of various issues and recommendations to improve data
    quality.
    """

    def __init__(self) -> None:
        self.validation_results: List[Dict[str, Any]] = []
        self.stats: Dict[str, int] = {
            'total': 0,
            'valid': 0,
            'missing_abstract': 0,
            'missing_title': 0,
            'invalid_year': 0,
            'invalid_id': 0,
            'enhanced': 0,
        }

    def validate_citations(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Validate each citation in a DataFrame.

        For each row the validator checks the ID, title, abstract and
        year fields.  Missing or invalid values are recorded and
        optionally repaired.  After all rows have been processed a
        summary report is generated.

        Args:
            df: DataFrame of citations to validate.

        Returns:
            A tuple of (validated DataFrame, report dictionary).
        """
        self.stats['total'] = len(df)
        validated_rows: List[Dict[str, Any]] = []
        for idx, row in df.iterrows():
            validated_row, issues = self.validate_single_citation(row.to_dict())
            if issues:
                self.validation_results.append({
                    'citation_id': validated_row.get('id', f'row_{idx}'),
                    'title': validated_row.get('title', 'Unknown')[:50],
                    'issues': issues,
                })
            validated_rows.append(validated_row)
        validated_df = pd.DataFrame(validated_rows)
        # Provide an opportunity to enrich missing abstracts
        initial_missing = self.stats['missing_abstract']
        if initial_missing > 0 and ENRICH_AVAILABLE:
            logger.info(f"Attempting to enrich {initial_missing} citations with missing abstracts via PubMed search")
            # This is a stub for enrichment; in practice you might call external APIs
            # to fetch abstracts.  Here we simply leave the abstracts blank.
            pass
        report = self.generate_validation_report()
        return validated_df, report

    def validate_single_citation(self, citation: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """Validate a single citation dictionary."""
        issues: List[str] = []
        # Validate ID
        citation_id = str(citation.get('id', '')).strip()
        if not citation_id or citation_id.lower() == 'nan':
            issues.append('Missing or invalid ID')
            self.stats['invalid_id'] += 1
            # Try to generate an ID from DOI or title
            doi = citation.get('doi')
            if doi and str(doi).lower() != 'nan':
                citation_id = str(doi)
            else:
                title = citation.get('title', '') or ''
                citation_id = f"hash_{hashlib.md5(title.encode()).hexdigest()[:12]}" if title else f"unknown_{self.stats['total']}"
            citation['id'] = citation_id
        # Validate title
        title = str(citation.get('title', '')).strip()
        if not title or len(title) < 5:
            issues.append('Missing or too short title')
            self.stats['missing_title'] += 1
        # Validate abstract
        abstract = str(citation.get('abstract', '')).strip()
        if not abstract or len(abstract) < 50:
            issues.append('Missing or insufficient abstract')
            self.stats['missing_abstract'] += 1
        # Validate year
        year = citation.get('year')
        if year:
            try:
                year_int = int(year)
                if year_int < 1900 or year_int > 2100:
                    issues.append(f'Invalid year: {year}')
                    self.stats['invalid_year'] += 1
                    citation['year'] = None
            except (ValueError, TypeError):
                issues.append(f'Invalid year format: {year}')
                self.stats['invalid_year'] += 1
                citation['year'] = None
        # Count as valid if title and abstract are sufficiently populated
        if title and len(title) >= 5 and abstract and len(abstract) >= 50:
            self.stats['valid'] += 1
        return citation, issues

    def generate_validation_report(self) -> Dict[str, Any]:
        """Compile a report of validation statistics and recommendations."""
        total = self.stats['total']
        quality_score = (self.stats['valid'] / total * 100) if total > 0 else 0
        report: Dict[str, Any] = {
            'summary': self.stats.copy(),
            'quality_score': quality_score,
            'critical_issues': {
                'missing_abstracts': self.stats['missing_abstract'],
                'missing_abstracts_pct': (self.stats['missing_abstract'] / total * 100) if total > 0 else 0,
            },
            'recommendations': [],
            'problematic_citations': self.validation_results[:10],
        }
        # Provide high‑level recommendations
        if report['critical_issues']['missing_abstracts_pct'] > 20:
            report['recommendations'].append(
                '⚠️ A high percentage of citations are missing abstracts. Consider using a different export format '
                'or retrieving abstracts from an external service.'
            )
        if self.stats['invalid_id'] > 0:
            report['recommendations'].append(
                f'Generated IDs for {self.stats["invalid_id"]} citations with missing/invalid identifiers.'
            )
        return report
