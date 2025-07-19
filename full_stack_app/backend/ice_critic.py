"""
ICE (Internal Consistency Evaluation) critic functions.

These helpers analyse screening decisions for systematic reviews,
looking for inconsistencies and potential quality issues.  The
functions operate on lists of result dictionaries returned by the
Deep Research API and can be used to flag common problems such as
missing PICOTT elements, mismatched confidence/decision pairs,
inconsistent exclusion reasons and overall inclusion rates that seem
unreasonable.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Tuple


def analyze_screening_consistency(
    screening_results: List[Dict[str, Any]],
    pico_criteria: Dict[str, str],
) -> Dict[str, Any]:
    """Analyse screening results for internal consistency issues.

    Args:
        screening_results: A list of result dictionaries returned by the
            Deep Research API.  Each dictionary should contain keys
            including ``id``, ``include`` (boolean), ``confidence`` and
            optional ``picott`` and ``reason`` fields.
        pico_criteria: The PICOTT criteria that were used for screening.

    Returns:
        A dictionary with ``issues`` (list of issue dictionaries) and
        ``summary`` (aggregated statistics).
    """
    issues: List[Dict[str, Any]] = []
    # Check for missing PICOTT elements on included citations
    for result in screening_results:
        include = result.get('include') or result.get('decision') == 'Include'
        picott = result.get('picott', {})
        if include and picott:
            missing: List[str] = []
            for key, criteria_val in pico_criteria.items():
                if criteria_val and criteria_val != 'Not specified':
                    element_key = key if key != 'study_type' else 'studyType'
                    if element_key in picott and (not picott[element_key] or picott[element_key] == 'Not found'):
                        missing.append(key)
            if missing:
                issues.append({
                    'type': 'PICOTT_elements_missing',
                    'citation_id': result.get('id'),
                    'severity': 'high',
                    'description': f"Citation included but missing PICOTT elements: {', '.join(missing)}",
                    'suggestion': 'Verify if abstract contains required PICOTT elements',
                })
    # Confidence vs decision consistency
    for result in screening_results:
        include = result.get('include') or result.get('decision') == 'Include'
        confidence = result.get('confidence', '').lower()
        if include and confidence == 'low':
            issues.append({
                'type': 'low_confidence_inclusion',
                'citation_id': result.get('id'),
                'severity': 'medium',
                'description': 'Citation included with low confidence',
                'suggestion': 'Consider full‑text review to confirm inclusion',
            })
        if not include and confidence == 'high':
            issues.append({
                'type': 'high_confidence_exclusion',
                'citation_id': result.get('id'),
                'severity': 'low',
                'description': 'Citation excluded with high confidence – verify exclusion reason',
                'suggestion': 'Double‑check exclusion criteria are correctly applied',
            })
    # Analyse exclusion reason wording
    exclusion_reasons = [r.get('reason') for r in screening_results if not (r.get('include') or r.get('decision') == 'Include')]
    reason_counts = Counter(exclusion_reasons)
    similar_groups = find_similar_reasons([r for r in reason_counts.keys() if r])
    for group in similar_groups:
        if len(group) > 1:
            issues.append({
                'type': 'inconsistent_exclusion_wording',
                'citation_id': 'Multiple',
                'severity': 'low',
                'description': f"Similar exclusion reasons with different wording: {group}",
                'suggestion': 'Standardise exclusion reason terminology',
            })
    # Inclusion rate analysis
    total = len(screening_results)
    inclusion_count = sum(1 for r in screening_results if r.get('include') or r.get('decision') == 'Include')
    inclusion_rate = (inclusion_count / total) if total > 0 else 0.0
    if inclusion_rate < 0.01:
        issues.append({
            'type': 'very_low_inclusion_rate',
            'citation_id': 'Overall',
            'severity': 'medium',
            'description': f'Inclusion rate is very low ({inclusion_rate * 100:.1f}%)',
            'suggestion': 'Verify that screening criteria are not too restrictive',
        })
    elif inclusion_rate > 0.5:
        issues.append({
            'type': 'high_inclusion_rate',
            'citation_id': 'Overall',
            'severity': 'medium',
            'description': f'Inclusion rate is high ({inclusion_rate * 100:.1f}%)',
            'suggestion': 'Verify that screening criteria are sufficiently specific',
        })
    # Build summary
    summary = {
        'total_issues': len(issues),
        'high_severity': sum(1 for i in issues if i['severity'] == 'high'),
        'medium_severity': sum(1 for i in issues if i['severity'] == 'medium'),
        'low_severity': sum(1 for i in issues if i['severity'] == 'low'),
        'inclusion_rate': inclusion_rate,
        'confidence_distribution': Counter(r.get('confidence', 'unknown') for r in screening_results),
        'unique_exclusion_reasons': len(reason_counts),
    }
    return {'issues': issues, 'summary': summary}


def find_similar_reasons(reasons: List[str]) -> List[List[str]]:
    """Group similar exclusion reason strings based on word overlap."""
    groups: List[List[str]] = []
    used: set[str] = set()
    for i, reason1 in enumerate(reasons):
        if reason1 in used:
            continue
        group = [reason1]
        used.add(reason1)
        for reason2 in reasons[i + 1:]:
            if reason2 in used:
                continue
            if calculate_reason_similarity(reason1, reason2) > 0.7:
                group.append(reason2)
                used.add(reason2)
        if len(group) > 1:
            groups.append(group)
    return groups


def calculate_reason_similarity(r1: str, r2: str) -> float:
    """Compute a simple similarity measure between two strings."""
    words1 = set(re.findall(r'\w+', r1.lower()))
    words2 = set(re.findall(r'\w+', r2.lower()))
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
        'with', 'by', 'from', 'not', 'no', 'does', 'did', 'is', 'was'
    }
    words1 = {w for w in words1 if w not in stop_words}
    words2 = {w for w in words2 if w not in stop_words}
    if not words1 or not words2:
        return 0.0
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    return intersection / union
