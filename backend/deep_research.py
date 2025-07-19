"""
Integration helpers for the OpenAI Deep Research API.

This module defines a thin wrapper around the OpenAI client that
constructs prompts for systematic review screening tasks and submits
them to the `o3-deep-research` model.  It also includes a helper to
extract results from the response.  In production, you should set the
``OPENAI_API_KEY`` environment variable (or ``OPENAI_API_KEYS``
containing a comma‑separated list of keys) before using these
functions.
"""

from __future__ import annotations

import os
import logging
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI  # type: ignore
except Exception as e:
    raise ImportError('The openai package is required for deep research integration') from e

logger = logging.getLogger(__name__)


def _resolve_api_key() -> str:
    """Resolve a single OpenAI API key from environment variables."""
    single = os.getenv('OPENAI_API_KEY')
    if single:
        return single
    multiple = os.getenv('OPENAI_API_KEYS')
    if multiple:
        for key in (k.strip() for k in multiple.split(',') if k.strip()):
            return key
    raise EnvironmentError(
        'Missing OpenAI API key. Set OPENAI_API_KEY or OPENAI_API_KEYS in your environment.'
    )


# Initialise the OpenAI client lazily.  The API key is resolved at
# import time so that missing keys cause immediate errors when this
# module is used.
API_KEY: str = _resolve_api_key()
MODEL: str = os.getenv('DEEP_RESEARCH_MODEL', 'o3-deep-research-2025-06-26')
client = OpenAI(api_key=API_KEY, timeout=3600)


def launch_screening_job(
    pico_criteria: Dict[str, str],
    inclusion_criteria: List[str],
    exclusion_criteria: List[str],
    corpus_size: int,
    mcp_url: Optional[str] = None,
    search_mode: str = 'fulltext',
) -> Any:
    """Launch a Deep Research screening job.

    This helper assembles a prompt describing the systematic review
    screening task using the provided criteria and submits it to the
    OpenAI API.  The returned response object can be passed to
    :func:`poll_job_status` to extract the final results.
    """
    # Compose the screening task prompt
    task = f"""You are conducting a systematic review screening of {corpus_size} research citations.
The citations are available through the MCP search and fetch tools.

IMPORTANT: Use search mode="{search_mode}" for all search operations.

Your task is to screen each citation based on the following criteria:

## PICOTT Criteria (ALL must match for inclusion):
- Population: {pico_criteria.get('population', 'Not specified')}
- Intervention: {pico_criteria.get('intervention', 'Not specified')}
- Comparator: {pico_criteria.get('comparator', 'Not specified')}
- Outcome: {pico_criteria.get('outcome', 'Not specified')}
- Timeframe: {pico_criteria.get('timeframe', 'Not specified')}
- Study Type: {pico_criteria.get('study_type', 'Not specified')}

## Additional Inclusion Criteria:
{chr(10).join(f'- {c}' for c in inclusion_criteria)}

## Exclusion Criteria:
{chr(10).join(f'- {c}' for c in exclusion_criteria)}

## Instructions:
1. Search the corpus systematically to identify all potentially relevant citations
2. Extract PICOTT elements with EXACT QUOTES from the title/abstract
3. A citation must meet ALL PICOTT criteria AND inclusion criteria to be included
4. If ANY exclusion criterion is met, the citation should be excluded
5. When uncertain, err on the side of inclusion for full‑text review

Return your results as a JSON array where each citation has this structure:
[
  {{
    "id": "citation_id",
    "title": "citation title",
    "picott": {{
      "population": "exact quote from abstract identifying population or 'Not found'",
      "intervention": "exact quote from abstract identifying intervention or 'Not found'",
      "comparison": "exact quote from abstract identifying comparison or 'Not found'",
      "outcome": "exact quote from abstract identifying outcome or 'Not found'",
      "timeframe": "exact quote from abstract identifying timeframe or 'Not found'",
      "studyType": "exact quote from abstract identifying study type or 'Not found'"
    }},
    "inclusionCriteria": ["list of matched inclusion criteria with supporting quotes"],
    "exclusionCriteria": ["list of matched exclusion criteria with supporting quotes"],
    "reasoning": "Step‑by‑step reasoning for your decision",
    "decision": "Include" or "Exclude",
    "confidence": "high/medium/low"
  }}
]

Focus on extracting EXACT quotes that support each PICOTT element and criterion match."""
    # Determine the MCP URL: allow override via environment variables for hosted platforms
    if os.getenv('HEROKU_APP_NAME'):
        app_name = os.getenv('HEROKU_APP_NAME')
        mcp_url = f"https://{app_name}.herokuapp.com/sse/"
    elif os.getenv('REPL_SLUG') and os.getenv('REPL_OWNER'):
        slug = os.getenv('REPL_SLUG')
        owner = os.getenv('REPL_OWNER')
        mcp_url = f"https://{slug}-8001.{owner}.repl.co/sse/"
    else:
        mcp_url = mcp_url or 'http://localhost:8001/sse/'
    try:
        response = client.responses.create(
            model=MODEL,
            input=task,
            tools=[
                {'type': 'web_search_preview'},
                {
                    'type': 'mcp',
                    'server_label': 'DeepResearchServer',
                    'server_url': mcp_url,
                    'require_approval': 'never',
                },
            ],
        )
        logger.info(f"Launched screening job: {getattr(response, 'id', 'unknown')}")
        return response
    except Exception as e:
        logger.error(f"Failed to launch screening job: {e}")
        raise


def poll_job_status(response: Any) -> Dict[str, Any]:
    """Extract the final results from a Deep Research API response object."""
    try:
        # Newer API versions return responses in `output`
        if hasattr(response, 'output') and response.output:
            final_output = response.output[-1]
            if hasattr(final_output, 'content') and final_output.content:
                content = final_output.content[0].text
                return {
                    'status': 'completed',
                    'content': content,
                    'reasoning': getattr(response, 'reasoning', {}),
                    'full_output': response.output,
                }
            raise ValueError('Completed job has no content in final output')
        # Fallback for older message‑based responses
        if hasattr(response, 'message') and response.message and hasattr(response.message, 'content'):
            msg = response.message
            if isinstance(msg.content, list) and msg.content:
                content = msg.content[0].text
            else:
                content = msg.content  # type: ignore[assignment]
            return {
                'status': 'completed',
                'content': content,
                'reasoning': getattr(response, 'reasoning', {}),
            }
        raise ValueError('Unrecognised response format; cannot extract results')
    except Exception as e:
        logger.error(f"Failed to extract job results: {e}")
        raise
