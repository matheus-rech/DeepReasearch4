"""
Streamlit user interface for the Deep Research systematic review app.

This module defines a simple multi‑step web application using
Streamlit.  Users can upload citation files in various formats,
validate and load them into the local database, specify inclusion
criteria based on the PICOTT framework, run an AI‑driven screening
task via the OpenAI Deep Research API, and review the results.

The UI is intentionally streamlined compared with the original
prototype: it focuses on the core tasks of loading citations,
defining screening criteria and displaying the final decisions.  If
no OpenAI API key is configured, the screening step will report an
error prompting the user to set the appropriate environment
variable.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd  # type: ignore
import streamlit as st  # type: ignore

from full_stack_app.backend import (
    database as db,
    parsers,
    data_validator,
    deep_research,
    ice_critic,
)


def _reset_session() -> None:
    """Initialise default values in the Streamlit session state.

    This helper ensures that expected keys exist in ``st.session_state``.
    """
    state_defaults = {
        "step": "upload",  # current UI step
        "citations_df": None,  # DataFrame of loaded citations
        "validation_report": None,  # Report from data validation
        "criteria": {},  # PICOTT and custom criteria
        "screening_results": None,  # Raw results from Deep Research
        "analysis": None,  # ICE critic analysis of screening results
    }
    for key, default in state_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


def show_upload_step() -> None:
    """Display the citation upload step.

    Users can upload a file in PubMed XML, RIS, CSV, EndNote XML or
    plain text format.  The file is parsed into a DataFrame using
    ``parsers.parse_citations`` and then inserted into the database.
    A validation report is shown to highlight data quality issues.
    """
    st.header("Step 1 – Load citations")
    st.write(
        "Upload your citation export file. Supported formats include PubMed XML, RIS, CSV, "
        "EndNote XML and plain text.  The parser will automatically detect the format."
    )
    uploaded = st.file_uploader(
        "Choose a citation file",
        type=["xml", "ris", "csv", "nbib", "txt"],
        help="Upload your exported citations from reference management software",
    )
    if uploaded:
        st.info(f"File uploaded: {uploaded.name} ({uploaded.size:,} bytes)")
        if st.button("Parse and load", type="primary"):
            with st.spinner("Parsing citations…"):
                try:
                    df = parsers.parse_citations(uploaded, uploaded.name)
                    st.session_state.citations_df = df
                    # Validate citations
                    validator = data_validator.CitationValidator()
                    validated_df, report = validator.validate_citations(df)
                    st.session_state.validation_report = report
                    # Insert into database
                    db.init_db()
                    stats = db.bulk_insert_citations(validated_df)
                    st.success(
                        f"Loaded {stats['inserted'] + stats['updated']} citations into the database."
                    )
                    if stats["skipped"]:
                        st.warning(f"Skipped {stats['skipped']} rows without IDs.")
                    # Move to next step
                    st.session_state.step = "criteria"
                except Exception as e:
                    st.error(f"Error parsing file: {e}")
        # Display validation report if available
        if st.session_state.validation_report:
            report = st.session_state.validation_report
            st.subheader("Data quality report")
            st.write(
                f"Quality score: **{report['quality_score']:.1f}%**", help="Percentage of fields present"
            )
            crit = report.get("critical_issues", {})
            if crit.get("missing_abstracts"):
                st.warning(
                    f"{crit['missing_abstracts']} citations are missing abstracts. Abstracts are important for accurate screening."
                )


def show_criteria_step() -> None:
    """Display the PICOTT and custom screening criteria form."""
    st.header("Step 2 – Define screening criteria")
    st.write(
        "Specify your PICOTT criteria and any additional inclusion or exclusion criteria. "
        "Leave a field blank if it is not relevant."
    )
    # PICOTT inputs arranged in two columns
    col1, col2, col3 = st.columns(3)
    with col1:
        pop = st.text_input("Population", value=st.session_state.criteria.get("population", ""))
        interv = st.text_input("Intervention", value=st.session_state.criteria.get("intervention", ""))
    with col2:
        comp = st.text_input("Comparator", value=st.session_state.criteria.get("comparator", ""))
        outcome = st.text_input("Outcome", value=st.session_state.criteria.get("outcome", ""))
    with col3:
        timeframe = st.text_input("Timeframe", value=st.session_state.criteria.get("timeframe", ""))
        study_type = st.text_input("Study type", value=st.session_state.criteria.get("study_type", ""))
    st.divider()
    st.write("Enter one criterion per line for inclusion and exclusion.")
    incl_text = st.text_area(
        "Additional inclusion criteria", value="\n".join(st.session_state.criteria.get("inclusion", [])), height=100
    )
    excl_text = st.text_area(
        "Exclusion criteria", value="\n".join(st.session_state.criteria.get("exclusion", [])), height=100
    )
    if st.button("Run screening", type="primary"):
        # Save criteria to session
        st.session_state.criteria = {
            "population": pop.strip() or "Not specified",
            "intervention": interv.strip() or "Not specified",
            "comparator": comp.strip() or "Not specified",
            "outcome": outcome.strip() or "Not specified",
            "timeframe": timeframe.strip() or "Not specified",
            "study_type": study_type.strip() or "Not specified",
            "inclusion": [c.strip() for c in incl_text.splitlines() if c.strip()],
            "exclusion": [c.strip() for c in excl_text.splitlines() if c.strip()],
        }
        # Launch screening job
        with st.spinner("Launching AI screening… this may take several minutes"):
            try:
                # Determine MCP URL; fall back to localhost
                mcp_url = os.getenv("MCP_URL", "http://localhost:8001") + "/sse/"
                response = deep_research.launch_screening_job(
                    pico_criteria={
                        "population": pop.strip() or "Not specified",
                        "intervention": interv.strip() or "Not specified",
                        "comparator": comp.strip() or "Not specified",
                        "outcome": outcome.strip() or "Not specified",
                        "timeframe": timeframe.strip() or "Not specified",
                        "study_type": study_type.strip() or "Not specified",
                    },
                    inclusion_criteria=[c.strip() for c in incl_text.splitlines() if c.strip()],
                    exclusion_criteria=[c.strip() for c in excl_text.splitlines() if c.strip()],
                    corpus_size=len(st.session_state.citations_df) if st.session_state.citations_df is not None else 0,
                    mcp_url=mcp_url,
                )
                # Poll for completion
                result = deep_research.poll_job_status(response)
                # The content is expected to be a JSON array
                content = result.get("content", "")
                try:
                    screening_results = json.loads(content)
                except json.JSONDecodeError:
                    st.error("Failed to parse AI response. Raw content shown below.")
                    st.code(content)
                    return
                st.session_state.screening_results = screening_results
                st.session_state.step = "results"
            except Exception as e:
                st.error(
                    f"Screening failed: {e}. Ensure your OPENAI_API_KEY is set and that the MCP server is reachable."
                )


def show_results_step() -> None:
    """Display the screening results and analysis."""
    st.header("Step 3 – Review results")
    results: Optional[List[Dict[str, Any]]] = st.session_state.get("screening_results")
    if not results:
        st.error("No screening results available. Please run the screening first.")
        return
    # Display decisions in a table
    df = pd.DataFrame(results)
    # Flatten decision fields for display
    if "decision" in df.columns:
        df["decision"] = df["decision"].astype(str)
    if "confidence" in df.columns:
        df["confidence"] = df["confidence"].astype(str)
    st.subheader("AI screening decisions")
    st.dataframe(
        df[[col for col in df.columns if col not in {"picott", "inclusionCriteria", "exclusionCriteria", "reasoning"}]],
        use_container_width=True,
    )
    # Summary statistics
    include_count = df[df.get("decision", "").str.lower() == "include"].shape[0]
    exclude_count = df[df.get("decision", "").str.lower() == "exclude"].shape[0]
    st.write(f"Included citations: {include_count}\n\nExcluded citations: {exclude_count}")
    # Perform ICE analysis
    try:
        analysis = ice_critic.analyze_screening_consistency(
            screening_results=results, pico_criteria=st.session_state.criteria
        )
        st.session_state.analysis = analysis
        st.subheader("Consistency analysis")
        st.write(f"Total issues detected: {analysis['summary']['total_issues']}")
        if analysis["issues"]:
            for issue in analysis["issues"]:
                st.warning(
                    f"{issue['type']}: {issue['description']} (severity: {issue['severity']})"
                )
    except Exception as e:
        st.error(f"Failed to analyse screening consistency: {e}")
    # Offer export
    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download results as CSV",
        data=csv_data,
        file_name="screening_results.csv",
        mime="text/csv",
    )


def main() -> None:
    """Entry point for the Streamlit application."""
    st.set_page_config(
        page_title="Systematic Review Screener",
        page_icon="📚",
        layout="wide",
    )
    _reset_session()
    # Sidebar navigation
    with st.sidebar:
        st.title("Navigation")
        steps = ["upload", "criteria", "results"]
        names = {
            "upload": "1. Upload",
            "criteria": "2. Criteria",
            "results": "3. Results",
        }
        current_step = st.session_state.step
        for step in steps:
            if step == current_step:
                st.success(f"→ {names[step]}")
            elif steps.index(step) < steps.index(current_step):
                st.info(f"✓ {names[step]}")
            else:
                st.text(names[step])
        st.divider()
        # Show corpus stats if citations are loaded
        if st.session_state.citations_df is not None:
            stats = db.get_corpus_stats()
            st.metric("Citations", stats["total_citations"])
            if stats["year_distribution"]:
                year_df = pd.DataFrame(stats["year_distribution"])
                if not year_df.empty:
                    st.bar_chart(year_df.set_index("year")["count"])
    # Main content area
    if st.session_state.step == "upload":
        show_upload_step()
    elif st.session_state.step == "criteria":
        show_criteria_step()
    elif st.session_state.step == "results":
        show_results_step()


if __name__ == "__main__":
    main()