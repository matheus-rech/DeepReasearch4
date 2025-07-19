"""
Citation file parsers for the Deep Research full‑stack application.

This module provides a collection of parsing helpers to ingest
citations from a variety of common reference manager export formats
including PubMed XML and MEDLINE text exports, RIS, EndNote XML and
generic CSV files.  A simple detection function chooses the right
parser based on file extension and content heuristics.  When optional
dependencies such as `rispy` or `llama_index` are not installed, the
corresponding functionality is disabled and the caller will receive
a clear `ImportError`.
"""

from __future__ import annotations

import io
import logging
import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

import pandas as pd  # type: ignore
from dateutil import parser as date_parser  # type: ignore
from bs4 import BeautifulSoup  # type: ignore

logger = logging.getLogger(__name__)

# Attempt to import rispy for RIS parsing.  If not available, an
# ImportError will be raised when RIS parsing is attempted.
try:
    import rispy  # type: ignore
    HAS_RISPY = True
except Exception:
    HAS_RISPY = False

# Attempt to import llama_index readers for ArXiv and PubMed search.  The
# functions that depend on these readers will raise if unavailable.
try:
    from llama_index.readers.papers import ArxivReader, PubmedReader  # type: ignore
    HAS_LLAMA_READERS = True
except Exception:
    HAS_LLAMA_READERS = False


def normalize_year(date_str: Any) -> Optional[int]:
    """Coerce a variety of date representations into a four‑digit year.

    Returns an integer year if one can be extracted and falls within
    1900–2100, otherwise returns ``None``.  Strings are parsed with
    `dateutil.parser.parse` and numeric values are cast directly.
    """
    if pd.isna(date_str) or date_str in (None, ''):
        return None
    try:
        if isinstance(date_str, (int, float)):
            year = int(date_str)
            return year if 1900 <= year <= 2100 else None
        parsed = date_parser.parse(str(date_str))
        return parsed.year
    except Exception:
        # fall back to regex search for a 4‑digit year
        match = re.search(r"\b(19|20)\d{2}\b", str(date_str))
        if match:
            return int(match.group())
    return None


def parse_pubmed_xml(file_obj: io.BufferedIOBase) -> pd.DataFrame:
    """Parse citations from PubMed XML export format."""
    tree = ET.parse(file_obj)
    root = tree.getroot()
    citations: List[Dict[str, Any]] = []
    for article in root.findall('.//PubmedArticle'):
        pmid = article.findtext('.//PMID')
        if not pmid:
            continue
        article_elem = article.find('.//Article')
        if article_elem is None:
            continue
        title = article_elem.findtext('.//ArticleTitle', default='')
        # gather abstract parts
        abstract_parts: List[str] = []
        abstract_elem = article_elem.find('.//Abstract')
        if abstract_elem is not None:
            for abstract_text in abstract_elem.findall('.//AbstractText'):
                text_content = abstract_text.text or ''
                label = abstract_text.get('Label', '') or ''
                abstract_parts.append(f"{label}: {text_content}".strip() if label else text_content)
        abstract = ' '.join(abstract_parts).strip()
        authors: List[str] = []
        for author in article_elem.findall('.//Author'):
            last_name = author.findtext('LastName', default='')
            fore_name = author.findtext('ForeName', default='')
            if last_name:
                authors.append(f"{last_name} {fore_name}".strip())
        journal = article_elem.findtext('.//Journal/Title', default='')
        pub_date = article_elem.find('.//Journal/JournalIssue/PubDate')
        year: Optional[int] = None
        if pub_date is not None:
            year_text = pub_date.findtext('Year')
            if not year_text:
                medline_date = pub_date.findtext('MedlineDate')
                year_text = medline_date
            year = normalize_year(year_text)
        doi: Optional[str] = None
        for eloc_id in article_elem.findall('.//ELocationID'):
            if eloc_id.get('EIdType') == 'doi':
                doi = eloc_id.text
                break
        mesh_terms = [m.text for m in article.findall('.//MeshHeading/DescriptorName') if m.text]
        keywords = [kw.text for kw in article.findall('.//Keyword') if kw.text]
        citations.append({
            'id': f'PMID:{pmid}',
            'title': title,
            'abstract': abstract,
            'year': year,
            'authors': authors,
            'journal': journal,
            'doi': doi,
            'mesh_terms': mesh_terms,
            'keywords': keywords,
            'raw_data': {
                'source': 'pubmed_xml',
                'pmid': pmid,
            },
        })
    return pd.DataFrame(citations)


def parse_ris(file_obj: io.TextIOBase) -> pd.DataFrame:
    """Parse citations from RIS format using rispy."""
    if not HAS_RISPY:
        raise ImportError('rispy library is required to parse RIS files')
    entries = rispy.load(file_obj)
    citations: List[Dict[str, Any]] = []
    for entry in entries:
        year = entry.get('year') or entry.get('publication_year')
        year = normalize_year(year)
        authors = entry.get('authors') or entry.get('first_authors') or []
        citation_id = entry.get('id') or entry.get('doi') or f"RIS_{len(citations)}"
        citations.append({
            'id': citation_id,
            'title': entry.get('title', ''),
            'abstract': entry.get('abstract', ''),
            'year': year,
            'authors': authors,
            'journal': entry.get('journal_name', ''),
            'doi': entry.get('doi', ''),
            'mesh_terms': [],
            'keywords': entry.get('keywords', []),
            'raw_data': {
                'source': 'ris',
                'entry': entry
            },
        })
    return pd.DataFrame(citations)


def parse_csv(file_obj: io.TextIOBase) -> pd.DataFrame:
    """Parse a generic CSV file containing citation information."""
    df = pd.read_csv(file_obj)
    column_map = {
        'pmid': 'id', 'PMID': 'id',
        'Title': 'title', 'title': 'title',
        'Abstract': 'abstract', 'abstract': 'abstract',
        'Year': 'year', 'Publication Year': 'year',
        'Authors': 'authors', 'authors': 'authors',
        'Journal': 'journal', 'Journal/Book': 'journal', 'journal': 'journal',
        'DOI': 'doi', 'doi': 'doi',
        'MeSH Terms': 'mesh_terms', 'mesh_terms': 'mesh_terms',
        'Keywords': 'keywords', 'keywords': 'keywords'
    }
    # Rename columns to canonical names
    df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})
    # Ensure expected columns exist
    for col in ['id', 'title', 'abstract', 'year', 'authors', 'journal', 'doi', 'mesh_terms', 'keywords']:
        if col not in df.columns:
            df[col] = None
    # Normalise authors into lists
    def _split_authors(val: Any) -> List[str]:
        if isinstance(val, list):
            return val
        if pd.isna(val) or not val:
            return []
        return [a.strip() for a in str(val).split(';') if a.strip()]
    df['authors'] = df['authors'].apply(_split_authors)
    # Normalise mesh_terms and keywords into lists
    def _split_terms(val: Any) -> List[str]:
        if isinstance(val, list):
            return val
        if pd.isna(val) or not val:
            return []
        return [t.strip() for t in str(val).split(';') if t.strip()]
    df['mesh_terms'] = df['mesh_terms'].apply(_split_terms)
    df['keywords'] = df['keywords'].apply(_split_terms)
    # Parse year
    df['year'] = df['year'].apply(normalize_year)
    return df


def parse_endnote_xml(file_obj: io.BufferedIOBase) -> pd.DataFrame:
    """Parse citations from EndNote XML export format using BeautifulSoup."""
    soup = BeautifulSoup(file_obj.read(), 'xml')
    citations: List[Dict[str, Any]] = []
    for record in soup.find_all('record'):
        rec_number = record.find('rec-number')
        rec_id = f"EndNote_{rec_number.text}" if rec_number else f"EndNote_{len(citations)}"
        title_elem = record.find('title')
        title = title_elem.text if title_elem else ''
        abstract_elem = record.find('abstract')
        abstract = abstract_elem.text if abstract_elem else ''
        authors: List[str] = []
        contributors = record.find('contributors')
        if contributors:
            for author in contributors.find_all('author'):
                if author.text:
                    authors.append(author.text)
        year_elem = record.find('year')
        year = normalize_year(year_elem.text if year_elem else None)
        journal_elem = record.find('secondary-title')
        journal = journal_elem.text if journal_elem else ''
        doi_elem = record.find('electronic-resource-num')
        doi = doi_elem.text if doi_elem else ''
        keywords: List[str] = []
        keywords_elem = record.find('keywords')
        if keywords_elem:
            for keyword in keywords_elem.find_all('keyword'):
                if keyword.text:
                    keywords.append(keyword.text)
        citations.append({
            'id': rec_id,
            'title': title,
            'abstract': abstract,
            'year': year,
            'authors': authors,
            'journal': journal,
            'doi': doi,
            'mesh_terms': [],
            'keywords': keywords,
            'raw_data': {
                'source': 'endnote_xml',
            },
        })
    return pd.DataFrame(citations)


def parse_pubmed_text(file_obj: io.BufferedIOBase) -> pd.DataFrame:
    """Parse citations from PubMed MEDLINE text (NBIB) export format."""
    content = file_obj.read().decode('utf-8', errors='ignore')
    entries = re.split(r'\n(?=PMID-)', content)
    citations: List[Dict[str, Any]] = []
    for entry in entries:
        if not entry.strip():
            continue
        lines = entry.strip().split('\n')
        if not lines:
            continue
        citation: Dict[str, Any] = {
            'id': '', 'title': '', 'abstract': '', 'year': None,
            'authors': [], 'journal': '', 'doi': '',
            'mesh_terms': [], 'keywords': [],
            'raw_data': {'source': 'pubmed_text'}
        }
        full_text = '\n'.join(lines)
        pmid_match = re.search(r'PMID-\s*(\d+)', full_text)
        if pmid_match:
            citation['id'] = f"PMID:{pmid_match.group(1)}"
        doi_match = re.search(r'LID\s*-\s*(10\.\S+)\s*\[doi\]', full_text, re.IGNORECASE)
        if doi_match:
            citation['doi'] = doi_match.group(1)
        title_match = re.search(r'TI\s*-\s*(.+?)(?=\n[A-Z]{2}\s*-)', full_text, re.DOTALL)
        if title_match:
            citation['title'] = title_match.group(1).strip().replace('\n', ' ')
        abstract_match = re.search(r'AB\s*-\s*(.+?)(?=\n[A-Z]{2}\s*-)', full_text, re.DOTALL)
        if abstract_match:
            citation['abstract'] = abstract_match.group(1).strip().replace('\n', ' ')
        authors = re.findall(r'FAU\s*-\s*(.+)', full_text)
        citation['authors'] = [a.strip() for a in authors]
        journal_match = re.search(r'TA\s*-\s*(.+)', full_text)
        if journal_match:
            citation['journal'] = journal_match.group(1).strip()
        date_match = re.search(r'DP\s*-\s*(\d{4})', full_text)
        if date_match:
            citation['year'] = int(date_match.group(1))
        mesh_terms = [m.strip() for m in re.findall(r'MH\s*-\s*([^\n]+)', full_text)]
        citation['mesh_terms'] = mesh_terms
        if citation['id'] or citation['title']:
            citations.append(citation)
    return pd.DataFrame(citations)


def detect_format(filename: str, content: bytes) -> str:
    """Attempt to detect the citation file format based on filename and content."""
    filename_lower = filename.lower()
    # extension based detection
    if filename_lower.endswith('.ris'):
        return 'ris'
    if filename_lower.endswith('.csv'):
        return 'csv'
    if filename_lower.endswith('.nbib'):
        return 'pubmed_text'
    if filename_lower.endswith('.xml'):
        content_str = content.decode('utf-8', errors='ignore')
        lower = content_str.lower()
        if '<pubmedarticle' in lower:
            return 'pubmed_xml'
        if '<record' in lower or '<records' in lower:
            return 'endnote_xml'
        return 'unknown_xml'
    if filename_lower.endswith('.txt'):
        snippet = content.decode('utf-8', errors='ignore')[:2000]
        if any(prefix in snippet for prefix in ['PMID:', 'PMID-', 'TI  -', 'AB  -', 'FAU  -', 'AU  -', 'LID  -', 'DP  -']):
            return 'pubmed_text'
    # content heuristics
    snippet = content.decode('utf-8', errors='ignore')[:2000].lower()
    if 'ty  -' in snippet:
        return 'ris'
    if '<pubmedarticle' in snippet:
        return 'pubmed_xml'
    if '<record' in snippet or '<records' in snippet:
        return 'endnote_xml'
    if snippet.startswith('pmid-') or 'pmid:' in snippet:
        return 'pubmed_text'
    return 'unknown'


def parse_citations(file_obj: io.BufferedIOBase, filename: str) -> pd.DataFrame:
    """Detect the format of a citation file and dispatch to the proper parser."""
    content = file_obj.read()
    file_format = detect_format(filename, content)
    # Reset file pointer to allow reading again
    file_obj = io.BytesIO(content)
    if file_format == 'pubmed_xml':
        return parse_pubmed_xml(file_obj)
    if file_format == 'ris':
        return parse_ris(io.StringIO(content.decode('utf-8', errors='ignore')))
    if file_format == 'csv':
        return parse_csv(io.StringIO(content.decode('utf-8', errors='ignore')))
    if file_format == 'endnote_xml':
        return parse_endnote_xml(file_obj)
    if file_format == 'pubmed_text':
        return parse_pubmed_text(file_obj)
    raise ValueError(f"Unsupported file format: {file_format}")


def parse_arxiv_search(search_query: str, max_results: int = 10) -> pd.DataFrame:
    """Fetch papers from ArXiv using the llama_index ArxivReader."""
    if not HAS_LLAMA_READERS:
        raise ImportError('llama_index.readers.papers is required for ArXiv search')
    loader = ArxivReader()
    documents, abstracts = loader.load_papers_and_abstracts(search_query=search_query, max_results=max_results)
    citations: List[Dict[str, Any]] = []
    for doc, abstract in zip(documents, abstracts):
        metadata = doc.metadata
        authors = metadata.get('authors', [])
        if isinstance(authors, str):
            authors = [authors]
        published = metadata.get('published', '')
        year = normalize_year(published) if published else None
        citations.append({
            'id': f"arxiv:{metadata.get('article_id', '')}",
            'title': metadata.get('title', ''),
            'abstract': abstract.text if abstract else metadata.get('summary', ''),
            'year': year,
            'authors': authors,
            'journal': 'arXiv',
            'doi': metadata.get('doi', ''),
            'mesh_terms': [],
            'keywords': metadata.get('categories', '').split() if metadata.get('categories') else [],
            'raw_data': metadata
        })
    return pd.DataFrame(citations)


def parse_pubmed_search(search_query: str, max_results: int = 10) -> pd.DataFrame:
    """Fetch papers from PubMed using the llama_index PubmedReader."""
    if not HAS_LLAMA_READERS:
        raise ImportError('llama_index.readers.papers is required for PubMed search')
    loader = PubmedReader()
    documents = loader.load_data(search_query=search_query, max_results=max_results)
    citations: List[Dict[str, Any]] = []
    for doc in documents:
        metadata = doc.metadata
        authors_str = metadata.get('Authors', '') or ''
        authors = [a.strip() for a in authors_str.split(';')] if authors_str else []
        year = normalize_year(metadata.get('PubDate', ''))
        citations.append({
            'id': f"PMID:{metadata.get('PubmedId', '')}",
            'title': metadata.get('Title', ''),
            'abstract': doc.text,
            'year': year,
            'authors': authors,
            'journal': metadata.get('Journal', ''),
            'doi': metadata.get('DOI', ''),
            'mesh_terms': metadata.get('MeshHeadings', '').split(';') if metadata.get('MeshHeadings') else [],
            'keywords': metadata.get('Keywords', '').split(';') if metadata.get('Keywords') else [],
            'raw_data': metadata
        })
    return pd.DataFrame(citations)
