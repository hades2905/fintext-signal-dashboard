"""
Named Entity Recognition using spaCy.

Extracts organisations, persons, locations, and financial instruments
from financial news text.

Requires:
    pip install spacy
    python -m spacy download en_core_web_sm
"""
from __future__ import annotations

import logging
from functools import lru_cache

import spacy
from spacy.language import Language

from .schemas import Article, Entity

logger = logging.getLogger(__name__)

# Entity types we care about in a financial context
RELEVANT_LABELS = {"ORG", "PERSON", "GPE", "LOC", "PRODUCT", "MONEY", "PERCENT", "FAC"}


@lru_cache(maxsize=1)
def _load_model() -> Language:
    try:
        return spacy.load("en_core_web_sm")
    except OSError as err:
        raise OSError(
            "spaCy model not found. Run: python -m spacy download en_core_web_sm"
        ) from err


def extract_entities(text: str) -> list[Entity]:
    """Return a deduplicated list of named entities from *text*."""
    nlp = _load_model()
    doc = nlp(text[:5000])  # cap to avoid very slow processing on huge texts
    seen: set[tuple[str, str]] = set()
    entities: list[Entity] = []
    for ent in doc.ents:
        if ent.label_ not in RELEVANT_LABELS:
            continue
        key = (ent.text.strip(), ent.label_)
        if key in seen:
            continue
        seen.add(key)
        entities.append(Entity(text=ent.text.strip(), label=ent.label_))
    return entities


def annotate_articles(articles: list[Article]) -> list[Article]:
    """
    Add NER entities to each article in-place.
    Returns the same list for convenience.
    """
    for article in articles:
        article.entities = extract_entities(article.title + " " + article.text)
    return articles
