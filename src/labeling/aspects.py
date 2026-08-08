"""Taxonomie d'aspects pour avis produits bébés (Amazon).

Alignée sur le cadrage (qualité, prix, service, livraison) et enrichie
pour le domaine baby products.
"""

from __future__ import annotations

import re

# Labels utilisés comme hypothèses NLI (anglais = langue des avis Amazon)
ASPECT_CANDIDATES: list[str] = [
    "product quality and performance",
    "price, cost and value for money",
    "design, appearance and looks",
    "size, dimensions and fit",
    "comfort, softness and texture",
    "durability, washability and longevity",
    "safety for babies and children",
    "ease of use, assembly and convenience",
    "packaging, shipping and delivery",
    "customer service and seller support",
]

# Template NLI : fonctionne bien avec les modèles MNLI
ASPECT_HYPOTHESIS_TEMPLATE: str = "This review sentence is about {}."

# Aspect de repli si aucun score ne dépasse le seuil
FALLBACK_ASPECT: str = "product quality and performance"

# Cues lexicales (weak supervision) pour corriger les ratés zero-shot évidents
_ASPECT_CUES: dict[str, tuple[str, ...]] = {
    "size, dimensions and fit": (
        r"\bsmall\b",
        r"\blarge\b",
        r"\bbig\b",
        r"\btiny\b",
        r"\bfit[s]?\b",
        r"\bsize\b",
        r"\btoo (big|small|tight|loose)\b",
        r"\bsnug\b",
        r"\btight\b",
        r"\bdimensions?\b",
    ),
    "price, cost and value for money": (
        r"\bprice\b",
        r"\bexpensive\b",
        r"\bcheap\b",
        r"\bworth\b",
        r"\bvalue\b",
        r"\bcost\b",
        r"\bmoney\b",
        r"\baffordable\b",
    ),
    "packaging, shipping and delivery": (
        r"\bship(?:ping|ped)?\b",
        r"\bdeliver(?:y|ed)?\b",
        r"\bpackage[d]?\b",
        r"\bpackaging\b",
        r"\barrived\b",
        r"\bbox\b",
    ),
    "customer service and seller support": (
        r"\brefund\b",
        r"\breturn\b",
        r"\bseller\b",
        r"\bcustomer service\b",
        r"\bsupport\b",
        r"\bwarranty\b",
    ),
    "comfort, softness and texture": (
        r"\bsoft(?:ness|er)?\b",
        r"\bcomfort(?:able)?\b",
        r"\bstiff\b",
        r"\btexture\b",
        r"\bcozy\b",
    ),
    "durability, washability and longevity": (
        r"\bwash(?:ing|ed|able)?\b",
        r"\bdurable\b",
        r"\bbroke\b",
        r"\bbreak\b",
        r"\blast(?:ed|ing)?\b",
        r"\brusted?\b",
        r"\bheld up\b",
    ),
    "design, appearance and looks": (
        r"\bcute\b",
        r"\badorable\b",
        r"\bpretty\b",
        r"\blook(?:s|ed)?\b",
        r"\bdesign\b",
        r"\bcolor\b",
        r"\bprint[s]?\b",
    ),
    "safety for babies and children": (
        r"\bsafe(?:ty)?\b",
        r"\btoxic\b",
        r"\bchok(?:e|ing)\b",
        r"\bhazard\b",
    ),
    "ease of use, assembly and convenience": (
        r"\beasy\b",
        r"\bassembl(?:y|e)\b",
        r"\binstall\b",
        r"\bconvenient\b",
        r"\bone hand\b",
    ),
}

_COMPILED_CUES: dict[str, tuple[re.Pattern[str], ...]] = {
    aspect: tuple(re.compile(p, re.IGNORECASE) for p in patterns)
    for aspect, patterns in _ASPECT_CUES.items()
}


def lexical_aspect_hits(sentence: str) -> list[str]:
    """Retourne les aspects dont au moins un cue lexical matche la phrase."""
    hits: list[str] = []
    for aspect, patterns in _COMPILED_CUES.items():
        if any(p.search(sentence) for p in patterns):
            hits.append(aspect)
    return hits
