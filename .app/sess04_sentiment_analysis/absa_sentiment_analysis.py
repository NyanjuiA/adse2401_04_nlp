"""
=========================================================================================
Python script to demonstrate ABSA (Aspect Based Sentiment Analysis)
=========================================================================================

WHAT IS ABSA?
-------------
Standard sentiment analysis assigns ONE sentiment to an entire sentence:
    "The battery is great but the screen is awful."  →  MIXED

ABSA assigns sentiment PER ASPECT (topic/feature):
    battery  →  POSITIVE
    screen   →  NEGATIVE

This gives far richer, more actionable insights — especially useful for
product reviews, customer feedback, and survey analysis.

HOW THIS DEMO WORKS
--------------------
  1. ASPECT EXTRACTION  — rule-based keyword matching across 9 product
                          categories (battery, screen, camera, etc.)
  2. CLAUSE SPLITTING   — the sentence is split at conjunctions (but, and,
                          however, …) so that each clause is scored in
                          isolation, preventing sentiments from bleeding
                          across aspects.
  3. SENTIMENT SCORING  — VADER (Valence Aware Dictionary and sEntiment
                          Reasoner) scores each clause.  VADER is purpose-
                          built for short, informal text and requires NO
                          model download or GPU.

Requirements:
    pip install vaderSentiment

Author: Nyanjui
Date: 15 May 2026
"""
# --------------------------------------------------------------------------------
# 0. Import required modules
# --------------------------------------------------------------------------------
from __future__ import annotations # Ensure this is the 1st import to avoid getting errors

import re,sys,textwrap
from dataclasses import dataclass, field
from typing import Optional

# --------------------------------------------------------------------------------
# 1. Check dependencies
# --------------------------------------------------------------------------------
def _require(package: str, install_cmd: str) -> None:

    import importlib.util
    if importlib.util.find_spec(package) is None:
        print(f"\n[ERROR] Required package {package} not found."
              f"\n      Install it with: {install_cmd}\n")
        sys.exit(1)

_require("vaderSentiment", "pip install vaderSentiment")

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --------------------------------------------------------------------------------
# 2. Aspect taxonomy
# --------------------------------------------------------------------------------

# Each key is the canonical aspect name shown in the output.
# Each value is a list of surface form keywords (singular, plural, synonyms).
# Matching is case-sentisitive and whole-word only (regext \b boundaries)
ASPECT_TAXONOMY: dict[str, list[str]] = {
    "battery life":   ["battery life", "battery", "charge", "charging"],
    "screen":         ["screen", "display", "monitor", "resolution", "brightness"],
    "camera":         ["camera", "cameras", "photo", "photos", "lens", "zoom"],
    "performance":    ["performance", "speed", "processor", "lag", "fast", "slow", "snappy"],
    "price":          ["price", "cost", "expensive", "cheap", "affordable", "value"],
    "service":        ["service", "services", "support", "staff", "customer service"],
    "build quality":  ["build quality", "build", "design", "materials", "durability", "sturdy"],
    "storage":        ["storage", "memory", "space", "capacity"],
    "audio":          ["audio", "sound", "speaker", "speakers", "headphone", "headphones", "bass"],
}

# VADER compound-score thresholds (industry standard values)
POSITIVE_THRESHOLD =  0.05
NEGATIVE_THRESHOLD = -0.05

# --------------------------------------------------------------------------------
# 3. Data classes
# --------------------------------------------------------------------------------
@dataclass
class AspectResult:

    aspect: str
    sentiment: str          # "POSITIVE

