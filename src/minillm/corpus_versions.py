"""Corpus-version constants shared by the V4 and V5 language profiles.

Only identifiers and pinned target arithmetic live here.  Pipeline policy stays
in the versioned configuration and is never inferred from a corpus name.
"""
from __future__ import annotations

V4_1M_VERSION = "minillm-corpus-v4-language-pilot-1m-v1"
V4_2M_VERSION = "minillm-corpus-v4-language-pilot-2m-v1"
V5_VERSION = "minillm-corpus-v5-language-10m-v1"

V5_SEQ_LEN = 1024
V5_TOKENS_PER_UPDATE = 8184  # 8 windows x (1024 - 1) supervised targets
V5_UPDATES = 1222
V5_TARGET = V5_UPDATES * V5_TOKENS_PER_UPDATE  # 10,000,848 supervised tokens
V5_CATEGORY_RATIOS = {
    "zh_general": 0.45,
    "zh_encyclopedic": 0.30,
    "zh_education": 0.15,
    "en_general": 0.10,
    "math_logic_code": 0.0,
}
V5_CANDIDATE_TARGET = 15_000_000
V5_TOLERANCE = 0.01
