import re
from typing import List


class ResponseFilter:
    """
    SAF Response Filter (Improved)

    Handles:
    - Light roleplay cleanup
    - Code / dummy content removal
    - Sycophancy soft correction (non-destructive)
    - Final response normalization

    IMPORTANT:
    - Does NOT overwrite valid answers
    - Preserves grounded factual content
    """

    def __init__(self):

        # --------------------------------------------------
        # ROLEPLAY / META PATTERNS
        # --------------------------------------------------
        self.roleplay_patterns: List[str] = [
            r"\bas an ai\b",
            r"\bas a language model\b",
            r"\bi am an ai\b",
            r"\bi cannot\b",
            r"\blet me roleplay\b",
            r"\bpretend that\b",
        ]

        # --------------------------------------------------
        # CODE / DUMMY PATTERNS
        # --------------------------------------------------
        self.code_patterns: List[str] = [
            r"```.*?```",               # code blocks
            r"\bexample code\b",
            r"\bsample code\b",
            r"\bdummy code\b",
            r"\bpseudo code\b",
            r"\bhere is.*code\b",
            r"\bdef\s+\w+\(",
            r"\bclass\s+\w+",
            r"\bimport\s+\w+",
        ]

        # --------------------------------------------------
        # SYCOPHANCY PATTERNS (PREFIX ONLY)
        # --------------------------------------------------
        self.sycophancy_prefixes: List[str] = [
            r"^\s*yes[, ]",
            r"^\s*i agree[, ]?",
            r"^\s*you are right[, ]?",
            r"^\s*that's correct[, ]?",
            r"^\s*of course[, ]?",
        ]

    # =========================================================
    # DETECTION
    # =========================================================
    def detect_roleplay(self, text: str) -> float:
        return self._pattern_score(text, self.roleplay_patterns)

    def detect_code(self, text: str) -> float:
        return self._pattern_score(text, self.code_patterns)

    def detect_sycophancy(self, text: str) -> float:
        return self._pattern_score(text, self.sycophancy_prefixes)

    def _pattern_score(self, text: str, patterns: List[str]) -> float:
        if not text:
            return 0.0
        text = text.lower()
        matches = sum(bool(re.search(p, text)) for p in patterns)
        return matches / max(len(patterns), 1)

    # =========================================================
    # CLEANING HELPERS
    # =========================================================
    def _remove_code(self, text: str) -> str:
        for p in self.code_patterns:
            text = re.sub(p, "", text, flags=re.IGNORECASE | re.DOTALL)
        return text

    def _remove_roleplay(self, text: str) -> str:
        for p in self.roleplay_patterns:
            text = re.sub(p, "", text, flags=re.IGNORECASE)
        return text

    def _remove_sycophancy_prefix(self, text: str) -> str:
        for p in self.sycophancy_prefixes:
            text = re.sub(p, "", text, flags=re.IGNORECASE)
        return text.strip()

    def _basic_cleanup(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\n{2,}", "\n", text)
        return text.strip()

    # =========================================================
    # MAIN FILTER (NON-DESTRUCTIVE)
    # =========================================================
    def clean(self, text: str) -> str:

        if not isinstance(text, str) or not text.strip():
            return "I don't know."

        original = text

        # -------------------------
        # REMOVE UNWANTED CONTENT
        # -------------------------
        text = self._remove_code(text)
        text = self._remove_roleplay(text)
        text = self._remove_sycophancy_prefix(text)

        text = self._basic_cleanup(text)

        # -------------------------
        # FALLBACK SAFETY
        # -------------------------
        if not text or len(text) < 5:
            return "I don't know."

        return text