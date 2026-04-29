import re
from typing import Dict


class BehaviorGuard:
    """
    SAF-RAG Behavior Guard (STABLE + NON-DESTRUCTIVE)

    Controls:
    - Sycophancy suppression (soft-first, not overblocking)
    - Roleplay mitigation (redirect, not reject)
    - Dummy/fake code filtering
    - Grounding enforcement (primary safety signal)
    """

    def __init__(
        self,
        sycophancy_threshold: float = 0.75,
        grounding_threshold: float = 0.35
    ):
        self.sycophancy_threshold = sycophancy_threshold
        self.grounding_threshold = grounding_threshold

        # Precompile regex for speed
        self.roleplay_patterns = [
            re.compile(p) for p in [
                r"\bpretend\b",
                r"\bimagine\b",
                r"\brole ?play\b",
                r"\bact as\b",
                r"\byou are now\b",
                r"\bsimulate\b"
            ]
        ]

        self.fake_code_patterns = [
            re.compile(p) for p in [
                r"```",
                r"def\s+\w+\(",
                r"class\s+\w+",
                r"#\s*(dummy|sample|example)\s*code",
                r"lorem ipsum"
            ]
        ]

        self.agreement_phrases = [
            "you are absolutely right",
            "i completely agree",
            "you're correct",
            "yes, exactly",
            "that is true"
        ]

    # =========================================================
    # DETECTORS
    # =========================================================
    def detect_roleplay(self, text: str) -> bool:
        if not text:
            return False
        t = text.lower()
        return any(p.search(t) for p in self.roleplay_patterns)

    def detect_fake_code(self, text: str) -> bool:
        if not text:
            return False
        t = text.lower()
        return any(p.search(t) for p in self.fake_code_patterns)

    def detect_over_agreement(self, text: str) -> bool:
        if not text:
            return False
        t = text.lower()
        return any(p in t for p in self.agreement_phrases)

    # =========================================================
    # CLEANERS
    # =========================================================
    def _remove_agreement_phrases(self, text: str) -> str:
        cleaned = text
        for phrase in self.agreement_phrases:
            cleaned = re.sub(re.escape(phrase), "", cleaned, flags=re.IGNORECASE)
        return cleaned.strip()

    def _safe_fallback(self) -> str:
        return "I don't know based on the available evidence."

    # =========================================================
    # MAIN CONTROL
    # =========================================================
    def apply(
        self,
        query: str,
        response: str,
        sycophancy_score: float,
        grounding_score: float
    ) -> Dict:

        if not isinstance(response, str) or not response.strip():
            return {
                "final_response": self._safe_fallback(),
                "action": "empty_response"
            }

        # -------------------------
        # 🔥 PRIORITY 1: GROUNDING (MOST IMPORTANT)
        # -------------------------
        if grounding_score < self.grounding_threshold:
            return {
                "final_response": self._safe_fallback(),
                "action": "blocked_low_grounding"
            }

        # -------------------------
        # 🔥 ROLEPLAY (SOFT REDIRECT)
        # -------------------------
        if self.detect_roleplay(query):
            return {
                "final_response": "I will provide a factual answer without roleplay.",
                "action": "redirected_roleplay"
            }

        # -------------------------
        # 🔥 FAKE CODE (HARD BLOCK)
        # -------------------------
        if self.detect_fake_code(response):
            return {
                "final_response": "Focus on the correct reasoning and final answer instead of example code.",
                "action": "blocked_fake_code"
            }

        # -------------------------
        # 🔥 SYCOPHANCY (SOFT FIRST)
        # -------------------------
        if sycophancy_score > self.sycophancy_threshold:
            # do NOT fully block — downgrade confidence instead
            cleaned = self._remove_agreement_phrases(response)

            if not cleaned:
                return {
                    "final_response": "The claim should be evaluated critically based on evidence.",
                    "action": "soft_blocked_sycophancy"
                }

            return {
                "final_response": cleaned,
                "action": "soft_corrected_sycophancy"
            }

        # -------------------------
        # 🔥 OVER-AGREEMENT CLEANUP
        # -------------------------
        if self.detect_over_agreement(response):
            cleaned = self._remove_agreement_phrases(response)

            return {
                "final_response": cleaned if cleaned else self._safe_fallback(),
                "action": "removed_agreement_bias"
            }

        # -------------------------
        # ✅ SAFE OUTPUT
        # -------------------------
        return {
            "final_response": response.strip(),
            "action": "accepted"
        }