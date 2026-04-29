import re


class SycophancyGuard:
    """
    Prevents the model from blindly agreeing with misleading or biased prompts.

    It rewrites high-risk queries into verification-oriented prompts.
    """

    def __init__(self):
        # patterns strongly correlated with sycophancy in your dataset
        self.bias_patterns = [
            r"isn'?t it",
            r"don'?t you think",
            r"right\?",
            r"obviously",
            r"clearly",
            r"everyone knows",
            r"assume .* correct",
            r"even if .* wrong"
        ]

    # =========================
    # DETECT BIAS
    # =========================
    def detect_bias(self, query: str) -> bool:
        q = query.lower()

        for pattern in self.bias_patterns:
            if re.search(pattern, q):
                return True

        return False

    # =========================
    # REWRITE QUERY
    # =========================
    def rewrite(self, query: str) -> str:
        """
        Converts misleading prompt into grounded reasoning task.
        """

        # remove strong assumption phrases
        query = re.sub(r"assume .*? correct[:,]?", "", query, flags=re.IGNORECASE)
        query = re.sub(r"even if .*? wrong[:,]?", "", query, flags=re.IGNORECASE)

        # normalize bias phrasing
        query = re.sub(r"isn'?t it\??", "", query, flags=re.IGNORECASE)
        query = re.sub(r"don'?t you think\??", "", query, flags=re.IGNORECASE)
        query = re.sub(r"right\?", "", query, flags=re.IGNORECASE)

        query = query.strip()

        #  critical transformation (THIS is your contribution)
        rewritten = (
            "Answer the following question using verified evidence. "
            "If the assumption is incorrect, correct it explicitly:\n\n"
            f"{query}"
        )

        return rewritten

    # =========================
    # APPLY GUARD
    # =========================
    def apply(self, query: str, risk: float, threshold: float = 0.5) -> str:
        """
        Applies guard only when risk is high.
        """

        if risk >= threshold or self.detect_bias(query):
            return self.rewrite(query)

        return query