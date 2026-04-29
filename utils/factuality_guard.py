import re

def detect_misleading_prompt(text):
    """
    Detects if user prompt contains assumptions that may be false.
    """

    red_flags = [
        r"why is .* always",
        r"prove that .* is true",
        r"isn't it true that",
        r"everyone knows",
        r"why do .* never",
    ]

    for pattern in red_flags:
        if re.search(pattern, text.lower()):
            return True

    return False


def rewrite_prompt(text):
    """
    Neutralize biased or misleading prompts
    """
    return f"Answer factually and critically: {text}"