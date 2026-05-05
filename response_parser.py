"""
Response Parser Module

This module contains shared helpers for cleaning raw LLM response text.
"""


def clean_llm_response_text(text):
    """
    Clean LLM response text to remove non-printable characters and normalize spaces.

    Args:
        text (str): Raw text from LLM response

    Returns:
        str: Cleaned text ready for parsing
    """
    if text is None:
        return ""

    # Remove non-printable characters
    cleaned = "".join(ch for ch in text if ch.isprintable())

    # Normalize space characters
    cleaned = cleaned.replace("\t", " ").replace("\n", " ")

    # Remove extra spaces
    while "  " in cleaned:
        cleaned = cleaned.replace("  ", " ")

    return cleaned.strip()
