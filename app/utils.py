"""
==============================================================
Utility Functions

Used by Streamlit UI

Tech Stack
----------
Pure Python
==============================================================
"""


def risk_level(score: float):
    """
    Convert predicted score into a risk label.
    """

    if score < 0.33:
        return "🟢 Low Risk"

    elif score < 0.66:
        return "🟠 Medium Risk"

    else:
        return "🔴 High Risk"


def automation_percentage(score: float):
    """
    Convert score into percentage.
    """

    return round(score * 100, 2)