"""
Rule-based recommendations. Deliberately simple and legible (if/elif on a
risk level) rather than a black box, since the output is operational
guidance a human is meant to act on - it should be auditable at a glance.
"""

STANDING_RISK_ACTIONS = {
    "High": [
        "Increase field monitoring frequency",
        "Pre-position coordination with emergency/security services",
        "Track scheduled demonstrations or political events in the county",
        "Cross-check against the latest news panel for emerging triggers",
    ],
    "Moderate": [
        "Maintain routine active monitoring",
        "Review local indicators weekly rather than monthly",
        "Flag for re-assessment if event frequency rises",
    ],
    "Low": [
        "Continue routine monitoring",
        "No elevated action indicated by current data",
    ],
}

EVENT_ACTIONS = {
    "High": [
        "Treat as a priority incident for situational awareness",
        "Verify against live reporting before acting operationally",
        "Notify relevant county-level coordination contacts",
    ],
    "Medium": [
        "Log and monitor for escalation over the following weeks",
        "Cross-reference with the county's standing vulnerability rank",
    ],
    "Low": [
        "Routine logging - no elevated response indicated",
    ],
    "Non-fatal": [
        "No fatality predicted - monitor as routine data only",
    ],
}


def severity_level_from_percentile(value: float, series) -> str:
    """Classify a county's predicted severity into High/Moderate/Low by
    percentile within the national distribution - keeps the threshold
    self-adjusting rather than a hardcoded number that goes stale."""
    if value >= series.quantile(0.8):
        return "High"
    elif value >= series.quantile(0.5):
        return "Moderate"
    return "Low"


def get_standing_recommendations(level: str) -> list:
    return STANDING_RISK_ACTIONS.get(level, STANDING_RISK_ACTIONS["Low"])


def get_event_recommendations(severity_class: str) -> list:
    return EVENT_ACTIONS.get(severity_class, EVENT_ACTIONS["Low"])
