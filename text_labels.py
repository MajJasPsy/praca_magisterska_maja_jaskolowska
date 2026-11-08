"""Wyłącznie etykiety wyświetlane uczestnikowi; identyfikatory danych bez zmian."""


def display_context(context, config=None):
    """Tłumaczenie etykiet stanów na polski, z możliwością nadpisania w YAML."""
    labels = (config or {}).get("contexts", {}).get("labels", {})
    defaults = {"high": "A", "neutral": "B", "low": "C"}
    return str(labels.get(context, defaults.get(context, context)))
