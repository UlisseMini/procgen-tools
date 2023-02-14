from typing import *

def format_label(label : str):
    """Format a label for display in the visualization."""
    return label.replace("embedder.", "")

def expand_label(label : str):
    if not (label.startswith("fc_policy") or label.startswith("fc_value")):
        return "embedder." + label
    else:
        return label

def format_labels(labels : List[str]):
    """Format labels for display in the visualization."""
    return list(map(format_label, labels))