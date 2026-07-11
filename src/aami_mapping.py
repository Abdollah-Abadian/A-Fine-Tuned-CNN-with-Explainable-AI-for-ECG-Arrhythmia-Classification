"""
Mapping of MIT-BIH annotation symbols to AAMI EC57:1998 superclasses.

Implements Table 1 of the paper exactly. The MIT-BIH database annotates
beats using single-character symbols (wfdb `annotation.symbol`); we map both
the raw wfdb symbols and the "original code" numbering used in the paper's
Table 1 to the five AAMI classes {N, S, V, F, Q}.
"""

from typing import Dict

# wfdb annotation symbol -> (original MIT-BIH numeric code, description, AAMI class)
SYMBOL_TABLE: Dict[str, tuple] = {
    "N": (0, "Normal (NOR)", "N"),
    "L": (1, "Left Bundle Branch Block (LBBB)", "N"),
    "R": (2, "Right Bundle Branch Block (RBBB)", "N"),
    "a": (3, "Aberrated Atrial Premature Beat (APB)", "S"),
    "V": (4, "Premature Ventricular Contraction (PVC)", "V"),
    "F": (5, "Fusion of Ventricular and Normal Beat", "F"),
    "J": (6, "Nodal (Junctional) Premature Beat", "S"),
    "A": (7, "Atrial Premature Beat (APB)", "S"),
    "S": (8, "Premature or Ectopic Supraventricular Beat", "S"),
    "E": (9, "Ventricular Escape Beat", "V"),
    "j": (10, "Nodal (Junctional) Escape Beat", "S"),
    "!": (11, "Ventricular Flutter Wave", "V"),
    "/": (12, "Paced Beat", "Q"),
    "f": (13, "Fusion of Paced and Normal Beat", "Q"),
    "Q": (14, "Unclassifiable Beat", "Q"),
    "x": (34, "Non-conducted P-wave (Blocked APB)", "S"),
}

# Symbols present in MIT-BIH .atr files that are NOT heartbeat annotations
# (rhythm change markers, signal quality markers, etc.) and must be skipped
# during beat extraction.
NON_BEAT_SYMBOLS = {
    "+", "~", "|", '"', "[", "]", "!", "x", "(", ")", "p", "t", "u", "`",
    "'", "^", "|", "s", "T", "*", "D", "=", "P",
}
# Note: '!' and 'x' also appear in SYMBOL_TABLE (ventricular flutter wave and
# blocked APB respectively, which ARE valid beat annotations in MIT-BIH); we
# therefore only treat the remaining non-beat symbols as skippable.
NON_BEAT_SYMBOLS = NON_BEAT_SYMBOLS - {"!", "x"}


def symbol_to_aami(symbol: str) -> str:
    """Map a raw wfdb annotation symbol to its AAMI class, or None if the
    symbol does not correspond to a classifiable heartbeat."""
    entry = SYMBOL_TABLE.get(symbol)
    if entry is None:
        return None
    return entry[2]


def is_beat_symbol(symbol: str) -> bool:
    return symbol in SYMBOL_TABLE and symbol not in NON_BEAT_SYMBOLS


AAMI_CLASS_TO_INDEX = {"N": 0, "S": 1, "V": 2, "F": 3, "Q": 4}
INDEX_TO_AAMI_CLASS = {v: k for k, v in AAMI_CLASS_TO_INDEX.items()}
