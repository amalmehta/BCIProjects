"""Phoneme alphabet shared by the neural data, the decoder and the lexicon.

The Brain-to-Text '25 data (Card et al., 2024) labels each sentence with a
sequence of 39 ARPAbet phonemes (stress removed) plus a word-boundary token,
and reserves index 0 for the CTC blank. We reuse that exact index layout so
labels from the HDF5 files can be used without remapping.
"""

from __future__ import annotations

BLANK = "<b>"
SIL = "|"  # word boundary / silence token (index 40 in the released data)

PHONEMES: list[str] = [
    "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH",
    "EH", "ER", "EY", "F", "G", "HH", "IH", "IY", "JH", "K",
    "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH",
    "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH",
]

#: Full CTC output alphabet: blank, 39 phonemes, word boundary (41 classes).
ALPHABET: list[str] = [BLANK] + PHONEMES + [SIL]
N_CLASSES = len(ALPHABET)
BLANK_ID = 0
SIL_ID = len(ALPHABET) - 1
PHONE_TO_ID = {p: i for i, p in enumerate(ALPHABET)}


def ids_to_phones(ids) -> list[str]:
    """Map integer class ids to phoneme strings (blank ids are dropped)."""
    return [ALPHABET[int(i)] for i in ids if int(i) != BLANK_ID]


def phones_to_ids(phones: list[str]) -> list[int]:
    """Map phoneme strings (ARPAbet, stress digits allowed) to class ids."""
    return [PHONE_TO_ID[p.rstrip("012")] for p in phones]


def strip_stress(phone: str) -> str:
    """Remove the CMUdict stress marker from a phoneme (``AH0`` -> ``AH``)."""
    return phone.rstrip("012")
