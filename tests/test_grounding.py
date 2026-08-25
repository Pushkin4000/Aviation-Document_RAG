from langchain_core.documents import Document

from app.config import Settings
from app.grounding import is_grounded
from app.models import REFUSAL_MESSAGE, RetrievedChunk

SETTINGS = Settings(_env_file=None)
CONTEXT = (
    "Cold Fronts. If cold air is replacing warm air, then the front is called a "
    "cold front. The passage of a cold front is marked by a sharp change in wind "
    "direction and a fall in temperature."
)


def _chunks(text=CONTEXT):
    return [RetrievedChunk(document=Document(page_content=text, metadata={}), score=0.9)]


def test_refusal_is_always_grounded():
    assert is_grounded(REFUSAL_MESSAGE, [], SETTINGS) is True


def test_verbatim_span_is_grounded():
    assert is_grounded("If cold air is replacing warm air, then the front is called a cold front.", _chunks(), SETTINGS)


def test_invented_content_is_not_grounded():
    assert not is_grounded("A cold front always produces severe hail and tornado activity.", _chunks(), SETTINGS)


def test_answer_without_context_is_not_grounded():
    assert not is_grounded("A cold front replaces warm air.", [], SETTINGS)


def test_question_shaped_answer_is_rejected():
    assert not is_grounded("What is a cold front?", _chunks(), SETTINGS)


def test_multiple_choice_residue_is_rejected():
    assert not is_grounded("The front is a. warm b. cold c. occluded", _chunks(), SETTINGS)


def test_answer_containing_question_mark_is_rejected():
    assert not is_grounded("Is the front called a cold front?", _chunks(), SETTINGS)


def test_mostly_numeric_answer_is_rejected():
    assert not is_grounded("1013 2992 1015 1020 1025 1030 1035 1040", _chunks(), SETTINGS)


# --- MIN-1: the tests above all short-circuit via the verbatim-substring or
# >=20-char clause shortcut (grounding.py lines ~46-51) and never reach the
# similarity/overlap computation below it. That leaves the `and` at
# grounding.py:63 -- the conjunction the whole abstractive gate (CRIT-1)
# depends on -- with zero coverage: flipping it to `or` left all tests
# green. These three cases are constructed to bypass the shortcut (verified
# not verbatim and no >=20-char clause is an exact substring of the
# context) and actually exercise fuzz.partial_ratio + token overlap.

def test_paraphrase_passing_similarity_and_overlap_is_grounded():
    # Reworded, not a verbatim span and no clause is an exact substring of
    # the context, but both similarity (~0.73) and overlap (~0.73) clear the
    # extractive thresholds (0.65 / 0.58) -- this must reach and pass the
    # similarity/overlap gate itself, not an earlier shortcut.
    answer = (
        "If cold air is replacing warm air, the front is known as a cold "
        "front, with a sharp change in wind direction and a fall in "
        "temperature."
    )
    assert is_grounded(answer, _chunks(), SETTINGS)


def test_high_similarity_low_overlap_paraphrase_is_rejected():
    # Similarity alone clears the bar (~0.69 >= 0.65: the phrasing tracks
    # the source closely) but token overlap does not (~0.37 < 0.58: half
    # the content words -- NOTAMs, slot times, de-icing schedules -- are
    # not in the context at all). Only the `and` at grounding.py:63 rejects
    # this; `or` would wrongly accept it. This is the direct proof of the
    # conjunction.
    answer = (
        "If cold air is replacing warm air, then the front is called a "
        "cold front, and pilots should also check NOTAMs, slot times, and "
        "de-icing schedules."
    )
    assert not is_grounded(answer, _chunks(), SETTINGS)


def test_abstractive_flag_selects_looser_thresholds():
    # CRIT-1: a paraphrase that fails the extractive pair (0.65/0.58) on
    # similarity alone (0.593 < 0.65) but clears the abstractive pair
    # (0.45/0.58) on both axes (0.593 >= 0.45, 0.786 >= 0.58). Must be
    # rejected under the default (extractive) call and accepted only when
    # abstractive=True is passed explicitly.
    answer = (
        "Warm air is replaced by cold air, giving a sharp wind direction "
        "change and a temperature fall at the front surface."
    )
    assert not is_grounded(answer, _chunks(), SETTINGS)
    assert is_grounded(answer, _chunks(), SETTINGS, abstractive=True)


def test_short_answer_relaxation_admits_overlap_below_strict_threshold():
    # 7 tokens (<=10), so the len(answer_token_list) <= 10 branch relaxes
    # the required overlap from 0.58 to max(0.5, 0.58 - 0.06) = 0.52. This
    # answer's overlap is ~0.57: below the strict 0.58 threshold but above
    # the relaxed 0.52 one, and its similarity (~0.73) clears 0.65. It must
    # be accepted -- proving the relaxation branch is live, not dead code.
    answer = "This front is called a cold front with wind nearby."
    assert is_grounded(answer, _chunks(), SETTINGS)
