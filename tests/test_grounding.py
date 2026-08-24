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
