from langchain_core.documents import Document

from app.ingest import chunk_fingerprint, clean_text, is_noise_chunk, split_documents


def test_clean_text_rejoins_hyphenated_line_breaks():
    assert "altimeter" in clean_text("alti-\nmeter setting")


def test_clean_text_drops_bare_page_numbers():
    assert clean_text("Chapter 3\n147\nCold Fronts") == "Chapter 3 Cold Fronts"


def test_noise_chunk_detects_exam_material():
    assert is_noise_chunk("Questions a. one b. two c. three d. four")
    assert not is_noise_chunk("A cold front occurs when cold air replaces warm air.")


def test_fingerprint_is_stable_across_whitespace():
    a = Document(page_content="Cold  front\n replaces warm air", metadata={"source": "s.pdf", "page": 1})
    b = Document(page_content="cold front replaces warm air", metadata={"source": "s.pdf", "page": 1})
    assert chunk_fingerprint(a) == chunk_fingerprint(b)


def test_fingerprint_differs_across_pages():
    a = Document(page_content="same text", metadata={"source": "s.pdf", "page": 1})
    b = Document(page_content="same text", metadata={"source": "s.pdf", "page": 2})
    assert chunk_fingerprint(a) != chunk_fingerprint(b)


def test_split_assigns_traceable_chunk_ids():
    docs = [Document(page_content="Cold fronts. " * 200, metadata={"source": "s.pdf", "page": 4})]
    chunks, dropped = split_documents(docs, chunk_size=200, chunk_overlap=20)
    assert chunks
    assert all(c.metadata["chunk_id"].startswith("s.pdf:p4:c") for c in chunks)
