"""The corpus, its passage ids, and the retriever that exists so the harness runs
with no index to build.

The passage-id tests matter more than they look: every `grounded-qa` task cites
passages by id, so an id scheme that is unstable between file orderings would make
the benchmark's hardest family depend on the filesystem.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import make_corpus

from bench.core.corpus import Corpus, slugify, tokenize
from bench.errors import TaskError


def test_a_document_becomes_one_passage_per_section_plus_preamble(tmp_path: Path):
    root = make_corpus(tmp_path)
    corpus = Corpus.load(root)
    ids = corpus.ids()
    assert "docs/incident.md" in ids  # the lede, before the first heading
    assert "docs/incident.md#root-cause" in ids
    assert "docs/incident.md#timeline" in ids
    assert len(corpus) == 3


def test_passage_ids_are_readable_and_stable(tmp_path: Path):
    """`incident-2026-08.md#root-cause` names a place a person can open. `p_00417`
    names a row in somebody's index."""
    root = make_corpus(tmp_path)
    corpus = Corpus.load(root)
    record = corpus.get("docs/incident.md#root-cause")
    assert record.heading == "Root cause"
    assert record.source == "docs/incident.md"
    assert "permission change" in record.text
    assert corpus.has("docs/incident.md#timeline")
    assert not corpus.has("docs/incident.md#nope")


def test_the_same_corpus_loads_to_the_same_digest(tmp_path: Path):
    """A result recorded against a corpus has to be checkable later, which means the
    digest cannot depend on inode order."""
    first = Corpus.load(make_corpus(tmp_path)).digest
    second = Corpus.load(make_corpus(tmp_path / "again")).digest
    assert first == second and len(first) == 16


def test_editing_a_passage_moves_the_digest(tmp_path: Path):
    before = Corpus.load(make_corpus(tmp_path)).digest
    after = Corpus.load(
        make_corpus(tmp_path / "edited", body="## Root cause\n\nSomething else entirely.\n")
    ).digest
    assert before != after


def test_two_headings_with_the_same_title_both_survive(tmp_path: Path):
    """Common in real docs — two "Examples" sections — and silently keeping one is
    how a task ends up citing a passage that exists but that nobody can find."""
    root = make_corpus(tmp_path, body="## Examples\n\nOne.\n\n## Examples\n\nTwo.\n")
    corpus = Corpus.load(root)
    assert "docs/incident.md#examples" in corpus.ids()
    assert "docs/incident.md#examples-2" in corpus.ids()
    assert "Two." in corpus.get("docs/incident.md#examples-2").text


def test_a_file_with_no_headings_is_one_passage(tmp_path: Path):
    root = make_corpus(tmp_path / "flat", body="just a note about tokens\n")
    corpus = Corpus.load(root)
    assert corpus.ids() == ["docs/incident.md"]


def test_markdown_and_text_are_read_and_other_files_ignored(tmp_path: Path):
    root = make_corpus(tmp_path)
    (root / "notes.txt").write_text("a plain note", encoding="utf-8")
    (root / "image.png").write_bytes(b"\x89PNG")
    corpus = Corpus.load(root)
    assert "notes.txt" in corpus.ids()
    assert not any("png" in passage_id for passage_id in corpus.ids())


def test_a_missing_corpus_says_which_directory(tmp_path: Path):
    with pytest.raises(TaskError, match="corpus directory not found"):
        Corpus.load(tmp_path / "nowhere")


def test_a_corpus_that_is_a_file_is_refused(tmp_path: Path):
    file = tmp_path / "corpus.md"
    file.write_text("text", encoding="utf-8")
    with pytest.raises(TaskError, match="should be a directory"):
        Corpus.load(file)


def test_a_corpus_with_nothing_retrievable_is_refused(tmp_path: Path):
    """An empty corpus would score every grounded task zero and read as "the entrant
    is bad at retrieval", which is the wrong conclusion about the wrong thing."""
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "data.bin").write_bytes(b"\x00")
    with pytest.raises(TaskError, match="nothing to retrieve"):
        Corpus.load(empty)


# --- retrieval ---------------------------------------------------------------


def test_search_finds_the_passage_that_answers_the_question(tmp_path: Path):
    corpus = Corpus.load(make_corpus(tmp_path))
    results = corpus.search("why did refunding stop", limit=3)
    assert results
    assert results[0].id == "docs/incident.md#root-cause"
    assert results[0].score is not None and results[0].score > 0


def test_search_returns_nothing_for_a_question_the_corpus_cannot_answer(tmp_path: Path):
    """The `answerable: false` tasks depend on this: the retriever's own honesty is
    what makes "I don't know" checkable without a judge.

    This is where the stopword list earns its place. Without it the query matches on
    "what", "is", "the" and "of", all of which are in the corpus, and the retriever
    answers every question ever asked with the nearest vaguely-worded passage.
    """
    corpus = Corpus.load(make_corpus(tmp_path))
    assert corpus.search("what is the capital of peru") == []


def test_a_question_made_only_of_stopwords_retrieves_nothing(tmp_path: Path):
    corpus = Corpus.load(make_corpus(tmp_path))
    assert corpus.search("what is it for") == []
    assert corpus.search("the of and to") == []


def test_one_content_word_that_is_in_the_corpus_still_retrieves(tmp_path: Path):
    """The other half of the rule: dropping stopwords must not make the retriever
    useless, only quiet about things it has never read."""
    corpus = Corpus.load(make_corpus(tmp_path))
    results = corpus.search("timeline")
    assert [passage.id for passage in results] == ["docs/incident.md#timeline"]


def test_the_stopword_list_is_the_benchmark_definition_and_is_small_on_purpose():
    """A dependency would hide it; a thousand-word list would change retrieval every
    time somebody added to it. Both are visible here instead."""
    from bench.core.corpus import STOPWORDS, content_tokens

    assert 50 < len(STOPWORDS) < 200
    assert "the" in STOPWORDS and "permission" not in STOPWORDS
    assert content_tokens("The refund of a permission") == ["refund", "permission"]


def test_search_respects_the_limit_and_orders_by_relevance(tmp_path: Path):
    corpus = Corpus.load(make_corpus(tmp_path))
    results = corpus.search("rollout refunding permission change", limit=2)
    assert len(results) <= 2
    if len(results) == 2:
        assert results[0].score >= results[1].score


def test_search_of_an_empty_query_returns_nothing(tmp_path: Path):
    corpus = Corpus.load(make_corpus(tmp_path))
    assert corpus.search("   ") == []


def test_the_retriever_is_reproducible(tmp_path: Path):
    """Two runs of the same query on the same corpus give the same order, tie-breaks
    included: a fixture target whose retrieval shuffles between repeats would show up
    as flakiness that belongs to the harness."""
    corpus = Corpus.load(make_corpus(tmp_path))
    first = [passage.id for passage in corpus.search("refund rollout", limit=3)]
    second = [passage.id for passage in corpus.search("refund rollout", limit=3)]
    assert first == second


def test_a_get_for_a_missing_passage_names_it(tmp_path: Path):
    corpus = Corpus.load(make_corpus(tmp_path))
    with pytest.raises(TaskError, match="no passage"):
        corpus.get("docs/incident.md#imaginary")


def test_a_passage_converts_to_the_trace_shape(tmp_path: Path):
    """The trace carries the passage text, not just the id, so a score can be
    explained a year later when the corpus has moved on."""
    corpus = Corpus.load(make_corpus(tmp_path))
    passage = corpus.search("permission change", limit=1)[0]
    assert passage.id.startswith("docs/incident.md#")
    assert "permission change" in passage.text


# --- slugs -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("Root cause", "root-cause"),
        ("Root cause: the permission change", "root-cause-the-permission-change"),
        ("v2.1 rollout", "v2-1-rollout"),
        ("What happened?", "what-happened"),
        ("!!!", "section"),
    ],
)
def test_slugs_survive_real_headings(title: str, expected: str):
    """Punctuation goes rather than becoming a dash, so `v2.1` and `v2-1` cannot
    collide into one passage id — which would silently merge two documents."""
    assert slugify(title) == expected


def test_tokenize_lowercases_and_drops_punctuation():
    assert tokenize("Root cause: the v2.1 rollout!") == [
        "root",
        "cause",
        "the",
        "v2",
        "1",
        "rollout",
    ]
