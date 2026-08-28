"""The retrieval corpus, and the retriever that ships with the harness.

Two decisions here, both of them about reproducibility rather than about retrieval
quality.

**Passage ids are derived from the file, not assigned.** `incident-2026-08.md#
root-cause` is the second `##` section of a file. That means a task can cite a
passage and the citation still resolves after somebody edits the document, because
the id names a place in a file that a person can open — rather than `p_00417`, which
names a row in somebody's index and is meaningless outside it. Renaming a heading
renames a passage id, which shows up as a failed citation and a diff, and that is
the honest outcome.

**The corpus is hashed, and the hash is part of a result.** A retrieval benchmark
whose corpus can change under it measures nothing across time. The digest covers
every passage id and text, so a result recorded against a corpus can be checked
later, and the board refuses to compare two entries that ran against different
corpora of the same name.

**Stopwords are dropped, and that is not a detail.** BM25 over three short passages
returns something for every question ever asked, because "what", "is" and "the" live
in the corpus too. The `answerable` tasks depend on the retriever being willing to
come back empty — "the corpus has nothing to say about this" is only checkable if the
retriever is allowed to say it — so the English function words are removed before
indexing and before searching. The list is small, English, and here rather than in a
dependency: a corpus in another language needs its own list, and that is a note in the
methodology, not a hidden assumption.

The bundled retriever is BM25-lite and exists so that the harness runs offline with
no index to build. It is explicitly not the thing being benchmarked: real entrants
bring their own retrieval, record it, and replay it (`--offline`). This one is for
the fixture tasks and for the example targets in `examples/`.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from ..errors import TaskError
from .trace import Passage

_WORD = re.compile(r"[a-z0-9]+")

# SIM905 wants a list literal here. A word list is the one place where the
# triple-quoted block is the readable form: a literal would be a 120-character line
# and every edit to it would show up as a whole-line diff.
STOPWORDS = frozenset("""
    a about after all also an and any are as at be because been before being between
    both but by can cannot could did do does doing done down during each few for from
    further had has have having he her here hers him his how i if in into is it its
    itself just me more most my no nor not of off on once only or other our out over
    own same she should so some such than that the their them then there these they
    this those through to too under until up very was we were what when where which
    while who whom why will with would you your
    """.split())  # noqa: SIM905
"""Function words, dropped before indexing and searching.

Hand-written and short on purpose: it is part of the benchmark's definition of
"relevant", so it belongs in the repository where a reviewer can read it and a diff
can show it changing. A corpus in a language other than English needs its own list,
which is a limitation stated in `docs/METHODOLOGY.md` rather than a silent one."""

_HEADING = re.compile(r"^##\s+(?P<title>.+?)\s*$", re.MULTILINE)


def slugify(title: str) -> str:
    """`Root cause: the permission change` → `root-cause-the-permission-change`.

    Punctuation goes rather than becoming dashes, so `v2.1` and `v2-1` cannot
    collide into the same passage id, and a heading that is entirely punctuation
    falls back to `section` rather than producing an empty anchor.
    """
    words = _WORD.findall(title.lower())
    return "-".join(words) or "section"


def tokenize(text: str) -> list[str]:
    """Every word, lowercased, punctuation gone. Stopwords included: this is what the
    tokeniser does, and `content_tokens` is what the retriever uses."""
    return _WORD.findall(text.lower())


def content_tokens(text: str) -> list[str]:
    """What the index and the queries are built from: stopwords removed, and a query
    that consists only of stopwords becomes empty rather than matching everything."""
    return [word for word in tokenize(text) if word not in STOPWORDS]


@dataclass(frozen=True)
class PassageRecord:
    id: str
    text: str
    source: str
    """Repo-relative path, kept beside the id so the report can say where to look."""

    heading: str | None = None

    def as_passage(self, *, score: float | None = None) -> Passage:
        return Passage(id=self.id, text=self.text, score=score)


@dataclass
class Corpus:
    root: Path
    passages: list[PassageRecord]
    _index: dict[str, list[str]] = field(default_factory=dict, repr=False)
    _length: dict[str, int] = field(default_factory=dict, repr=False)
    _avg_length: float = 0.0

    # --- construction -------------------------------------------------------

    @classmethod
    def load(cls, root: str | Path) -> Corpus:
        """Read every `.md` and `.txt` under `root`, in sorted order.

        Sorted rather than filesystem order: passage ordering ends up in the digest,
        and a digest that depends on inode order is a digest that differs between two
        checkouts of the same commit.
        """
        path = Path(root)
        if not path.exists():
            raise TaskError(f"corpus directory not found: {path}")
        if not path.is_dir():
            raise TaskError(f"corpus should be a directory, got a file: {path}")

        records: list[PassageRecord] = []
        for file in sorted(path.rglob("*")):
            if not file.is_file() or file.suffix not in {".md", ".txt"}:
                continue
            relative = file.relative_to(path).as_posix()
            records.extend(_split_file(file, relative))
        if not records:
            raise TaskError(
                f"{path} has nothing to retrieve: no .md or .txt files",
                hint="a corpus is a directory of markdown; each `##` section becomes a passage",
            )
        corpus = cls(root=path, passages=records)
        corpus._build_index()
        return corpus

    # --- lookup -------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.passages)

    def get(self, passage_id: str) -> PassageRecord:
        for record in self.passages:
            if record.id == passage_id:
                return record
        raise TaskError(f"no passage {passage_id!r} in the corpus at {self.root}")

    def has(self, passage_id: str) -> bool:
        return any(record.id == passage_id for record in self.passages)

    def ids(self) -> list[str]:
        return [record.id for record in self.passages]

    @property
    def digest(self) -> str:
        payload = "\n".join(f"{record.id}\t{record.text}" for record in self.passages)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    # --- retrieval ----------------------------------------------------------

    def search(self, query: str, *, limit: int = 5) -> list[Passage]:
        """BM25 over the passages, tuned for small documents.

        `k1=1.5, b=0.75` are the textbook middle values. They are written here rather
        than exposed as configuration because a benchmark's own retriever must not be
        a variable: if these numbers were tunable, somebody would tune them, and the
        fixture results would stop being comparable between two clones of this repo.
        """
        terms = content_tokens(query)
        if not terms:
            return []
        k1, b = 1.5, 0.75
        total = len(self.passages)
        scored: list[tuple[float, PassageRecord]] = []
        for record in self.passages:
            score = 0.0
            frequencies = Counter(content_tokens(record.text))
            length = sum(frequencies.values()) or 1
            for term in set(terms):
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                document_frequency = len(self._index.get(term, []))
                if document_frequency == 0:
                    continue
                idf = math.log(1 + (total - document_frequency + 0.5) / (document_frequency + 0.5))
                denominator = frequency + k1 * (1 - b + b * length / self._avg_length)
                score += idf * frequency * (k1 + 1) / denominator
            if score > 0:
                scored.append((score, record))
        scored.sort(key=lambda item: (-item[0], item[1].id))
        return [record.as_passage(score=round(score, 6)) for score, record in scored[:limit]]

    def _build_index(self) -> None:
        index: dict[str, list[str]] = {}
        lengths: dict[str, int] = {}
        for record in self.passages:
            tokens = content_tokens(record.text)
            lengths[record.id] = len(tokens)
            for term in set(tokens):
                index.setdefault(term, []).append(record.id)
        self._index = index
        self._length = lengths
        self._avg_length = (sum(lengths.values()) / len(lengths)) if lengths else 1.0


def _split_file(file: Path, relative: str) -> list[PassageRecord]:
    """Turn one markdown file into passages, one per `##` section.

    Text before the first heading becomes a passage with the bare file id, which is
    usually the file's lede and is exactly what a question about "what is this
    service" should retrieve. Files with no headings at all become a single passage:
    a short note does not need to be cut up.
    """
    text = file.read_text(encoding="utf-8")
    matches = list(_HEADING.finditer(text))
    records: list[PassageRecord] = []
    if not matches:
        body = text.strip()
        if body:
            records.append(PassageRecord(id=relative, text=body, source=relative))
        return records

    preamble = text[: matches[0].start()].strip()
    if preamble:
        records.append(PassageRecord(id=relative, text=preamble, source=relative))

    used: Counter[str] = Counter()
    for position, match in enumerate(matches):
        end = matches[position + 1].start() if position + 1 < len(matches) else len(text)
        body = text[match.start() : end].strip()
        if not body:
            continue
        anchor = slugify(match.group("title"))
        used[anchor] += 1
        if used[anchor] > 1:
            # Two headings with the same title in one file is common (two
            # "Examples" sections) and silently dropping one is how a task ends up
            # citing a passage that exists but that nobody can find.
            anchor = f"{anchor}-{used[anchor]}"
        records.append(
            PassageRecord(
                id=f"{relative}#{anchor}",
                text=body,
                source=relative,
                heading=match.group("title").strip(),
            )
        )
    return records
