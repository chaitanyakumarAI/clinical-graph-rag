"""run_stage1.py: every PDF becomes its own Document, like POST /ingest."""

import hashlib
import sys
import types

import pytest

sys.modules.setdefault("fitz", types.ModuleType("fitz"))   # parse_pdf is stubbed below
import run_stage1   # noqa: E402


class FakeNeo4j:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def verify_connection(self):
        return "ok"

    def print_summary(self):
        pass


@pytest.fixture
def cli(tmp_path, monkeypatch):
    """Two PDFs in a folder; records every pipeline call in order."""
    for name in ("a.pdf", "b.pdf"):
        (tmp_path / name).write_bytes(f"pdf bytes of {name}".encode())

    events, already = [], set()

    def parse_pdf(path):
        stem = path.replace("\\", "/").rsplit("/", 1)[-1][:-4]
        return [{"page": 1, "text": f"text of {stem}", "source": stem}]

    def chunk_pages(pages, *_):
        return [{"chunk_id": pages[0]["source"] + "-c0", "source": pages[0]["source"],
                 "page": 1, "text": pages[0]["text"]}]

    def extract(chunks, **_):
        events.append(("extract", chunks[0]["source"]))
        return [f"result for {chunks[0]['source']}"]

    def load_to_graph(**kw):
        events.append(("graph", kw["filename"], kw["sha256"], kw["chunks"][0]["source"]))
        return {"entities": 1, "relations": 0, "chunks": 1, "mentions": 0}

    def register_document(**kw):
        events.append(("register", kw["filename"], kw["sha256"]))

    monkeypatch.setattr(run_stage1, "parse_pdf", parse_pdf)
    monkeypatch.setattr(run_stage1, "chunk_pages", chunk_pages)
    monkeypatch.setattr(run_stage1, "extract_from_chunks", extract)
    monkeypatch.setattr(run_stage1, "deduplicate", lambda results: (["e"], []))
    monkeypatch.setattr(run_stage1, "Neo4jClient", FakeNeo4j)
    monkeypatch.setattr(run_stage1, "load_to_graph", load_to_graph)
    monkeypatch.setattr(run_stage1, "register_document", register_document)
    monkeypatch.setattr(run_stage1, "is_already_ingested", lambda sha, client: sha in already)
    monkeypatch.setattr(run_stage1, "resolve_provider", lambda p: "fake")
    monkeypatch.setattr(run_stage1, "model_for", lambda p: "fake-model")

    def run(*argv):
        monkeypatch.setattr(sys, "argv", ["run_stage1.py", "--pdf-dir", str(tmp_path),
                                          "--no-qdrant", *argv])
        run_stage1.main()
        return events

    run.sha = lambda name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
    run.already = already
    return run


def test_each_pdf_gets_its_own_document(cli):
    events = cli()
    graph = [e for e in events if e[0] == "graph"]
    assert graph == [("graph", "a.pdf", cli.sha("a.pdf"), "a"),
                     ("graph", "b.pdf", cli.sha("b.pdf"), "b")]
    assert [e for e in events if e[0] == "register"] == [
        ("register", "a.pdf", cli.sha("a.pdf")), ("register", "b.pdf", cli.sha("b.pdf"))]


def test_all_extraction_happens_before_any_write(cli):
    kinds = [e[0] for e in cli()]
    assert kinds[:2] == ["extract", "extract"]          # all-or-nothing is kept
    assert "extract" not in kinds[2:]


def test_already_ingested_pdf_is_skipped(cli):
    cli.already.add(cli.sha("a.pdf"))
    events = cli()
    assert ("extract", "a") not in events                # no LLM calls for it
    assert [e[1] for e in events if e[0] == "graph"] == ["b.pdf"]


def test_dry_run_writes_nothing(cli):
    events = cli("--dry-run")
    assert {e[0] for e in events} == {"extract"}
