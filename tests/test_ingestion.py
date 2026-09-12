import src.ingestion as ingestion


class FakeEngine:
    def __init__(self):
        self.unloaded = False

    def unload_db(self):
        self.unloaded = True


def test_recursive_splitter_uses_project_chunk_settings():
    splitter = ingestion.get_text_splitter("recursive")

    assert splitter._chunk_size == ingestion.CHUNK_SIZE
    assert splitter._chunk_overlap == ingestion.CHUNK_OVERLAP


def test_splitter_rejects_unknown_method():
    try:
        ingestion.get_text_splitter("unknown")
    except ValueError as error:
        assert "chunking method" in str(error)
    else:
        raise AssertionError("unknown method should fail")


def test_ingest_data_returns_empty_list_when_paper_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(ingestion, "PAPER_PATH", tmp_path / "missing.pdf")

    assert ingestion.ingest_data() == []


def test_create_vector_db_ignores_empty_chunk_list(monkeypatch):
    monkeypatch.setattr(
        ingestion,
        "HuggingFaceEmbeddings",
        lambda **_: (_ for _ in ()).throw(AssertionError("must not load model")),
    )

    assert ingestion.create_vector_db([]) is None


def test_clear_existing_db_unloads_engine_and_removes_directory(tmp_path, monkeypatch):
    database_path = tmp_path / "chroma"
    database_path.mkdir()
    (database_path / "index").write_text("generated", encoding="utf-8")
    engine = FakeEngine()
    monkeypatch.setattr(ingestion, "CHROMA_PATH", database_path)
    monkeypatch.setattr(ingestion.RetrievalEngine, "get_instance", lambda: engine)

    assert ingestion.clear_existing_db() is True
    assert engine.unloaded is True
    assert not database_path.exists()


def test_db_setup_skips_existing_database(tmp_path, monkeypatch):
    database_path = tmp_path / "chroma"
    database_path.mkdir()
    (database_path / "index").write_text("generated", encoding="utf-8")
    monkeypatch.setattr(ingestion, "CHROMA_PATH", database_path)
    monkeypatch.setattr(
        ingestion,
        "clear_existing_db",
        lambda: (_ for _ in ()).throw(AssertionError("must not rebuild")),
    )

    assert ingestion.db_setup(rebuild_db=False) is None


def test_db_setup_builds_missing_database(tmp_path, monkeypatch):
    database_path = tmp_path / "chroma"
    calls = []
    monkeypatch.setattr(ingestion, "CHROMA_PATH", database_path)
    monkeypatch.setattr(ingestion, "clear_existing_db", lambda: True)
    monkeypatch.setattr(ingestion, "ingest_data", lambda method: ["chunk"])
    monkeypatch.setattr(ingestion, "create_vector_db", lambda chunks: calls.append(chunks))

    ingestion.db_setup(rebuild_db=False, chunking_method="recursive")

    assert calls == [["chunk"]]
