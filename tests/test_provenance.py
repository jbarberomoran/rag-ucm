import json
from types import SimpleNamespace

import pytest

from src.provenance import experiment_config, index_config, load_annotations, prepare_manifest


def test_manifest_roundtrip_rejects_different_configuration(tmp_path):
    path = tmp_path / "manifest.json"
    original = prepare_manifest(path, {"model": "example"})
    assert prepare_manifest(path, {"model": "example"}, resume=True) == original
    assert "packages" in original
    assert "GOOGLE_API_KEY" not in path.read_text()
    with pytest.raises(ValueError, match="differs"):
        prepare_manifest(path, {"model": "other"}, resume=True)
    with pytest.raises(ValueError, match="without"):
        prepare_manifest(tmp_path / "missing", {}, resume=True)


def test_input_hashes_and_configuration_are_recorded(tmp_path):
    paper = tmp_path / "paper"
    paper.write_text("first")
    before = index_config(paper, "recursive", 1200, 350)
    paper.write_text("changed")
    assert index_config(paper, "recursive", 1200, 350) != before
    args = SimpleNamespace(
        methods=["baseline"], questions=[0], runs=2, chunking="semantic", annotations=None
    )
    config = experiment_config(args, paper, paper)
    assert config["runs"] == 2
    assert config["annotation_sha256"] is None
    annotation = tmp_path / "annotations.json"
    annotation.write_text(json.dumps({"1": ["chunk"]}))
    args.annotations = annotation
    assert experiment_config(args, paper, paper)["annotation_sha256"]


def test_annotations_reject_stale_chunk_ids(tmp_path):
    annotation = tmp_path / "labels.json"
    catalog = tmp_path / "chunks.json"
    questions = tmp_path / "questions.json"
    catalog.write_text('[{"chunk_id": "current"}]')
    questions.write_text('[{}]')
    annotation.write_text('{"1": ["stale"]}')
    with pytest.raises(ValueError, match="Unknown chunk"):
        load_annotations(annotation, catalog, questions)
    annotation.write_text('{"1": ["current"]}')
    assert load_annotations(annotation, catalog, questions) == {"1": ["current"]}
    assert load_annotations(None, catalog, questions) == {}
