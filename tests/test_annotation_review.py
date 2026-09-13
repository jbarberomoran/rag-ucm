import json

import pytest

from src.annotation_review import export_review, prepare_review


def test_unreviewed_or_stale_packets_cannot_be_exported(tmp_path):
    questions, catalog = tmp_path / "q.json", tmp_path / "c.json"
    questions.write_text('[{"question": "Q"}]')
    catalog.write_text('[{"chunk_id": "a", "text": "evidence"}]')
    packet = prepare_review(questions, catalog)
    with pytest.raises(ValueError, match="human review"):
        export_review(packet, questions, catalog)
    packet["questions"][0].update(reviewed=True, reviewer="Reviewer", relevant_chunk_ids=["a"])
    assert export_review(packet, questions, catalog) == {"1": ["a"]}
    packet["questions"][0]["relevant_chunk_ids"] = ["unknown"]
    with pytest.raises(ValueError, match="Unknown"):
        export_review(packet, questions, catalog)
    packet["questions"] = []
    with pytest.raises(ValueError, match="exactly once"):
        export_review(packet, questions, catalog)
    questions.write_text(json.dumps([{"question": "Changed"}]))
    with pytest.raises(ValueError, match="does not match"):
        export_review(packet, questions, catalog)
