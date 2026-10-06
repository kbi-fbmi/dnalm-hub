"""No-network tests for GenericMcpClient's response parsing (transport is stubbed)."""

import json

import pytest
from dnalm_client import GenericMcpClient


class _StubClient(GenericMcpClient):
    """Replays canned JSON-RPC messages instead of talking HTTP."""

    def __init__(self, messages):
        super().__init__("http://stub")
        self._messages = list(messages)
        self.sent = []

    def _ensure_session(self):
        return "session-1"

    def _post_mcp_sse_json(self, path, payload, *, session_id=None):
        self.sent.append(payload)
        return self._messages.pop(0), {}


def test_structured_content_is_preferred():
    c = _StubClient(
        [{"result": {"structuredContent": {"a": 1}, "content": [{"text": '{"a": 2}'}]}}]
    )
    assert c.get_model_info("x") == {"a": 1}


def test_falls_back_to_json_text_content():
    c = _StubClient(
        [{"result": {"content": [{"type": "text", "text": json.dumps({"embedding": [1.0]})}]}}]
    )
    assert c.embed_sequence("ACGT", checkpoint="x") == {"embedding": [1.0]}


def test_tool_error_raises_with_server_message():
    c = _StubClient([{"result": {"isError": True, "content": [{"text": "Sequence too long"}]}}])
    with pytest.raises(RuntimeError, match="Sequence too long"):
        c.embed_sequence("ACGT", checkpoint="x")


def test_batch_embedding_sends_list():
    c = _StubClient([{"result": {"structuredContent": {"embeddings": [[1.0], [2.0]]}}}])
    c.embed_sequence(["ACGT", "GGCC"], checkpoint="x")
    assert c.sent[0]["params"]["arguments"]["sequence"] == ["ACGT", "GGCC"]


def test_list_tools_returns_tool_definitions():
    c = _StubClient([{"result": {"tools": [{"name": "embed_sequence"}, {"name": "score_snp"}]}}])
    assert [t["name"] for t in c.list_tools()] == ["embed_sequence", "score_snp"]
    assert c.sent[0]["method"] == "tools/list"


def test_species_sent_only_when_given():
    c = _StubClient([{"result": {"structuredContent": {}}}] * 2)
    c.embed_sequence("ACGT", checkpoint="x")
    c.embed_sequence("ACGT", checkpoint="x", species="mouse")
    assert "species" not in c.sent[0]["params"]["arguments"]
    assert c.sent[1]["params"]["arguments"]["species"] == "mouse"


def test_call_passes_kwargs_and_parses_result():
    c = _StubClient([{"result": {"content": [{"type": "text", "text": '{"region_start": 5}'}]}}])
    assert c.call("annotate_sequence", sequence="ACGT", species="human") == {"region_start": 5}
    assert c.sent[0]["params"] == {
        "name": "annotate_sequence",
        "arguments": {"sequence": "ACGT", "species": "human"},
    }


def test_optional_arguments_sent_only_when_given():
    c = _StubClient([{"result": {"structuredContent": {}}}] * 4)
    c.embed_sequence("ACGT")
    c.compare_sequences("ACGT", "ACGA", checkpoint="x", layer_name="6")
    c.score_snp("ACGT", "T")
    c.generate_sequence("ACGT")
    args = [m["params"]["arguments"] for m in c.sent]
    assert "checkpoint" not in args[0] and "layer_name" not in args[0]  # server defaults
    assert args[1]["checkpoint"] == "x" and args[1]["layer_name"] == "6"
    assert "checkpoint" not in args[2]
    assert "checkpoint" not in args[3] and "top_k" not in args[3]


def test_request_ids_are_unique_across_threads():
    from concurrent.futures import ThreadPoolExecutor

    c = _StubClient([{"result": {"structuredContent": {}}}] * 64)
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda _: c.embed_sequence("ACGT"), range(64)))
    ids = [m["id"] for m in c.sent]
    assert len(set(ids)) == 64  # a shared session must not see two requests with one id
