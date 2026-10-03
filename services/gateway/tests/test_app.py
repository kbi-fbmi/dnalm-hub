"""The gateway against fake model services: no network, no model weights."""

import base64
import struct
import urllib.error
from typing import ClassVar

import openai
import pytest
from dnalm_gateway.app import create_app
from dnalm_gateway.backends import Backends, parse_backends
from starlette.testclient import TestClient

CHECKPOINTS = {
    "http://ntv3": [
        {"name": "100m-pre", "description": "NTv3 100M", "repo_id": "InstaDeepAI/NTv3_100M_pre"},
        {"name": "shared", "description": "in both", "repo_id": "x/shared-3"},
    ],
    "http://ntv2": [
        {"name": "500m", "description": "NTv2 500M", "repo_id": "InstaDeepAI/nt-v2-500m"},
        {"name": "shared", "description": "in both", "repo_id": "x/shared-2"},
    ],
}


class FakeClient:
    """Stands in for GenericMcpClient; records embed_sequence calls."""

    calls: ClassVar[list[dict]] = []
    down: ClassVar[set[str]] = set()

    def __init__(self, base_url, auth_token=None, timeout=30.0):
        self.base_url = base_url
        self.auth_token = auth_token

    def _check_up(self):
        if self.base_url in self.down:
            raise urllib.error.URLError("connection refused")

    def list_available_checkpoints(self):
        self._check_up()
        return CHECKPOINTS[self.base_url]

    def embed_sequence(self, sequence, checkpoint, layer_name, pooling, species=None):
        self._check_up()
        self.calls.append(
            {
                "url": self.base_url,
                "sequence": sequence,
                "checkpoint": checkpoint,
                "layer_name": layer_name,
                "pooling": pooling,
                "species": species,
                "token": self.auth_token,
            }
        )
        if any("X" in s for s in sequence):
            raise RuntimeError("Invalid DNA characters: X")
        return {"embeddings": [[float(len(s)), 0.5, -1.0] for s in sequence]}


@pytest.fixture(autouse=True)
def reset_fake():
    FakeClient.calls = []
    FakeClient.down = set()


def make_client(auth_token=None, max_batch=32):
    backends = Backends(
        urls={"ntv3": "http://ntv3", "ntv2": "http://ntv2"},
        auth_token="backend-token",
        max_batch=max_batch,
        client_factory=FakeClient,
    )
    return TestClient(create_app(backends, auth_token=auth_token))


def test_parse_backends():
    assert parse_backends(" ntv3=http://a:8000/ , ntv2=http://b:8000") == {
        "ntv3": "http://a:8000",
        "ntv2": "http://b:8000",
    }
    with pytest.raises(ValueError):
        parse_backends("")
    with pytest.raises(ValueError):
        parse_backends("http://a:8000")


def test_list_models():
    resp = make_client().get("/v1/models")
    assert resp.status_code == 200
    ids = [m["id"] for m in resp.json()["data"]]
    assert ids == ["ntv3/100m-pre", "ntv3/shared", "ntv2/500m", "ntv2/shared"]


def test_list_models_skips_unreachable_service():
    FakeClient.down = {"http://ntv2"}
    ids = [m["id"] for m in make_client().get("/v1/models").json()["data"]]
    assert ids == ["ntv3/100m-pre", "ntv3/shared"]


def test_retrieve_model():
    client = make_client()
    assert client.get("/v1/models/ntv2/500m").json()["owned_by"] == "ntv2"
    assert client.get("/v1/models/ntv2/nope").status_code == 404


def test_embeddings_single_string():
    resp = make_client().post("/v1/embeddings", json={"model": "ntv3/100m-pre", "input": "ACGT"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    assert body["model"] == "ntv3/100m-pre"
    assert body["data"] == [{"object": "embedding", "index": 0, "embedding": [4.0, 0.5, -1.0]}]
    assert body["usage"] == {"prompt_tokens": 4, "total_tokens": 4}
    call = FakeClient.calls[0]
    assert call["url"] == "http://ntv3"
    assert call["sequence"] == ["ACGT"]
    assert call["pooling"] == "mean"
    assert call["layer_name"] == "last"
    assert call["species"] is None
    assert call["token"] == "backend-token"


def test_embeddings_batch_is_chunked_and_ordered():
    seqs = ["A" * n for n in range(1, 6)]
    resp = make_client(max_batch=2).post(
        "/v1/embeddings", json={"model": "ntv2/500m", "input": seqs}
    )
    data = resp.json()["data"]
    assert [d["index"] for d in data] == [0, 1, 2, 3, 4]
    assert [d["embedding"][0] for d in data] == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert [len(c["sequence"]) for c in FakeClient.calls] == [2, 2, 1]


def test_embeddings_extra_fields_are_forwarded():
    make_client().post(
        "/v1/embeddings",
        json={"model": "ntv3/100m-post", "input": "ACGT", "layer": 6, "species": "mouse"},
    )
    call = FakeClient.calls[0]
    assert call["checkpoint"] == "100m-post"  # not listed, still passed through
    assert call["layer_name"] == "6"
    assert call["species"] == "mouse"


def test_embeddings_hf_repo_id_passthrough():
    model = "ntv2/InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
    make_client().post("/v1/embeddings", json={"model": model, "input": "ACGT"})
    assert FakeClient.calls[0]["url"] == "http://ntv2"
    assert (
        FakeClient.calls[0]["checkpoint"]
        == "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
    )


@pytest.mark.parametrize(
    ("model", "url", "checkpoint"),
    [("500m", "http://ntv2", "500m"), ("InstaDeepAI/NTv3_100M_pre", "http://ntv3", "100m-pre")],
)
def test_embeddings_bare_alias_or_repo_id(model, url, checkpoint):
    resp = make_client().post("/v1/embeddings", json={"model": model, "input": "ACGT"})
    assert resp.json()["model"] == f"{url.removeprefix('http://')}/{checkpoint}"
    assert (FakeClient.calls[0]["url"], FakeClient.calls[0]["checkpoint"]) == (url, checkpoint)


def test_embeddings_ambiguous_or_unknown_model():
    client = make_client()
    resp = client.post("/v1/embeddings", json={"model": "shared", "input": "ACGT"})
    assert resp.status_code == 404
    assert "ntv3/shared" in resp.json()["error"]["message"]
    resp = client.post("/v1/embeddings", json={"model": "nope", "input": "ACGT"})
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "model_not_found"


def test_embeddings_base64():
    resp = make_client().post(
        "/v1/embeddings",
        json={"model": "ntv3/100m-pre", "input": "ACG", "encoding_format": "base64"},
    )
    raw = base64.b64decode(resp.json()["data"][0]["embedding"])
    assert struct.unpack("<3f", raw) == (3.0, 0.5, -1.0)


@pytest.mark.parametrize(
    ("body", "param"),
    [
        ({"input": "ACGT"}, "model"),
        ({"model": "ntv3/100m-pre"}, "input"),
        ({"model": "ntv3/100m-pre", "input": []}, "input"),
        ({"model": "ntv3/100m-pre", "input": [[1, 2, 3]]}, "input"),
        ({"model": "ntv3/100m-pre", "input": "ACGT", "encoding_format": "int8"}, "encoding_format"),
        ({"model": "ntv3/100m-pre", "input": "ACGT", "dimensions": 64}, "dimensions"),
        ({"model": "ntv3/100m-pre", "input": "ACGT", "layer": [1]}, "layer"),
    ],
)
def test_embeddings_invalid_request(body, param):
    resp = make_client().post("/v1/embeddings", json=body)
    assert resp.status_code == 400
    assert resp.json()["error"]["param"] == param
    assert FakeClient.calls == []


def test_embeddings_service_error_is_400_with_its_message():
    resp = make_client().post("/v1/embeddings", json={"model": "ntv3/100m-pre", "input": "ACXT"})
    assert resp.status_code == 400
    assert resp.json()["error"]["message"] == "Invalid DNA characters: X"


def test_embeddings_unreachable_service_is_502():
    FakeClient.down = {"http://ntv3"}
    resp = make_client().post("/v1/embeddings", json={"model": "ntv3/100m-pre", "input": "ACGT"})
    assert resp.status_code == 502
    assert resp.json()["error"]["type"] == "server_error"


def test_auth():
    client = make_client(auth_token="s3cret")
    assert client.get("/health").status_code == 200
    assert client.get("/v1/models").status_code == 401
    assert client.get("/v1/models", headers={"Authorization": "Bearer wrong"}).status_code == 401
    ok = client.get("/v1/models", headers={"Authorization": "Bearer s3cret"})
    assert ok.status_code == 200


def test_official_openai_sdk():
    """The OpenAI SDK requests base64 by default and decodes it; check it round-trips."""
    sdk = openai.OpenAI(
        api_key="s3cret",
        base_url="http://testserver/v1",
        http_client=make_client(auth_token="s3cret"),
    )
    result = sdk.embeddings.create(model="ntv3/100m-pre", input=["ACGT", "AC"])
    assert [d.embedding for d in result.data] == [[4.0, 0.5, -1.0], [2.0, 0.5, -1.0]]
    result = sdk.embeddings.create(
        model="ntv3/100m-post", input="ACGT", extra_body={"species": "human"}
    )
    assert FakeClient.calls[-1]["species"] == "human"
    assert [m.id for m in sdk.models.list()][:1] == ["ntv3/100m-pre"]
    with pytest.raises(openai.NotFoundError):
        sdk.embeddings.create(model="nope", input="ACGT")
    with pytest.raises(openai.BadRequestError):
        sdk.embeddings.create(model="ntv3/100m-pre", input="ACXT")
