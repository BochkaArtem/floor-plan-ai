"""Smoke tests for the Flask HTTP API."""

import io
import json
from pathlib import Path

import pytest
from PIL import Image

from backend.app import create_app


@pytest.fixture
def client(tmp_path: Path):
    app = create_app()
    app.config["UPLOAD_DIR"] = tmp_path / "uploads"
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def _png_bytes(width: int = 64, height: int = 64) -> bytes:
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_health(client) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["status"] == "ok"
    assert "segmenter" in payload and "generator" in payload


def test_upload_and_detect_round_trip(client) -> None:
    data = {"image": (io.BytesIO(_png_bytes()), "test.png")}
    resp = client.post("/api/upload", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    payload = resp.get_json()
    image_id = payload["image_id"]
    assert payload["width"] == 64 and payload["height"] == 64

    resp = client.post("/api/detect", json={"image_id": image_id})
    assert resp.status_code == 200
    payload = resp.get_json()
    assert "polygons" in payload
    assert "model" in payload


def test_export_coco(client) -> None:
    body = {
        "images": [
            {
                "file_name": "x.png",
                "width": 100,
                "height": 100,
                "polygons": [
                    {
                        "category": "room",
                        "points": [[0, 0], [50, 0], [50, 50], [0, 50]],
                    }
                ],
            }
        ]
    }
    resp = client.post("/api/export/coco", json=body)
    assert resp.status_code == 200
    coco = json.loads(resp.data.decode("utf-8"))
    assert len(coco["images"]) == 1
    assert len(coco["annotations"]) == 1
    assert coco["annotations"][0]["category_id"]


def test_generate_endpoint(client) -> None:
    resp = client.post(
        "/api/generate",
        json={"width": 320, "height": 240, "num_rooms": 3, "seed": 42},
    )
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["width"] == 320 and payload["height"] == 240
    assert payload["data_url"].startswith("data:image/png;base64,")
    assert isinstance(payload["polygons"], list)


def test_generate_with_room_types_and_shape(client) -> None:
    resp = client.post(
        "/api/generate",
        json={
            "width": 320,
            "height": 240,
            "num_rooms": 5,
            "boundary_shape": "L",
            "room_types": ["hall", "living", "kitchen", "bedroom", "bathroom"],
            "backend": "procedural",
            "seed": 13,
        },
    )
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["boundary_shape"] == "L"
    # Each room polygon carries a subcategory.
    room_polys = [p for p in payload["polygons"] if p["category"] == "room"]
    assert room_polys
    assert any(p.get("subcategory") in {"hall", "living", "kitchen", "bedroom", "bathroom"} for p in room_polys)


def test_health_reports_backends(client) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    payload = resp.get_json()
    assert "boundary_shapes" in payload and "rect" in payload["boundary_shapes"]
    assert "room_types" in payload and "kitchen" in payload["room_types"]
    assert "generator_backends_available" in payload
