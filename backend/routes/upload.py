"""Endpoints for uploading floor plan images."""

from __future__ import annotations

import base64
import io
import uuid
from pathlib import Path

from flask import Blueprint, current_app, jsonify, request
from PIL import Image

bp = Blueprint("upload", __name__)


@bp.route("/api/upload", methods=["POST"])
def upload():
    """Accept a multipart upload OR a JSON ``{ "data_url": "data:image/..." }``.

    Returns ``{ image_id, width, height, data_url }`` so the frontend can
    render the image immediately and reference it in subsequent calls.
    """
    payload = request.get_json(silent=True) or {}
    data_url: str | None = payload.get("data_url")
    file_storage = request.files.get("image")

    raw: bytes
    if data_url:
        _, _, b64 = data_url.partition(",")
        raw = base64.b64decode(b64)
    elif file_storage is not None:
        raw = file_storage.read()
    else:
        return jsonify({"error": "No image provided. Send multipart `image` or JSON `data_url`."}), 400

    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:  # noqa: BLE001 — show parse errors to the client
        return jsonify({"error": f"Could not parse image: {exc}"}), 400

    image_id = uuid.uuid4().hex
    upload_dir: Path = current_app.config["UPLOAD_DIR"]
    upload_dir.mkdir(parents=True, exist_ok=True)
    path = upload_dir / f"{image_id}.png"
    image.save(path, format="PNG")

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return jsonify(
        {
            "image_id": image_id,
            "width": image.width,
            "height": image.height,
            "data_url": f"data:image/png;base64,{encoded}",
        }
    )
