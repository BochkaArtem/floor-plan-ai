"""Auto-detect endpoints (segmentation)."""

from __future__ import annotations

from pathlib import Path

from flask import Blueprint, current_app, jsonify, request

from backend.ml.segmentation import UNetSegmenter
from backend.utils.image import load_image

bp = Blueprint("detect", __name__)


@bp.route("/api/detect", methods=["POST"])
def detect():
    payload = request.get_json(force=True)
    image_id = payload.get("image_id")
    if not image_id:
        return jsonify({"error": "Missing image_id"}), 400

    upload_dir: Path = current_app.config["UPLOAD_DIR"]
    image_path = upload_dir / f"{image_id}.png"
    if not image_path.exists():
        return jsonify({"error": f"Unknown image_id: {image_id}"}), 404

    image = load_image(str(image_path))
    segmenter: UNetSegmenter = current_app.config["SEGMENTER"]
    polygons = segmenter.segment(image)

    return jsonify(
        {
            "model": "unet" if segmenter.available() else "classical-cv",
            "polygons": [
                {"category": p.category, "points": [[float(x), float(y)] for x, y in p.points]}
                for p in polygons
            ],
        }
    )
