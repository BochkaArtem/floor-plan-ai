"""COCO export / import endpoints."""

from __future__ import annotations

import io
import json

from flask import Blueprint, jsonify, request, send_file

from backend.utils.coco import CocoImage, Polygon, build_coco_dict, load_coco

bp = Blueprint("export", __name__)


@bp.route("/api/export/coco", methods=["POST"])
def export_coco():
    """Convert an array of editor annotations into a downloadable COCO JSON."""
    payload = request.get_json(force=True)
    images_payload = payload.get("images", [])
    if not images_payload:
        return jsonify({"error": "No images provided"}), 400

    images: list[CocoImage] = []
    for item in images_payload:
        polygons = [
            Polygon(
                points=[(float(p[0]), float(p[1])) for p in poly["points"]],
                category=poly["category"],
            )
            for poly in item.get("polygons", [])
        ]
        images.append(
            CocoImage(
                file_name=item.get("file_name", "image.png"),
                width=int(item["width"]),
                height=int(item["height"]),
                polygons=polygons,
            )
        )

    data = build_coco_dict(images)
    raw = json.dumps(data, indent=2).encode("utf-8")
    return send_file(
        io.BytesIO(raw),
        mimetype="application/json",
        as_attachment=True,
        download_name="annotations.coco.json",
    )


@bp.route("/api/import/coco", methods=["POST"])
def import_coco():
    """Parse an uploaded COCO JSON and return its annotations as a list."""
    file_storage = request.files.get("file")
    if file_storage is None:
        return jsonify({"error": "Missing file"}), 400
    try:
        # Save to a temp buffer and parse.
        raw = file_storage.read()
        path = io.BytesIO(raw)

        # ``load_coco`` expects a filesystem path, so we round-trip via tmp.
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp.write(path.getvalue())
            tmp_path = tmp.name
        coco_images = load_coco(tmp_path)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Failed to parse COCO: {exc}"}), 400

    return jsonify(
        {
            "images": [
                {
                    "file_name": img.file_name,
                    "width": img.width,
                    "height": img.height,
                    "polygons": [
                        {"category": p.category, "points": [[x, y] for x, y in p.points]}
                        for p in img.polygons
                    ],
                }
                for img in coco_images
            ]
        }
    )
