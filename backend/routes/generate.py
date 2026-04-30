"""Floor plan generation endpoint."""

from __future__ import annotations

from flask import Blueprint, current_app, jsonify, request

from backend.ml.generation import GenerationConditions
from backend.utils.image import image_data_url

bp = Blueprint("generate", __name__)


@bp.route("/api/generate", methods=["POST"])
def generate():
    payload = request.get_json(force=True)
    cond = GenerationConditions(
        width=int(payload.get("width", 640)),
        height=int(payload.get("height", 480)),
        num_rooms=int(payload.get("num_rooms", 4)),
        seed=payload.get("seed"),
        area_m2=payload.get("area_m2"),
        boundary_shape=payload.get("boundary_shape", "rect"),
    )

    generator = current_app.config["GENERATOR"]
    image, polygons = generator.generate(cond)
    return jsonify(
        {
            "model": generator.name,
            "width": image.shape[1],
            "height": image.shape[0],
            "data_url": image_data_url(image),
            "polygons": [
                {"category": p.category, "points": [[float(x), float(y)] for x, y in p.points]}
                for p in polygons
            ],
        }
    )
