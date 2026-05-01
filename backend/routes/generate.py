"""Floor plan generation endpoint."""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from backend.ml.generation import GenerationConditions, select_generator
from backend.utils.image import image_data_url

bp = Blueprint("generate", __name__)


@bp.route("/api/generate", methods=["POST"])
def generate():
    payload = request.get_json(force=True)

    raw_types = payload.get("room_types") or []
    if isinstance(raw_types, str):
        raw_types = [t.strip() for t in raw_types.split(",") if t.strip()]

    cond = GenerationConditions(
        width=int(payload.get("width", 640)),
        height=int(payload.get("height", 480)),
        num_rooms=int(payload.get("num_rooms", 5)),
        seed=payload.get("seed"),
        area_m2=payload.get("area_m2"),
        boundary_shape=payload.get("boundary_shape", "auto"),
        room_types=list(raw_types),
        backend=payload.get("backend", "auto"),
    )

    generator = select_generator(cond.backend)
    image, polygons = generator.generate(cond)
    return jsonify(
        {
            "model": generator.name,
            "width": image.shape[1],
            "height": image.shape[0],
            "boundary_shape": cond.boundary_shape,
            "room_types": cond.room_types,
            "data_url": image_data_url(image),
            "polygons": [
                {
                    "category": p.category,
                    "subcategory": p.subcategory,
                    "points": [[float(x), float(y)] for x, y in p.points],
                }
                for p in polygons
            ],
        }
    )
