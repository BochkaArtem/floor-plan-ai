"""Flask application entrypoint for Floor Plan AI."""

from __future__ import annotations

from pathlib import Path

from flask import Flask, send_from_directory

from backend.ml.generation import MaskUNetGenerator, get_default_generator
from backend.ml.segmentation import UNetSegmenter
from backend.routes import detect, export, generate, upload

ROOT_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = ROOT_DIR / "data" / "uploads"
STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app() -> Flask:
    app = Flask(
        __name__,
        static_folder=str(STATIC_DIR),
        static_url_path="/static",
    )
    app.config["UPLOAD_DIR"] = UPLOAD_DIR
    app.config["SEGMENTER"] = UNetSegmenter()
    app.config["GENERATOR"] = get_default_generator()

    app.register_blueprint(upload.bp)
    app.register_blueprint(detect.bp)
    app.register_blueprint(export.bp)
    app.register_blueprint(generate.bp)

    @app.route("/")
    def index() -> object:
        return send_from_directory(STATIC_DIR, "index.html")

    @app.route("/health")
    def health() -> dict[str, object]:
        nn_gen = MaskUNetGenerator()
        return {
            "status": "ok",
            "segmenter": "unet" if app.config["SEGMENTER"].available() else "classical-cv",
            "generator": "mask-unet" if nn_gen.available() else "procedural",
            "generator_backends_available": {
                "procedural": True,
                "nn": nn_gen.available(),
            },
            "boundary_shapes": ["rect", "L", "T", "U", "plus", "random", "auto"],
            "room_types": [
                "hall", "living", "kitchen", "bedroom", "bathroom", "balcony",
            ],
        }

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
