from pathlib import Path

from backend.utils.coco import CocoImage, Polygon, build_coco_dict, load_coco, save_coco


def test_build_and_load_coco_round_trip(tmp_path: Path) -> None:
    images = [
        CocoImage(
            file_name="plan_001.png",
            width=640,
            height=480,
            polygons=[
                Polygon(points=[(10.0, 10.0), (110.0, 10.0), (110.0, 110.0), (10.0, 110.0)], category="room"),
                Polygon(points=[(120.0, 50.0), (140.0, 50.0), (140.0, 90.0), (120.0, 90.0)], category="door"),
            ],
        ),
    ]
    out = tmp_path / "out.json"
    save_coco(images, out)
    assert out.exists()

    loaded = load_coco(out)
    assert len(loaded) == 1
    img = loaded[0]
    assert img.file_name == "plan_001.png"
    assert img.width == 640 and img.height == 480
    assert len(img.polygons) == 2
    cats = sorted(p.category for p in img.polygons)
    assert cats == ["door", "room"]


def test_polygon_bbox_and_area() -> None:
    poly = Polygon(points=[(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)], category="room")
    assert poly.bbox() == (0.0, 0.0, 4.0, 3.0)
    assert poly.area() == 12.0


def test_build_coco_dict_categories() -> None:
    data = build_coco_dict([])
    cat_names = sorted(c["name"] for c in data["categories"])
    assert cat_names == ["door", "room", "wall", "window"]
