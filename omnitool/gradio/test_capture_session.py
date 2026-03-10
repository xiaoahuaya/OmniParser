import json
from pathlib import Path

from PIL import Image

from omnitool.gradio import capture_session


class _FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "som_image_base64": "",
            "parsed_content_list": [
                {"type": "text", "content": "云顶之弈"},
                {"type": "icon", "content": "开始"},
            ],
            "latency": 0.25,
        }


def _write_dummy_png(path: Path):
    image = Image.new("RGB", (1280, 720), color=(12, 34, 56))
    image.save(path)
    return image


def test_capture_session_saves_manifest_and_parse(monkeypatch, tmp_path):
    def fake_get_screenshot(*, windows_host_url=None, output_dir=None, **_kwargs):
        output_path = Path(output_dir) / "screenshot_testframe.png"
        image = _write_dummy_png(output_path)
        return image, output_path

    monkeypatch.setattr(capture_session, "_get_screenshot", fake_get_screenshot)
    monkeypatch.setattr(capture_session.requests, "post", lambda *args, **kwargs: _FakeResponse())

    manifest_path = capture_session.capture_session(
        windows_host_url="localhost:5000",
        omniparser_url="http://127.0.0.1:9000/parse/",
        interval_sec=0.01,
        duration_sec=0,
        output_dir=tmp_path,
        session_name="tft_materials",
        max_frames=1,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["frame_count"] == 1
    assert len(manifest["frames"]) == 1
    frame = manifest["frames"][0]
    assert Path(frame["image_path"]).exists()
    assert Path(frame["parse_path"]).exists()

    parse_payload = json.loads(Path(frame["parse_path"]).read_text(encoding="utf-8"))
    assert parse_payload["parsed_content_list"][0]["content"] == "云顶之弈"
    assert "ID: 0, Text: 云顶之弈" in parse_payload["screen_info"]
