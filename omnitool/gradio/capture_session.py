import argparse
import base64
import json
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path

import pyautogui
import requests
from PIL import Image


def _timestamp_label() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _resolve_output_dir(base_dir: str | Path | None, session_name: str | None) -> Path:
    root = Path(base_dir or "./tmp/capture_sessions")
    name = session_name.strip() if session_name else _timestamp_label()
    output_dir = root / name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _encode_image(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def _get_screenshot(*, windows_host_url: str | None, output_dir: Path) -> tuple[Image.Image, Path]:
    output_path = output_dir / f"screenshot_{datetime.now().strftime('%H%M%S_%f')}.png"
    if windows_host_url:
        host = windows_host_url if windows_host_url.startswith("http") else f"http://{windows_host_url}"
        response = requests.get(
            f"{host}/screenshot",
            timeout=30,
            proxies={"http": "", "https": ""},
        )
        response.raise_for_status()
        screenshot = Image.open(BytesIO(response.content))
    else:
        screenshot = pyautogui.screenshot()
    screenshot.save(output_path)
    return screenshot, output_path


def _parse_frame(omniparser_url: str, screenshot_path: Path) -> dict:
    image_base64 = _encode_image(screenshot_path)
    response = requests.post(
        omniparser_url,
        json={"base64_image": image_base64},
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    payload["screen_info"] = _build_screen_info(payload.get("parsed_content_list") or [])
    return payload


def _build_screen_info(parsed_content_list: list[dict]) -> str:
    lines: list[str] = []
    for idx, element in enumerate(parsed_content_list):
        kind = str(element.get("type") or "").strip() or "unknown"
        content = str(element.get("content") or "").strip()
        label = "Text" if kind == "text" else "Icon" if kind == "icon" else kind
        lines.append(f"ID: {idx}, {label}: {content}")
    return "\n".join(lines)


def capture_session(
    *,
    windows_host_url: str | None,
    omniparser_url: str | None,
    interval_sec: float,
    duration_sec: float,
    output_dir: str | Path | None,
    session_name: str | None,
    max_frames: int | None,
) -> Path:
    if interval_sec <= 0:
        raise ValueError("interval_sec must be > 0")
    if duration_sec <= 0 and (max_frames is None or max_frames <= 0):
        raise ValueError("duration_sec or max_frames must allow at least one frame")

    session_dir = _resolve_output_dir(output_dir, session_name)
    frames_dir = session_dir / "frames"
    parses_dir = session_dir / "parses"
    frames_dir.mkdir(parents=True, exist_ok=True)
    parses_dir.mkdir(parents=True, exist_ok=True)

    started_at = time.time()
    deadline = started_at + duration_sec if duration_sec > 0 else None
    frame_count = 0
    manifest: dict = {
        "session_dir": str(session_dir.resolve()),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "windows_host_url": windows_host_url or "",
        "omniparser_url": omniparser_url or "",
        "interval_sec": interval_sec,
        "duration_sec": duration_sec,
        "max_frames": max_frames,
        "frames": [],
    }

    while True:
        if deadline is not None and time.time() >= deadline:
            break
        if max_frames is not None and frame_count >= max_frames:
            break

        frame_started = time.time()
        screenshot, screenshot_path = _get_screenshot(windows_host_url=windows_host_url, output_dir=frames_dir)
        screenshot_path = Path(screenshot_path).resolve()
        frame_id = screenshot_path.stem.replace("screenshot_", "")
        frame_entry = {
            "frame_id": frame_id,
            "captured_at": datetime.now().isoformat(timespec="seconds"),
            "image_path": str(screenshot_path),
            "width": screenshot.size[0],
            "height": screenshot.size[1],
        }

        if omniparser_url:
            parse_payload = _parse_frame(omniparser_url, screenshot_path)
            parse_path = parses_dir / f"{frame_id}.json"
            parse_path.write_text(
                json.dumps(parse_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            frame_entry["parse_path"] = str(parse_path.resolve())
            frame_entry["parsed_count"] = len(parse_payload.get("parsed_content_list") or [])
            frame_entry["latency"] = parse_payload.get("latency")

        manifest["frames"].append(frame_entry)
        frame_count += 1

        elapsed = time.time() - frame_started
        sleep_for = interval_sec - elapsed
        if sleep_for > 0:
            time.sleep(sleep_for)

    manifest["ended_at"] = datetime.now().isoformat(timespec="seconds")
    manifest["frame_count"] = frame_count
    manifest_path = session_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="后台采集截图素材，可选调用 OmniParser 做离线分析。")
    parser.add_argument("--windows-host-url", default="localhost:5000", help="截图节点地址，如 localhost:5000")
    parser.add_argument("--omniparser-url", default="", help="OmniParser 解析地址，如 http://127.0.0.1:9000/parse/")
    parser.add_argument("--interval-sec", type=float, default=2.0, help="采样间隔秒数")
    parser.add_argument("--duration-sec", type=float, default=1800.0, help="采集总时长秒数，0 表示只看 max_frames")
    parser.add_argument("--max-frames", type=int, default=0, help="最大帧数，0 表示不限制")
    parser.add_argument("--output-dir", default="./tmp/capture_sessions", help="输出根目录")
    parser.add_argument("--session-name", default="", help="会话目录名，默认按时间戳生成")
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()
    manifest_path = capture_session(
        windows_host_url=(args.windows_host_url or "").strip() or None,
        omniparser_url=(args.omniparser_url or "").strip() or None,
        interval_sec=float(args.interval_sec),
        duration_sec=float(args.duration_sec),
        output_dir=args.output_dir,
        session_name=(args.session_name or "").strip() or None,
        max_frames=(args.max_frames if args.max_frames > 0 else None),
    )
    print(f"[capture_session] done -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
