"""
启动多个 OmniParser Server 实例，每个节点对应一个独立服务。

用法:
    python start_multi.py
    python start_multi.py --device cuda --BOX_TRESHOLD 0.05

节点与端口映射:
    192.168.31.134 -> port 9001
    192.168.31.135 -> port 9002
"""

import subprocess
import sys
import os
import signal
import argparse

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SERVER_SCRIPT = os.path.join(ROOT_DIR, "omnitool", "omniparserserver", "omniparserserver.py")

sys.path.insert(0, os.path.join(ROOT_DIR, "omnitool", "gradio"))
from node_config import NODE_OMNIPARSER_PORT_MAP as NODE_PORT_MAP


def parse_arguments():
    parser = argparse.ArgumentParser(description="启动多个 OmniParser Server 实例")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--BOX_TRESHOLD", type=float, default=0.05)
    parser.add_argument(
        "--som_model_path",
        type=str,
        default=os.path.join(ROOT_DIR, "weights", "icon_detect", "model.pt"),
    )
    parser.add_argument(
        "--caption_model_name", type=str, default="florence2"
    )
    parser.add_argument(
        "--caption_model_path",
        type=str,
        default=os.path.join(ROOT_DIR, "weights", "icon_caption_florence"),
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    python_exe = sys.executable
    processes = []

    for node_host, port in NODE_PORT_MAP.items():
        cmd = [
            python_exe,
            SERVER_SCRIPT,
            "--host", "0.0.0.0",
            "--port", str(port),
            "--device", args.device,
            "--BOX_TRESHOLD", str(args.BOX_TRESHOLD),
            "--som_model_path", args.som_model_path,
            "--caption_model_name", args.caption_model_name,
            "--caption_model_path", args.caption_model_path,
        ]
        print(f"[启动] 节点 {node_host} -> port {port}")
        proc = subprocess.Popen(cmd)
        processes.append((node_host, port, proc))

    print(f"\n共启动 {len(processes)} 个 OmniParser Server 实例")
    for node_host, port, _ in processes:
        print(f"  {node_host} -> http://0.0.0.0:{port}")
    print("\n按 Ctrl+C 停止所有服务\n")

    try:
        for _, _, proc in processes:
            proc.wait()
    except KeyboardInterrupt:
        print("\n[停止] 正在关闭所有服务...")
        for node_host, port, proc in processes:
            proc.terminate()
            print(f"  已停止 {node_host}:{port}")
        for _, _, proc in processes:
            proc.wait()
        print("[完成] 所有服务已关闭")


if __name__ == "__main__":
    main()
