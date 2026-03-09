"""
GLM 模型淘宝测试脚本
使用 OmniParser + GLM-4.5V 模型测试打开淘宝网站
"""
import sys
import os
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from anthropic.types import TextBlock
from loop import sampling_loop_sync


def _output_callback(message, sender="bot"):
    """输出回调,打印消息"""
    print(f"[{sender.upper()}] {message}")


def _tool_output_callback(tool_result, tool_id):
    """工具输出回调"""
    print(f"[TOOL {tool_id}] {tool_result}")


def _api_response_callback(response):
    """API 响应回调"""
    print("[API] Response received")


def run_test(task="打开淘宝网站", omniparser_server_url="localhost:8000", model_name="glm-4.6"):
    """
    运行测试

    Args:
        task: 要执行的任务描述
        omniparser_server_url: OmniParser Server 地址
        model_name: GLM 模型名称,默认 glm-4.6
    """
    model = f"omniparser + {model_name}"
    provider = "zhipu"
    api_key = os.getenv("ZHIPU_API_KEY", "")
    only_n_most_recent_images = 2
    max_tokens = 16384

    print("=" * 80)
    print("GLM 淘宝测试开始")
    print(f"模型: {model}")
    print(f"任务: {task}")
    print("=" * 80)

    messages = [
        {
            "role": "user",
            "content": [TextBlock(type="text", text=task)],
        }
    ]

    try:
        # 运行采样循环
        for loop_msg in sampling_loop_sync(
            model=model,
            provider=provider,
            messages=messages,
            output_callback=_output_callback,
            tool_output_callback=_tool_output_callback,
            api_response_callback=_api_response_callback,
            api_key=api_key,
            only_n_most_recent_images=only_n_most_recent_images,
            max_tokens=max_tokens,
            omniparser_url=omniparser_server_url,
        ):
            if loop_msg is None:
                print("\n[INFO] 任务完成或循环结束")
                break

    except KeyboardInterrupt:
        print("\n[INFO] 用户中断测试")
    except Exception as e:
        print(f"\n[ERROR] 测试过程中出现异常: {type(e).__name__}: {str(e)}")
        import traceback

        traceback.print_exc()
    finally:
        print("=" * 80)
        print("GLM 淘宝测试结束")
        print("=" * 80)


if __name__ == "__main__":
    run_test("打开淘宝网站")
