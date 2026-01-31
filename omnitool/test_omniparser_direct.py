"""
直接测试 OmniParser 而不通过 HTTP 服务器
"""
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from util.omniparser import Omniparser
import traceback

# 添加 gradio 目录到路径以导入工具
import sys
gradio_dir = project_root / 'omnitool' / 'gradio'
sys.path.insert(0, str(gradio_dir))

from tools.screen_capture import get_screenshot
from agent.llm_utils.utils import encode_image

def test_omniparser_direct():
    """直接测试 OmniParser,绕过 HTTP 服务器"""
    print("=" * 80)
    print("开始直接测试 OmniParser")
    print("=" * 80)

    # 配置
    config = {
        'som_model_path': str(project_root / 'weights' / 'icon_detect' / 'model.pt'),
        'caption_model_name': 'florence2',
        'caption_model_path': str(project_root / 'weights' / 'icon_caption_florence'),
        'device': 'cpu',
        'BOX_TRESHOLD': 0.05
    }

    try:
        # 初始化 OmniParser
        print("\n[INFO] 正在初始化 OmniParser...")
        omniparser = Omniparser(config)
        print("[SUCCESS] OmniParser 初始化成功!\n")

        # 获取屏幕截图
        print("[INFO] 正在获取屏幕截图...")
        screenshot, screenshot_path = get_screenshot()
        screenshot_path = str(screenshot_path)
        print(f"[SUCCESS] 截图保存在: {screenshot_path}\n")

        # 编码图像为 base64
        print("[INFO] 正在将截图编码为 base64...")
        image_base64 = encode_image(screenshot_path)
        print(f"[SUCCESS] 图像编码成功,长度: {len(image_base64)}\n")

        # 调用 omniparser.parse()
        print("[INFO] 正在调用 omniparser.parse()...")
        print("[INFO] 这可能需要一些时间,请耐心等待...")

        dino_labled_img, parsed_content_list = omniparser.parse(image_base64)

        print("\n[SUCCESS] OmniParser 解析成功!")
        print(f"[INFO] 解析到 {len(parsed_content_list)} 个元素")
        print(f"[INFO] 前 5 个元素:")
        for i, item in enumerate(parsed_content_list[:5]):
            print(f"  {i}: {item}")

        print("\n" + "=" * 80)
        print("测试成功完成!")
        print("=" * 80)

    except Exception as e:
        print(f"\n[ERROR] 发生异常: {type(e).__name__}: {str(e)}")
        print("\n完整堆栈追踪:")
        traceback.print_exc()
        print("\n" + "=" * 80)
        print("测试失败")
        print("=" * 80)
        return False

    return True

if __name__ == "__main__":
    test_omniparser_direct()
