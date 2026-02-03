# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

**OmniParser** 是微软开发的屏幕解析工具，将 UI 截图转换为结构化元素，增强视觉语言模型（如 GPT-4V）生成可定位操作的能力。

三个主要组件：
- **OmniParser Core** (`util/`): 核心解析引擎，包含 YOLO 图标检测 + Florence2/BLIP2 caption + OCR
- **OmniParser Server** (`omnitool/omniparserserver/`): FastAPI REST API 封装
- **OmniTool** (`omnitool/`): 完整的 GUI 代理系统
  - `gradio/`: Gradio UI + 代理循环
  - `omnibox/`: Docker 中的 Windows 11 VM

## 常用命令

### 环境设置
```bash
conda create -n "omni" python==3.12
conda activate omni
pip install -r requirements.txt
```

### 下载模型权重
```bash
for f in icon_detect/{train_args.yaml,model.pt,model.yaml} icon_caption/{config.json,generation_config.json,model.safetensors}; do huggingface-cli download microsoft/OmniParser-v2.0 "$f" --local-dir weights; done
mv weights/icon_caption weights/icon_caption_florence
```

### 运行演示
```bash
python gradio_demo.py
```

### OmniParser Server
```bash
cd omnitool/omniparserserver
python -m omniparserserver --som_model_path ../../weights/icon_detect/model.pt --caption_model_name florence2 --caption_model_path ../../weights/icon_caption_florence --device cuda
```

### Gradio UI
```bash
cd omnitool/gradio
python app.py --windows_host_url localhost:8006 --omniparser_server_url localhost:8000
```

### 测试
```bash
pytest                                      # 运行所有测试
pytest omnitool/test_omniparser_direct.py   # 运行 OmniParser 直接测试
```

### 代码检查
```bash
ruff check .
```

## 代码架构

### 核心解析流程 (`util/`)

`Omniparser.parse(image_base64)` 执行流程：
1. `check_ocr_box()` → OCR 文本检测 (EasyOCR/PaddleOCR)
2. YOLO 模型检测 UI 元素边界框
3. `get_parsed_content_icon()` → 批量 caption 生成 (batch_size: GPU=128, CPU=16)
4. `get_som_labeled_img()` → 生成带标签的 SOM (Screen Object Model) 图像

关键文件：
- `util/omniparser.py`: `Omniparser` 类，封装完整解析逻辑
- `util/utils.py`: 模型加载、图像处理、OCR、caption 生成

### 代理循环 (`omnitool/gradio/loop.py`)

`sampling_loop_sync()` 支持两种模式：

**Anthropic Computer Use 模式** (claude-3-5-sonnet):
- `AnthropicActor` 生成工具调用 → `AnthropicExecutor` 执行
- OmniParser 结构化信息注入 system prompt

**VLM Agent 模式** (gpt-4o, o1, o3-mini, deepseek-r1, qwen2.5-vl, glm-4.5v):
- `VLMAgent` / `VLMOrchestratedAgent` 使用 SOM 图像 + 结构化信息
- 输出 JSON 格式操作，由 Executor 解析执行

### 工具系统 (`omnitool/gradio/tools/`)

`ComputerTool` 提供键盘/鼠标/截图操作：
- Windows: PyAutoGUI + UIAutomation
- 支持比例坐标和绝对坐标转换
- 坐标缩放适配不同分辨率

### LLM 客户端 (`omnitool/gradio/agent/llm_utils/`)

- `oaiclient.py`: OpenAI API (GPT-4o, O1, O3-mini)
- `groqclient.py`: Groq API (DeepSeek R1)
- `omniparserclient.py`: OmniParser Server 客户端
- `proxy_client.py`: 统一中转 API 客户端 (OpenAI 兼容格式)

### 配置管理 (`omnitool/gradio/llm_config.py`)

中转 API 配置管理，支持多 provider 配置（如 Codex/Claude 中转）。配置文件：`omnitool/gradio/llm_config.json`

## 关键设计决策

1. **设备管理**: 自动检测 CUDA，回退到 CPU
2. **批处理**: caption 推理使用批处理优化 GPU 利用率
3. **图像过滤**: 代理循环仅保留最近 N 张图像管理 context 长度
4. **坐标系统**: 统一使用比例坐标 (0-1) 便于跨分辨率适配
5. **屏幕变化检测**: 代理循环通过下采样灰度像素比较检测屏幕是否变化，避免重复无效操作

## 环境变量

- `OMNITOOL_DEBUG`: 设为 `1`/`true`/`yes` 启用调试日志

## 注意事项

- 模型权重文件夹必须命名为 `icon_caption_florence`（不是 `icon_caption`）
- `BOX_TRESHOLD` 参数拼写如此（项目约定）
- OmniBox 需要 KVM 支持，仅 Linux 可用
- icon_detect 模型使用 AGPL 许可证（继承自 YOLO）
