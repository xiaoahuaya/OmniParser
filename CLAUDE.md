# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

**OmniParser** 是微软开发的屏幕解析工具，将 UI 截图转换为结构化元素，增强视觉语言模型（如 GPT-4V）生成可定位操作的能力。

三个主要组件：
- **OmniParser Core** (`util/`): 核心解析引擎，包含 YOLO 图标检测 + Florence2/BLIP2 caption + OCR
- **OmniParser Server** (`omnitool/omniparserserver/`): FastAPI REST API 封装
- **OmniTool** (`omnitool/`): 完整的 GUI 代理系统
  - `gradio/`: Gradio UI + 代理循环 + 任务管理
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
pytest omnitool/gradio/test_loop_helpers.py # 运行循环控制逻辑测试
```

### 代码检查
```bash
ruff check .
```

## 代码架构

### 端到端工作流

```
用户输入 → app.py (Gradio UI / 任务管理)
         → loop.py (sampling_loop_sync)
         → OmniParserClient → omniparserserver → util/omniparser.parse()
         → Actor (Anthropic/VLM) 生成动作
         → Executor 执行工具 (ComputerTool)
         → loop_helpers 验证结果 (屏幕变化检测/循环防护)
         → task_state_store 持久化状态
         → 反馈 Actor 继续循环
```

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

**API Provider 映射**:
- `ANTHROPIC` → claude-3-5-sonnet, `OPENAI` → gpt-4o, `ZHIPU` → glm-4.5v
- `CODEX_PROXY` / `CLAUDE_PROXY` → 中转代理 (OpenAI 兼容格式)

### 循环控制与恢复 (`omnitool/gradio/loop_helpers.py`)

智能恢复机制，防止代理陷入无效循环：
- **屏幕变化检测**: 下采样灰度像素比较 (阈值 `SCREEN_DIFF_THRESHOLD=0.015`)
- **动作门 (action gate)**: 执行后验证动作是否生效 (阈值 `0.012`)
- **重复坐标拦截**: 连续 3 次相同点击自动拦截
- **导航异常恢复**: 检测 Bing/Challenge 页面并自动处理
- **滚动兜底**: 滚轮无变化时自动切换键盘翻页
- **焦点探针 (focus probe)**: 输入框有效性检测

### 工具系统 (`omnitool/gradio/tools/`)

`ComputerTool` (`computer.py`) 提供键盘/鼠标/截图操作：
- 动作: left_click, right_click, double_click, key, type, drag, scroll, screenshot 等
- Windows: PyAutoGUI + UIAutomation
- 支持比例坐标 (0-1) 和绝对坐标转换，坐标缩放适配不同分辨率
- 输入法支持（分组输入，速率优化）
- 支持本地和远程截图 (HTTP 请求到 Windows 主机)

### 任务状态管理 (`omnitool/gradio/task_state_store.py`)

JSON 持久化的任务状态系统，支持：
- 任务创建/加载/保存/恢复
- 连续循环模式 (`continuous_mode`)
- 最大步数和运行时间限制
- 执行计划与步骤跟踪

相关文件：
- `task_policy.py`: 任务策略（发布检测、互动任务判断）
- `node_config.py`: 多节点 Windows 主机配置
- `run_limits.py`: 运行时长限制

### LLM 客户端 (`omnitool/gradio/agent/llm_utils/`)

- `oaiclient.py`: OpenAI API (GPT-4o, O1, O3-mini)
- `groqclient.py`: Groq API (DeepSeek R1)
- `omniparserclient.py`: OmniParser Server 客户端（截图 → base64 → 解析）
- `proxy_client.py`: 统一中转 API 客户端 (OpenAI 兼容格式)

### 配置管理 (`omnitool/gradio/llm_config.py`)

中转 API 配置管理，支持多 provider 配置。配置文件：`omnitool/gradio/llm_config.json`

### 运行时监控 (`omnitool/gradio/runtime_log_monitor.py`)

日志噪声过滤、信号检测（错误/超时/故障切换）、后端日志汇聚与摘要。

## 关键设计决策

1. **设备管理**: 自动检测 CUDA，回退到 CPU
2. **批处理**: caption 推理使用批处理优化 GPU 利用率
3. **图像过滤**: 代理循环仅保留最近 N 张图像管理 context 长度
4. **坐标系统**: 统一使用比例坐标 (0-1) 便于跨分辨率适配
5. **屏幕变化检测**: 下采样灰度像素比较，避免重复无效操作
6. **状态持久化**: JSON 存储任务状态，支持断点恢复
7. **多节点架构**: 支持多个远程 Windows VM 并行执行任务

## 环境变量

- `OMNITOOL_DEBUG`: 设为 `1`/`true`/`yes` 启用调试日志
- `OMNITOOL_WINDOWS_HOST_URL`: Windows 主机 URL
- `OMNITOOL_PREFLIGHT_ENABLED`: 启用动作前置验证
- `OMNITOOL_FOCUS_PROBE_ENABLED`: 启用焦点探针
- `OMNITOOL_OUTPUT_MAX_FILES`: 最大输出文件数
- `OMNITOOL_OUTPUT_MAX_AGE_HOURS`: 输出文件最大保留时间
- `OMNITOOL_RUN_MAX_ATTEMPTS`: 最大重试次数

## 编码规范

- Python 4 空格缩进；函数/变量 `snake_case`，类名 `PascalCase`
- 提交前运行 `ruff check .`；遵循邻近文件的既有模式
- 修改解析或代理循环逻辑时优先补回归测试

## 注意事项

- 模型权重文件夹必须命名为 `icon_caption_florence`（不是 `icon_caption`）
- `BOX_TRESHOLD` 参数拼写如此（项目约定）
- OmniBox 需要 KVM 支持，仅 Linux 可用
- icon_detect 模型使用 AGPL 许可证（继承自 YOLO）
- `docs/flows/` 包含各平台操作流程文档（小红书、抖音、快手等）
