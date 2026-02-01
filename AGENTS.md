# Repository Guidelines（仓库指南）

## 项目结构与模块组织
- `util/`：OmniParser 核心解析逻辑（模型加载、解析、标注）。
- `omnitool/`：完整工具链：`omniparserserver/`（FastAPI 服务）、`gradio/`（UI 与代理）、`omnibox/`（Windows VM 工具）。
- `docs/` 与 `eval/`：评测文档与脚本。
- `imgs/`：文档/演示图片资源。
- `weights/`：模型权重目录（默认不随仓库分发）。
- 入口示例：`gradio_demo.py` 与 `demo.ipynb`。

## 构建、测试与开发命令
```bash
conda create -n "omni" python==3.12
conda activate omni
pip install -r requirements.txt
```
```bash
python gradio_demo.py
```
```bash
python -m omniparserserver --som_model_path weights/icon_detect/model.pt \
  --caption_model_name florence2 --caption_model_path weights/icon_caption_florence \
  --device cuda --BOX_TRESHOLD 0.05
```
```bash
ruff check .
pytest
```
OmniBox VM 管理由 `omnitool/omnibox/scripts/` 下的 `manage_vm.sh` / `manage_vm.ps1` 提供。

## 编码风格与命名规范
- Python 4 空格缩进；函数/变量 `snake_case`，类名 `PascalCase`。
- 文件/模块名保持小写与下划线风格（参考 `util/`、`omnitool/gradio/`）。
- 提交前运行 `ruff check .`；遵循邻近文件的既有模式。

## 测试规范
- 测试框架：`pytest`。
- 测试较少，示例见 `omnitool/test_omniparser_direct.py`、`omnitool/gradio/test_glm_taobao.py`。
- 修改解析或代理循环逻辑时优先补回归测试，测试函数以 `test_*` 命名。

## 提交与 PR 指南
- 历史提交多为简短动词开头（如 “update readme”、“add …”），无强制格式。
- 提交应聚焦单一变更点，避免重构与功能混在一起。
- PR 需包含：变更说明、验证方式；UI 改动请附截图或 GIF（`omnitool/gradio/`）。
- 关联相关 Issue；不要提交模型权重或 API 密钥。

## 安全与配置提示
- 安全漏洞报告流程见 `SECURITY.md`。
- 模型权重放置在 `weights/`，路径命名需保持一致以免影响演示与脚本。
