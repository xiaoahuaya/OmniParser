# 流程索引（任务启动前先查）

说明：

- 系统在每次任务启动时优先读取本索引。
- 若任务命中关键词，则优先读取 `docs` 列中的详细操作文档。
- `keywords` 使用逗号分隔；`docs` 使用分号分隔（相对仓库路径）。

| id | description | keywords | docs |
| --- | --- | --- | --- |
| xhs_nurture | 小红书养号流程（互动优先，按节奏发布） | 小红书,xiaohongshu,xhs,养号,日常运营,浏览,点赞,收藏,评论,互动,开发相关,养狗相关,教育相关,nurture,engage | docs/flows/xhs_nurture_flow.md;docs/flows/flow_reference_guidelines.md;docs/flows/vm134_quick_flow.md |
| xhs_publish_text_note | 小红书发布纯文本笔记（写长文） | 小红书,xiaohongshu,xhs,发布,发笔记,纯文本,写长文,项目进度,publish,post | docs/flows/xhs_text_note_publish_flow.md;docs/flows/vm134_quick_flow.md |
| generic_web_task | 通用网页任务执行指引（有文档优先） | 浏览,打开网站,提交,发布,点赞,收藏,搜索,操作流程,workflow,guide | docs/flows/flow_reference_guidelines.md |
