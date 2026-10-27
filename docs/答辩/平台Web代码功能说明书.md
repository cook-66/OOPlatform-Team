# 平台 Web 侧代码功能说明书

> 适用范围：本机实际运行的 Web 相关代码。所有行为均以当前代码为准（2026-10-26）。
> 阅读顺序建议：第 2 节路由 → 第 3 节数据模型 → 第 4 节接口 → 第 5 节前端 → 第 6 节大模型 → 第 7 节端到端流程。

## 1. 文件清单与角色

| 文件 | 角色 | 说明 |
|---|---|---|
| `apps/odp_manage/odp_manage/urls.py` | 根路由 | 挂载 admin、业务路由、开发环境媒体文件 |
| `apps/odp_manage/odp_manage/settings.py` | 项目配置 | `DEBUG=True`、`TIME_ZONE='Asia/Shanghai'`、`USE_TZ=True`、`MEDIA_URL='/media/'`、`MEDIA_ROOT=apps/odp_manage/media`、`STATICFILES_DIRS=[BASE_DIR/"static"]` |
| `apps/odp_manage/platform_web/urls.py` | 业务路由 | 7 条路由（页面 + 6 个 API） |
| `apps/odp_manage/platform_web/models.py` | 数据模型 | `InferenceRecord` 一条记录 = 一次推理 + 一次解释 + 一笔费用 |
| `apps/odp_manage/platform_web/views.py` | 核心后端 | 模型单例、推理、解释、历史、详情、趋势、计费汇总 |
| `apps/odp_manage/platform_web/apps.py` | 启动钩子 | runserver 时后台线程预热模型 |
| `apps/odp_manage/platform_web/templates/index.html` | 前端单页 | Vue3（CDN）+ Tailwind + ECharts，5 个模块 + 原生 JS 业务逻辑 |
| `apps/platform/src/od_platform/assistant/llm_service.py` | LLM 服务 | 防呆 Prompt、token/费用计算、视频分析（队友维护） |
| `platform_web/migrations/0001~0003` | 迁移 | 建表；增加 `llm_cost`；增加 `llm_tokens` |
| `apps/odp_manage/static/training_results.png` | 静态资源 | 训练曲线图，供"模型训练"模块展示 |

## 2. 路由表

| 方法 | 路径 | 视图 | 功能 |
|---|---|---|---|
| GET | `/` | `index` | 渲染单页控制台 |
| POST | `/api/infer/` | `api_run_inference` | 上传图片/视频 → YOLO 推理 → 落库 |
| POST | `/api/explain/` | `api_generate_explanation` | 类别 → 大模型解释 → 写回记录 |
| GET | `/api/record/<id>/` | `api_record_detail` | 单条记录全字段（详情弹窗） |
| GET | `/api/history/` | `api_history` | 历史列表 + 今日花费/Token 汇总 |
| GET | `/api/trend/` | `api_trend` | 最近 7 天每日推理数 |
| GET | `/api/today-cost/` | `api_today_cost` | 今日花费与付费记录数 |

媒体文件：`/media/...` 由根路由 `static()` 提供（仅 `DEBUG=True` 时有效）。

## 3. 数据模型 `InferenceRecord`

| 字段 | 类型 | 含义 |
|---|---|---|
| `original_img` | ImageField → `original/` | 上传原图（也用于存视频路径） |
| `result_img` | ImageField → `result/` | 结果图/结果视频相对路径 |
| `detected_classes` | CharField(200) | 双语类别串，如 `易拉罐（Soda Can）×2` |
| `confidence` | Float | 本次最高置信度 |
| `llm_explanation` | TextField | 大模型分析报告 |
| `llm_cost` | Float | 本次预估花费（元） |
| `llm_tokens` | Int | 本次 token 用量 |
| `created_at` | DateTime(auto_now_add) | 检测时间（库里 UTC，展示按本地时区） |

设计要点：**一次推理一条记录**，解释与费用都挂在同一条记录上，因此"详情、历史、花费、趋势"全部来自同一张表，不存在两套账。

## 4. 后端 `views.py`

### 4.1 常量与工具

| 名称 | 作用 |
|---|---|
| `_YOLO_MODEL_PATH` | 当前推理权重：`D:\OOPlatform-Team\models\trained\train-20261026-091721-dataset_new-yolo11s-best.pt` |
| `INFER_CONF = 0.25` | 默认置信度阈值（与前端滑块默认值一致） |
| `CLASS_NAME_ZH` | 24 类英文 → 中文映射 |
| `_bilingual()` | 输出「中文（英文）」，未知类回退英文原样 |
| `_parse_conf()` | 读取请求里的 `conf`，非法回退默认，夹在 0.05–0.95 |
| `_is_video()` | 按 `content_type` + 扩展名判断视频 |
| `_parse_class_counts()` | 把 `dict` / `类:数量` / `类,类` 统一成 `{类名: 数量}` |
| `_count_classes(boxes, names)` | 从检测框统计每类数量 |
| `get_yolo_model()` | **进程内单例**，双检锁，首次加载，之后复用 |
| `warmup_yolo_model()` | 空图推理一次，把初始化开销移到启动阶段 |
| `_infer_image()` | 单帧推理 → `result.plot()` 写结果图 |
| `_infer_video()` | 逐帧推理 → H.264(`avc1`) 写 mp4，失败回退 `mp4v`，写后回读校验 |

`_infer_video` 细节：宽高裁成偶数（yuv420p 要求）；`VideoWriter.isOpened()` 不可靠，所以额外用 `VideoCapture` 回读帧数，`<=0` 直接报错；`_PREDICT_LOCK` 串行化预测，避免 ultralytics 实例并发问题。

### 4.2 `api_run_inference`

流程：判断图片/视频 → 统一存 `original/` → 建一条空记录 → 调 `get_yolo_model()` → 推理 → 写结果文件 → 回填类别/置信度 → 返回。

响应字段：`status`、`record_id`、`media_type`、`classes`、`class_counts`、`confidence`、`conf_threshold`、`frames`、`infer_time`，以及二选一的 `result_img_url` / `result_video_url`。
异常处理：任何推理异常都会删除这条半成品记录并返回 500，避免脏数据。

### 4.3 `api_generate_explanation`

流程：解析 JSON → 校验 `analyze_detection` 是否可用 → 解析类别 → 转双语 → 读取 `conf_threshold` → **优先按 `record_id` 精确找记录**（找不到只记 warning，不报错）→ 调 `analyze_detection` → 写入 `llm_explanation / llm_cost / llm_tokens`。

容错：大模型异常返回 **502** + 中文提示（附带 `explanation` 字段供前端直接展示）；模块没导入成功返回 503；类别为空 400；JSON 非法 400。

### 4.4 查询类接口

- `api_record_detail`：返回 `id / created_at / original_img / result_img / detected_classes / confidence / llm_explanation / llm_cost / total_tokens`，不存在返回 404。
- `api_history`：`limit` 默认 50、上限 200；返回 `records[]`（含 `tokens`）、`total_count`、`today.total_cost / today.total_tokens`。
- `api_trend`：`TruncDate + Count`，最近 7 天补 0，日期在服务端按本地时区生成。
- `api_today_cost`：`Sum(llm_cost)` + 付费记录数（前端预留接口，当前主要用 `/api/history/` 的 today 字段）。

## 5. 前端 `index.html`

### 5.1 框架分工

- **Vue3**：只负责视图切换（`currentTab`，默认 `infer`）与导航高亮，5 个模块全部 `v-if`。
- **Tailwind**：布局与深色主题。
- **ECharts**：数据理解柱状图、仪表盘折线/环形图。
- **原生 JS**：上传、fetch、进度、媒体渲染、表格、弹窗——事件用 `document` 委托，避免 Vue 重建 DOM 后监听丢失。

### 5.2 五大模块

| Tab | 内容 | 数据来源 |
|---|---|---|
| `intro` 项目介绍 | 平台定位、闭环链路、技术栈标签 | 静态文案 |
| `data` 数据理解 | 9,663 图 / 24 类卡片、24 条类别占比进度条、类别柱状图 | 真实标注统计（dataset_new，15,976 实例） |
| `train` 模型训练 | mAP50 89.7%、100 轮、yolo11s、真实训练曲线 | `runs/training/20261026-091721/` |
| `infer` 模型推理 | 上传、置信度滑块、推理、结果图/视频、LLM 解释、花费 | `/api/infer/`、`/api/explain/` |
| `dashboard` 仪表盘 | 4 张 KPI、7 天折线、类别环形图、历史表格 | `/api/history/`、`/api/trend/` |

### 5.3 关键函数

| 函数 | 功能 |
|---|---|
| `refreshHistory()` | 拉 `/api/history/`，渲染表格、今日花费/Token、累计推理数 |
| `loadTrend()` | 拉 `/api/trend/`，真实 7 天数据 |
| `refreshDashboard()` | 数据就绪后再初始化图表 |
| `initDashboardCharts()` | 折线 + 环形图（实例失效自动重建） |
| `initDataBarChart()` | 解析进度条 DOM 生成柱状图，**保证图表与进度条数值同源** |
| `initDataBarChart` 数据来源 | `#class-bars` 内每行第一个 span（名称）、第二个 span（`1,826 个`） |
| `openRecordDetail(id)` | 拉 `/api/record/<id>/`，填充弹窗 |
| `closeRecordDetail()` | 关闭弹窗、暂停并卸载视频 |
| `setDetailMedia()` | 详情弹窗里自动判断图片/视频 |
| `showVideo()` / `showVideoFallback()` | 视频播放，失败降级为下载 |
| `resetVideoPreview()` | 换视频前解绑 `onerror`、清空 `src`，防串台 |

### 5.4 推理交互

1. 滑块 `0.05–0.9`，实时显示数值，提交时 `formData.set("conf", 当前值)`。
2. 推理成功后记录 `currentRecordId`，再请求 `/api/explain/` 并带 `record_id` + `conf_threshold`。
3. 结果预览：图片进 `<img>`，视频进 `<video playsinline webkit-playsinline>`；视频 `onerror` 时显示"不支持该编码，请下载查看"+ 下载按钮。
4. 推理完成、解释完成都会刷新历史与今日花费。
5. 再次推理**不需要重新选文件**，表单里的文件会复用。

## 6. 大模型模块 `llm_service.py`

| 名称 | 功能 |
|---|---|
| `client` | `OpenAI` 客户端，`base_url` 指向 DeepSeek，Key 从环境变量 `LLM_API_KEY` 读取（import 时创建） |
| `GUARDRAIL_PROMPT` | 防呆人设：海洋环境工程专家，信息不足必须表达不确定，禁止编造画面细节 |
| `_ask_llm()` | 拼 Prompt → 调 `deepseek-chat` → 返回 `{text, cost, tokens}`；空检测直接返回"无法确认"，不调用 API |
| `analyze_detection()` | Web 用入口：强制防呆 Prompt，透传平均置信度 |
| `generate_explanation()` | 只要文本的包装（历史兼容） |
| `estimate_cost()` | 按文件类型/视频时长预估 token 与费用（**Web 端未接入**） |
| `analyze_video()` | 独立流程：自己跑 YOLO → 汇总类别与置信度 → 调 LLM → 写 `billing_records.json`（**Web 端未使用**） |
| `_save_billing_record()` | 追加本地 JSON 账单（Streamlit 流程用） |

计费公式（`_ask_llm` 内）：`cost = prompt_tokens × 0.000001 + completion_tokens × 0.000002`（元，四舍五入 6 位）。

Prompt 结构：物品推测 → 确定性评估（引用真实置信度）→ 环境危害科普 → 治理建议。返回的 `cost/tokens` 由 `api_generate_explanation` 入库，成为"本次花费/今日花费/今日 Token"的唯一来源。

**已知待清理**：该文件里 `generate_explanation` 定义了两次（约第 110 行与第 192 行），Python 后定义覆盖前定义，实际生效的是第 192 行版本；`_ask_llm` 的 `conf_threshold` 参数目前未被 Prompt 使用（只用了 `avg_confidence`）。

## 7. 端到端流程

1. 用户上传 → `/api/infer/` → 建记录 → 单例模型推理 → 结果文件写 `media/results/` → 返回 `record_id` + 预览 URL。
2. 前端带着 `record_id` 请求 `/api/explain/` → `analyze_detection` → 返回文本/tokens/费用 → 写回**同一条**记录 → 前端显示解释与花费。
3. 前端刷新 `/api/history/` → 表格、今日花费、今日 Token、累计推理数。
4. 切到仪表盘 → `/api/trend/` 拉 7 天真实数据 → ECharts 折线；环形图展示数据集类别分布。
5. 点"查看详情" → `/api/record/<id>/` → 弹窗展示原图/结果、类别、置信度、解释、费用。

## 8. 运行与配置

```
# 1) 依赖：openai（其余 Django/opencv/numpy/ultralytics 已具备）
pip install "openai>=1.0.0"

# 2) 数据库迁移（已有 0001~0003）
python manage.py migrate

# 3) 大模型 Key（必须在启动进程的环境里）
$env:LLM_API_KEY="sk-..."
python manage.py runserver
```

- 权重路径写死在 `views._YOLO_MODEL_PATH`，换模型只改这一处。
- 启动日志出现"模型预热完成"表示单例已就绪。
- 媒体文件仅在 `DEBUG=True` 时由 Django 提供；生产需交给 Nginx。

## 9. 功能 → 代码速查表（答辩用）

| 若被问到 | 指向 |
|---|---|
| 为什么推理快 | `views.get_yolo_model` + `apps.ready` + `_PREDICT_LOCK` |
| 视频黑屏怎么修 | `_infer_video`（`avc1` + 回读校验）+ `showVideoFallback` |
| 多人会不会写串 | `api_generate_explanation` 里按 `record_id` 精确更新 |
| 费用从哪来 | `_ask_llm` 的 `cost/tokens` → `InferenceRecord.llm_cost/llm_tokens` → `api_history.today` |
| 滑块真的生效吗 | `_parse_conf` → `_infer_image/_infer_video` 的 `model.predict(conf=...)` |
| 为什么图表数据可信 | `api_trend` 真实聚合；柱状图解析同一份进度条 DOM |
| 大模型胡说怎么办 | `GUARDRAIL_PROMPT` + 空检测短路 + 502 优雅降级 |

## 10. 已知限制（主动说明）

1. 无登录/权限，记录不区分用户。
2. SQLite + 单进程；多 worker 部署时每个进程各有一份模型。
3. LLM Key 依赖环境变量，未配置时解释接口返回 502。
4. 视频逐帧 CPU 推理，时长与耗时近似线性。
5. 前端依赖 CDN（Vue/Tailwind/ECharts），离线环境需改为本地静态资源。
6. 环形图是数据集类别分布，不是在线推理识别占比。
7. `Metal Rod` 标注实例为 0，该类指标不可信。
8. `llm_service.py` 存在重复定义与未使用参数（见第 6 节），不影响运行，建议由 LLM 同学清理。
