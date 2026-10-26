# ODPlatform · 水下智能视觉中枢 🌊

**ODPlatform** 是一个工业级、全栈闭环的水下目标检测与生态分析平台。系统涵盖了从**数据流水线、模型训练、大模型智能研判到成本控制（FinOps）**的全生命周期管理。

本项目采用标准的 Monorepo 架构，以极高的工程素养解决了 AI 落地过程中的脏数据、模型泛化、前后端解耦以及 API 调用成本黑盒等痛点。

## ✨ 核心特性

- 🏭 **自动化数据流水线 (D3/D4)**：支持 COU/Brackish 等复杂水下数据集的自动清洗、智能分层划分。内置 15 项工业级数据质检，生成防篡改的数据指纹，确保“带病数据”绝不进模型。
- 🎯 **高精度模型训练 (D5/D6)**：深度集成 YOLO11s，支持动态配置超参数（通过 Pydantic 实现三源合并）。带有完整的训练审计与指标回溯，最终模型精度（mAP50）高达 **94.2%**。
- 🧠 **多模态智能研判 (D7/D8)**：Web 调度中心内嵌 DeepSeek 大模型。YOLO 识别出目标后，大模型会自动根据水下生态的上下文，生成专业的巡检与生态分析报告。
- 💳 **FinOps 商业级计费 (Web 端)**：系统自动追踪每一次大模型调用的 Token 消耗，并实时计算预估费用入库。支持今日花费汇总与历史账单查询，让 AI 成本彻底透明。
- 🖥️ **现代化 Web 调度中心 (Django + Tailwind)**：基于 Django 构建核心枢纽，前端采用 TailwindCSS 打造深色科技风控制台。支持图片/视频双模态上传、实时推理、历史记录追踪与数据看板展示，完全前后端分离。

## 🛠️ 环境准备

推荐使用虚拟环境进行开发：

```bash
python -m venv .venv
source .venv/bin/activate  # Windows 下使用 .venv\Scripts\activate

# 安装核心引擎及开发依赖
pip install -e "./apps/platform[dev]"

# 安装 Web 后端依赖
pip install django pillow openai
🚀 快速启动
本项目分为底层算法引擎和 Web 调度中心两部分。

1. 启动 Web 调度中心 (前端 + 后端)
bash
cd apps/odp_manage
# 配置大模型 API Key
$env:LLM_API_KEY="sk-你的DeepSeek_Key"

# 数据库迁移并启动
python manage.py migrate
python manage.py runserver
访问 http://127.0.0.1:8000/ 即可进入 Web 控制台。

2. 执行数据流水线与模型训练 (底层引擎)
bash
# 洗数据
odp-transform --dataset cou_raw --format yolo

# 跑质检
odp-validate apps/platform/configs/datasets/cou_raw.yaml

# 开启 GPU 炼丹 (支持自定义参数)
odp-train --data cou_raw --model yolo11s.pt --epochs 100 --batch 0.8 --imgsz 640 --device 0
📂 核心目录结构
text
OOPlatform-Team/
├── apps/
│   ├── platform/             # 核心算法引擎 (od_platform)
│   │   ├── src/              # 包含 data_pipeline, inference, training 等
│   │   └── configs/          # 数据集与运行配置
│   ├── odp_manage/           # Web 调度中心 (Django)
│   │   ├── platform_web/     # MVT 业务逻辑 (包含计费、推理路由)
│   │   └── templates/        # TailwindCSS 前端界面
│   └── web-backend/          # (旧版) FastAPI 实时推流服务
├── data/                     # 数据仓库 (raw 为只读，processed 为产物)
├── models/                   # 模型权重 (pretrained 与 trained 分离)
├── runs/                     # 运行现场日志与审计快照
└── README.md
🎓 架构与设计哲学
流转卡与数据指纹：每一批数据都有唯一的指纹，训练前必须通过 D4 核验，杜绝使用脏数据。

环境隔离与资产分离：模型权重与数据集不进入 Git 仓库，代码与运行现场（runs/）严格隔离。

全局单例与防崩溃：Web 后端对模型加载采用单例模式，大模型接口全面捕获异常并提供降级兜底，保证长跑服务“永不抛异常”。
```
