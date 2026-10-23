ODPlatform 🚀
ODPlatform 是一个专为计算机视觉（CV）任务打造的工业级、可插拔开发平台。它涵盖了从数据流水线、运行配置、模型训练到多线程实时推理的全生命周期管理。

本项目采用标准的 Monorepo 架构，以极高的工程素养解决了 AI 落地过程中的脏数据、配置混乱和推理性能瓶颈问题。

✨ 核心特性
📦 统一配置 (D5)：基于 Pydantic v2 的三源合并配置（CLI > YAML > Default），附带完整的参数溯源链。

🏭 数据流水线 (D3)：一键将 VOC/COCO/YOLO 格式转换为标准训练集，支持智能分层划分（专治长尾稀有类），并生成防篡改的数据指纹。

🛡️ 数据质检 (D4)：15 项工业级数据体检（如出界框、孤儿标签、数据泄漏），确保垃圾数据绝不进模型。

🎯 模型训练 (D6)：基于数据流转卡（Lineage Stamp）的审计训练链路，自动记录环境快照与参数血缘。

⚡ 实时推理引擎 (D7)：四级多线程流水线（采集 -> 推理 -> 渲染 -> 显示），主线程零阻塞。内置 5 维 FPS 监控和 HUD 面板，支持断线重连与程序化取消。

🧩 可插拔帧源 (frame_source)：统一摄像头、视频流、本地图片目录的读取接口，支持极高的扩展性。

🛠️ 环境准备
推荐使用虚拟环境进行开发：

bash
python -m venv .venv
source .venv/bin/activate # Windows 下使用 .venv\Scripts\activate
pip install -e "./apps/platform[dev]"
🚀 快速开始 (CLI 命令一览)
本项目基于 hatchling 打包，安装后会在环境中注册以下核心命令：

命令 功能说明
odp-init 初始化工作区目录结构（包含共享数据、模型与私有配置目录）。
odp-reset 安全重置。清理运行时产物（默认 dry-run 模式防误触）。
odp-transform 数据格式转换、清洗、以及智能划分。
odp-validate 对 D3 产出的数据集进行 15 项严格质检。
odp-gen-config 从 Pydantic 反射生成带详细注释的 YAML 配置模板。
odp-train 启动 YOLO 模型训练并生成完整的审计快照。
odp-infer 启动多线程实时推理（支持图片、视频、摄像头）。
一个典型的端到端工作流：
bash

# 1. 转换并清洗数据

odp-transform --dataset coco8 --format yolo --classes person bicycle car

# 2. 数据质量检查

odp-validate apps/platform/configs/datasets/coco8.yaml

# 3. 开始模型训练

odp-train --data coco8 --epochs 50 --batch 16

# 4. 使用本地摄像头进行实时推理 (会弹出 GUI 和 HUD 面板)

odp-infer --source 0 --show
📂 目录结构
text
ODPlatform/
├── apps/
│ └── platform/ # 核心引擎 (od_platform 包)
│ ├── configs/ # 数据集与运行配置
│ ├── src/ # 核心源码 (包含 data_pipeline, inference 等)
│ └── tests/ # 单元测试与集成测试
├── data/ # 数据仓库 (raw 为只读，processed 为产物)
├── models/ # 模型权重 (pretrained 与 trained 分离)
├── runs/ # 运行现场日志与审计快照
└── README.md
📜 架构与设计哲学
本项目严格遵循 “生产材料”与“审计材料”分离、“幂等性” 与 “永不抛异常（长跑服务）” 等企业级原则。
详细的架构演进与设计决策，请参阅 docs/architecture/ 下的 ADR 文档。

Built with ❤️ by ODPlatform Team
