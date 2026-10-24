# 数据目录

`data/raw/`、`data/processed/` 和 `runs/` 是本地运行目录，不提交 Git。团队成员从官方来源取得原始数据后，在自己的电脑上执行 D3/D4，以生成可追溯的 YAML、manifest、数据指纹和验证印章。

## 正式数据集

### Brackish Underwater

作为 6 类水下生物回归基准。原始 Roboflow YOLO 导出放入 `data/raw/brackish/`，保留其中的 `train/valid/test` 目录，然后运行：

```powershell
.\.venv\Scripts\odp-transform.exe --dataset brackish --format yolo --split-strategy preserve_source
.\.venv\Scripts\odp-validate.exe apps/platform/configs/datasets/brackish.yaml
```

### COU: Common Objects Underwater

作为 24 类水下常见物体扩展集。官方来源：University of Minnesota DRUM，DOI [`10.13020/1vwe-2707`](https://doi.org/10.13020/1vwe-2707)，许可为 CC BY-SA 4.0。

COCO 压缩包校验信息：

- 文件名：`coco.zip`
- 文件大小：`3,844,669,717` 字节
- MD5：`7b7bdcf81dc6cd388b0ff492dfda888a`

将压缩包解压到 `data/raw/cou/` 下任意层级；D3 会定位官方的 `coco/coco/images/` 与三份 split JSON。不要改动官方标注，然后运行：

```powershell
.\.venv\Scripts\odp-transform.exe --dataset cou --format coco --split-strategy preserve_source
.\.venv\Scripts\odp-validate.exe apps/platform/configs/datasets/cou.yaml
```

当前真实验证结论为 `WARNING`（无 `ERROR`），按 D4 分级规则允许训练并保留风险审计。详细结果见 `docs/答辩/D3_D4_数据工程实现说明.md`。
