# MACR

MACR 用于评价 ChatGPT 生成代码的可靠性，主流程为 **MASE → RRN（LACF + RD-CVAE）→ RV**。仓库包含原实验代码、两份正式数据表、原特征缓存和两个数据集各五组正式模型权重。

## 目录

| 路径 | 含义 |
| --- | --- |
| [code_generation](code_generation) | 生成待评价代码 |
| [MASE](MASE) | 多角色定性评分、原 lead/worker/checker 提示词及调用流程 |
| [objective_evaluation](objective_evaluation) | 原测试执行、静态指标与指标归一化 |
| [reference_similarity](reference_similarity) | CodeBERT CLS 向量余弦相似度、Levenshtein 编辑距离相似度 |
| [RRN/LACF](RRN/LACF) | 原注意力与循环卷积特征融合 |
| [RRN/RD_CVAE](RRN/RD_CVAE) | 条件融合、训练及 GCRS 评分 |
| [RV](RV) | CTC、ATC 结果验证 |
| [data](data) | HumanEval 164 条、LeetCode 309 条正式输入及角色提示词来源材料 |
| [outputs/models/fusion](outputs/models/fusion) | 两数据集 × 五种子的正式模型及配套 result.json |
| [outputs/feature_cache](outputs/feature_cache) | 原实验特征 |
| [outputs/models/codebert-base](outputs/models/codebert-base) | CodeBERT 配置和分词器；大权重按下文获取 |
| [run_pipeline.py](run_pipeline.py) | 原模型复核、特征核对和重新训练入口 |

代码沿用原项目实现，仅调整目录导入、默认工作文件、API 环境变量和 CodeBERT 加载类兼容。没有额外加入新的联合训练模型、评分检查算法、三项资源指标、对比实验或消融实验。

## 安装与复核

```powershell
git clone https://github.com/347agatha1/MACR.git
Set-Location MACR
python -m pip install -r requirements.txt
python -B run_pipeline.py
```

该命令加载已有正式权重，重新计算测试集得分并核对保存结果；不调用大模型 API。模型文件的 result.json 还包含归一化参数，需与 model.pth 一起保留。

重新从正式表提取特征并核对原缓存：

```powershell
python -B run_pipeline.py --extract-features
```

按原参数从随机初始化重新训练两个数据集、每个数据集五个种子：

```powershell
python -B run_pipeline.py --train
```

新权重和 summary.json 写入 outputs/runs/随机编号，不覆盖正式权重。此入口使用原特征缓存，不是任意新工作表的独立预测器。

## 已验证的原实验结果

下表为五个种子的均值 ± 样本标准差，单位为百分比。已在当前验证环境中完成十次从随机初始化的重训，新旧模型的全部参数张量、测试预测和指标完全相同。其他软件环境的重训结果应重新核对。

| 数据集 | CodeXGLUE_CTC | CodeXGLUE_ATC | Levenshtein_CTC | Levenshtein_ATC |
| --- | ---: | ---: | ---: | ---: |
| HumanEval | 68.00 ± 4.00 | 65.83 ± 4.56 | 68.80 ± 7.16 | 76.67 ± 3.73 |
| LeetCode | 63.83 ± 7.52 | 61.74 ± 5.00 | 66.81 ± 5.34 | 63.04 ± 2.66 |

原解析流程分别使用 163/164 和 308/309 条任务：HumanEval/130、graph-valid-tree 的定性矩阵无法解析，按原逻辑记录并剔除。任务按文件顺序划分为训练集、验证集、测试集，比例为 70%/15%/15%；五个种子使用相同划分。

## 主要训练设置

- 随机种子：1、5、21、42、3407。
- LACF：2 个注意力头，卷积核 3，dropout=0。
- RD-CVAE：编码器 128/64/32，潜变量 16，双分支解码器各 64/32。
- Adam：学习率 0.001，权重衰减 0.0001。
- 重构/KL/评分损失权重：4/4/2，α=0.1。
- 最多 300 轮，验证指标选模，连续 30 轮未改善时早停。
- 原结果流程使用预先提取的 LACF 特征；重训入口训练 RD-CVAE 和评分层。

## 获取参考相似度模型

仅在重新计算 CodeBERT 参考相似度时需要。模型来源为 [Microsoft CodeBERT-base](https://huggingface.co/microsoft/codebert-base)，固定版本为 `3b0952feddeffad0063f274080e3c23d75e7eb39`。配置和分词器已包含在仓库中，498,627,950 字节的大权重单独下载：

```powershell
Invoke-WebRequest -Uri 'https://huggingface.co/microsoft/codebert-base/resolve/3b0952feddeffad0063f274080e3c23d75e7eb39/pytorch_model.bin' -OutFile '.\outputs\models\codebert-base\pytorch_model.bin'
Get-FileHash -LiteralPath '.\outputs\models\codebert-base\pytorch_model.bin' -Algorithm SHA256
```

预期 SHA-256：`28b61fd8fa069f6bc966f4cb9572a4026ab2a784fca8fb224020d91b744e32d6`。

## 生成和评价新代码

以下为 HumanEval 原脚本入口。先准备工作副本；正式输入表不直接改写：

```powershell
New-Item -ItemType Directory -Force '.\outputs\runs' | Out-Null
Copy-Item -LiteralPath '.\data\human-eval.xlsx' -Destination '.\outputs\runs\working.xlsx'
$env:MACR_API_KEY = '你的密钥'
$env:MACR_API_URL = '兼容接口的完整地址'
$env:MACR_MODEL = '实际使用的模型名称'
python -B .\code_generation\generate.py
python -B .\MASE\scoring.py
python -B .\objective_evaluation\metrics.py
python -B .\reference_similarity\edit_distance.py
python -B .\reference_similarity\codexglue.py
```

保留原脚本行为：生成示例默认处理前两条；定性阶段交互选择数量，跳过已有定性数据；定量阶段跳过已有定量数据。上面是各阶段调用顺序，不会自动清除旧结果。重新生成某条代码后，需先在工作副本中清空该条旧定性、定量和参考相似度数据，再运行下游阶段。

编辑距离输出 working_with_levenshtein.xlsx，CodeBERT 接续该文件，输出 working_with_levenshtein_with_codexglue.xlsx。HumanEval 参考列为 canonical_solution；LeetCode 调用原函数时使用 completion。API 配置通过环境变量提供，.env.example 不会自动加载。

本地依赖目录、下载缓存、临时运行结果和历史审计记录不纳入仓库。
