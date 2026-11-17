# 模型与实验特征

- models/fusion：两个数据集各五组正式权重。每组 model.pth 和 result.json 配套，后者包含归一化参数及原结果。
- feature_cache：原实验特征，模型复核及原方法重训使用。
- models/codebert-base：参考相似度模型的配置与分词器；大权重下载方式见[根目录说明](../README.md#获取参考相似度模型)。
- runs：执行特征提取或重训时自动创建的结果目录，不纳入版本管理。

加载正式模型复核使用 `python -B run_pipeline.py`；重新训练使用 `python -B run_pipeline.py --train`。
