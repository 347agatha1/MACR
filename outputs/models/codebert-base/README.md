# CodeBERT 参考相似度模型

配置与分词器来自 [microsoft/codebert-base](https://huggingface.co/microsoft/codebert-base)，固定版本 `3b0952feddeffad0063f274080e3c23d75e7eb39`。

本目录还需 [pytorch_model.bin](https://huggingface.co/microsoft/codebert-base/resolve/3b0952feddeffad0063f274080e3c23d75e7eb39/pytorch_model.bin)，大小为498,627,950字节。将其下载到本目录。

SHA-256：`28b61fd8fa069f6bc966f4cb9572a4026ab2a784fca8fb224020d91b744e32d6`。

仅在重新计算CodeBERT相似度时需要；读取原特征、复核已有结果和重训RD-CVAE不需要这个大文件。操作命令见[根目录说明](../../../README.md#获取参考相似度模型)。
