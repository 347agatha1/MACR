# 原参考相似度

[edit_distance.py](edit_distance.py)来自DEMO1/score_validation/compareCode/textdistance.py；[codexglue.py](codexglue.py)来自同目录CodeXGLUE.py。保留原编辑距离及CLS向量余弦计算。CodeXGLUE仅调整本地模型目录及按roberta配置选择加载类；没有新增相似度算法。默认HumanEval参考列为canonical_solution，LeetCode调用原函数时使用completion。见[根目录README](../README.md)。
