import numpy as np
SIMILARITY_KEYS = ['ngramMatch', 'weightedNgramMatch', 'syntaxMatch', 'dataflowMatch', 'CodeBLEU', 'CodeXGLUE', 'difflib', '编辑距离相似度']
def compute_ctc_atc(pred, ref):
    """按冻结口径计算 CTC/ATC（0~1 小数）。

    pred / ref：一维等长数组，顺序即文件行序。
    """
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    ref = np.asarray(ref, dtype=np.float64).reshape(-1)
    n = pred.shape[0]
    if n == 0:
        return 0.0, 0.0

    ctc = float(np.mean((pred > pred.mean()) == (ref > ref.mean())))

    if n > 1:
        d_pred = np.diff(pred)
        d_ref = np.diff(ref)
        consistent = ((d_pred > 0) & (d_ref > 0)) | ((d_pred < 0) & (d_ref < 0))
        atc = float(np.mean(consistent))
    else:
        atc = 0.0
    return ctc, atc
