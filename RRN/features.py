import contextlib
import json
import os
import sys
import numpy as np
import torch
import openpyxl
from RV.metrics import SIMILARITY_KEYS
def resolve_excel_path(script_dir, cli_excel_path=None):
    if cli_excel_path:
        if os.path.isabs(cli_excel_path):
            return cli_excel_path
        return os.path.join(script_dir, cli_excel_path)
    for candidate in ('human-eval.xlsx', 'LeetCodeDataset.xlsx'):
        candidate_path = os.path.join(script_dir, candidate)
        if os.path.exists(candidate_path):
            return candidate_path
    raise FileNotFoundError('脚本目录下未找到 human-eval.xlsx / LeetCodeDataset.xlsx')

def load_rows_in_file_order(excel_file_path):
    """按 Excel 文件原始行序读入全部样本（不做任何排序）。"""
    wb = openpyxl.load_workbook(excel_file_path)
    ws = wb.active
    headers = [cell.value for cell in ws[1]]
    if 'QualitativeMatrix' not in headers or 'QuantitativeMetric' not in headers:
        raise ValueError('Excel 文件缺少 QualitativeMatrix / QuantitativeMetric 列')
    col = {name: headers.index(name) for name in headers}
    sim_cols = {key: col[key] for key in SIMILARITY_KEYS if key in col}
    rows = []
    for r in range(2, ws.max_row + 1):
        row_cells = ws[r]
        qualitative_matrix_value = row_cells[col['QualitativeMatrix']].value
        if not qualitative_matrix_value:
            continue
        quantitative_metric_value = row_cells[col['QuantitativeMetric']].value
        quantitative_metric = {}
        if quantitative_metric_value:
            for line in str(quantitative_metric_value).strip().split('\n'):
                if ':' in line:
                    (key, value) = line.split(':', 1)
                    try:
                        quantitative_metric[key.strip()] = float(value.strip())
                    except (TypeError, ValueError):
                        pass
        if not quantitative_metric:
            continue
        task_id = row_cells[col['task_id']].value if 'task_id' in col else f'row_{r}'
        sim_values = {}
        skip = False
        for key in SIMILARITY_KEYS:
            if key not in sim_cols:
                sim_values[key] = 0.0
                continue
            try:
                sim_values[key] = float(row_cells[sim_cols[key]].value)
            except (TypeError, ValueError):
                sim_values[key] = 0.0
        if skip:
            continue
        rows.append({'task_id': str(task_id), 'qualitative_matrix': qualitative_matrix_value, 'quantitative_metric': quantitative_metric, 'similarity': sim_values})
    return rows

def extract_features(rows, script_dir, run_root, feature_seed, cache_tag='dataset'):
    """提取 x(8 维 LoopCNN 定性特征) / c(16 维定量指标) / similarity(8 维标签)。

    所有 seed / mode 共享同一份缓存，保证 5 个训练 seed 使用完全相同的上游特征；
    定性矩阵解析失败的样本丢弃并记录 task_id。缓存按数据集区分（cache_tag）。
    """
    cache_path = os.path.join(run_root, f'feature_cache_{cache_tag}_seed{feature_seed}.npz')
    meta_path = os.path.join(run_root, f'feature_cache_{cache_tag}_seed{feature_seed}.json')
    if os.path.exists(cache_path) and os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        if meta.get('n_rows_raw') == len(rows) and meta.get('first_task_id') == rows[0]['task_id'] and (meta.get('last_task_id') == rows[-1]['task_id']):
            data = np.load(cache_path, allow_pickle=False)
            print(f"[特征缓存] 命中: {cache_path}（{data['x'].shape[0]} 样本，丢弃 {len(meta['dropped_task_ids'])} 条）")
            return (data['x'], data['c'], data['similarity'], [str(t) for t in data['task_ids']], meta['dropped_task_ids'])
        print('[特征缓存] 数据集已变化，重新提取...')
    os.makedirs(run_root, exist_ok=True)
    log_path = os.path.join(run_root, 'feature_extraction.log')
    print(f'[特征提取] 共 {len(rows)} 行，LoopCNN 日志写入 {log_path}')
    np.random.seed(feature_seed)
    torch.manual_seed(feature_seed)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    from RRN.LACF.fusion import loop_cnn_fusion_with_fusion_layer, parse_qualitative_matrix_string
    from objective_evaluation.normalization import CrossAttationNormalizer
    normalizer = CrossAttationNormalizer()
    (x_list, c_list, sim_list, task_ids, dropped) = ([], [], [], [], [])
    with open(log_path, 'w', encoding='utf-8') as logf, contextlib.redirect_stdout(logf):
        for (i, row) in enumerate(rows):
            try:
                matrix = parse_qualitative_matrix_string(row['qualitative_matrix'])
                if matrix is None:
                    raise ValueError('定性矩阵解析失败')
                matrix = np.clip(matrix, 1, 5).astype(np.float32)
                loop_out = loop_cnn_fusion_with_fusion_layer(matrix, num_heads=2, kernel_size=3, cnn_model_path=None)
                loop_out = np.asarray(loop_out, dtype=np.float32).reshape(1, -1)
                if loop_out.shape[1] != len(SIMILARITY_KEYS):
                    raise ValueError(f'LoopCNN 输出维度异常: {loop_out.shape}')
                normalized_metrics = normalizer.normalize_all(row['quantitative_metric'])
                c_vec = np.array(list(normalized_metrics.values()), dtype=np.float32).reshape(1, -1)
                sim_vec = np.array([np.clip(float(row['similarity'].get(key, 0.0)), 0.0, 1.0) for key in SIMILARITY_KEYS], dtype=np.float32).reshape(1, -1)
            except Exception as exc:
                dropped.append({'task_id': row['task_id'], 'reason': str(exc)})
                print(f"[丢弃] task_id={row['task_id']} 原因: {exc}")
                continue
            x_list.append(loop_out)
            c_list.append(c_vec)
            sim_list.append(sim_vec)
            task_ids.append(row['task_id'])
            if (i + 1) % 50 == 0:
                print(f'[特征提取] {i + 1}/{len(rows)}')
    x = np.vstack(x_list).astype(np.float32)
    c = np.vstack(c_list).astype(np.float32)
    similarity = np.vstack(sim_list).astype(np.float32)
    task_ids_arr = np.array([str(t) for t in task_ids], dtype=np.str_)
    np.savez(cache_path, x=x, c=c, similarity=similarity, task_ids=task_ids_arr)
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump({'n_rows_raw': len(rows), 'first_task_id': rows[0]['task_id'], 'last_task_id': rows[-1]['task_id'], 'n_kept': int(x.shape[0]), 'dropped_task_ids': dropped, 'feature_seed': feature_seed}, f, ensure_ascii=False, indent=2)
    print(f'[特征提取] 完成: 保留 {x.shape[0]} 条，丢弃 {len(dropped)} 条，缓存至 {cache_path}')
    return (x, c, similarity, task_ids, dropped)
