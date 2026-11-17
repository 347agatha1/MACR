"""Only the full method matching the requested eight metrics."""
from pathlib import Path
import argparse
import json
import statistics
import sys
import uuid
sys.dont_write_bytecode = True
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import torch
from RRN.RD_CVAE.model import ScoringModel
from RRN.RD_CVAE.train import train_one, evaluate

ROOT = Path(__file__).resolve().parent
SEEDS = [1, 5, 21, 42, 3407]
CONFIG = dict(rec=4., kl=4., score=2., alpha=.1, temperature=1., output='sigmoid', readout='original', lr=.001, upstream='cached', semantic_rows='all', family='original_v13_full_rows')
KEYS = ['CTC_CodeXGLUE', 'ATC_CodeXGLUE', 'CTC_edit', 'ATC_edit']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Retrain only the full method, 10 runs')
    parser.add_argument('--extract-features', action='store_true', help='Regenerate upstream features from supplied scoring matrices; verify against historical cache')
    args = parser.parse_args()
    torch.set_num_threads(1)
    output = ROOT / 'outputs' / 'runs' / uuid.uuid4().hex[:8]
    summary = {}
    for ds in ['humanEval', 'leetcode']:
        cache = np.load(ROOT / 'outputs' / 'feature_cache' / f'feature_cache_{ds}_6_30_30_12_12_seed5.npz', allow_pickle=False)
        if args.extract_features:
            from RRN.features import load_rows_in_file_order, resolve_excel_path, extract_features
            sys.path.insert(0, str(ROOT / 'RRN' / 'LACF'))
            rows = load_rows_in_file_order(str(ROOT / 'data' / ('human-eval.xlsx' if ds == 'humanEval' else 'LeetCodeDataset.xlsx')))
            regenerated = extract_features(rows, str(ROOT / 'RRN' / 'LACF'), str(output / 'features'), 5, cache_tag=ds)
            for index, key in enumerate(['x', 'c', 'similarity']):
                np.testing.assert_allclose(regenerated[index], cache[key], rtol=1e-5, atol=1e-6)
            assert regenerated[3] == cache['task_ids'].tolist()
            print(ds, 'upstream features verified')
        data = [cache['x'], cache['c'], cache['similarity'], cache['task_ids'].tolist()]
        runs = []
        for seed in SEEDS:
            if args.train:
                record = train_one(data, 'full', 'reconstruction', seed, CONFIG, output / ds / f'seed{seed}')
            else:
                folder = ROOT / 'outputs' / 'models' / 'fusion' / ds / f'seed{seed}'
                record = json.loads((folder / 'result.json').read_text(encoding='utf-8'))
                model = ScoringModel('full', 'reconstruction', seed, CONFIG)
                model.load_state_dict(torch.load(folder / 'model.pth', map_location='cpu', weights_only=True))
                model.eval()
                norm = record['normalization']
                a = int(len(cache['x']) * .7)
                b = a + int(len(cache['x']) * .15)
                # Preserve float32 arithmetic of the original training path.
                xm = cache['x'][:a].mean(0)
                xs = np.maximum(cache['x'][:a].std(0), 1e-6)
                cm = cache['c'][:a].mean(0)
                cs = np.maximum(cache['c'][:a].std(0), 1e-6)
                x = torch.tensor(((cache['x'] - xm) / xs)[b:], dtype=torch.float32)
                c = torch.tensor(((cache['c'] - cm) / cs)[b:], dtype=torch.float32)
                with torch.no_grad():
                    scores = model(x, c)[0].numpy()
                np.testing.assert_allclose(scores, record['test']['final_score'], rtol=1e-6, atol=1e-5)
                actual = evaluate(scores, cache['similarity'][b:])
                for key in KEYS:
                    assert abs(actual[key] - record['test']['metrics'][key]) < 1e-10
            runs.append(record)
        summary[ds] = {key: {'mean': statistics.mean(r['test']['metrics'][key] for r in runs), 'std': statistics.stdev(r['test']['metrics'][key] for r in runs)} for key in KEYS}
        print(ds, ' | '.join(f"{key}: {summary[ds][key]['mean']:.2f} +/- {summary[ds][key]['std']:.2f}" for key in KEYS))
    if args.train:
        (output / 'summary.json').write_text(json.dumps({'config': CONFIG, 'results': summary}, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Full method only. No ablation variants.')


if __name__ == '__main__':
    main()
