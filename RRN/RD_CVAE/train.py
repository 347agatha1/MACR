import copy
import json
import numpy as np
import torch
from torch.nn import functional as F
from torch.nn import functional
from .model import ScoringModel
from RV.metrics import SIMILARITY_KEYS, compute_ctc_atc
KEYS = ['CTC_edit', 'ATC_edit', 'CTC_CodeXGLUE', 'ATC_CodeXGLUE']
INDICES = [SIMILARITY_KEYS.index('编辑距离相似度'), SIMILARITY_KEYS.index('CodeXGLUE')]

def evaluate(scores, targets):
    metrics = []
    for index in INDICES:
        metrics.extend(100 * value for value in compute_ctc_atc(scores, targets[:, index]))
    return dict(zip(KEYS, metrics))

def losses(scores, targets):
    normalized = (scores - scores.mean()) / scores.std(unbiased=False).clamp_min(1e-4)
    components = []
    for index in INDICES:
        reference = targets[:, index]
        components.append(functional.binary_cross_entropy_with_logits(
            normalized, (reference > reference.mean()).float()))
        delta = torch.diff(reference)
        valid = delta != 0
        components.append(functional.softplus(-torch.diff(normalized)[valid] * delta[valid].sign()).mean()
                          if valid.any() else scores.sum() * 0)
    return torch.stack(components)

def train_one(data, group, arm, seed, cfg, out):
    assert group == 'full' and arm == 'reconstruction'
    (x, c, y, ids) = data
    torch.manual_seed(seed)
    a = int(len(x) * 0.7)
    b = a + int(len(x) * 0.15)
    xm = x[:a].mean(0)
    xs = np.maximum(x[:a].std(0), 1e-06)
    cm = c[:a].mean(0)
    cs = np.maximum(c[:a].std(0), 1e-06)
    xt = torch.tensor((x - xm) / xs, dtype=torch.float32)
    ct = torch.tensor((c - cm) / cs, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    model = ScoringModel(group, arm, seed, cfg)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.get('lr', 0.001), weight_decay=0.0001)
    best = (-np.inf, -np.inf)
    state = None
    stale = 0
    audit = []
    for epoch in range(300):
        model.train()
        opt.zero_grad()
        (s, recon, mu, lv) = model(xt[:a], ct[:a])
        parts = losses(s, yt[:a])
        ls = cfg['alpha'] * parts.mean() + (1 - cfg['alpha']) * parts.max()
        lr = F.mse_loss(recon, model.rec_target.detach()) if model.has_cvae else s.sum() * 0
        lk = -0.5 * (1 + lv - mu.square() - lv.exp()).mean() if model.has_cvae else s.sum() * 0
        total = cfg['score'] * ls + cfg['rec'] * lr + cfg['kl'] * lk
        assert torch.isfinite(total)
        total.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            v = model(xt[a:b], ct[a:b])[0].numpy()
        metrics = evaluate(v, y[a:b])
        key = (min(metrics.values()), float(np.mean(list(metrics.values()))))
        audit.append({'epoch': epoch + 1, 'total': float(total.detach()), 'components': parts.detach().tolist(), 'score_loss': float(ls.detach()), 'reconstruction': float(lr.detach()), 'kl': float(lk.detach()), 'val': metrics})
        if key > best:
            best = key
            state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            stale = 0
        else:
            stale += 1
        if stale >= 30:
            break
    model.load_state_dict(state)
    model.eval()
    result = {'config': dict(cfg, group=group, arm=arm, seed=seed), 'best_epoch': best_epoch, 'audit': audit, 'normalization': {'input_mean': xm.tolist(), 'input_scale': xs.tolist(), 'condition_mean': cm.tolist(), 'condition_scale': cs.tolist()}}
    for (split, sub) in [('train', slice(0, a)), ('val', slice(a, b)), ('test', slice(b, None))]:
        with torch.no_grad():
            s = model(xt[sub], ct[sub])[0].numpy()
            repeat = model(xt[sub], ct[sub])[0].numpy()
        assert np.array_equal(s, repeat) and np.isfinite(s).all()
        if cfg['output'] != 'raw':
            assert ((s >= 0) & (s <= 100)).all()
        result[split] = {'metrics': evaluate(s, y[sub]), 'final_score': s.tolist(), 'task_ids': ids[sub], 'targets': y[sub][:, INDICES].tolist()}
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / 'model.pth')
    (out / 'result.json').write_text(json.dumps(result, ensure_ascii=False), encoding='utf-8')
    return result
