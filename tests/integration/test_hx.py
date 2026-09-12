"""HateXPlain integration: RoT-text explanations vs human rationales.

Needs the external cache (``fetch_external.py``); skips without it. The
committed artifacts hold the RoT weights (3000-post slice fit), the eval
indices and the SHAP reference values — the RoT surrogate is never refit
here and the black box is never retrained.
"""

import os

import numpy as np
import pytest
from _external import load_hx, rebuild_text_explainer

ARTIFACTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")

pytest.importorskip("transformers")
pytest.importorskip("sklearn")


def _offsets(tokenizer, text, max_len=96):
    return tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=max_len)["offset_mapping"]


def _word_scores(imp_row, text, offsets):
    words = text.split()
    bounds, s = [], 0
    for w in words:
        s = text.index(w, s)
        bounds.append((s, s + len(w)))
        s += len(w)
    ws = np.zeros(len(words))
    for j, (a, b) in enumerate(bounds):
        vals = [imp_row[k] for k, (x, y) in enumerate(offsets)
                if y > a and x < b and not (x == 0 and y == 0)]
        ws[j] = float(np.mean(vals)) if vals else 0.0
    return ws


def _wauroc(pairs, rats):
    from sklearn.metrics import roc_auc_score

    aucs, w = [], []
    for (ws, p), r in zip(pairs, rats):
        r = np.asarray(r, dtype=float)
        if len(ws) != len(r):
            continue
        gt = (r > 0.5).astype(int)
        if gt.min() == gt.max():
            continue
        s = ws if p == 1 else -ws
        aucs.append(roc_auc_score(gt, s))
        w.append(len(r))
    aucs, w = np.array(aucs), np.array(w, dtype=float)
    return float(aucs.mean()), float((aucs * w).sum() / w.sum()), len(aucs)


@pytest.fixture(scope="module")
def hx():
    data = load_hx()
    ev = np.load(os.path.join(ARTIFACTS, "hx_eval_idx.npy"))
    exp, _ = rebuild_text_explainer(ARTIFACTS)
    transformers = pytest.importorskip("transformers")
    import ruleofthumb

    full_texts = data["full"]["test"]["texts"]
    emb = ruleofthumb.embed_texts(full_texts, max_length=96, batch_size=32)
    imp = exp.get_explanation(emb.embeddings, mask=emb.attention_mask, sample_chunk=300)
    pred = np.asarray(exp.predict(emb.embeddings, mask=emb.attention_mask, sample_chunk=300))
    tok = transformers.AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    pairs = []
    for j, t in enumerate(full_texts):
        pairs.append((_word_scores(imp[j], t, _offsets(tok, t)), int(pred[j])))
    rats = data["full"]["test"]["rat"]
    ev_pairs = [pairs[i] for i in ev]
    ev_rats = [rats[i] for i in ev]
    return {"data": data, "ev": ev, "pairs": pairs, "rats": rats,
            "ev_pairs": ev_pairs, "ev_rats": ev_rats, "pred": pred}


def test_hx_fidelity(hx):
    ybb = hx["data"]["ybb"]["test"]
    assert float(np.mean(hx["pred"] == ybb)) >= 0.72


def test_hx_wauroc_beats_random(hx):
    mean_auc, _, n = _wauroc(hx["pairs"], hx["rats"])
    assert n >= 1000
    assert mean_auc >= 0.65


def test_hx_shap_parity(hx):
    from sklearn.metrics import roc_auc_score

    sv = np.load(os.path.join(ARTIFACTS, "hx_shap_eval.npy"))
    data = hx["data"]
    texts = [data["full"]["test"]["texts"][i] for i in hx["ev"]]
    voc = data["vec"].vocabulary_
    aucs = []
    for j, t in enumerate(texts):
        row = np.asarray(sv[j]).ravel()
        ws = np.array([row[voc[w.lower()]] if w.lower() in voc else 0.0 for w in t.split()])
        r = np.asarray(hx["ev_rats"][j], dtype=float)
        if len(ws) != len(r):
            continue
        gt = (r > 0.5).astype(int)
        if gt.min() == gt.max():
            continue
        s = ws if int(data["bb"].predict(data["vec"].transform([t]))[0]) == 1 else -ws
        aucs.append(roc_auc_score(gt, s))
    rot_auc, _, _ = _wauroc(hx["ev_pairs"], hx["ev_rats"])
    assert abs(rot_auc - float(np.mean(aucs))) <= 0.08


def test_hx_faithfulness(hx):
    rng = np.random.RandomState(0)
    data = hx["data"]
    n_test = len(data["full"]["test"]["texts"])
    sel = rng.choice(n_test, 200, replace=False)
    vec, bb = data["vec"], data["bb"]
    texts = [data["full"]["test"]["texts"][i] for i in sel]
    X = vec.transform(texts)
    p0 = bb.predict_proba(X)
    c = bb.predict(X)
    for k, dmargin, strict_ins in ((3, 0.02, True), (10, 0.02, True)):
        dr, dn, ir, inn = [], [], [], []
        for ii, i in enumerate(sel):
            ws, p = hx["pairs"][i]
            words = texts[ii].split()
            o = set(np.argsort(-np.abs(ws))[:k])
            sgn = ws if p == 1 else -ws
            so = set(np.argsort(-sgn)[:k])
            ro = set(rng.permutation(len(words))[:k])
            dall = set(range(len(words)))
            for store, kept in ((dr, dall - o), (dn, dall - ro), (ir, so), (inn, ro)):
                txt = " ".join(w for l, w in enumerate(words) if l in kept)
                p1 = bb.predict_proba(vec.transform([txt]))[0, c[ii]]
                store.append(abs(p0[ii, c[ii]] - p1))
        assert float(np.mean(dr)) > float(np.mean(dn)) + dmargin
        if strict_ins:
            assert float(np.mean(ir)) < float(np.mean(inn))


def test_hx_target_slices(hx):
    from collections import Counter

    tgt = [hx["data"]["full"]["test"]["tgt"][i] for i in hx["ev"]]
    ybb = hx["data"]["ybb"]["test"][hx["ev"]]
    pred = hx["pred"][hx["ev"]]
    groups = [g for g, n in Counter(tgt).most_common() if n >= 10]
    assert len(groups) >= 3
    for g in groups:
        m = np.array([t == g for t in tgt])
        fid = float(np.mean(pred[m] == ybb[m]))
        assert 0.0 <= fid <= 1.0


def test_hx_renders(hx):
    import matplotlib

    matplotlib.use("Agg")
    from ruleofthumb import plot

    ws, _ = hx["ev_pairs"][0]
    words = hx["data"]["full"]["test"]["texts"][hx["ev"][0]].split()
    assert plot.text_html(ws, words) is not None
    assert plot.text_matplotlib(ws, words) is not None
