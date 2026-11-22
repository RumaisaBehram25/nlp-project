"""Small web server for the LogLens demo.

Run:  python3 server.py            (then open http://localhost:8000)
Needs: torch, numpy, pandas, scikit-learn  (and sentence-transformers for the SBERT model)
"""
import argparse
import json
import math
import os
import random
import sys
import threading
import time
import traceback
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, 'models')
sys.path.insert(0, HERE)
import logparse as lp  # noqa: E402

MODEL_LIST = [
    ('Transformer-sbert', 'transformer_sbert', 'Transformer + SBERT'),
    ('Transformer-id', 'transformer_id', 'Transformer + IDs'),
    ('TFIDF-text', 'tfidf_text', 'TF-IDF (text)'),
    ('TFIDF-id', 'tfidf_id', 'TF-IDF (IDs)'),
    ('DeepLog', 'deeplog', 'DeepLog LSTM'),
]
FILES = {m[0]: m[1] for m in MODEL_LIST}
NAMES = {m[0]: m[2] for m in MODEL_LIST}

LOCK = threading.Lock()
S = dict(common=None, error=None, models={}, sbert=None, sbert_error=None, emb_cache={}, last=None)
MAX_BODY = 300 * 1024 * 1024


def ensure_backend():
    if S['common'] is not None or S['error']:
        return
    try:
        import numpy  # noqa: F401
        import torch  # noqa: F401
        import common
        S['common'] = common
    except Exception as e:  # missing packages
        S['error'] = f'{type(e).__name__}: {e}'


def get_model(mid, train):
    key = (mid, train)
    if key not in S['models']:
        import torch
        path = os.path.join(MODELS_DIR, f'{FILES[mid]}_{train}.pt')
        if not os.path.exists(path):
            raise FileNotFoundError(f'model file missing: models/{FILES[mid]}_{train}.pt')
        S['models'][key] = torch.load(path, weights_only=False, map_location='cpu')
    return S['models'][key]


def get_sbert():
    if S['sbert'] is None:
        if S['sbert_error']:
            raise RuntimeError(S['sbert_error'])
        try:
            from sentence_transformers import SentenceTransformer
            S['sbert'] = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        except Exception as e:
            S['sbert_error'] = f'SBERT is not available ({type(e).__name__}: {e}). Install sentence-transformers and check the internet connection.'
            raise RuntimeError(S['sbert_error'])
    return S['sbert']


def embed(templates):
    import numpy as np
    common = S['common']
    texts = [common.clean_template(t) for t in templates[2:]]
    missing = [t for t in dict.fromkeys(texts) if t not in S['emb_cache']]
    if missing:
        vecs = get_sbert().encode(missing, batch_size=64, normalize_embeddings=True)
        for t, v in zip(missing, vecs):
            S['emb_cache'][t] = v.astype('float32')
    E = np.zeros((len(templates), 384), dtype='float32')
    for i, t in enumerate(texts):
        E[i + 2] = S['emb_cache'][t]
    return E


def status():
    ensure_backend()
    models = []
    for mid, f, name in MODEL_LIST:
        models.append(dict(id=mid, name=name, trained={
            ds: os.path.exists(os.path.join(MODELS_DIR, f'{f}_{ds}.pt')) for ds in ('HDFS', 'BGL')}))
    try:
        import sentence_transformers  # noqa: F401
        sb = 'ready' if S['sbert'] is not None else 'loads on first use'
    except Exception:
        sb = 'missing'
    return dict(ok=S['error'] is None, error=S['error'], models=models, sbert=sb)


def clean(x):
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x


def run(req):
    ensure_backend()
    if S['error']:
        raise RuntimeError('The model packages are not installed on the server: ' + S['error'])
    fmt = req.get('format', 'hdfs')
    train = req.get('train', 'HDFS')
    model_ids = [m for m in req.get('models', []) if m in FILES]
    if not model_ids:
        raise ValueError('Pick at least one model.')
    if not req.get('log', '').strip():
        raise ValueError('Upload a log file first.')

    groups, notes = lp.parse(fmt, req['log'], req.get('labels') or None, req.get('window', 20))
    if not groups:
        raise ValueError('No sequences could be built from this file. Check that the format matches the file.')

    models = {mid: get_model(mid, train) for mid in model_ids}
    templates0 = models[model_ids[0]].templates
    seqs, templates, stats = lp.to_sequences(groups, templates0)
    n0 = len(templates0)
    ids_before = [list(s) for s in seqs]

    frac = float(req.get('reword', 0) or 0)
    if frac > 0:
        pick = lp.pick_reword(len(templates), frac)
        templates = [lp.reword(t) if pick[i] else t for i, t in enumerate(templates)]
    noise = req.get('noise') or None
    p = float(req.get('noise_p', 0) or 0)
    if noise and p > 0:
        seqs = lp.add_noise(seqs, noise, p, random.Random(42))

    emb = None
    if 'Transformer-sbert' in model_ids:
        emb = embed(templates)
    view = lp.View(seqs, templates, emb)

    labels = [g['label'] for g in groups]
    have = [i for i, y in enumerate(labels) if y is not None]
    out_models, scores_by, preds_by = [], {}, {}
    for mid in model_ids:
        m = models[mid]
        t0 = time.time()
        sc = [float(x) for x in m.score(view)]
        thr = float(m.thr)
        pr = [int(s >= thr) for s in sc]
        scores_by[mid], preds_by[mid] = sc, pr
        entry = dict(id=mid, name=NAMES[mid], threshold=thr, flagged=sum(pr), seconds=round(time.time() - t0, 2))
        if have:
            y = [labels[i] for i in have]
            met, conf = lp.metrics(y, [pr[i] for i in have], [sc[i] for i in have])
            entry['metrics'] = {k: clean(v) for k, v in met.items()}
            entry['confusion'] = conf
        out_models.append(entry)

    rows = []
    for i, g in enumerate(groups):
        rows.append(dict(
            i=i, key=g['key'], n=len(g['lines']), label=g['label'],
            unknown=sum(1 for t in ids_before[i] if t >= n0),
            scores={mid: round(scores_by[mid][i], 4) for mid in model_ids},
            preds={mid: preds_by[mid][i] for mid in model_ids}))

    S['last'] = dict(groups=groups, ids=ids_before, templates0=templates0)
    return dict(
        n_groups=len(groups), n_lines=stats['total'], unknown_lines=stats['unknown'],
        new_templates=stats['new_templates'], has_labels=bool(have),
        n_anomalies=sum(1 for y in labels if y == 1), n_labeled=len(have),
        notes=notes, models=out_models, rows=rows,
        settings=dict(format=fmt, train=train, reword=frac, noise=noise, noise_p=p))


def lines_of(i):
    last = S['last']
    if not last or not (0 <= i < len(last['groups'])):
        raise ValueError('Run an analysis first.')
    g, ids, t0 = last['groups'][i], last['ids'][i], last['templates0']
    out = []
    for (_, raw), tid in zip(g['lines'], ids):
        out.append(dict(text=raw[:300], known=tid < len(t0), template=t0[tid] if tid < len(t0) else None))
    return dict(key=g['key'], label=g['label'], lines=out[:200], total=len(out))


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=HERE, **kw)

    def log_message(self, fmt, *args):
        line = fmt % args
        if '/api/' in line or ' 4' in line or ' 5' in line:
            sys.stderr.write('%s %s\n' % (self.address_string(), line))

    def end_headers(self):
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()

    def send_json(self, obj, code=200):
        body = json.dumps(obj).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == '/api/status':
            return self.send_json(status())
        if u.path == '/api/lines':
            try:
                i = int(parse_qs(u.query).get('i', ['-1'])[0])
                with LOCK:
                    return self.send_json(lines_of(i))
            except Exception as e:
                return self.send_json(dict(error=str(e)), 400)
        low = u.path.lower()
        if low.endswith(('.py', '.pt', '.pyc')) or low.startswith('/models'):
            return self.send_error(403)
        return super().do_GET()

    def do_POST(self):
        if urlparse(self.path).path != '/api/run':
            return self.send_error(404)
        try:
            n = int(self.headers.get('Content-Length', 0))
            if n > MAX_BODY:
                return self.send_json(dict(error='File too large for the demo (limit 300 MB).'), 413)
            req = json.loads(self.rfile.read(n).decode('utf-8', errors='ignore'))
            with LOCK:
                return self.send_json(run(req))
        except (ValueError, FileNotFoundError, RuntimeError) as e:
            return self.send_json(dict(error=str(e)), 400)
        except Exception as e:
            traceback.print_exc()
            return self.send_json(dict(error=f'{type(e).__name__}: {e}'), 500)


def main(port=8000, host='127.0.0.1'):
    ensure_backend()
    if S['error']:
        print('WARNING: model packages missing ->', S['error'])
        print('The dashboard will open, but the Try it page cannot run models until they are installed.')
    srv = ThreadingHTTPServer((host, port), Handler)
    print(f'LogLens running at http://localhost:{port}')
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print('bye')


def start_in_background(port=8000, host='0.0.0.0'):
    ensure_backend()
    srv = ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8000)
    ap.add_argument('--host', default='127.0.0.1')
    a = ap.parse_args()
    main(a.port, a.host)
