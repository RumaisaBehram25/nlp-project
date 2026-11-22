import os, re, glob, copy, pickle, shutil, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (precision_recall_fscore_support, accuracy_score,
                             average_precision_score, roc_auc_score)

ROOT = '/content/drive/MyDrive/nlp_log_anomaly'
RAW = f'{ROOT}/data/raw'
PROC = f'{ROOT}/data/processed'
MODELS = f'{ROOT}/models'
RESULTS = f'{ROOT}/results'
LOCAL = '/content/raw'
MAX_LEN = 100
SEED = 42
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

MODEL_FILES = {
    'TFIDF-id': 'tfidf_id',
    'TFIDF-text': 'tfidf_text',
    'DeepLog': 'deeplog',
    'Transformer-id': 'transformer_id',
    'Transformer-sbert': 'transformer_sbert',
}


def set_seed(s=SEED):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


# ---------- raw files ----------

def ensure_raw():
    os.makedirs(LOCAL, exist_ok=True)
    for f in glob.glob(f'{RAW}/*.zip') + glob.glob(f'{RAW}/*.tar.gz'):
        stem = os.path.basename(f).split('.')[0]
        dest = f'{LOCAL}/{stem}'
        if not os.path.isdir(dest):
            print('extracting', f)
            shutil.unpack_archive(f, dest)


def find_file(fname):
    for base in (LOCAL, RAW):
        hits = glob.glob(f'{base}/**/{fname}', recursive=True)
        if hits:
            return hits[0]
    raise FileNotFoundError(f'{fname} not found in {LOCAL} or {RAW}')


# ---------- processed data ----------

def save_ds(name, d):
    os.makedirs(PROC, exist_ok=True)
    with open(f'{PROC}/{name}.pkl', 'wb') as f:
        pickle.dump(d, f)


def load_ds(name):
    with open(f'{PROC}/{name}.pkl', 'rb') as f:
        return pickle.load(f)


def build_ds(templates, seqs, labels, split):
    return dict(templates=['<pad>', '<unk>'] + list(templates), seqs=seqs,
                labels=np.array(labels, dtype=np.int64), split=np.array(split))


class View:
    def __init__(self, seqs, templates, emb=None):
        self.seqs, self.templates, self.emb = seqs, templates, emb


def make_view(ds, split, reword=0.0, noise=None, p=0.0, seed=SEED):
    idx = np.where(ds['split'] == split)[0]
    seqs = [ds['seqs'][i] for i in idx]
    y = ds['labels'][idx]
    templates, emb = ds['templates'], ds.get('emb')
    if reword > 0:
        rng = np.random.RandomState(seed)
        pick = rng.rand(len(templates)) < reword
        pick[:2] = False
        templates = [ds['templates_rw'][i] if pick[i] else templates[i] for i in range(len(templates))]
        if emb is not None:
            emb = np.where(pick[:, None], ds['emb_rw'], emb)
    if noise:
        seqs = add_noise(seqs, noise, p, np.random.RandomState(seed))
    return View(seqs, templates, emb), y


def pad(seqs, max_len=MAX_LEN):
    L = min(max_len, max(len(s) for s in seqs))
    out = np.zeros((len(seqs), L), dtype=np.int64)
    for i, s in enumerate(seqs):
        s = s[:L]
        out[i, :len(s)] = s
    return out


def remap(seqs, src_t, dst_t):
    lut = {t: i for i, t in enumerate(dst_t) if i >= 2}
    m = np.array([lut.get(t, 1) if i >= 2 else i for i, t in enumerate(src_t)])
    return [m[np.array(s)].tolist() for s in seqs]


# ---------- noise and rewording ----------

def add_noise(seqs, kind, p, rng):
    out = []
    for s in seqs:
        s = list(s)
        if kind == 'drop':
            k = [e for e in s if rng.rand() >= p]
            s = k if k else s[:1]
        elif kind == 'dup':
            k = []
            for e in s:
                k.append(e)
                if rng.rand() < p:
                    k.append(e)
            s = k
        elif kind == 'reorder':
            for i in range(len(s) - 1):
                if rng.rand() < p:
                    s[i], s[i + 1] = s[i + 1], s[i]
        out.append(s)
    return out


SYN = {
    'receiving': 'getting', 'received': 'got', 'receive': 'get',
    'block': 'data chunk', 'blocks': 'data chunks',
    'deleting': 'removing', 'deleted': 'removed', 'delete': 'remove',
    'terminating': 'shutting down', 'served': 'delivered', 'serving': 'delivering',
    'exception': 'error', 'error': 'fault', 'failed': 'did not succeed',
    'addstoredblock': 'registered stored block', 'updated': 'refreshed',
    'ask': 'request', 'verification': 'check', 'verified': 'checked',
    'corrected': 'fixed', 'detected': 'found', 'generating': 'creating',
    'instruction': 'command', 'cache': 'buffer', 'parity': 'checksum',
    'interrupt': 'signal', 'connection': 'link', 'unable': 'not able',
    'starting': 'beginning', 'started': 'began', 'finished': 'completed',
    'transmitting': 'sending', 'packet': 'message', 'lost': 'dropped',
}


def reword(t):
    toks = t.split()
    new = [SYN.get(w.lower(), w) for w in toks]
    out = ' '.join(new)
    if out == t:
        out = 'Event: ' + t
    return out


def reword_all(templates):
    return [t if i < 2 else reword(t) for i, t in enumerate(templates)]


def clean_template(t):
    return re.sub(r'\s+', ' ', t.replace('<*>', ' ')).strip()


# ---------- metrics and logging ----------

def metrics(y, pred, score=None):
    p, r, f, _ = precision_recall_fscore_support(y, pred, average='binary', pos_label=1, zero_division=0)
    m = dict(precision=p, recall=r, f1=f, accuracy=accuracy_score(y, pred))
    if score is not None and len(set(y)) > 1:
        m['auc'] = roc_auc_score(y, score)
    return m


def best_threshold(scores, y):
    cand = np.unique(scores)
    if len(cand) > 300:
        cand = np.unique(np.quantile(scores, np.linspace(0, 1, 300)))
    best_f, best_t = -1, cand[0]
    for t in cand:
        f = metrics(y, (scores >= t).astype(int))['f1']
        if f > best_f:
            best_f, best_t = f, t
    return best_t


def evaluate(model, view, y):
    s = model.score(view)
    return metrics(y, (s >= model.thr).astype(int), s)


def log_result(model, train, test, condition, m):
    os.makedirs(RESULTS, exist_ok=True)
    path = f'{RESULTS}/results.csv'
    row = dict(model=model, train=train, test=test, condition=condition)
    row.update({k: round(float(v), 4) for k, v in m.items()})
    new = pd.DataFrame([row])
    if os.path.exists(path):
        old = pd.read_csv(path)
        key = ['model', 'train', 'test', 'condition']
        same = (old[key] == new[key].iloc[0]).all(axis=1)
        new = pd.concat([old[~same], new], ignore_index=True)
    new.to_csv(path, index=False)


def save_model(obj, name):
    os.makedirs(MODELS, exist_ok=True)
    if hasattr(obj, 'net'):
        obj.net.cpu()
    torch.save(obj, f'{MODELS}/{name}.pt')


def load_model(name):
    return torch.load(f'{MODELS}/{name}.pt', weights_only=False)


# ---------- baseline: TF-IDF + logistic regression ----------

class Baseline:
    def __init__(self, kind):
        self.kind = kind

    def _docs(self, view):
        if self.kind == 'id':
            seqs = remap(view.seqs, view.templates, self.templates)
            return [' '.join(f'e{i}' for i in s) for s in seqs]
        txt = [clean_template(t) for t in view.templates]
        return [' '.join(txt[i] for i in s) for s in view.seqs]

    def fit(self, view, y):
        self.templates = view.templates
        if self.kind == 'id':
            self.vec = TfidfVectorizer(token_pattern=r'\S+', ngram_range=(1, 2), sublinear_tf=True)
        else:
            self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=50000)
        X = self.vec.fit_transform(self._docs(view))
        self.clf = LogisticRegression(class_weight='balanced', max_iter=2000)
        self.clf.fit(X, y)

    def score(self, view):
        return self.clf.predict_proba(self.vec.transform(self._docs(view)))[:, 1]

    def set_threshold(self, view, y):
        self.thr = best_threshold(self.score(view), y)


# ---------- DeepLog style LSTM ----------

class LSTMNet(nn.Module):
    def __init__(self, vocab, emb=64, hid=128, layers=2):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hid, layers, batch_first=True)
        self.out = nn.Linear(hid, vocab)

    def forward(self, x):
        o, _ = self.lstm(self.emb(x))
        return self.out(o[:, -1])


def windows(seqs, h):
    X, T, S = [], [], []
    for k, s in enumerate(seqs):
        p = [0] * h + list(s)
        for i in range(len(s)):
            X.append(p[i:i + h])
            T.append(s[i])
            S.append(k)
    return np.array(X, dtype=np.int64), np.array(T, dtype=np.int64), np.array(S)


class DeepLogDetector:
    def __init__(self, h=10, g=9, epochs=10, bs=1024, lr=1e-3):
        self.h, self.g, self.epochs, self.bs, self.lr = h, g, epochs, bs, lr

    def fit(self, view, y):
        set_seed()
        self.templates = view.templates
        normal = [s for s, l in zip(view.seqs, y) if l == 0]
        X, T, _ = windows(normal, self.h)
        self.net = LSTMNet(len(self.templates)).to(DEV)
        dl = DataLoader(TensorDataset(torch.tensor(X), torch.tensor(T)), batch_size=self.bs, shuffle=True)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss()
        for ep in range(self.epochs):
            self.net.train()
            tot = 0
            for xb, tb in dl:
                xb, tb = xb.to(DEV), tb.to(DEV)
                opt.zero_grad()
                loss = loss_fn(self.net(xb), tb)
                loss.backward()
                opt.step()
                tot += loss.item() * len(xb)
            print(f'epoch {ep + 1} loss {tot / len(X):.4f}')

    def score(self, view):
        seqs = remap(view.seqs, view.templates, self.templates)
        X, T, S = windows(seqs, self.h)
        g = min(self.g, len(self.templates))
        self.net.to(DEV).eval()
        miss = []
        with torch.no_grad():
            for i in range(0, len(X), 4096):
                xb = torch.tensor(X[i:i + 4096]).to(DEV)
                tb = torch.tensor(T[i:i + 4096]).to(DEV)
                top = self.net(xb).topk(g, dim=1).indices
                miss.append((~(top == tb[:, None]).any(1)).cpu().numpy())
        miss = np.concatenate(miss).astype(float)
        return np.bincount(S, weights=miss, minlength=len(seqs))

    def set_threshold(self, view, y):
        self.thr = best_threshold(self.score(view), y)


# ---------- BERT style transformer on event sequences ----------

class SeqTransformer(nn.Module):
    def __init__(self, mode, vocab, in_dim=384, d=128, heads=4, layers=2):
        super().__init__()
        self.mode = mode
        self.inp = nn.Embedding(vocab, d, padding_idx=0) if mode == 'id' else nn.Linear(in_dim, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.randn(1, MAX_LEN + 1, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, d * 4, dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Linear(d, 1)

    def forward(self, ids, E=None):
        x = self.inp(ids) if self.mode == 'id' else self.inp(E[ids])
        B = x.size(0)
        x = torch.cat([self.cls.expand(B, -1, -1), x], 1)
        x = x + self.pos[:, :x.size(1)]
        mask = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=ids.device), ids == 0], 1)
        x = self.enc(x, src_key_padding_mask=mask)
        return self.head(x[:, 0]).squeeze(-1)


class TransformerDetector:
    def __init__(self, mode, epochs=10, bs=128, lr=1e-3):
        self.mode, self.epochs, self.bs, self.lr = mode, epochs, bs, lr

    def _prep(self, view):
        if self.mode == 'id':
            ids = pad(remap(view.seqs, view.templates, self.templates))
            return ids, None
        E = torch.tensor(view.emb, dtype=torch.float32, device=DEV)
        return pad(view.seqs), E

    def fit(self, view, y, vview, vy):
        set_seed()
        self.templates = view.templates
        self.net = SeqTransformer(self.mode, len(self.templates)).to(DEV)
        ids, E = self._prep(view)
        dl = DataLoader(TensorDataset(torch.tensor(ids), torch.tensor(y, dtype=torch.float32)),
                        batch_size=self.bs, shuffle=True)
        pw = torch.tensor((y == 0).sum() / max((y == 1).sum(), 1), dtype=torch.float32, device=DEV)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        best, best_state = -1, None
        for ep in range(self.epochs):
            self.net.train()
            tot = 0
            for xb, yb in dl:
                xb, yb = xb.to(DEV), yb.to(DEV)
                opt.zero_grad()
                loss = loss_fn(self.net(xb, E), yb)
                loss.backward()
                opt.step()
                tot += loss.item() * len(xb)
            ap = average_precision_score(vy, self.score(vview))
            print(f'epoch {ep + 1} loss {tot / len(ids):.4f} val AP {ap:.4f}')
            if ap > best:
                best = ap
                best_state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
        self.net.load_state_dict(best_state)

    def score(self, view):
        self.net.to(DEV).eval()
        ids, E = self._prep(view)
        out = []
        with torch.no_grad():
            for i in range(0, len(ids), 512):
                xb = torch.tensor(ids[i:i + 512]).to(DEV)
                out.append(torch.sigmoid(self.net(xb, E)).cpu().numpy())
        return np.concatenate(out)

    def set_threshold(self, view, y):
        self.thr = best_threshold(self.score(view), y)
