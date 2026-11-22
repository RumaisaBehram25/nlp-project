"""Log parsing, template matching, noise and metrics for the demo app.

Only the standard library is used here, so it can be tested without torch.
The masks match the ones used when the models were trained.
"""
import csv
import io
import random
import re

BLK = re.compile(r'blk_-?\d+')
HDFS_HDR = re.compile(r'^\d+ \d+ \d+ \w+ [^:]+: (.*)$')
NODE = re.compile(r'R\d+-M\d+(-[NL]\d+)?(-[A-Z]:J\d+-U\d+|-U\d+-[A-Z])?')
IP = re.compile(r'/?\d+\.\d+\.\d+\.\d+(:\d+)?')

MAX_LINES = 2_000_000
MAX_GROUPS = 20000


def mask_hdfs(c):
    return IP.sub('<*>', BLK.sub('<*>', c))


def mask_bgl(c):
    c = NODE.sub('<*>', c)
    c = re.sub(r'0x[0-9a-fA-F]+', '<*>', c)
    return re.sub(r'\b\d+\b', '<*>', c)


# ---------- parsing into groups ----------

def parse_labels_csv(text):
    out = {}
    for row in csv.reader(io.StringIO(text)):
        if len(row) < 2 or row[0].lower() in ('blockid', 'block_id', 'id'):
            continue
        out[row[0].strip()] = 1 if row[1].strip().lower() in ('anomaly', 'anomalous', '1', 'true') else 0
    return out


def parse(fmt, log_text, labels_text=None, window=20):
    """Return (groups, notes). A group is dict(key, lines=[(masked, raw)], label)."""
    notes = []
    raw_lines = log_text.splitlines()
    if len(raw_lines) > MAX_LINES:
        raw_lines = raw_lines[:MAX_LINES]
        notes.append(f'Only the first {MAX_LINES:,} lines were used.')
    groups = []

    if fmt == 'hdfs':
        by = {}
        skipped = 0
        for line in raw_lines:
            h = HDFS_HDR.match(line)
            blks = list(dict.fromkeys(BLK.findall(line)))
            if not h or not blks:
                skipped += 1
                continue
            masked = mask_hdfs(h.group(1))
            for b in blks:
                by.setdefault(b, []).append((masked, line))
        labels = parse_labels_csv(labels_text) if labels_text else {}
        for b, lines in by.items():
            groups.append(dict(key=b, lines=lines, label=labels.get(b) if labels else None))
        if skipped:
            notes.append(f'{skipped:,} lines did not look like HDFS lines and were skipped.')
        if labels_text and not any(g['label'] is not None for g in groups):
            notes.append('No block id in the labels file matched the log.')
    else:
        masked_lines, line_labels, skipped = [], [], 0
        for line in raw_lines:
            if not line.strip():
                continue
            if fmt == 'bgl':
                p = line.split(None, 9)
                if len(p) < 10:
                    skipped += 1
                    continue
                masked_lines.append((mask_bgl(p[9]), line))
                line_labels.append(0 if p[0] == '-' else 1)
            else:
                masked_lines.append((mask_bgl(line.strip()), line))
                line_labels.append(None)
        if skipped:
            notes.append(f'{skipped:,} lines did not look like BGL lines and were skipped.')
        w = max(2, int(window))
        for k in range(len(masked_lines) // w):
            chunk = masked_lines[k * w:(k + 1) * w]
            ll = line_labels[k * w:(k + 1) * w]
            label = None if ll[0] is None else int(any(ll))
            groups.append(dict(key=f'window {k + 1}', lines=chunk, label=label))
        if len(masked_lines) % w:
            notes.append(f'The last {len(masked_lines) % w} lines did not fill a window and were dropped.')

    if len(groups) > MAX_GROUPS:
        groups = groups[:MAX_GROUPS]
        notes.append(f'Only the first {MAX_GROUPS:,} groups were used.')
    return groups, notes


# ---------- matching messages to the templates the models were trained on ----------

class Matcher:
    def __init__(self, templates):
        self.by_len = {}
        for i, t in enumerate(templates):
            if i < 2:
                continue
            toks = t.split()
            lit = sum(x != '<*>' for x in toks)
            self.by_len.setdefault(len(toks), []).append((lit, i, toks))
        for v in self.by_len.values():
            v.sort(key=lambda x: -x[0])
        self.cache = {}

    def match(self, msg):
        if msg in self.cache:
            return self.cache[msg]
        toks = msg.split()
        found = None
        for _, i, tt in self.by_len.get(len(toks), []):
            if all(a == '<*>' or a == b for a, b in zip(tt, toks)):
                found = i
                break
        self.cache[msg] = found
        return found


class View:
    def __init__(self, seqs, templates, emb=None):
        self.seqs, self.templates, self.emb = seqs, templates, emb


def to_sequences(groups, train_templates):
    """Map every message to a training template id, or to a new id when it is unknown."""
    matcher = Matcher(train_templates)
    templates = list(train_templates)
    new_ids = {}
    seqs, matched_ids = [], []
    total = unknown = 0
    for g in groups:
        ids = []
        for msg, _ in g['lines']:
            total += 1
            i = matcher.match(msg)
            if i is None:
                unknown += 1
                i = new_ids.get(msg)
                if i is None:
                    i = len(templates)
                    templates.append(msg)
                    new_ids[msg] = i
            ids.append(i)
        seqs.append(ids)
        matched_ids.append(ids)
    return seqs, templates, dict(total=total, unknown=unknown, new_templates=len(new_ids))


# ---------- rewording and noise (same rules as the notebook) ----------

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
    out = ' '.join(SYN.get(w.lower(), w) for w in t.split())
    return 'Event: ' + t if out == t else out


def add_noise(seqs, kind, p, rng):
    out = []
    for s in seqs:
        s = list(s)
        if kind == 'drop':
            k = [e for e in s if rng.random() >= p]
            s = k if k else s[:1]
        elif kind == 'dup':
            k = []
            for e in s:
                k.append(e)
                if rng.random() < p:
                    k.append(e)
            s = k
        elif kind == 'reorder':
            for i in range(len(s) - 1):
                if rng.random() < p:
                    s[i], s[i + 1] = s[i + 1], s[i]
        out.append(s)
    return out


def pick_reword(n_templates, frac, seed=42):
    rng = random.Random(seed)
    return [i >= 2 and rng.random() < frac for i in range(n_templates)]


# ---------- metrics (no sklearn needed) ----------

def confusion(y, pred):
    tp = sum(1 for a, b in zip(y, pred) if a == 1 and b == 1)
    fp = sum(1 for a, b in zip(y, pred) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y, pred) if a == 1 and b == 0)
    tn = sum(1 for a, b in zip(y, pred) if a == 0 and b == 0)
    return dict(tp=tp, fp=fp, fn=fn, tn=tn)


def auc(y, scores):
    pos = [s for a, s in zip(y, scores) if a == 1]
    neg = [s for a, s in zip(y, scores) if a == 0]
    if not pos or not neg:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    rank_sum = sum(ranks[i] for i, a in enumerate(y) if a == 1)
    return (rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def metrics(y, pred, scores):
    c = confusion(y, pred)
    p = c['tp'] / (c['tp'] + c['fp']) if c['tp'] + c['fp'] else 0.0
    r = c['tp'] / (c['tp'] + c['fn']) if c['tp'] + c['fn'] else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return dict(precision=p, recall=r, f1=f, accuracy=(c['tp'] + c['tn']) / len(y), auc=auc(y, scores)), c
