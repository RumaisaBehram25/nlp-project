(() => {
  const $ = (s, r = document) => r.querySelector(s);
  const R = window.RESULTS;

  const MODELS = [
    { id: 'Transformer-sbert', name: 'Transformer + SBERT', short: 'Transformer + SBERT', color: 'var(--s1)', desc: 'reads the meaning of each message' },
    { id: 'Transformer-id', name: 'Transformer + IDs', short: 'Transformer + IDs', color: 'var(--s2)', desc: 'same transformer, event IDs only' },
    { id: 'TFIDF-text', name: 'TF-IDF (text)', short: 'TF-IDF (text)', color: 'var(--s3)', desc: 'word counts + logistic regression' },
    { id: 'TFIDF-id', name: 'TF-IDF (IDs)', short: 'TF-IDF (IDs)', color: 'var(--s4)', desc: 'event counts + logistic regression' },
    { id: 'DeepLog', name: 'DeepLog LSTM', short: 'DeepLog LSTM', color: 'var(--s5)', desc: 'predicts the next event' },
  ];
  const M = Object.fromEntries(MODELS.map(m => [m.id, m]));

  const index = new Map();
  R.forEach(r => index.set([r.model, r.train, r.test, r.condition].join('|'), r));
  const get = (m, tr, te, c) => index.get([m, tr, te, c].join('|'));
  const val = (m, tr, te, c, k) => { const r = get(m, tr, te, c); return r ? r[k] : null; };
  const f2 = v => (v == null || Number.isNaN(v)) ? '–' : v.toFixed(2);
  const f3 = v => (v == null || Number.isNaN(v)) ? '–' : v.toFixed(3);

  /* ---------- tooltip ---------- */
  const tip = $('#tip');
  function showTip(html, cx, cy) {
    tip.innerHTML = html;
    tip.style.display = 'block';
    const w = tip.offsetWidth, h = tip.offsetHeight;
    let x = cx + 14, y = cy + 14;
    if (x + w > innerWidth - 8) x = cx - w - 14;
    if (y + h > innerHeight - 8) y = cy - h - 14;
    tip.style.left = Math.max(8, x) + 'px';
    tip.style.top = Math.max(8, y) + 'px';
  }
  const hideTip = () => { tip.style.display = 'none'; };

  /* ---------- controls ---------- */
  function segmented(el, opts, init, cb) {
    let cur = init;
    el.classList.add('seg');
    el.setAttribute('role', 'group');
    const draw = () => {
      el.innerHTML = opts.map(o => `<button type="button" data-v="${o.v}" aria-pressed="${o.v === cur}">${o.l}</button>`).join('');
    };
    el.addEventListener('click', e => {
      const b = e.target.closest('button');
      if (!b) return;
      cur = b.dataset.v;
      draw();
      cb(cur);
    });
    draw();
    cb(cur);
  }

  /* ---------- horizontal bars ---------- */
  function hbar(el, rows) {
    rows = rows.filter(x => x.v != null).sort((a, b) => b.v - a.v);
    el.innerHTML = rows.map(x => `
      <div class="hrow">
        <div class="hlabel"><i class="dot" style="background:${x.m.color}"></i>${x.m.name}</div>
        <div class="htrack"><div class="hbar" style="background:${x.m.color}"></div></div>
        <div class="hval">${f2(x.v)}</div>
      </div>`).join('');
    void el.offsetWidth;
    const bars = el.querySelectorAll('.hbar');
    const rowEls = el.querySelectorAll('.hrow');
    rows.forEach((x, i) => {
      bars[i].style.width = (Math.max(x.v, 0) * 100) + '%';
      if (x.v > 0) bars[i].style.minWidth = '3px';
      const r = x.r;
      const html = r
        ? `<b>${x.m.name}</b>
           <div class="tr"><span>F1</span><em>${f3(r.f1)}</em></div>
           <div class="tr"><span>Precision</span><em>${f3(r.precision)}</em></div>
           <div class="tr"><span>Recall</span><em>${f3(r.recall)}</em></div>
           <div class="tr"><span>Accuracy</span><em>${f3(r.accuracy)}</em></div>
           <div class="tr"><span>AUC</span><em>${f3(r.auc)}</em></div>`
        : `<b>${x.m.name}</b>`;
      rowEls[i].addEventListener('pointermove', e => showTip(html, e.clientX, e.clientY));
      rowEls[i].addEventListener('pointerleave', hideTip);
    });
  }

  /* ---------- hero ---------- */
  function hero() {
    const sb = 'Transformer-sbert';
    const rw = val(sb, 'HDFS', 'HDFS', 'reword_1.0', 'f1');
    const rwIds = Math.max(val('Transformer-id', 'HDFS', 'HDFS', 'reword_1.0', 'f1'), val('TFIDF-id', 'HDFS', 'HDFS', 'reword_1.0', 'f1'));
    const cr = val(sb, 'HDFS', 'BGL', 'cross', 'f1');
    const crIds = Math.max(...['Transformer-id', 'TFIDF-id', 'DeepLog'].map(m => val(m, 'HDFS', 'BGL', 'cross', 'f1')));
    const clean = R.filter(r => r.train === r.test && r.condition === 'clean' && !(r.model === 'DeepLog' && r.train === 'HDFS')).map(r => r.f1);
    const dl = val('DeepLog', 'HDFS', 'HDFS', 'clean', 'f1');
    $('#tiles').innerHTML = `
      <div class="tile"><div class="big">${f2(rw)}</div><div class="lab">F1 when every message is reworded</div><div class="cmp">HDFS · ID-based models: ${f2(rwIds)}</div></div>
      <div class="tile"><div class="big">${f2(cr)}</div><div class="lab">F1 on a system it never saw</div><div class="cmp">HDFS → BGL · ID-based models: ${f2(crIds)}</div></div>
      <div class="tile"><div class="big">${f2(Math.min(...clean))}–${f2(Math.max(...clean))}</div><div class="lab">F1 on clean data</div><div class="cmp">all models on both systems, except DeepLog on HDFS (${f2(dl)})</div></div>`;
    $('#scale').textContent = 'The first two numbers are for Transformer + SBERT. HDFS: 50,000 sampled blocks (2.9% anomalous). BGL: first 1M lines cut into windows of 20.';
  }

  /* ---------- chips ---------- */
  $('#model-chips').innerHTML = MODELS.map(m =>
    `<div class="chip"><i class="dot" style="background:${m.color}"></i><b>${m.name}</b><span class="mini">${m.desc}</span></div>`).join('');

  /* ---------- leaderboard ---------- */
  const METRICS = [
    { v: 'f1', l: 'F1' }, { v: 'precision', l: 'Precision' }, { v: 'recall', l: 'Recall' },
    { v: 'accuracy', l: 'Accuracy' }, { v: 'auc', l: 'AUC' },
  ];
  const LNOTE = {
    f1: 'F1 on the anomaly class: one number that balances precision and recall.',
    precision: 'Precision: when the model says anomaly, how often it is right. Low precision means false alarms.',
    recall: 'Recall: of the real anomalies, how many the model found. Low recall means missed problems.',
    accuracy: 'Accuracy looks high for almost every model because anomalies are rare. This is why we judge on the anomaly class.',
    auc: 'AUC: how well the model ranks anomalies above normal cases, before any threshold is chosen. 0.5 is a coin flip.',
  };
  const L = { ds: 'HDFS', metric: 'f1' };
  function renderLeader() {
    hbar($('#leader'), MODELS.map(m => {
      const r = get(m.id, L.ds, L.ds, 'clean');
      return { m, r, v: r ? r[L.metric] : null };
    }));
    $('#leader-note').textContent = LNOTE[L.metric];
  }
  segmented($('#seg-l-ds'), [{ v: 'HDFS', l: 'HDFS' }, { v: 'BGL', l: 'BGL' }], 'HDFS', v => { L.ds = v; renderLeader(); });
  segmented($('#seg-l-metric'), METRICS, 'f1', v => { L.metric = v; renderLeader(); });

  /* ---------- line chart ---------- */
  function lineChart(el, labels, series, o) {
    const W = Math.max(el.clientWidth, 300), H = 360, narrow = W < 620;
    const m = { t: 14, r: narrow ? 18 : 160, b: 50, l: 46 };
    const iw = W - m.l - m.r, ih = H - m.t - m.b, n = labels.length;
    const X = i => m.l + iw * i / (n - 1);
    const Y = v => m.t + ih * (1 - v);
    let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${o.aria}">`;
    [0, .25, .5, .75, 1].forEach(t => {
      s += `<line class="grid" x1="${m.l}" x2="${m.l + iw}" y1="${Y(t)}" y2="${Y(t)}"/>`;
      s += `<text class="tick" x="${m.l - 8}" y="${Y(t) + 4}" text-anchor="end">${t.toFixed(2)}</text>`;
    });
    labels.forEach((l, i) => {
      s += `<text class="tick" x="${X(i)}" y="${m.t + ih + 20}" text-anchor="middle">${l}</text>`;
    });
    s += `<text class="axis-title" x="${m.l + iw / 2}" y="${H - 8}" text-anchor="middle">${o.xTitle}</text>`;
    series.forEach(se => {
      let d = '', pen = false;
      se.values.forEach((v, i) => {
        if (v == null) { pen = false; return; }
        d += `${pen ? 'L' : 'M'}${X(i).toFixed(1)},${Y(v).toFixed(1)} `;
        pen = true;
      });
      s += `<g class="ser" data-m="${se.id}"><path d="${d}" fill="none" style="stroke:${se.color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>`;
      se.values.forEach((v, i) => {
        if (v != null) s += `<circle class="pt" data-i="${i}" cx="${X(i).toFixed(1)}" cy="${Y(v).toFixed(1)}" r="4" style="fill:${se.color};stroke:var(--surface)" stroke-width="2"/>`;
      });
      s += '</g>';
    });
    if (!narrow) {
      const last = series.map(se => {
        let i = se.values.length - 1;
        while (i > 0 && se.values[i] == null) i--;
        return { se, y: Y(se.values[i] ?? 0) };
      }).sort((a, b) => a.y - b.y);
      for (let i = 1; i < last.length; i++) if (last[i].y - last[i - 1].y < 15) last[i].y = last[i - 1].y + 15;
      last.forEach(l => {
        s += `<text class="dlabel" x="${m.l + iw + 12}" y="${l.y + 4}">${l.se.short}</text>`;
      });
    }
    s += `<line class="xh" y1="${m.t}" y2="${m.t + ih}" style="display:none"/>`;
    s += `<rect class="ov" x="${m.l - 10}" y="${m.t - 6}" width="${iw + 20}" height="${ih + 12}" fill="transparent"/>`;
    s += '</svg>';
    el.innerHTML = s;

    const svg = el.querySelector('svg'), xh = svg.querySelector('.xh'), ov = svg.querySelector('.ov');
    ov.addEventListener('pointermove', e => {
      const b = svg.getBoundingClientRect();
      const px = (e.clientX - b.left) * (W / b.width);
      const i = Math.max(0, Math.min(n - 1, Math.round((px - m.l) / iw * (n - 1))));
      xh.setAttribute('x1', X(i));
      xh.setAttribute('x2', X(i));
      xh.style.display = '';
      svg.querySelectorAll('.pt').forEach(c => c.setAttribute('r', +c.dataset.i === i ? 6 : 4));
      const rows = series.filter(se => se.values[i] != null).sort((a, b) => b.values[i] - a.values[i]);
      showTip(`<b>${o.xTitle}: ${labels[i]}</b>` + rows.map(se =>
        `<div class="tr"><span><i class="dot" style="background:${se.color}"></i>${se.name}</span><em>${f3(se.values[i])}</em></div>`).join(''), e.clientX, e.clientY);
    });
    ov.addEventListener('pointerleave', () => {
      xh.style.display = 'none';
      svg.querySelectorAll('.pt').forEach(c => c.setAttribute('r', 4));
      hideTip();
    });
  }

  /* ---------- robustness ---------- */
  const PERT = {
    drop: { l: 'Drop events', xTitle: 'share of events removed', levels: ['0.1', '0.2', '0.3'], key: 'drop', note: 'Each event is removed with the given chance, like a log pipeline that loses lines.' },
    dup: { l: 'Repeat events', xTitle: 'share of events repeated', levels: ['0.1', '0.2', '0.3'], key: 'dup', note: 'Each event may be written twice, like a retry or a noisy logger.' },
    reorder: { l: 'Swap neighbours', xTitle: 'share of neighbours swapped', levels: ['0.1', '0.2', '0.3'], key: 'reorder', note: 'Neighbouring events are swapped, like logs that arrive out of order.' },
    reword: { l: 'Reword messages', xTitle: 'share of messages reworded', levels: ['0.3', '0.6', '1.0'], key: 'reword', note: 'A share of message templates is replaced with reworded text, like after a software update. ID-based models see brand-new events they have never been trained on.' },
  };
  const RB = { ds: 'HDFS', pert: 'drop', metric: 'f1' };
  function robustData() {
    const P = PERT[RB.pert];
    const labels = ['0%', ...P.levels.map(l => Math.round(l * 100) + '%')];
    const series = MODELS.map(m => ({
      id: m.id, name: m.name, short: m.short, color: m.color,
      values: [val(m.id, RB.ds, RB.ds, 'clean', RB.metric), ...P.levels.map(l => val(m.id, RB.ds, RB.ds, `${P.key}_${l}`, RB.metric))],
    }));
    return { P, labels, series };
  }
  function renderRobust() {
    const { P, labels, series } = robustData();
    lineChart($('#r-chart'), labels, series, {
      xTitle: P.xTitle,
      aria: `${RB.metric.toUpperCase()} of five models on ${RB.ds} as logs are changed: ${P.l}`,
    });
    $('#r-legend').innerHTML = MODELS.map(m => `<span data-m="${m.id}"><i style="background:${m.color}"></i>${m.name}</span>`).join('');
    $('#r-legend').querySelectorAll('span').forEach(sp => {
      sp.addEventListener('pointerenter', () => $('#r-chart').querySelectorAll('.ser').forEach(g => { g.style.opacity = g.dataset.m === sp.dataset.m ? 1 : .15; }));
      sp.addEventListener('pointerleave', () => $('#r-chart').querySelectorAll('.ser').forEach(g => { g.style.opacity = 1; }));
    });
    $('#r-table').innerHTML = `<table><thead><tr><th>${P.xTitle}</th>${series.map(s => `<th class="num">${s.name}</th>`).join('')}</tr></thead><tbody>` +
      labels.map((l, i) => `<tr><td>${l}</td>${series.map(s => `<td class="num">${f3(s.values[i])}</td>`).join('')}</tr>`).join('') + '</tbody></table>';
    $('#r-note').textContent = P.note + (RB.metric === 'auc' ? ' AUC ignores the decision threshold, so it shows whether the model can still rank anomalies correctly.' : '');
  }
  segmented($('#seg-r-ds'), [{ v: 'HDFS', l: 'HDFS' }, { v: 'BGL', l: 'BGL' }], 'HDFS', v => { RB.ds = v; renderRobust(); });
  segmented($('#seg-r-pert'), Object.entries(PERT).map(([v, p]) => ({ v, l: p.l })), 'drop', v => { RB.pert = v; renderRobust(); });
  segmented($('#seg-r-metric'), [{ v: 'f1', l: 'F1' }, { v: 'auc', l: 'AUC' }], 'f1', v => { RB.metric = v; renderRobust(); });
  let rt;
  addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(renderRobust, 120); });

  /* ---------- cross-system ---------- */
  const CNOTE = {
    cross_f1: 'The decision threshold is the one chosen on the training system, as it would be in real use.',
    tuned_f1: 'The threshold is re-chosen on the other system\'s validation data. This shows what is left once the threshold problem is removed.',
    auc: 'AUC ignores the threshold. 0.50 means the model cannot tell anomalies from normal cases at all. ID-based models sit at 0.50 because every message from the other system is an unknown event.',
  };
  function renderCross() {
    const mode = C.mode;
    const cond = mode === 'tuned_f1' ? 'cross_tuned' : 'cross';
    const key = mode === 'auc' ? 'auc' : 'f1';
    [['HDFS', 'BGL', '#c1'], ['BGL', 'HDFS', '#c2']].forEach(([a, b, sel]) => {
      hbar($(sel), MODELS.map(m => {
        const r = get(m.id, a, b, cond);
        return { m, r, v: r ? r[key] : null };
      }));
    });
    $('#c-note').textContent = CNOTE[mode];
  }
  const C = { mode: 'cross_f1' };
  segmented($('#seg-c'), [{ v: 'cross_f1', l: 'F1 · trained threshold' }, { v: 'tuned_f1', l: 'F1 · threshold re-tuned' }, { v: 'auc', l: 'AUC' }], 'cross_f1', v => { C.mode = v; renderCross(); });

  /* ---------- similarity ---------- */
  const SIM = [
    { name: 'HDFS', same: 0.824, other: 0.319 },
    { name: 'BGL', same: 0.801, other: 0.203 },
  ];
  $('#sim').innerHTML = SIM.map(s => `
    <div class="srow"><div class="slab">${s.name}</div>
      <div class="sbar"><span>Original vs reworded</span><div class="t"><div class="b" style="width:${s.same * 100}%;background:var(--s1)"></div></div><b>${s.same.toFixed(2)}</b></div>
      <div class="sbar"><span>Two different messages</span><div class="t"><div class="b" style="width:${s.other * 100}%;background:var(--ink-3)"></div></div><b>${s.other.toFixed(2)}</b></div>
    </div>`).join('');

  /* ---------- findings ---------- */
  function findings() {
    const sb = 'Transformer-sbert';
    const v = (m, tr, te, c, k = 'f1') => val(m, tr, te, c, k);
    const othersCross = Math.max(...MODELS.filter(m => m.id !== sb).map(m => v(m.id, 'HDFS', 'BGL', 'cross')));
    $('#f-good').innerHTML = [
      `<b>ID-based models fall apart when messages are reworded.</b> With every message reworded on HDFS, Transformer + IDs drops to F1 ${f2(v('Transformer-id', 'HDFS', 'HDFS', 'reword_1.0'))} and TF-IDF (IDs) to ${f2(v('TFIDF-id', 'HDFS', 'HDFS', 'reword_1.0'))}. Transformer + SBERT keeps ${f2(v(sb, 'HDFS', 'HDFS', 'reword_1.0'))}.`,
      `<b>Only the SBERT transformer carries over to a new system.</b> Trained on HDFS and tested on BGL it reaches F1 ${f2(v(sb, 'HDFS', 'BGL', 'cross'))}. Every other model stays at ${f2(othersCross)} or below.`,
      `<b>On clean data the detectors are strong.</b> Apart from DeepLog on HDFS, every model reaches F1 of 0.86 or more on its own system.`,
    ].map(t => `<li>${t}</li>`).join('');
    $('#f-care').innerHTML = [
      `<b>Plain word counts are just as robust to rewording.</b> TF-IDF (text) keeps F1 ${f2(v('TFIDF-text', 'HDFS', 'HDFS', 'reword_1.0'))} on HDFS and ${f2(v('TFIDF-text', 'BGL', 'BGL', 'reword_1.0'))} on BGL. Our rewording is a synonym list that leaves many words untouched, so word counting still works. Real software updates may change more.`,
      `<b>DeepLog is weak on HDFS</b> (F1 ${f2(v('DeepLog', 'HDFS', 'HDFS', 'clean'))}, recall ${f2(v('DeepLog', 'HDFS', 'HDFS', 'clean', 'recall'))}) but fine on BGL (${f2(v('DeepLog', 'BGL', 'BGL', 'clean'))}). This looks like a setup or tuning issue, so do not read it as a verdict on the method.`,
      `<b>The SBERT transformer is fragile to repeated events on HDFS.</b> With 10% repeats its F1 is ${f2(v(sb, 'HDFS', 'HDFS', 'dup_0.1'))} and AUC ${f2(v(sb, 'HDFS', 'HDFS', 'dup_0.1', 'auc'))}. Swapped neighbours mostly hurt through the chosen threshold (AUC stays ${f2(v(sb, 'HDFS', 'HDFS', 'reorder_0.1', 'auc'))}).`,
      `<b>BGL clean scores are close to a tie.</b> The test set has about 2% anomalies against 34% in training (the split is by time). All results use samples: 50,000 HDFS blocks and the first 1M BGL lines.`,
    ].map(t => `<li>${t}</li>`).join('');
  }

  /* ---------- data table ---------- */
  const COLS = [
    ['model', 'Model', false], ['train', 'Trained on', false], ['test', 'Tested on', false], ['condition', 'Condition', false],
    ['precision', 'Precision', true], ['recall', 'Recall', true], ['f1', 'F1', true], ['accuracy', 'Accuracy', true], ['auc', 'AUC', true],
  ];
  const T = { sortKey: 'model', dir: 1, model: '', train: '', test: '', cond: '' };
  const opt = (id, label, values) => {
    $(id).innerHTML = `<option value="">All</option>` + values.map(x => `<option value="${x}">${label ? label(x) : x}</option>`).join('');
  };
  opt('#f-model', x => (M[x] ? M[x].name : x), MODELS.map(m => m.id));
  opt('#f-train', null, ['HDFS', 'BGL']);
  opt('#f-test', null, ['HDFS', 'BGL']);
  function renderTable() {
    const c = T.cond.trim().toLowerCase();
    let rows = R.filter(r => (!T.model || r.model === T.model) && (!T.train || r.train === T.train) && (!T.test || r.test === T.test) && (!c || r.condition.toLowerCase().includes(c)));
    rows = rows.slice().sort((a, b) => {
      const x = a[T.sortKey], y = b[T.sortKey];
      if (x == null) return 1;
      if (y == null) return -1;
      return (x < y ? -1 : x > y ? 1 : 0) * T.dir;
    });
    $('#tbl').innerHTML = '<thead><tr>' + COLS.map(([k, l, num]) =>
      `<th data-k="${k}" class="${num ? 'num' : ''}">${l}${T.sortKey === k ? (T.dir > 0 ? ' ↑' : ' ↓') : ''}</th>`).join('') + '</tr></thead><tbody>' +
      rows.map(r => '<tr>' + COLS.map(([k, , num]) =>
        `<td class="${num ? 'num' : ''}">${k === 'model' ? (M[r.model] ? M[r.model].name : r.model) : num ? f3(r[k]) : r[k]}</td>`).join('') + '</tr>').join('') + '</tbody>';
    $('#rowcount').textContent = `${rows.length} of ${R.length} results`;
    $('#tbl').querySelectorAll('th').forEach(th => th.addEventListener('click', () => {
      const k = th.dataset.k;
      T.dir = T.sortKey === k ? -T.dir : 1;
      T.sortKey = k;
      renderTable();
    }));
  }
  $('#f-model').addEventListener('change', e => { T.model = e.target.value; renderTable(); });
  $('#f-train').addEventListener('change', e => { T.train = e.target.value; renderTable(); });
  $('#f-test').addEventListener('change', e => { T.test = e.target.value; renderTable(); });
  $('#f-cond').addEventListener('input', e => { T.cond = e.target.value; renderTable(); });
  $('#csv').addEventListener('click', () => {
    const head = ['model', 'train', 'test', 'condition', 'precision', 'recall', 'f1', 'accuracy', 'auc'];
    const csv = [head.join(',')].concat(R.map(r => head.map(k => r[k] ?? '').join(','))).join('\n');
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
    a.download = 'results.csv';
    a.click();
    URL.revokeObjectURL(a.href);
  });

  /* ---------- theme ---------- */
  const root = document.documentElement;
  try {
    const saved = localStorage.getItem('loglens-theme');
    if (saved) root.dataset.theme = saved;
  } catch (e) { /* storage blocked */ }
  $('#theme').addEventListener('click', () => {
    const dark = root.dataset.theme ? root.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
    root.dataset.theme = dark ? 'light' : 'dark';
    try { localStorage.setItem('loglens-theme', root.dataset.theme); } catch (e) { /* storage blocked */ }
  });

  hero();
  findings();
  renderTable();
})();
