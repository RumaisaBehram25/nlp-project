(() => {
  if (!document.querySelector('#p-table') || !document.querySelector('#r-sum')) {
    if (!/[?&]v=/.test(location.search)) { location.replace(location.pathname + '?v=' + Date.now()); return; }
  }
  const $ = (s, r = document) => r.querySelector(s);
  const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const f2 = v => (v == null ? '–' : v.toFixed(2));
  const f3 = v => (v == null ? '–' : v.toFixed(3));

  const MODELS = [
    { id: 'Transformer-sbert', name: 'Transformer + SBERT', color: 'var(--s1)', desc: 'reads the meaning of each message' },
    { id: 'Transformer-id', name: 'Transformer + IDs', color: 'var(--s2)', desc: 'same transformer, event IDs only' },
    { id: 'TFIDF-text', name: 'TF-IDF (text)', color: 'var(--s3)', desc: 'word counts + logistic regression' },
    { id: 'TFIDF-id', name: 'TF-IDF (IDs)', color: 'var(--s4)', desc: 'event counts + logistic regression' },
    { id: 'DeepLog', name: 'DeepLog LSTM', color: 'var(--s5)', desc: 'predicts the next event' },
  ];
  const M = Object.fromEntries(MODELS.map(m => [m.id, m]));
  const ICON_OK = '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12.5l4.5 4.5L19 7.5"/></svg>';
  const ICON_BAD = '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M6 6l12 12M18 6L6 18"/></svg>';

  const st = {
    fmt: 'hdfs', train: 'HDFS', log: null, labels: null, picked: new Set(['Transformer-sbert', 'Transformer-id', 'TFIDF-text', 'TFIDF-id', 'DeepLog']),
    status: null, res: null, view: 'all', q: '', page: 0, open: null,
  };
  const PAGE = 50;

  /* ---------- helpers ---------- */
  function segmented(el, opts, init, cb) {
    let cur = init;
    el.classList.add('seg');
    const draw = () => { el.innerHTML = opts.map(o => `<button type="button" data-v="${o.v}" aria-pressed="${o.v === cur}">${o.l}</button>`).join(''); };
    el.addEventListener('click', e => {
      const b = e.target.closest('button');
      if (!b) return;
      cur = b.dataset.v; draw(); cb(cur);
    });
    el.set = v => { cur = v; draw(); };
    draw();
    return el;
  }
  function banner(kind, label, html) {
    const b = $('#banner');
    b.hidden = !html;
    b.className = 'banner ' + kind;
    const icon = kind === 'bad' ? ICON_BAD : ICON_OK;
    b.innerHTML = `${icon}<div><span class="lbl">${label}</span> ${html}</div>`;
  }
  const fmtSize = n => n > 1e6 ? (n / 1e6).toFixed(1) + ' MB' : Math.max(1, Math.round(n / 1e3)) + ' KB';

  /* ---------- data inputs ---------- */
  function setLog(name, text) {
    st.log = { name, text };
    const lines = text.split('\n').length;
    $('#name-log').textContent = `${name} · ${fmtSize(text.length)} · ${lines.toLocaleString()} lines`;
    $('#drop-log').classList.add('has');
  }
  function setLabels(name, text) {
    st.labels = text ? { name, text } : null;
    $('#name-labels').textContent = text ? `${name} · ${text.split('\n').length - 1} labels` : 'CSV with BlockId,Label (Normal or Anomaly)';
    $('#drop-labels').classList.toggle('has', !!text);
  }
  function readFile(file, cb) {
    if (file.size > 150e6) { banner('bad', 'File too large.', 'The demo accepts files up to about 150 MB.'); return; }
    const r = new FileReader();
    r.onload = () => cb(file.name, String(r.result));
    r.readAsText(file);
  }
  function wireDrop(zone, input, cb) {
    input.addEventListener('change', () => input.files[0] && readFile(input.files[0], cb));
    ['dragenter', 'dragover'].forEach(ev => zone.addEventListener(ev, e => { e.preventDefault(); zone.classList.add('over'); }));
    ['dragleave', 'drop'].forEach(ev => zone.addEventListener(ev, e => { e.preventDefault(); zone.classList.remove('over'); }));
    zone.addEventListener('drop', e => { const f = e.dataTransfer.files[0]; if (f) readFile(f, cb); });
  }
  wireDrop($('#drop-log'), $('#in-log'), setLog);
  wireDrop($('#drop-labels'), $('#in-labels'), setLabels);

  async function loadSample(kind) {
    try {
      if (kind === 'hdfs') {
        const [log, lab] = await Promise.all([fetch('samples/hdfs_sample.log').then(r => r.text()), fetch('samples/hdfs_sample_labels.csv').then(r => r.text())]);
        segFmt.set('hdfs'); onFmt('hdfs');
        setLog('hdfs_sample.log', log); setLabels('hdfs_sample_labels.csv', lab);
      } else {
        const log = await fetch('samples/bgl_sample.log').then(r => r.text());
        segFmt.set('bgl'); onFmt('bgl');
        setLog('bgl_sample.log', log); setLabels('', '');
      }
      banner('', '', '');
    } catch (e) {
      banner('bad', 'Could not load the sample.', 'Open this page through the server, not as a file.');
    }
  }
  $('#s-hdfs').addEventListener('click', () => loadSample('hdfs'));
  $('#s-bgl').addEventListener('click', () => loadSample('bgl'));

  function onFmt(v) {
    st.fmt = v;
    $('#f-labels').style.display = v === 'hdfs' ? '' : 'none';
    $('#f-window').style.display = v === 'hdfs' ? 'none' : '';
    $('#hint-labels').textContent = v === 'hdfs'
      ? 'Without a labels file you still get predictions, but no scores.'
      : v === 'bgl' ? 'Labels are read from the first column of each line: "-" means normal, anything else is an alert.'
        : 'Plain text has no labels, so you get predictions only.';
    segTrain.set(v === 'hdfs' ? 'HDFS' : 'BGL'); st.train = v === 'hdfs' ? 'HDFS' : 'BGL';
    renderModels();
  }
  const segFmt = segmented($('#seg-fmt'), [{ v: 'hdfs', l: 'HDFS' }, { v: 'bgl', l: 'BGL' }, { v: 'plain', l: 'Plain text' }], 'hdfs', onFmt);
  const segTrain = segmented($('#seg-train'), [{ v: 'HDFS', l: 'HDFS' }, { v: 'BGL', l: 'BGL' }], 'HDFS', v => { st.train = v; renderModels(); });

  /* ---------- models ---------- */
  function renderModels() {
    const avail = {};
    (st.status ? st.status.models : MODELS.map(m => ({ id: m.id, trained: { HDFS: true, BGL: true } }))).forEach(m => { avail[m.id] = m.trained[st.train]; });
    const sbertMissing = st.status && st.status.sbert === 'missing';
    $('#models').innerHTML = MODELS.map(m => {
      const off = !avail[m.id] || (m.id === 'Transformer-sbert' && sbertMissing);
      const note = !avail[m.id] ? 'model file missing' : (m.id === 'Transformer-sbert' && sbertMissing) ? 'needs sentence-transformers' : m.desc;
      return `<label class="check ${off ? 'off' : ''}"><input type="checkbox" value="${m.id}" ${st.picked.has(m.id) && !off ? 'checked' : ''} ${off ? 'disabled' : ''}>
        <span class="t"><b><i class="dot" style="background:${m.color}"></i>${m.name}</b><span>${note}</span></span></label>`;
    }).join('');
    $('#models').querySelectorAll('input').forEach(i => i.addEventListener('change', () => {
      i.checked ? st.picked.add(i.value) : st.picked.delete(i.value);
    }));
  }

  /* ---------- stress controls ---------- */
  $('#reword').addEventListener('input', e => { $('#v-reword').textContent = e.target.value + '%'; });
  $('#noise-p').addEventListener('input', e => { $('#v-noise').textContent = e.target.value + '%'; });
  $('#noise').addEventListener('change', e => { $('#f-noisep').style.display = e.target.value ? '' : 'none'; });
  $('#f-noisep').style.display = 'none';

  /* ---------- server status ---------- */
  async function checkServer() {
    try {
      const r = await fetch('/api/status', { cache: 'no-store' });
      st.status = await r.json();
      if (!st.status.ok) {
        banner('bad', 'Models cannot run yet.', `The server is up, but Python packages are missing: <code>${esc(st.status.error)}</code>. Install them with <code>pip install torch numpy pandas scikit-learn sentence-transformers</code> and restart the server.`);
      } else if (st.status.sbert === 'missing') {
        banner('bad', 'SBERT is not installed.', 'The other models work. For Transformer + SBERT run <code>pip install sentence-transformers</code> and restart the server.');
      } else {
        banner('ok', 'Server connected.', 'Pick a sample or upload a file to begin.');
      }
    } catch (e) {
      st.status = null;
      banner('bad', 'No server found.', 'This page needs the LogLens server to run the models. Start it with <code>python3 server.py</code> inside the app folder and open <code>http://localhost:8000/try.html</code>.');
    }
    renderModels();
  }

  /* ---------- run ---------- */
  $('#run').addEventListener('click', async () => {
    if (!st.log) { banner('bad', 'No data.', 'Choose a log file or load a sample first.'); return; }
    if (!st.picked.size) { banner('bad', 'No models.', 'Tick at least one model.'); return; }
    const btn = $('#run');
    btn.disabled = true; btn.textContent = 'Running…';
    $('#runhint').textContent = 'The first run is slower while the models load.';
    try {
      const body = {
        format: st.fmt, train: st.train, models: [...st.picked], log: st.log.text,
        labels: st.fmt === 'hdfs' && st.labels ? st.labels.text : null,
        window: +$('#window').value || 20,
        reword: +$('#reword').value / 100,
        noise: $('#noise').value || null, noise_p: +$('#noise-p').value / 100,
      };
      const r = await fetch('/api/run', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
      const j = await r.json();
      if (!r.ok || j.error) throw new Error(j.error || 'Server error');
      st.res = j; st.page = 0; st.open = null; st.view = 'all'; st.q = '';
      banner('', '', '');
      renderResults();
      $('#results').hidden = false;
      $('#results').scrollIntoView({ behavior: 'smooth' });
    } catch (e) {
      banner('bad', 'Run failed.', esc(e.message));
    } finally {
      btn.disabled = false; btn.textContent = 'Run models'; $('#runhint').textContent = '';
    }
  });

  /* ---------- results ---------- */
  function renderResults() {
    const R = st.res, labeled = R.has_labels;
    const unkPct = R.n_lines ? (100 * R.unknown_lines / R.n_lines) : 0;
    $('#r-tiles').innerHTML = `
      <div class="tile"><div class="big">${R.n_groups.toLocaleString()}</div><div class="lab">Sequences</div><div class="cmp">${R.n_lines.toLocaleString()} log lines</div></div>
      <div class="tile"><div class="big">${unkPct.toFixed(0)}%</div><div class="lab">Unknown messages</div><div class="cmp">${R.new_templates} message types the models never saw in training</div></div>
      <div class="tile"><div class="big">${labeled ? R.n_anomalies.toLocaleString() : '–'}</div><div class="lab">True anomalies</div><div class="cmp">${labeled ? `of ${R.n_labeled.toLocaleString()} labeled sequences` : 'no labels given'}</div></div>
      <div class="tile"><div class="big">${R.settings.train}</div><div class="lab">Models trained on</div><div class="cmp">${stressText(R.settings)}</div></div>`;
    $('#r-notes').innerHTML = R.notes.map(n => `<li>${esc(n)}</li>`).join('');

    if (labeled) {
      $('#r-compare').style.display = '';
      const best = Math.max(...R.models.map(m => m.metrics.f1));
      $('#r-sub').textContent = 'Scores are for the anomaly class. F1 balances precision (few false alarms) and recall (few misses).';
      const rows = R.models.slice().sort((a, b) => b.metrics.f1 - a.metrics.f1);
      const top = rows[0], tc = top.confusion;
      $('#r-sum').innerHTML = `<b>${top.name}</b> did best: it caught <b>${tc.tp}</b> of <b>${tc.tp + tc.fn}</b> real anomalies and raised <b>${tc.fp}</b> false alarm${tc.fp === 1 ? '' : 's'}.` +
        (rows.length > 1 ? ` The weakest was <b>${rows[rows.length - 1].name}</b> (F1 ${f2(rows[rows.length - 1].metrics.f1)}).` : '');
      $('#r-bars').innerHTML = rows.map(m => `
        <div class="hrow"><div class="hlabel"><i class="dot" style="background:${M[m.id].color}"></i>${m.name}</div>
        <div class="htrack"><div class="hbar" style="background:${M[m.id].color};width:0"></div></div><div class="hval">${f2(m.metrics.f1)}</div></div>`).join('');
      void $('#r-bars').offsetWidth;
      $('#r-bars').querySelectorAll('.hbar').forEach((b, i) => { b.style.width = (rows[i].metrics.f1 * 100) + '%'; if (rows[i].metrics.f1 > 0) b.style.minWidth = '3px'; });
      $('#r-table').innerHTML = `<thead><tr><th>Model</th><th class="num">F1</th><th class="num">Precision</th><th class="num">Recall</th><th class="num">Accuracy</th><th class="num">AUC</th><th class="num">Flagged</th><th class="num">Threshold</th></tr></thead><tbody>` +
        rows.map(m => `<tr><td><i class="dot" style="background:${M[m.id].color}"></i> ${m.name}${m.metrics.f1 === best ? '<span class="best">best</span>' : ''}</td>
          <td class="num">${f3(m.metrics.f1)}</td><td class="num">${f3(m.metrics.precision)}</td><td class="num">${f3(m.metrics.recall)}</td>
          <td class="num">${f3(m.metrics.accuracy)}</td><td class="num">${f3(m.metrics.auc)}</td><td class="num">${m.flagged}</td><td class="num">${+m.threshold.toPrecision(3)}</td></tr>`).join('') + '</tbody>';
      $('#r-cms').innerHTML = rows.map(m => { const c = m.confusion; return `
        <div class="cm"><h4><i class="dot" style="background:${M[m.id].color}"></i>${m.name}</h4><div class="cmg">
          <div class="good"><b>${c.tp}</b><span>anomalies caught</span></div><div class="badc"><b>${c.fp}</b><span>false alarms</span></div>
          <div class="badc"><b>${c.fn}</b><span>anomalies missed</span></div><div class="good"><b>${c.tn}</b><span>normal, correct</span></div></div></div>`; }).join('');
    } else {
      $('#r-sum').textContent = '';
      $('#r-compare').style.display = 'none';
    }
    renderAgree();
    buildViewSeg();
    renderPred();
  }
  function renderAgree() {
    const R = st.res, n = R.models.length, ids = R.models.map(m => m.id);
    const bins = Array.from({ length: n + 1 }, () => ({ bad: 0, ok: 0, unk: 0 }));
    R.rows.forEach(r => {
      const k = ids.filter(m => r.preds[m] === 1).length;
      bins[k][r.label == null ? 'unk' : r.label ? 'bad' : 'ok']++;
    });
    const mx = Math.max(1, ...bins.map(b => b.bad + b.ok + b.unk));
    const lab = R.has_labels;
    const seg = (c, v, name, k) => v ? `<div class="ag-seg ${c}" style="height:${(100 * v / mx).toFixed(1)}%" title="${v} ${name} flagged by ${k} of ${n} models"></div>` : '';
    $('#r-agree').innerHTML = `<div class="ag-legend">${lab ? '<span><i class="sw bad"></i>Real problems</span><span><i class="sw ok"></i>Normal</span>' : '<span><i class="sw unk"></i>Sequences</span>'}</div>` +
      `<div class="ag-chart">` + bins.map((b, k) => `<div class="ag-col"><div class="ag-num">${b.bad + b.ok + b.unk}</div><div class="ag-stack">${seg('unk', b.unk, 'sequences', k)}${seg('ok', b.ok, 'normal sequences', k)}${seg('bad', b.bad, 'real problems', k)}</div><div class="ag-x">${k}</div></div>`).join('') + `</div><div class="ag-cap">Number of models that flagged the sequence (0 to ${n})</div>`;
  }
  function stressText(s) {
    const p = [];
    if (s.reword > 0) p.push(`${Math.round(s.reword * 100)}% reworded`);
    if (s.noise) p.push(`${Math.round(s.noise_p * 100)}% ${({ drop: 'dropped', dup: 'repeated', reorder: 'swapped' })[s.noise]}`);
    return p.length ? 'stress: ' + p.join(', ') : 'no stress test';
  }

  function buildViewSeg() {
    const opts = [{ v: 'all', l: 'All' }, { v: 'flagged', l: 'Flagged by any model' }];
    if (st.res.has_labels) opts.push({ v: 'anom', l: 'True anomalies' }, { v: 'mistakes', l: 'Mistakes' });
    const el = $('#seg-view');
    el.classList.add('seg');
    const draw = () => { el.innerHTML = opts.map(o => `<button type="button" data-v="${o.v}" aria-pressed="${o.v === st.view}">${o.l}</button>`).join(''); };
    el.onclick = e => {
      const b = e.target.closest('button');
      if (!b) return;
      st.view = b.dataset.v; st.page = 0; st.open = null; draw(); renderPred();
    };
    st.view = 'all';
    draw();
  }

  function filtered() {
    const R = st.res, ids = Object.keys(R.rows[0].preds), q = st.q.trim().toLowerCase();
    return R.rows.filter(r => {
      if (q && !r.key.toLowerCase().includes(q)) return false;
      if (st.view === 'flagged') return ids.some(m => r.preds[m] === 1);
      if (st.view === 'anom') return r.label === 1;
      if (st.view === 'mistakes') return r.label != null && ids.some(m => r.preds[m] !== r.label);
      return true;
    });
  }

  function renderPred() {
    const R = st.res, ids = R.models.map(m => m.id);
    const rows = filtered();
    const pages = Math.max(1, Math.ceil(rows.length / PAGE));
    st.page = Math.min(st.page, pages - 1);
    const slice = rows.slice(st.page * PAGE, (st.page + 1) * PAGE);
    const tip = t => ` title="${t}"`;
    const head = `<thead><tr><th${tip('One group of log lines (an HDFS block or a BGL window)')}>Sequence</th><th class="num"${tip('Number of log lines in this sequence')}>Lines</th><th class="num"${tip('Lines whose message type was not seen in training')}>Unseen</th>${R.has_labels ? `<th${tip('The correct answer from your labels')}>Truth</th>` : ''}<th class="num"${tip('How many models flagged it')}>Votes</th>${ids.map(m => `<th${tip(M[m].name)}><i class="dot" style="background:${M[m].color}"></i> ${M[m].name}</th>`).join('')}</tr></thead>`;
    const body = slice.map(r => {
      const cells = ids.map(m => {
        const p = r.preds[m];
        const verdict = p ? 'flagged' : 'normal';
        let mark = '';
        if (r.label != null) mark = p === r.label ? `<span class="mk good">${ICON_OK}right</span>` : `<span class="mk bad">${ICON_BAD}wrong</span>`;
        const bad = r.label != null && p !== r.label;
        return `<td class="${bad ? 'wrongc' : ''}"><span class="verdict ${p ? 'fl' : ''}">${verdict}</span> ${mark}</td>`;
      }).join('');
      const lab = r.label == null ? '' : `<td><span class="lab-pill ${r.label ? 'anom' : ''}">${r.label ? 'anomaly' : 'normal'}</span></td>`;
      const votes = ids.filter(m => r.preds[m] === 1).length;
      let out = `<tr class="clickable ${r.label === 1 ? 'anomrow' : ''}" data-i="${r.i}"><td>${esc(r.key)}</td><td class="num">${r.n}</td><td class="num">${r.unknown}</td>${lab}<td class="num"><span class="votes">${votes}/${ids.length}</span></td>${cells}</tr>`;
      if (st.open === r.i) out += `<tr class="detail"><td colspan="${4 + (R.has_labels ? 1 : 0) + ids.length}" id="detail">Loading…</td></tr>`;
      return out;
    }).join('');
    $('#p-table').innerHTML = head + `<tbody>${body || '<tr><td colspan="12">Nothing matches this filter.</td></tr>'}</tbody>`;
    $('#pinfo').textContent = `${rows.length.toLocaleString()} sequences · page ${st.page + 1} of ${pages}`;
    $('#prev').disabled = st.page === 0;
    $('#next').disabled = st.page >= pages - 1;
    $('#p-table').querySelectorAll('tr.clickable').forEach(tr => tr.addEventListener('click', () => {
      const i = +tr.dataset.i;
      st.open = st.open === i ? null : i;
      renderPred();
      if (st.open != null) loadDetail(st.open);
    }));
  }

  async function loadDetail(i) {
    const cell = $('#detail');
    try {
      const j = await (await fetch('/api/lines?i=' + i)).json();
      if (j.error) throw new Error(j.error);
      const row = st.res.rows.find(r => r.i === i);
      const sc = st.res.models.map(m => `<span><i class="dot" style="background:${M[m.id].color}"></i>${m.name}: score <b>${+row.scores[m.id].toPrecision(3)}</b> (flag at ${+m.threshold.toPrecision(3)})</span>`).join('');
      cell.innerHTML = `<div class="sc">${sc}</div><div class="dl">` + j.lines.map((l, k) =>
        `<div><span class="k">${k + 1}</span>${l.known ? '' : '<span class="unk">unknown</span>'}<span>${esc(l.text)}</span></div>`).join('') +
        (j.total > j.lines.length ? `<div>… ${j.total - j.lines.length} more lines</div>` : '') + '</div>';
    } catch (e) {
      cell.textContent = 'Could not load the lines: ' + e.message;
    }
  }

  $('#prev').addEventListener('click', () => { st.page--; st.open = null; renderPred(); });
  $('#next').addEventListener('click', () => { st.page++; st.open = null; renderPred(); });
  $('#q').addEventListener('input', e => { st.q = e.target.value; st.page = 0; st.open = null; renderPred(); });
  $('#dl').addEventListener('click', () => {
    const R = st.res, ids = R.models.map(m => m.id);
    const head = ['sequence', 'events', 'unknown_lines', 'true_label'].concat(ids.flatMap(m => [m + '_score', m + '_flagged']));
    const lines = [head.join(',')].concat(R.rows.map(r => [`"${r.key}"`, r.n, r.unknown, r.label ?? ''].concat(ids.flatMap(m => [r.scores[m], r.preds[m]])).join(',')));
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([lines.join('\n')], { type: 'text/csv' }));
    a.download = 'predictions.csv'; a.click(); URL.revokeObjectURL(a.href);
  });

  /* ---------- theme ---------- */
  const root = document.documentElement;
  try { const s = localStorage.getItem('loglens-theme'); if (s) root.dataset.theme = s; } catch (e) { /* storage blocked */ }
  $('#theme').addEventListener('click', () => {
    const dark = root.dataset.theme ? root.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
    root.dataset.theme = dark ? 'light' : 'dark';
    try { localStorage.setItem('loglens-theme', root.dataset.theme); } catch (e) { /* storage blocked */ }
  });

  onFmt('hdfs');
  checkServer();
})();
