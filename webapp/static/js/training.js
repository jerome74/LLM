/* Training console — SSE consumer + Chart.js + form handling */

// ── Chart setup ──────────────────────────────────────────────────────────────

const ctx = document.getElementById('loss-chart').getContext('2d');
const lossChart = new Chart(ctx, {
  type: 'line',
  data: {
    labels: [],
    datasets: [
      {
        label: 'Train Loss',
        data: [],
        borderColor: '#60a5fa',
        backgroundColor: 'rgba(96,165,250,0.1)',
        tension: 0.3,
        pointRadius: 3,
      },
      {
        label: 'Val Loss',
        data: [],
        borderColor: '#34d399',
        backgroundColor: 'rgba(52,211,153,0.1)',
        tension: 0.3,
        pointRadius: 3,
      },
    ],
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    scales: {
      x: { ticks: { color: '#9ca3af' }, grid: { color: '#374151' } },
      y: { ticks: { color: '#9ca3af' }, grid: { color: '#374151' } },
    },
    plugins: {
      legend: { labels: { color: '#e5e7eb' } },
    },
  },
});

// ── State ────────────────────────────────────────────────────────────────────

let eventSource = null;
let uploadedText = null;
let uploadedFilename = null;

// ── Corpus UI ────────────────────────────────────────────────────────────────

async function loadBuiltinList() {
  const res = await fetch('/api/corpus/builtin');
  const files = await res.json();
  const sel = document.getElementById('builtin_file');
  sel.innerHTML = '';
  if (files.length === 0) {
    sel.innerHTML = '<option value="">No built-in files found</option>';
    document.getElementById('btn-start').disabled = true;
    return;
  }
  files.forEach(f => {
    const opt = document.createElement('option');
    opt.value = f;
    opt.textContent = f;
    sel.appendChild(opt);
  });
  sel.addEventListener('change', () => fetchBuiltinStats());
  fetchBuiltinStats();
}

async function fetchBuiltinStats() {
  const fname = document.getElementById('builtin_file').value;
  if (!fname) return;
  // We need the text to get stats — fetch it via upload-style read on the server
  // Instead, call stats with dummy text length approximation or fetch the file
  // For built-in files, fetch text stats directly from server by reading file
  const res = await fetch('/api/corpus/stats', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      filename: fname,
      context_length: parseInt(document.getElementById('context_length').value),
      train_ratio: parseFloat(document.getElementById('train_ratio').value),
    }),
  });
  if (!res.ok) return;
  // Server-side stats for builtin files
  const data = await res.json();
  showStats(data);
  enableStartIfReady();
}

function toggleCorpusUI() {
  const isBuiltin = document.getElementById('src_builtin').checked;
  document.getElementById('builtin-ui').classList.toggle('d-none', !isBuiltin);
  document.getElementById('upload-ui').classList.toggle('d-none', isBuiltin);
  enableStartIfReady();
}

async function handleUpload(input) {
  const file = input.files[0];
  if (!file) return;

  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch('/api/corpus/upload', { method: 'POST', body: formData });
  const data = await res.json();
  if (data.error) { setStatus(data.error, 'danger'); return; }

  uploadedText = data.text;
  uploadedFilename = data.filename;
  document.getElementById('upload-info').textContent =
    `${data.filename} — ${data.length.toLocaleString()} chars`;

  // Get stats
  const statsRes = await fetch('/api/corpus/stats', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      text: uploadedText,
      context_length: parseInt(document.getElementById('context_length').value),
      train_ratio: parseFloat(document.getElementById('train_ratio').value),
    }),
  });
  const stats = await statsRes.json();
  showStats(stats);
  enableStartIfReady();
}

function showStats(data) {
  document.getElementById('corpus-stats').classList.remove('d-none');
  document.getElementById('stat-total').textContent = (data.total_tokens || 0).toLocaleString();
  document.getElementById('stat-train').textContent = (data.train_tokens || 0).toLocaleString();
  document.getElementById('stat-val').textContent = (data.val_tokens || 0).toLocaleString();
}

// ── Validation ───────────────────────────────────────────────────────────────

function validateHeads() {
  const emb = parseInt(document.getElementById('emb_dim').value);
  const heads = parseInt(document.getElementById('n_heads').value);
  const warn = document.getElementById('heads-warning');
  const invalid = emb % heads !== 0;
  warn.classList.toggle('d-none', !invalid);
  return !invalid;
}

function enableStartIfReady() {
  const isBuiltin = document.getElementById('src_builtin').checked;
  const corpusReady = isBuiltin
    ? !!document.getElementById('builtin_file').value
    : !!uploadedText;
  const headsOk = validateHeads();
  document.getElementById('btn-start').disabled = !(corpusReady && headsOk);
}

// ── Training control ─────────────────────────────────────────────────────────

async function startTraining() {
  if (eventSource) { eventSource.close(); eventSource = null; }

  // Reset chart
  lossChart.data.labels = [];
  lossChart.data.datasets[0].data = [];
  lossChart.data.datasets[1].data = [];
  lossChart.update();

  const isBuiltin = document.getElementById('src_builtin').checked;
  const payload = {
    n_layers: parseInt(document.getElementById('n_layers').value),
    n_heads: parseInt(document.getElementById('n_heads').value),
    emb_dim: parseInt(document.getElementById('emb_dim').value),
    context_length: parseInt(document.getElementById('context_length').value),
    drop_rate: parseFloat(document.getElementById('drop_rate').value),
    num_epochs: parseInt(document.getElementById('num_epochs').value),
    batch_size: parseInt(document.getElementById('batch_size').value),
    eval_freq: parseInt(document.getElementById('eval_freq').value),
    learning_rate: parseFloat(document.getElementById('learning_rate').value),
    train_ratio: parseFloat(document.getElementById('train_ratio').value),
    start_context: document.getElementById('start_context').value,
    corpus_source: isBuiltin ? 'builtin' : 'upload',
    corpus_filename: isBuiltin ? document.getElementById('builtin_file').value : uploadedFilename,
  };

  if (!isBuiltin && uploadedText) {
    payload.corpus_text = uploadedText;
  }

  setStatus('Starting training…', 'secondary');
  setButtons(true);

  const res = await fetch('/api/train/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  const data = await res.json();
  if (data.error) {
    setStatus('Error: ' + data.error, 'danger');
    setButtons(false);
    return;
  }

  setStatus(`Training started — ${(data.params / 1e6).toFixed(2)}M params on ${data.device}`, 'secondary');

  // Open SSE stream
  eventSource = new EventSource('/api/events');
  eventSource.onmessage = handleEvent;
  eventSource.onerror = () => {
    setStatus('SSE connection lost', 'danger');
    setButtons(false);
  };
}

async function stopTraining() {
  await fetch('/api/train/stop', { method: 'POST' });
  setStatus('Stop requested…', 'secondary');
}

async function saveCheckpoint() {
  const res = await fetch('/api/train/save', { method: 'POST' });
  const data = await res.json();
  if (data.error) {
    setStatus('Save failed: ' + data.error, 'danger');
    return;
  }
  const msg = document.getElementById('save-msg');
  msg.textContent = 'Saved: ' + data.filename;
  msg.classList.remove('d-none');
  setTimeout(() => msg.classList.add('d-none'), 4000);
}

// ── SSE event handler ────────────────────────────────────────────────────────

function handleEvent(e) {
  const msg = JSON.parse(e.data);

  if (msg.type === 'heartbeat') return;

  if (msg.type === 'loss') {
    document.getElementById('m-epoch').textContent = msg.epoch;
    document.getElementById('m-step').textContent = msg.step;
    document.getElementById('m-tloss').textContent = msg.train_loss.toFixed(4);
    document.getElementById('m-vloss').textContent = msg.val_loss.toFixed(4);

    lossChart.data.labels.push(msg.step);
    lossChart.data.datasets[0].data.push(msg.train_loss);
    lossChart.data.datasets[1].data.push(msg.val_loss);
    lossChart.update('none');

    setStatus(
      `Epoch ${msg.epoch} · Step ${msg.step} · Tokens ${msg.tokens_seen.toLocaleString()}`,
      'secondary'
    );
  }

  if (msg.type === 'sample') {
    document.getElementById('sample-text').textContent = msg.text;
  }

  if (msg.type === 'done') {
    setStatus(`Done — ${msg.step} steps, ${msg.tokens_seen.toLocaleString()} tokens seen`, 'success');
    finishTraining();
  }

  if (msg.type === 'stopped') {
    setStatus('Training stopped.', 'secondary');
    finishTraining();
  }

  if (msg.type === 'error') {
    setStatus('Error: ' + msg.message, 'danger');
    finishTraining();
  }
}

function finishTraining() {
  if (eventSource) { eventSource.close(); eventSource = null; }
  setButtons(false);
  document.getElementById('btn-save').disabled = false;
}

// ── UI helpers ───────────────────────────────────────────────────────────────

function setStatus(text, type) {
  const bar = document.getElementById('status-bar');
  bar.className = `alert alert-${type} py-2 small mb-3`;
  bar.textContent = text;
}

function setButtons(running) {
  document.getElementById('btn-start').disabled = running;
  document.getElementById('btn-stop').disabled = !running;
  document.getElementById('btn-save').disabled = running;
}

// ── System resource monitor ───────────────────────────────────────────────────

async function pollResources() {
  try {
    const res = await fetch('/api/system/stats');
    if (!res.ok) return;
    const d = await res.json();

    // CPU
    document.getElementById('res-cpu-pct').textContent = d.cpu_pct.toFixed(1) + '%';
    document.getElementById('res-cpu-bar').style.width = d.cpu_pct + '%';

    // RAM
    document.getElementById('res-ram-val').textContent =
      `${d.ram_used_gb.toFixed(1)} / ${d.ram_total_gb.toFixed(1)} GB  (${d.ram_pct.toFixed(1)}%)`;
    document.getElementById('res-ram-bar').style.width = d.ram_pct + '%';

    // GPU / MPS
    const gpu = d.gpu;
    if (gpu.available) {
      document.getElementById('res-gpu-label').textContent = gpu.type;
      if (gpu.mem_pct !== null) {
        document.getElementById('res-gpu-val').textContent =
          `${gpu.mem_used_mb} / ${gpu.mem_total_mb} MB  (${gpu.mem_pct}%)`;
        document.getElementById('res-gpu-bar').style.width = gpu.mem_pct + '%';
      } else {
        // MPS — no total VRAM exposed
        document.getElementById('res-gpu-val').textContent = `${gpu.mem_used_mb} MB allocated`;
        document.getElementById('res-gpu-bar').style.width = '0%';
      }
      document.getElementById('res-gpu-row').classList.remove('d-none');
    } else {
      document.getElementById('res-gpu-row').classList.add('d-none');
    }

    const now = new Date();
    document.getElementById('res-updated').textContent =
      'updated ' + now.toLocaleTimeString();
  } catch (_) { /* silently ignore network errors */ }
}

// ── Init ─────────────────────────────────────────────────────────────────────

window.addEventListener('DOMContentLoaded', () => {
  loadBuiltinList();
  validateHeads();
  pollResources();
  setInterval(pollResources, 2000);
});
