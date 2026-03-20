/* Generate page — checkpoint load + text generation */

async function loadCheckpoints() {
  const res = await fetch('/api/checkpoints');
  const files = await res.json();
  const sel = document.getElementById('ckpt-select');
  sel.innerHTML = '';
  if (files.length === 0) {
    sel.innerHTML = '<option value="">No checkpoints found</option>';
    return;
  }
  files.forEach(f => {
    const opt = document.createElement('option');
    opt.value = f;
    opt.textContent = f;
    sel.appendChild(opt);
  });
}

async function loadCheckpoint() {
  const filename = document.getElementById('ckpt-select').value;
  if (!filename) return;

  document.getElementById('load-status').textContent = 'Loading…';
  document.getElementById('btn-load').disabled = true;

  const res = await fetch('/api/checkpoint/load', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ filename }),
  });

  const data = await res.json();
  document.getElementById('btn-load').disabled = false;

  if (data.error) {
    document.getElementById('load-status').textContent = 'Error: ' + data.error;
    return;
  }

  document.getElementById('load-status').textContent =
    `Loaded — epoch ${data.epoch}, step ${data.step}`;

  // Populate model info
  const cfg = data.gpt_config;
  document.getElementById('info-params').textContent =
    (data.params / 1e6).toFixed(2) + 'M';
  document.getElementById('info-layers').textContent = cfg.n_layers;
  document.getElementById('info-heads').textContent = cfg.n_heads;
  document.getElementById('info-emb').textContent = cfg.emb_dim;

  document.getElementById('btn-generate').disabled = false;
}

async function generateText() {
  const payload = {
    prompt: document.getElementById('prompt').value,
    max_new_tokens: parseInt(document.getElementById('max_new_tokens').value),
    temperature: parseFloat(document.getElementById('temperature').value),
    top_k: parseInt(document.getElementById('top_k').value),
  };

  document.getElementById('gen-spinner').classList.remove('d-none');
  document.getElementById('btn-generate').disabled = true;
  document.getElementById('gen-output').textContent = '…';

  const res = await fetch('/api/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  document.getElementById('gen-spinner').classList.add('d-none');
  document.getElementById('btn-generate').disabled = false;

  const data = await res.json();
  if (data.error) {
    document.getElementById('gen-output').textContent = 'Error: ' + data.error;
    return;
  }
  document.getElementById('gen-output').textContent = data.text;
}

window.addEventListener('DOMContentLoaded', loadCheckpoints);
