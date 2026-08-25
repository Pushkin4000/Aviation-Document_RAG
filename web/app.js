'use strict';

// Thresholds are a Global Constraint with a single source of truth: GET /health.
// These are fallback values only, used if /health is unreachable — a change to
// .env must flow through to the UI without editing this file.
let clarifyThreshold = 0.34;
let answerThreshold = 0.48;
// Fallback only, used if /health is unreachable. Once /health responds,
// loadHealth() overwrites this with the server's actual refusal_message —
// same single-source-of-truth reason as the thresholds above.
let REFUSAL = 'This information is not available in the provided document(s).';

const $ = (id) => document.getElementById(id);

function placeTicks() {
  $('tick-clarify').style.left = (clarifyThreshold * 100) + '%';
  $('tick-answer').style.left = (answerThreshold * 100) + '%';
}

async function loadHealth() {
  const el = $('health');
  try {
    const r = await fetch('/health');
    const body = await r.json();
    el.dataset.state = body.status;
    const vectors = body.index.vector_loaded ? 'VECTOR+LEXICAL' : 'LEXICAL ONLY';
    el.textContent = `${body.status.toUpperCase()} · ${vectors} · ${body.generation_path.toUpperCase()}`;
    el.title = body.index.degraded_reason || '';

    if (body.thresholds) {
      if (typeof body.thresholds.clarify === 'number') clarifyThreshold = body.thresholds.clarify;
      if (typeof body.thresholds.answer === 'number') answerThreshold = body.thresholds.answer;
    }
    if (typeof body.refusal_message === 'string' && body.refusal_message) {
      REFUSAL = body.refusal_message;
    }
  } catch (err) {
    el.dataset.state = 'degraded';
    el.textContent = 'UNREACHABLE';
  } finally {
    placeTicks();
  }
}

// Motion exists only to report request state.
let stageTimers = [];
function runStages() {
  clearStages();
  const stages = ['retrieve', 'route', 'generate', 'verify'];
  stages.forEach((name, i) => {
    stageTimers.push(setTimeout(() => {
      stages.slice(0, i).forEach((prev) => {
        document.querySelector(`[data-stage="${prev}"]`).dataset.done = 'true';
        document.querySelector(`[data-stage="${prev}"]`).dataset.active = 'false';
      });
      const el = document.querySelector(`[data-stage="${name}"]`);
      if (el) el.dataset.active = 'true';
    }, i * 200));
  });
}

function clearStages() {
  stageTimers.forEach(clearTimeout);
  stageTimers = [];
  document.querySelectorAll('.stage').forEach((el) => {
    el.dataset.active = 'false';
    el.dataset.done = 'false';
  });
}

function renderChunks(chunks, citations) {
  const host = $('chunks');
  host.textContent = '';
  if (!chunks || !chunks.length) {
    const empty = document.createElement('div');
    empty.className = 'chunk-snippet';
    empty.textContent = 'No chunks retrieved.';
    host.appendChild(empty);
    return;
  }
  const cited = new Set(citations || []);
  chunks.forEach((c) => {
    const div = document.createElement('div');
    div.className = 'chunk';
    div.dataset.cited = cited.has(`${c.source} (Page ${c.page})`) ? 'true' : 'false';

    const id = document.createElement('div');
    id.className = 'chunk-id';
    id.textContent = c.chunk_id;

    const score = document.createElement('div');
    score.className = 'chunk-score';
    score.textContent = `score ${c.score.toFixed(3)} · overlap ${(c.lexical_overlap ?? 0).toFixed(3)}`;

    const snippet = document.createElement('div');
    snippet.className = 'chunk-snippet';
    snippet.textContent = (c.content_snippet || '').slice(0, 190);

    div.append(id, score, snippet);
    host.appendChild(div);
  });
}

function renderCitations(citations) {
  const host = $('citations');
  host.textContent = '';
  if (!citations || !citations.length) return;
  const h = document.createElement('h2');
  h.textContent = 'Sources';
  const ul = document.createElement('ul');
  citations.forEach((c) => {
    const li = document.createElement('li');
    li.textContent = c;
    ul.appendChild(li);
  });
  host.append(h, ul);
}

async function ask(question) {
  const button = $('submit');
  button.disabled = true;
  runStages();
  const started = performance.now();

  try {
    const res = await fetch('/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, debug: true }),
    });
    const elapsed = Math.round(performance.now() - started);

    if (!res.ok) {
      const detail = await res.text();
      throw new Error(`${res.status}: ${detail.slice(0, 200)}`);
    }
    const body = await res.json();
    clearStages();

    const answerEl = $('answer');
    answerEl.textContent = body.answer;
    answerEl.dataset.refusal = body.answer === REFUSAL ? 'true' : 'false';

    if (body.follow_up_question) {
      const hint = document.createElement('span');
      hint.style.color = 'var(--muted)';
      hint.style.fontSize = '15px';
      hint.textContent = ' ' + body.follow_up_question;
      answerEl.appendChild(hint);
    }

    $('m-route').textContent = (body.route || '—').toUpperCase();
    $('m-decision').textContent = (body.decision || '—').toUpperCase();
    $('m-latency').textContent = elapsed + ' ms';

    const confidence = body.confidence ?? 0;
    $('gauge-fill').style.width = (confidence * 100) + '%';
    $('gauge-value').textContent = confidence.toFixed(3);

    renderCitations(body.citations);
    renderChunks(body.retrieved_chunks, body.citations);
  } catch (err) {
    clearStages();
    const answerEl = $('answer');
    answerEl.dataset.refusal = 'false';
    answerEl.textContent = 'Request failed: ' + err.message;
  } finally {
    button.disabled = false;
  }
}

$('form').addEventListener('submit', (event) => {
  event.preventDefault();
  const question = $('question').value.trim();
  if (question) ask(question);
});

placeTicks();
loadHealth();
