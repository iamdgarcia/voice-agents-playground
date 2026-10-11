// Voice Agents Playground — live Silero VAD endpointing visualizer.
// Client-side only: mic → 16kHz mono → silero-vad (transformers.js) → endpointing FSM → UI.
import { AutoModel, PretrainedConfig, Tensor, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1';
env.allowLocalModels = false;

// ---------- config ----------
const WINDOW = 512;            // samples per VAD window (32ms at 16kHz)
const SR = 16000;
const $ = (id) => document.getElementById(id);

const params = {
  threshold: 0.5,   // speech prob gate
  minSpeech: 250,   // ms of speech before segment opens
  hangover: 400,    // ms of silence before segment closes
};

// ---------- state ----------
const fsm = {
  inSpeech: false,
  speechMs: 0,       // accumulated speech ms in current candidate
  silenceMs: 0,      // accumulated silence ms after speech
  segStart: 0,       // ms timestamp of segment open (candidate start)
};
const stats = { speechTotalMs: 0, segments: [], latencies: [] };

// ---------- model ----------
let vadModel = null;
async function loadModel(onProgress) {
  const model = await AutoModel.from_pretrained('onnx-community/silero-vad', {
    config: new PretrainedConfig({ model_type: 'custom' }), // no config.json in repo
    dtype: 'fp32',
    progress_callback: onProgress,
  });
  return model;
}

const srTensor = new Tensor('int64', [BigInt(SR)], []);
let state = new Tensor('float32', new Float32Array(256), [2, 1, 128]);
let inferChain = Promise.resolve(); // serialize inferences

function inferWindow(samples) {
  const run = inferChain.then(async () => {
    const input = new Tensor('float32', samples, [1, WINDOW]);
    const out = await vadModel({ input, sr: srTensor, state });
    state = out.stateN; // carry LSTM state across windows
    const p = Number(out.output.data[0]);
    input.dispose();
    return p;
  });
  inferChain = run.catch(() => {});
  return run;
}

// ---------- audio capture ----------
async function startAudio(onWindow) {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
  });
  const ctx = new AudioContext({ sampleRate: 48000 });
  const src = ctx.createMediaStreamSource(stream);

  let targetRate = SR;
  const node = ctx.createScriptProcessor(4096, 1, 1); // broad-compat capture path
  const inBuf = new Float32Array(0);

  // linear resampler 48k -> 16k, emits 512-sample windows
  const resample = {
    pos: 0.0,
    buf: new Float32Array(0),
    push(chunk) {
      const merged = new Float32Array(this.buf.length + chunk.length);
      merged.set(this.buf); merged.set(chunk, this.buf.length);
      this.buf = merged;
      const ratio = ctx.sampleRate / targetRate;
      const out = [];
      while ((this.pos + 1) / ratio < this.buf.length) {
        const i = Math.floor(this.pos / ratio);
        const frac = this.pos / ratio - i;
        out.push(this.buf[i] * (1 - frac) + this.buf[Math.min(i + 1, this.buf.length - 1)] * frac);
        this.pos += ratio;
      }
      // consumed prefix
      const consumed = Math.floor(this.pos / ratio);
      if (consumed > 0) { this.buf = this.buf.slice(consumed); this.pos -= consumed * ratio; }
      return out;
    },
  };

  let pending = [];
  node.onaudioprocess = (e) => {
    pending.push(...resample.push(e.inputBuffer.getChannelData(0)));
    while (pending.length >= WINDOW) {
      const w = new Float32Array(pending.slice(0, WINDOW));
      pending = pending.slice(WINDOW);
      onWindow(w);
    }
  };
  src.connect(node);
  node.connect(ctx.destination); // scriptProcessor needs a destination; mute via zero gain
  const mute = ctx.createGain(); mute.gain.value = 0;
  node.disconnect(); node.connect(mute); mute.connect(ctx.destination);

  return async () => {
    node.disconnect(); src.disconnect(); mute.disconnect();
    stream.getTracks().forEach((t) => t.stop());
    await ctx.close();
  };
}

// ---------- endpointing FSM ----------
// Speech opens when prob >= threshold sustained for minSpeech ms.
// Segment closes after hangover ms of silence. Latency estimate = hangover + minSpeech/2.
function feedProb(p, nowMs) {
  const isSpeech = p >= params.threshold;
  if (!fsm.inSpeech) {
    fsm.speechMs = isSpeech ? fsm.speechMs + 32 : 0;
    fsm.silenceMs = 0;
    if (fsm.speechMs >= params.minSpeech && isSpeech) {
      fsm.inSpeech = true;
      fsm.segStart = nowMs - fsm.speechMs;
      fsm.silenceMs = 0;
      onSegmentStart(nowMs);
    }
  } else {
    fsm.silenceMs = isSpeech ? 0 : fsm.silenceMs + 32;
    if (fsm.silenceMs >= params.hangover) {
      const dur = nowMs - fsm.silenceMs - fsm.segStart;
      onSegmentEnd(nowMs - fsm.silenceMs, dur, fsm.silenceMs);
      fsm.inSpeech = false;
      fsm.speechMs = 0;
    }
  }
  return isSpeech;
}

// ---------- UI wiring ----------
const els = {
  wave: $('wave'), timeline: $('timeline'), probFill: $('probFill'),
  probMark: $('probMark'), probNum: $('probNum'), chip: $('stateChip'),
  log: $('log'), stSpeech: $('stSpeech'), stCount: $('stCount'),
  stMedian: $('stMedian'), stLatency: $('stLatency'),
};

// waveform ring buffer (about 12s at 16kHz)
const WAVE_SECONDS = 12;
const wave = new Float32Array(WAVE_SECONDS * SR);
let waveFill = 0;

function drawWave() {
  const c = els.wave, ctx = c.getContext('2d');
  const W = c.width, H = c.height, mid = H / 2;
  ctx.fillStyle = '#0d1320'; ctx.fillRect(0, 0, W, H);
  // threshold line color hint
  ctx.strokeStyle = '#232c40'; ctx.beginPath(); ctx.moveTo(0, mid); ctx.lineTo(W, mid); ctx.stroke();
  const n = Math.max(waveFill, 1);
  const step = n / W;
  ctx.strokeStyle = '#4f8cff'; ctx.lineWidth = 1; ctx.beginPath();
  for (let x = 0; x < W; x++) {
    let peak = 0;
    const from = Math.floor(x * step), to = Math.min(Math.floor((x + 1) * step) + 1, n);
    for (let i = from; i < to; i++) peak = Math.max(peak, Math.abs(wave[i]));
    const y = mid - peak * (mid - 4) * Math.sign(1);
    if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

// segment timeline: last 30s, speech spans as green blocks
const TL_SECONDS = 30;
let segmentsTl = []; // [{start, end}]
function drawTimeline(nowMs) {
  const c = els.timeline, ctx = c.getContext('2d');
  const W = c.width, H = c.height;
  ctx.fillStyle = '#0d1320'; ctx.fillRect(0, 0, W, H);
  ctx.strokeStyle = '#232c40'; ctx.beginPath();
  for (let s = 0; s <= TL_SECONDS; s += 5) {
    const x = W - (s / TL_SECONDS) * W;
    ctx.moveTo(x, 0); ctx.lineTo(x, H);
  }
  ctx.stroke();
  const cutoff = nowMs - TL_SECONDS * 1000;
  ctx.fillStyle = '#35d49a';
  for (const seg of segmentsTl) {
    const x1 = W - ((nowMs - seg.start) / (TL_SECONDS * 1000)) * W;
    const x2 = W - ((nowMs - (seg.end || nowMs)) / (TL_SECONDS * 1000)) * W;
    if (x2 < 0) continue;
    ctx.fillRect(Math.max(x2, 0), 8, Math.min(x1, W) - Math.max(x2, 0), H - 16);
  }
}

function pushLog(kind, nowMs, meta) {
  if (els.log.firstElementChild?.classList.contains('muted')) els.log.innerHTML = '';
  const li = document.createElement('li');
  li.className = kind;
  const t = (nowMs / 1000).toFixed(2);
  li.innerHTML = `<span class="t">${t}s</span><span class="ev">${kind === 'start' ? '▶ speech-start' : '⏹ speech-end'}</span> <span class="meta">${meta}</span>`;
  els.log.prepend(li);
  while (els.log.children.length > 60) els.log.lastElementChild.remove();
}

function onSegmentStart(nowMs) {
  pushLog('start', nowMs, 'agent would start listening / buffer audio');
  segmentsTl.push({ start: nowMs, end: 0 });
  updateStats();
}
function onSegmentEnd(nowMs, durMs, silenceMs) {
  pushLog('end', nowMs, `dur ${durMs}ms · silence-after ${silenceMs}ms`);
  const seg = segmentsTl[segmentsTl.length - 1];
  if (seg) seg.end = nowMs;
  stats.latencies.push(silenceMs);
  if (stats.latencies.length > 200) stats.latencies.shift();
  updateStats();
}

function updateStats() {
  const durs = segmentsTl.filter((s) => s.end > 0).map((s) => s.end - s.start);
  els.stSpeech.textContent = (stats.speechTotalMs / 1000).toFixed(1) + ' s';
  els.stCount.textContent = String(segmentsTl.length);
  els.stMedian.textContent = durs.length
    ? `${Math.round(median(durs))} ms` : '—';
  els.stLatency.textContent = stats.latencies.length
    ? `${Math.round(median(stats.latencies))} ms` : '—';
}
function median(a) { const s = [...a].sort((x, y) => x - y); const m = Math.floor(s.length / 2); return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; }

// ---------- main loop ----------
let stopAudio = null;
let running = false;

function onWindow(samples) {
  const now = performance.now();
  // waveform ring
  wave.set(samples, waveFill % (wave.length - WINDOW));
  waveFill += WINDOW;
  if (waveFill > wave.length) { wave.copyWithin(0, WINDOW); waveFill = wave.length; }

  inferWindow(samples).then((p) => {
    const now2 = performance.now();
    const isSpeech = feedProb(p, now2);
    els.probFill.style.width = `${(p * 100).toFixed(1)}%`;
    els.probNum.textContent = p.toFixed(2);
    els.probMark.style.left = `${params.threshold * 100}%`;
    els.chip.textContent = isSpeech ? 'SPEECH' : (fsm.inSpeech ? 'HANGOVER' : 'SILENCE');
    els.chip.className = isSpeech || fsm.inSpeech ? 'speech' : 'silence';
    if (isSpeech) stats.speechTotalMs += 32;
    drawWave(); drawTimeline(now2);
  }).catch((e) => console.error('vad', e));
}

async function start() {
  if (running) return;
  if (!vadModel) {
    $('loadOverlay').classList.remove('hidden');
    vadModel = await loadModel((info) => {
      if (info.status === 'progress' && info.total) {
        $('loadBar').value = (info.loaded / info.total) * 100;
        $('loadFile').textContent = info.file || '';
      }
    });
    $('loadOverlay').classList.add('hidden');
  }
  try {
    stopAudio = await startAudio(onWindow);
    running = true;
    $('startBtn').disabled = true; $('stopBtn').disabled = false;
  } catch (e) {
    alert('Microphone access denied: ' + e.message);
  }
}
function stop() {
  if (stopAudio) stopAudio();
  stopAudio = null; running = false;
  $('startBtn').disabled = false; $('stopBtn').disabled = true;
  if (fsm.inSpeech) { onSegmentEnd(performance.now(), performance.now() - fsm.segStart, 0); fsm.inSpeech = false; }
}

$('startBtn').onclick = start;
$('stopBtn').onclick = stop;
$('threshold').oninput = (e) => { params.threshold = +e.target.value; $('thVal').textContent = params.threshold.toFixed(2); };
$('minSpeech').oninput = (e) => { params.minSpeech = +e.target.value; $('msVal').textContent = params.minSpeech; };
$('hangover').oninput = (e) => { params.hangover = +e.target.value; $('hoVal').textContent = params.hangover; };

// device badge: webgpu vs wasm
(async () => {
  let device = 'wasm';
  if (navigator.gpu) {
    try { if (await navigator.gpu.requestAdapter()) device = 'webgpu'; } catch { /* wasm */ }
  }
  $('device-badge').textContent = `device: ${device}`;
})();
