---
title: Voice Agents Playground
emoji: 🎙️
colorFrom: blue
colorTo: indigo
sdk: static
app_file: index.html
short_description: Live Silero VAD endpointing visualizer for voice agents
---

# Voice Agents Playground

Real-time, in-browser visualization of **endpointing** — the decision a realtime voice agent
makes about when a user has finished speaking. Runs [Silero VAD](https://huggingface.co/onnx-community/silero-vad)
fully client-side via [transformers.js](https://huggingface.co/docs/transformers.js) (WebGPU when available, WASM otherwise).
No audio ever leaves your machine.

**What you see:** scrolling waveform, live speech probability, speech segments on a timeline,
and a decision log of `speech-start` / `speech-end` events with durations and silence-after —
the same signals an agent loop uses to decide when to answer.

Tune threshold, min-speech and hangover live and watch endpointing behavior change.

Built by [iamdgarcia](https://huggingface.co/iamdgarcia) — realtime voice agent engineering.
Related research: semantic VAD & conversation dynamics ([iamdgarcia.com](https://iamdgarcia.com)).
