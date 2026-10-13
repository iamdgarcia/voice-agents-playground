---
title: Voice Agents Playground
emoji: 🎙️
colorFrom: blue
colorTo: indigo
sdk: static
app_file: index.html
short_description: Live Silero VAD endpointing visualizer for voice agents
tags:
- voice-agent
- vad
- endpointing
- transformers-js
- webgpu
---

# Voice Agents Playground
**▶ Try it live:** <https://iamdgarcia-voice-agents-playground.static.hf.space> · **Code on GitHub:** <https://github.com/iamdgarcia/voice-agents-playground>


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

Further reading: [FD-VAD: Semantic Endpoint Detection for Streaming Full-Duplex Speech](https://www.alphaxiv.org/abs/2609.35791) (UMD, 2026) — why acoustic-only endpointing (what this demo shows) is the floor that semantic detectors must beat.

📖 Full writeup: [https://iamdgarcia.medium.com/watching-your-voice-agent-think-three-instruments-you-can-open-in-a-browser-tab-f7ed16ec2a2a](https://iamdgarcia.medium.com/watching-your-voice-agent-think-three-instruments-you-can-open-in-a-browser-tab-f7ed16ec2a2a) — *Watching your voice agent think* (D. García Peña).
