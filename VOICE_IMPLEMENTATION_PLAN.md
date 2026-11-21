# JERO i2 Voice Mode - Implementation Plan

## Current Architecture Summary

**Message Flow (Text Chat):**
1. User types in `MessageInput` → `send_message()` in `window.py`
2. Creates `GenerationWorker` (QThread) with brain + message
3. Worker calls `brain.generate_stream()` → streams from llama.cpp server
4. Chunks emitted via `worker.chunk` signal → `receive_chunk()` updates streaming QLabel
5. On finish: `receive_finished()` saves to `memory.db`, updates UI

**Key Files:**
- `main.py` - Entry point
- `jero/app.py` - QApplication wrapper
- `jero/brain/local_qwen.py` - Brain with llama.cpp server management
- `jero/brain/worker.py` - GenerationWorker (QThread)
- `jero/brain/base.py` - Base Brain class (empty)
- `jero/memory/database.py` - SQLite for conversations & memories
- `jero/memory/commands.py` - "remember that" handler
- `jero/ui/window.py` - Main window, mic button (disabled, "coming soon")
- `JERO i1.spec` - PyInstaller spec bundling runtime/llama-server.exe + DLLs

---

## New Files to Create

### 1. Config & Settings
- `jero/config.py` - Centralized settings (model paths, phrases, thresholds, voice settings)

### 2. Voice Core Package: `jero/voice/`
- `jero/voice/__init__.py`
- `jero/voice/state_machine.py` - State enum + StateMachine class (Idle, Listening, Hearing, Thinking, Speaking, Standby)
- `jero/voice/audio_input.py` - Microphone capture (sounddevice), VAD integration, volume level callback
- `jero/voice/stt.py` - faster-whisper tiny (English) wrapper, streaming transcription
- `jero/voice/tts.py` - Piper TTS wrapper, sentence-by-sentence streaming, interrupt support
- `jero/voice/wake_detector.py` - Standby/wake phrase detection (case-insensitive, fuzzy match)
- `jero/voice/worker.py` - VoiceConversationWorker (QThread) orchestrating STT → Brain → TTS

### 3. Voice UI: `jero/ui/voice_window.py`
- `VoiceWindow` - Separate window, dark theme (#111111)
- `OrbWidget` - QPainter animated orb + revolving arcs (60 FPS QTimer)
  - States: Listening (calm pulse, slow arcs), Hearing (orb grows with volume), Thinking (fast arcs), Speaking (orb pulses with audio), Standby (dimmed, nearly still)
- State label, live caption toggle, mute button, close button

### 4. Main Window Integration
- Modify `jero/ui/window.py`:
  - Enable mic button, update tooltip
  - Add `open_voice_window()` method
  - Track voice window instance (single instance)

---

## Files to Modify

1. **`jero/ui/window.py`** - Enable mic button, add voice window launcher
2. **`jero/brain/local_qwen.py`** - Add `generate_stream()` usage from voice (already exists), ensure thread-safe
3. **`JERO i1.spec`** → **`JERO i2.spec`** - Add voice models, binaries, ONNX runtime, Piper, sounddevice
4. **`requirements.txt`** - Add new dependencies
5. **`jero/app.py`** - Optional: pass voice config to brain

---

## New Dependencies (requirements.txt additions)

```
# Audio I/O
sounddevice==0.5.1
numpy==2.2.0

# STT: faster-whisper (CTranslate2 backend)
faster-whisper==1.1.1
ctranslate2==4.5.0
huggingface-hub==0.28.0

# VAD: Silero VAD (ONNX)
onnxruntime==1.21.0

# TTS: Piper
piper-tts==1.2.0

# Config
pyyaml==6.0.1
```

**Estimated size increase:**
- faster-whisper tiny model: ~75 MB
- Silero VAD ONNX model: ~5 MB
- Piper voice (en_US-lessac-medium): ~50 MB
- ONNX Runtime: ~15 MB
- PyTorch (transitive via ctranslate2): ~200 MB (but already pulled by faster-whisper deps)
- sounddevice + numpy: ~30 MB
- **Total additional: ~375 MB** (compressed in PyInstaller)

---

## Model Files to Bundle (in `./models/voice/`)

```
models/
└── voice/
    ├── stt/
    │   └── faster-whisper-tiny/     # ~75 MB (auto-downloaded on first run, then bundled)
    ├── vad/
    │   └── silero_vad.onnx          # ~5 MB
    └── tts/
        └── en_US-lessac-medium.onnx # ~50 MB + .json config
```

**PyInstaller bundling strategy:**
- Models placed in `./models/voice/` (local to project)
- Spec file includes `models/voice/` as datas
- At runtime (frozen), resolve to `sys._MEIPASS/models/voice/`
- At runtime (dev), resolve to project root `models/voice/`

---

## Voice Conversation Flow

```
User speaks
    ↓
AudioInput (sounddevice) → VAD (Silero) detects speech start/end
    ↓
STT (faster-whisper) transcribes segment → text
    ↓
WakeDetector checks: "stay on standby" → enter Standby
                        "wake up jero"  → exit Standby
    ↓
If not Standby: Brain.generate_stream() (same as text chat)
    ↓
Streaming tokens → TTS (Piper) sentence-by-sentence
    ↓
AudioOutput (sounddevice) plays
    ↓
While TTS playing: AudioInput paused/gated (echo prevention)
    ↓
User can interrupt → TTS stops, back to Listening
```

---

## State Machine

```
Idle → (mic opens) → Listening
Listening → (VAD detects speech) → Hearing
Hearing → (VAD detects silence) → Thinking
Thinking → (first token from LLM) → Speaking
Speaking → (TTS finishes) → Listening
Speaking → (interrupt) → Listening
Listening/Thinking/Speaking → (wake phrase "stay on standby") → Standby
Standby → (wake phrase "wake up jero") → Listening
Any → (close window) → Idle → cleanup
```

---

## Error Handling

| Error | Handling |
|-------|----------|
| No microphone / permission denied | Show message in voice window, disable mic button, stay in Idle |
| STT model missing | Show message, fallback to text-only, log error |
| TTS model missing | Show message, use text-only replies, log error |
| LLM server not ready | Say/show "JERO is starting up...", retry |
| Background noise false triggers | VAD threshold tuning, min speech duration |
| Orphaned threads on close | `closeEvent` stops all workers, closes audio streams |

---

## Implementation Steps

### Phase 1: Foundation
1. Create `jero/config.py` with all voice settings
2. Create `jero/voice/` package structure
3. Implement `state_machine.py`
4. Implement `audio_input.py` with sounddevice + Silero VAD
5. Update `requirements.txt`

### Phase 2: STT & TTS
6. Implement `stt.py` (faster-whisper tiny, English only)
7. Implement `tts.py` (Piper, sentence streaming, interrupt)
8. Implement `wake_detector.py`

### Phase 3: Voice Worker
9. Implement `voice/worker.py` - VoiceConversationWorker orchestrating the pipeline

### Phase 4: Voice UI
10. Create `jero/ui/voice_window.py` with OrbWidget animation
11. Modify `jero/ui/window.py` - enable mic button, launch voice window

### Phase 5: Integration & Packaging
12. Update `JERO i1.spec` → `JERO i2.spec` with all new assets
13. Test run from source
14. Build exe and test packaged version

---

## Testing Checklist

- [ ] Mic button opens VoiceWindow (only one instance)
- [ ] Orb animation: 60 FPS, low CPU (<10% idle)
- [ ] State transitions visually correct (Listening/Hearing/Thinking/Speaking/Standby)
- [ ] Speech recognition works (English, faster-whisper tiny)
- [ ] VAD detects start/end of speech reliably
- [ ] TTS speaks naturally (Piper, sentence streaming)
- [ ] Replies start speaking before full answer generated
- [ ] Interrupt: speaking stops, JERO listens
- [ ] "stay on standby" → Standby state, only wake phrase works
- [ ] "wake up jero" → exits Standby, confirms aloud
- [ ] Text chat still works exactly as i1
- [ ] Conversation history saved to memory.db (both text & voice)
- [ ] No microphone: graceful error message
- [ ] Missing models: graceful error message
- [ ] Close voice window: mic stops, TTS stops, threads cleaned up
- [ ] Packaged exe works identically to source run