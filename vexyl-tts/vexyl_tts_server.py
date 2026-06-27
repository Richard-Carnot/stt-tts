"""
vexyl_tts_server.py
VEXYL-TTS Server
------------------------------------------------------
Wraps ai4bharat/indic-parler-tts in a WebSocket server.
Accepts JSON text requests, returns base64-encoded WAV audio.
Also exposes a batch synthesis API (POST /batch/synthesize).

Usage:
    pip install git+https://github.com/huggingface/parler-tts.git
    pip install transformers torch soundfile websockets numpy
    python vexyl_tts_server.py

Optional env vars:
    PORT                      (default: 8080, Cloud Run injects this)
    VEXYL_TTS_HOST            (default: 0.0.0.0)
    VEXYL_TTS_PORT            (fallback if PORT unset)
    VEXYL_TTS_DEVICE          (default: auto)  options: auto, cpu, cuda, mps
    VEXYL_TTS_CACHE_SIZE      (default: 200)   LRU cache capacity
    VEXYL_TTS_API_KEY         (default: empty)  shared secret; if set, clients must send X-API-Key header
    VEXYL_TTS_MAX_CONN        (default: 50)     max concurrent WebSocket connections
"""

import asyncio
import websockets
from websockets.asyncio.server import ServerConnection
import json
import base64
import hashlib
import numpy as np
import torch
import soundfile as sf
import os
import io
import scipy.signal
import logging
import time
import signal
import threading
import hmac
import uuid
import requests as _http
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from http import HTTPStatus
from queue import Empty
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [VexylTTS] %(levelname)s %(message)s"
)
log = logging.getLogger("vexyl_tts")

# ─── Config ────────────────────────────────────────────────────────────────────
HOST            = os.getenv("VEXYL_TTS_HOST",   "0.0.0.0")
PORT            = int(os.getenv("PORT", os.getenv("VEXYL_TTS_PORT", "8080")))
DEVICE_PREF     = os.getenv("VEXYL_TTS_DEVICE", "auto")
CACHE_SIZE      = int(os.getenv("VEXYL_TTS_CACHE_SIZE", "200"))
API_KEY         = os.getenv("VEXYL_TTS_API_KEY", "")
MAX_CONNECTIONS = int(os.getenv("VEXYL_TTS_MAX_CONN", "50"))
OUTPUT_SAMPLE_RATE = int(os.getenv("VEXYL_TTS_SAMPLE_RATE", "0"))  # 0 = native (44100), set 8000 for Asterisk
MAX_ACTIVE_GENERATIONS = int(os.getenv("VEXYL_TTS_MAX_ACTIVE_GENERATIONS", "1"))
MAX_QUEUE_SIZE = int(os.getenv("VEXYL_TTS_MAX_QUEUE_SIZE", "100"))
STREAM_PLAY_STEPS = int(os.getenv("VEXYL_TTS_STREAM_PLAY_STEPS", "40"))
STREAMER_TIMEOUT_SECONDS = float(os.getenv("VEXYL_TTS_STREAMER_TIMEOUT", "30"))
MODEL_ID = "ai4bharat/indic-parler-tts"
ENABLE_TORCH_COMPILE = os.getenv("VEXYL_TTS_TORCH_COMPILE", "auto")  # auto, true, false

# ─── Provider config ──────────────────────────────────────────────────────────
# TTS_PROVIDER=local    → on-premise indic-parler-tts + Kokoro (default)
# TTS_PROVIDER=bhashini → Bhashini cloud TTS; no local models loaded
TTS_PROVIDER: str = os.getenv("TTS_PROVIDER", "local").lower().strip()

BHASHINI_USER_ID: str            = os.getenv("BHASHINI_USER_ID", "")
BHASHINI_API_KEY: str            = os.getenv("BHASHINI_API_KEY", "")
BHASHINI_PIPELINE_ID: str        = os.getenv("BHASHINI_PIPELINE_ID", "64392f96daac500b55c543cd")
BHASHINI_AUTH_TOKEN: str         = os.getenv("BHASHINI_AUTH_TOKEN", "")
BHASHINI_INFERENCE_URL: str      = os.getenv(
    "BHASHINI_INFERENCE_URL",
    "https://dhruva-api.bhashini.gov.in/services/inference/pipeline",
)
BHASHINI_PIPELINE_CONFIG_URL: str = os.getenv(
    "BHASHINI_PIPELINE_CONFIG_URL",
    "https://meity-auth.ulcacontrib.org/ulca/apis/v0/model/getModelsPipeline",
)

# Batch synthesis config
BATCH_MAX_TEXT_LENGTH = 5000          # max characters per request
BATCH_MAX_JOBS       = 1000
BATCH_JOB_TTL        = 3600          # 1 hour
BATCH_MAX_BODY_SIZE  = 64 * 1024     # 64KB max POST body

# ─── Voice presets per language ────────────────────────────────────────────────
# Tuned for healthcare IVR: calm, clear, professional.
VOICE_PRESETS = {
    "ml-IN": {
        "default": "Anjali speaks in a calm, clear, and professional tone with a moderate speed and low pitch. The recording is of very high quality with no background noise.",
        "warm":    "Anjali speaks in a warm and empathetic tone, slightly slow-paced for clarity. The recording is of very high quality with no background noise.",
        "formal":  "Harish speaks in a formal, neutral tone with precise diction and moderate speed. The recording is of very high quality with no background noise.",
    },
    "hi-IN": {
        "default": "Divya speaks in a calm, clear, and professional tone with a moderate speed and neutral pitch. The recording is of very high quality with no background noise.",
        "warm":    "Divya speaks in a warm and friendly tone, slightly slow-paced. The recording is of very high quality with no background noise.",
        "formal":  "Rohit speaks in a formal, neutral tone with precise diction and moderate speed. The recording is of very high quality with no background noise.",
    },
    "ta-IN": {
        "default": "Kavitha speaks in a calm, clear, and professional tone with a moderate speed. The recording is of very high quality with no background noise.",
        "warm":    "Kavitha speaks in a warm and empathetic tone, slightly slow-paced for clarity. The recording is of very high quality with no background noise.",
        "formal":  "Jaya speaks in a formal, neutral tone with precise diction and moderate speed. The recording is of very high quality with no background noise.",
    },
    "te-IN": {
        "default": "Lalitha speaks in a calm, clear, and professional tone with a moderate speed. The recording is of very high quality with no background noise.",
        "warm":    "Lalitha speaks in a warm and empathetic tone, slightly slow-paced for clarity. The recording is of very high quality with no background noise.",
        "formal":  "Prakash speaks in a formal, neutral tone with precise diction and moderate speed. The recording is of very high quality with no background noise.",
    },
    "kn-IN": {
        "default": "Anu speaks in a calm, clear, and professional tone with a moderate speed. The recording is of very high quality with no background noise.",
        "warm":    "Anu speaks in a warm and empathetic tone, slightly slow-paced for clarity. The recording is of very high quality with no background noise.",
        "formal":  "Suresh speaks in a formal, neutral tone with precise diction and moderate speed. The recording is of very high quality with no background noise.",
    },
    "bn-IN": {
        "default": "Aditi speaks in a calm, clear, and professional tone with a moderate speed. The recording is of very high quality with no background noise.",
        "warm":    "Aditi speaks in a warm and empathetic tone, slightly slow-paced for clarity. The recording is of very high quality with no background noise.",
        "formal":  "Arjun speaks in a formal, neutral tone with precise diction and moderate speed. The recording is of very high quality with no background noise.",
    },
    "gu-IN": {
        "default": "Neha speaks in a calm, clear, and professional tone with a moderate speed. The recording is of very high quality with no background noise.",
        "warm":    "Neha speaks in a warm and empathetic tone, slightly slow-paced for clarity. The recording is of very high quality with no background noise.",
        "formal":  "Yash speaks in a formal, neutral tone with precise diction and moderate speed. The recording is of very high quality with no background noise.",
    },
    "mr-IN": {
        "default": "Sunita speaks in a calm, clear, and professional tone with a moderate speed. The recording is of very high quality with no background noise.",
        "warm":    "Sunita speaks in a warm and empathetic tone, slightly slow-paced for clarity. The recording is of very high quality with no background noise.",
        "formal":  "Sanjay speaks in a formal, neutral tone with precise diction and moderate speed. The recording is of very high quality with no background noise.",
    },
    "en-IN": {
        "default": "Mary speaks in a calm, clear, and professional tone with a moderate speed. The recording is of very high quality with no background noise.",
        "warm":    "Mary speaks in a warm and empathetic tone, slightly slow-paced for clarity. The recording is of very high quality with no background noise.",
        "formal":  "Thoma speaks in a formal, neutral tone with precise diction and moderate speed. The recording is of very high quality with no background noise.",
    },
    "pa-IN": {
        "default": "Divjot speaks in a calm, clear, and professional tone with a moderate speed. The recording is of very high quality with no background noise.",
        "warm":    "Divjot speaks in a warm and empathetic tone, slightly slow-paced for clarity. The recording is of very high quality with no background noise.",
        "formal":  "Gurpreet speaks in a formal, neutral tone with precise diction and moderate speed. The recording is of very high quality with no background noise.",
    },
    "or-IN": {
        "default": "Debjani speaks in a calm, clear, and professional tone with a moderate speed. The recording is of very high quality with no background noise.",
        "warm":    "Debjani speaks in a warm and empathetic tone, slightly slow-paced for clarity. The recording is of very high quality with no background noise.",
        "formal":  "Manas speaks in a formal, neutral tone with precise diction and moderate speed. The recording is of very high quality with no background noise.",
    },
    "as-IN": {
        "default": "Sita speaks in a calm, clear, and professional tone with a moderate speed. The recording is of very high quality with no background noise.",
        "warm":    "Sita speaks in a warm and empathetic tone, slightly slow-paced for clarity. The recording is of very high quality with no background noise.",
        "formal":  "Amit speaks in a formal, neutral tone with precise diction and moderate speed. The recording is of very high quality with no background noise.",
    },
    "ur-IN": {
        "default": "Zainab speaks in a calm, clear, and professional tone with a moderate speed. The recording is of very high quality with no background noise.",
        "warm":    "Zainab speaks in a warm and empathetic tone, slightly slow-paced for clarity. The recording is of very high quality with no background noise.",
        "formal":  "Rohit speaks in a formal, neutral tone with precise diction and moderate speed. The recording is of very high quality with no background noise.",
    },
    "ne-IN": {
        "default": "Amrita speaks in a calm, clear, and professional tone with a moderate speed. The recording is of very high quality with no background noise.",
        "warm":    "Amrita speaks in a warm and empathetic tone, slightly slow-paced for clarity. The recording is of very high quality with no background noise.",
        "formal":  "Ram speaks in a formal, neutral tone with precise diction and moderate speed. The recording is of very high quality with no background noise.",
    },
    "sa-IN": {
        "default": "Vasudha speaks in a calm, clear, and professional tone with a moderate speed. The recording is of very high quality with no background noise.",
        "warm":    "Vasudha speaks in a warm and empathetic tone, slightly slow-paced for clarity. The recording is of very high quality with no background noise.",
        "formal":  "Aryan speaks in a formal, neutral tone with precise diction and moderate speed. The recording is of very high quality with no background noise.",
    },
    "brx-IN": {
        "default": "Bimala speaks in a calm, clear, and professional tone with a moderate speed. The recording is of very high quality with no background noise.",
        "warm":    "Bimala speaks in a warm and empathetic tone, slightly slow-paced for clarity. The recording is of very high quality with no background noise.",
        "formal":  "Bikram speaks in a formal, neutral tone with precise diction and moderate speed. The recording is of very high quality with no background noise.",
    },
    "doi-IN": {
        "default": "Meena speaks in a calm, clear, and professional tone with a moderate speed. The recording is of very high quality with no background noise.",
        "warm":    "Meena speaks in a warm and empathetic tone, slightly slow-paced for clarity. The recording is of very high quality with no background noise.",
        "formal":  "Vikram speaks in a formal, neutral tone with precise diction and moderate speed. The recording is of very high quality with no background noise.",
    },
    "kok-IN": {
        "default": "Priya speaks in a calm, clear, and professional tone with a moderate speed. The recording is of very high quality with no background noise.",
        "warm":    "Priya speaks in a warm and empathetic tone, slightly slow-paced for clarity. The recording is of very high quality with no background noise.",
        "formal":  "Kaustubh speaks in a formal, neutral tone with precise diction and moderate speed. The recording is of very high quality with no background noise.",
    },
    "mai-IN": {
        "default": "Shruti speaks in a calm, clear, and professional tone with a moderate speed. The recording is of very high quality with no background noise.",
        "warm":    "Shruti speaks in a warm and empathetic tone, slightly slow-paced for clarity. The recording is of very high quality with no background noise.",
        "formal":  "Saurabh speaks in a formal, neutral tone with precise diction and moderate speed. The recording is of very high quality with no background noise.",
    },
    "mni-IN": {
        "default": "Leima speaks in a calm, clear, and professional tone with a moderate speed. The recording is of very high quality with no background noise.",
        "warm":    "Leima speaks in a warm and empathetic tone, slightly slow-paced for clarity. The recording is of very high quality with no background noise.",
        "formal":  "Tomba speaks in a formal, neutral tone with precise diction and moderate speed. The recording is of very high quality with no background noise.",
    },
    "sat-IN": {
        "default": "Sumitra speaks in a calm, clear, and professional tone with a moderate speed. The recording is of very high quality with no background noise.",
        "warm":    "Sumitra speaks in a warm and empathetic tone, slightly slow-paced for clarity. The recording is of very high quality with no background noise.",
        "formal":  "Raju speaks in a formal, neutral tone with precise diction and moderate speed. The recording is of very high quality with no background noise.",
    },
    "sd-IN": {
        "default": "Hema speaks in a calm, clear, and professional tone with a moderate speed. The recording is of very high quality with no background noise.",
        "warm":    "Hema speaks in a warm and empathetic tone, slightly slow-paced for clarity. The recording is of very high quality with no background noise.",
        "formal":  "Mohan speaks in a formal, neutral tone with precise diction and moderate speed. The recording is of very high quality with no background noise.",
    },
    "_default": {
        "default": "A female speaker delivers calm, clear, and professional speech with moderate speed. The recording is of very high quality with no background noise.",
        "warm":    "A female speaker delivers warm and empathetic speech, slightly slow-paced for clarity. The recording is of very high quality with no background noise.",
        "formal":  "A male speaker delivers formal, neutral speech with precise diction and moderate speed. The recording is of very high quality with no background noise.",
    }
}

# VEXYL language codes -> Parler-TTS language names
LANG_MAP = {
    "ml-IN": "malayalam", "hi-IN": "hindi",      "ta-IN": "tamil",
    "te-IN": "telugu",    "kn-IN": "kannada",    "bn-IN": "bengali",
    "gu-IN": "gujarati",  "mr-IN": "marathi",    "pa-IN": "punjabi",
    "or-IN": "odia",      "as-IN": "assamese",   "ur-IN": "urdu",
    "ne-IN": "nepali",    "sa-IN": "sanskrit",   "en-IN": "english",
    "brx-IN": "bodo",     "doi-IN": "dogri",     "kok-IN": "konkani",
    "mai-IN": "maithili", "mni-IN": "manipuri",  "sat-IN": "santali",
    "sd-IN": "sindhi",    "en-US": "english",    "en-GB": "english",
}

def _is_english(lang_code: str) -> bool:
    if not lang_code:
        return False
    return LANG_MAP.get(lang_code) == "english"

# ─── Connection Limits ────────────────────────────────────────────────────────
_conn_semaphore: asyncio.Semaphore   # initialized in main()
_generation_semaphore: asyncio.Semaphore = None
active_connections: int = 0
_server_start_time: float = 0.0

# ─── Model globals ─────────────────────────────────────────────────────────────
model          = None
tokenizer      = None
desc_tokenizer = None
device         = None

# Bhashini: TTS service IDs cached per ISO language code
_bhashini_tts_cache: dict[str, str] = {}
_bhashini_tts_cache_lock = threading.Lock()

# ─── LRU Cache ────────────────────────────────────────────────────────────────
class LRUCache:
    def __init__(self, capacity):
        self.cache    = OrderedDict()
        self.capacity = capacity

    def get(self, key):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

    def __len__(self):
        return len(self.cache)

audio_cache = LRUCache(CACHE_SIZE)
cache_hits  = 0
cache_total = 0

# ─── Batch Job Types ─────────────────────────────────────────────────────────

class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class BatchJob:
    job_id: str
    status: JobStatus
    text: str
    language: str
    style: str
    created_at: float
    description: Optional[str] = None
    audio_b64: Optional[str] = None
    sample_rate: Optional[int] = None
    latency_ms: Optional[int] = None
    completed_at: Optional[float] = None
    error_message: Optional[str] = None

_batch_jobs: dict[str, BatchJob] = {}
_batch_queue: asyncio.Queue = None       # initialized in main()
_batch_worker_task: asyncio.Task = None
_batch_cleanup_task: asyncio.Task = None

# ─── Model Loader ──────────────────────────────────────────────────────────────
def load_model():
    global model, tokenizer, desc_tokenizer, device

    if DEVICE_PREF == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = DEVICE_PREF

    if TTS_PROVIDER == "bhashini":
        log.info("[Bhashini mode] Skipping local model load — synthesis handled by Bhashini cloud API")
        return

    log.info(f"Loading {MODEL_ID} on {device}...")
    start = time.time()

    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer

    # Select optimal dtype: bfloat16 for CUDA (A100/H100), float32 for CPU/MPS
    if device == "cuda":
        if torch.cuda.is_bf16_supported():
            model_dtype = torch.bfloat16
        else:
            model_dtype = torch.float16
        log.info(f"Using {model_dtype} for CUDA inference")
    else:
        model_dtype = torch.float32

    model = ParlerTTSForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=model_dtype,
    ).to(device)
    model.eval()

    # MPS workaround: DAC audio decoder uses conv1d with >65536 output channels
    # which MPS doesn't support. Move the audio encoder to CPU and wrap its
    # decode method to automatically transfer tensors from MPS → CPU.
    if device == "mps":
        model.audio_encoder = model.audio_encoder.to("cpu")
        _original_decode = model.audio_encoder.decode
        def _cpu_decode(*args, **kwargs):
            args = tuple(a.to("cpu") if isinstance(a, torch.Tensor) else a for a in args)
            kwargs = {k: v.to("cpu") if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
            return _original_decode(*args, **kwargs)
        model.audio_encoder.decode = _cpu_decode
        log.info("Moved audio_encoder to CPU (MPS conv1d channel limit workaround)")

    tokenizer      = AutoTokenizer.from_pretrained(MODEL_ID)
    desc_tokenizer = AutoTokenizer.from_pretrained(
        model.config.text_encoder._name_or_path
    )

    # Optional torch.compile for CUDA (significant speedup after warmup)
    _should_compile = (
        ENABLE_TORCH_COMPILE == "true"
        or (ENABLE_TORCH_COMPILE == "auto" and device == "cuda")
    )
    if _should_compile and hasattr(torch, "compile"):
        try:
            model.generate = torch.compile(
                model.generate,
                mode="reduce-overhead",
            )
            log.info("Enabled torch.compile for model.generate() (reduce-overhead mode)")
        except Exception as e:
            log.warning(f"torch.compile failed, falling back to eager mode: {e}")

    elapsed = time.time() - start
    log.info(f"Model loaded in {elapsed:.1f}s | device={device} | dtype={model_dtype} | sample_rate={model.config.sampling_rate}Hz")


# ─── TTS Core ─────────────────────────────────────────────────────────────────
def get_voice_description(lang_code, style="default"):
    presets = VOICE_PRESETS.get(lang_code, VOICE_PRESETS["_default"])
    return presets.get(style, presets.get("default"))


def clamp_play_steps(value):
    try:
        return max(10, min(120, int(value)))
    except (TypeError, ValueError):
        return clamp_play_steps(STREAM_PLAY_STEPS) if value != STREAM_PLAY_STEPS else 40


def _generation_params(play_steps=None):
    return {
        "do_sample": True,
        "play_steps": clamp_play_steps(play_steps),
    }


def _audio_to_wav(audio_arr, native_rate):
    audio_arr = audio_arr.astype(np.float32)
    peak = np.abs(audio_arr).max() if len(audio_arr) else 0.0
    if peak > 1.0:
        audio_arr = audio_arr / peak

    sample_rate = native_rate
    if OUTPUT_SAMPLE_RATE and OUTPUT_SAMPLE_RATE != native_rate:
        # Use polyphase resampling for proper anti-aliased downsampling
        # (np.interp does linear interpolation which causes aliasing at 44100→8000)
        from math import gcd
        up = OUTPUT_SAMPLE_RATE // gcd(OUTPUT_SAMPLE_RATE, native_rate)
        down = native_rate // gcd(OUTPUT_SAMPLE_RATE, native_rate)
        audio_arr = scipy.signal.resample_poly(audio_arr, up, down).astype(np.float32)
        sample_rate = OUTPUT_SAMPLE_RATE

    buf = io.BytesIO()
    sf.write(buf, audio_arr, sample_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read(), sample_rate


def build_audio_end_message(
    request_id,
    total_chunks,
    first_chunk_ms,
    latency_ms,
    sample_rate,
    full_audio_b64="",
    include_full_audio=False,
    cancelled=False,
):
    msg = {
        "type": "audio_end",
        "request_id": request_id,
        "total_chunks": total_chunks,
        "first_chunk_ms": first_chunk_ms,
        "latency_ms": latency_ms,
        "sample_rate": sample_rate,
    }
    if cancelled:
        msg["cancelled"] = True
    if include_full_audio and full_audio_b64:
        msg["full_audio_b64"] = full_audio_b64
    return msg


class RAGSegmentBuffer:
    SENTENCE_ENDINGS = (".", "?", "!", "।", "॥", "\n")
    SOFT_BOUNDARIES = (",", ";", ":", "،")

    def __init__(self, min_chars=24):
        self.text = ""
        self.cancelled = False
        self.min_chars = min_chars

    def push(self, delta):
        if self.cancelled or not delta:
            return []
        self.text += str(delta)
        return self._drain_ready_segments()

    def flush(self):
        if self.cancelled:
            self.text = ""
            return []
        segment = self.text.strip()
        self.text = ""
        return [segment] if segment else []

    def cancel(self):
        self.cancelled = True
        self.text = ""

    def _drain_ready_segments(self):
        segments = []
        while True:
            boundary = self._find_boundary()
            if boundary == -1:
                break
            segment = self.text[:boundary + 1].strip()
            self.text = self.text[boundary + 1:].lstrip()
            if segment:
                segments.append(segment)
        return segments

    def _find_boundary(self):
        for idx, ch in enumerate(self.text):
            if ch in self.SENTENCE_ENDINGS:
                return idx
            if ch in self.SOFT_BOUNDARIES and idx + 1 >= self.min_chars:
                return idx
        return -1


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences using sentence endings (. ? ! । ॥ \n)."""
    if not text:
        return []
    sentences = []
    current = []
    endings = {".", "?", "!", "।", "॥", "\n"}
    for char in text:
        current.append(char)
        if char in endings:
            sentences.append("".join(current).strip())
            current = []
    if current:
        s = "".join(current).strip()
        if s:
            sentences.append(s)
    return [s for s in sentences if s]


_kokoro_pipeline = None
KOKORO_VOICES = {
    "af_bella", "af_nicole", "af_sarah", "af_sky",
    "am_adam", "am_michael",
    "bf_emma", "bf_isabella",
    "bm_george", "bm_lewis"
}

def _get_kokoro_voice(style: str) -> str:
    style_lower = style.strip().lower()
    if style_lower in KOKORO_VOICES:
        return style_lower
    voice_map = {
        "default": "af_bella",
        "warm": "af_sarah",
        "formal": "am_adam"
    }
    return voice_map.get(style_lower, "af_bella")

def _get_kokoro_pipeline():
    global _kokoro_pipeline
    if _kokoro_pipeline is None:
        from kokoro import KPipeline
        log.info("Initializing Kokoro-82M pipeline for English...")
        _kokoro_pipeline = KPipeline(lang_code='a', device=device)
        # Warm up Kokoro
        log.info("Running Kokoro warm-up...")
        list(_kokoro_pipeline("Warmup", voice="af_bella", speed=1.0))
        log.info("Kokoro warm-up complete")
    return _kokoro_pipeline

def _synthesize_kokoro_sync(text: str, voice: str = "af_bella") -> tuple[bytes, int]:
    pipeline = _get_kokoro_pipeline()
    generator = pipeline(text, voice=voice, speed=1.0)
    all_audio = []
    for _, _, audio in generator:
        if audio is not None:
            if hasattr(audio, "numpy"):
                audio_arr = audio.cpu().numpy()
            else:
                audio_arr = np.array(audio)
            if len(audio_arr) > 0:
                all_audio.append(audio_arr)
    
    if not all_audio:
        raise ValueError("Kokoro generated no audio")
    
    combined = np.concatenate(all_audio)
    return _audio_to_wav(combined, 24000)

async def synthesize_kokoro_full(text: str, style: str = "default"):
    voice = _get_kokoro_voice(style)
    return await asyncio.to_thread(_synthesize_kokoro_sync, text, voice)

def _synthesize_sync(text, lang_code, style="default", custom_description=None):
    """Run full synthesis. Returns (WAV bytes, sample_rate)."""
    description = custom_description or get_voice_description(lang_code, style)

    desc_inputs   = desc_tokenizer(description, return_tensors="pt").to(device)
    prompt_inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.inference_mode():
        generation = model.generate(
            input_ids=desc_inputs.input_ids,
            attention_mask=desc_inputs.attention_mask,
            prompt_input_ids=prompt_inputs.input_ids,
            prompt_attention_mask=prompt_inputs.attention_mask,
        )

    audio_arr = generation.cpu().numpy().squeeze().astype(np.float32)
    return _audio_to_wav(audio_arr, model.config.sampling_rate)

# ─── Bhashini TTS helpers ─────────────────────────────────────────────────────

def _bcp47_to_iso(lang_code: str) -> str:
    """Convert BCP-47 code (hi-IN) to ISO 639 short code (hi) for Bhashini."""
    return lang_code.split("-")[0]


def _style_to_gender(style: str) -> str:
    """Map Vexyl voice style to Bhashini gender parameter."""
    return "male" if style == "formal" else "female"


def _get_bhashini_tts_service_id(lang_code: str) -> str:
    """
    Return the Bhashini TTS service ID for *lang_code* (ISO 639 short code).
    Results are cached per language; first call hits the pipeline config API.
    Thread-safe.
    """
    with _bhashini_tts_cache_lock:
        if lang_code in _bhashini_tts_cache:
            return _bhashini_tts_cache[lang_code]

    resp = _http.post(
        BHASHINI_PIPELINE_CONFIG_URL,
        headers={
            "userID": BHASHINI_USER_ID,
            "ulcaApiKey": BHASHINI_API_KEY,
            "Content-Type": "application/json",
        },
        json={
            "pipelineTasks": [{
                "taskType": "tts",
                "config": {"language": {"sourceLanguage": lang_code}},
            }],
            "pipelineRequestConfig": {"pipelineId": BHASHINI_PIPELINE_ID},
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    configs = data["pipelineResponseConfig"][0]["config"]
    all_ids = [c["serviceId"] for c in configs]
    log.info(f"[Bhashini] TTS available service IDs for '{lang_code}': {all_ids}")
    service_id = next((sid for sid in all_ids if "ai4bharat" in sid), all_ids[0])

    with _bhashini_tts_cache_lock:
        _bhashini_tts_cache[lang_code] = service_id

    log.info(f"[Bhashini] TTS selected service ID for '{lang_code}': {service_id}")
    return service_id


def _bhashini_synthesize_sync(text: str, lang_code: str, style: str = "default") -> tuple[bytes, int]:
    """
    Synchronous Bhashini TTS call — intended to run in asyncio.to_thread().

    1. Calls Bhashini inference pipeline with taskType=tts
    2. Decodes base64 audio from the response
    3. Normalises and returns (WAV bytes, sample_rate)
    """
    # Skip text that has no speakable content (numbers/punctuation only, very short)
    import re as _re
    if not text or not text.strip() or not _re.search(r'\w', text) or len(text.strip()) < 2:
        raise ValueError(f"Text too short or non-speakable: {text!r}")

    iso_code = _bcp47_to_iso(lang_code)
    gender   = _style_to_gender(style)
    service_id = _get_bhashini_tts_service_id(iso_code)

    resp = _http.post(
        BHASHINI_INFERENCE_URL,
        headers={
            "Authorization": BHASHINI_AUTH_TOKEN,
            "Content-Type": "application/json",
        },
        json={
            "pipelineTasks": [{
                "taskType": "tts",
                "config": {
                    "language": {"sourceLanguage": iso_code},
                    "serviceId": service_id,
                    "gender": gender,
                },
            }],
            "inputData": {
                "input": [{"source": text}],
                "audio": [{"audioContent": None}],
            },
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()

    audio_b64: str = data["pipelineResponse"][0]["audio"][0]["audioContent"]
    audio_bytes = base64.b64decode(audio_b64)

    # Normalise to WAV (handles WAV, FLAC, OGG — whatever Bhashini returns)
    buf = io.BytesIO(audio_bytes)
    audio_arr, native_rate = sf.read(buf, dtype="float32")
    return _audio_to_wav(audio_arr, native_rate)


def _init_bhashini() -> None:
    """Validate Bhashini credentials at startup by pre-fetching Hindi TTS service ID."""
    if not BHASHINI_USER_ID or not BHASHINI_API_KEY:
        log.error(
            "[Bhashini] BHASHINI_USER_ID or BHASHINI_API_KEY not set — "
            "Bhashini TTS will not work."
        )
        return
    try:
        _get_bhashini_tts_service_id("hi")
        log.info("[Bhashini] Credentials validated, Hindi TTS service ID cached")
    except Exception as exc:
        log.error(f"[Bhashini] Startup validation failed: {exc}")


async def synthesize_full(text, lang_code, style="default", custom_description=None):
    """Async wrapper for full synthesis."""
    if TTS_PROVIDER == "bhashini":
        return await asyncio.to_thread(_bhashini_synthesize_sync, text, lang_code, style)
    if _is_english(lang_code):
        return await synthesize_kokoro_full(text, style)
    return await asyncio.to_thread(_synthesize_sync, text, lang_code, style, custom_description)


async def synthesize(text, lang_code, style="default", custom_description=None):
    """Backward-compatible alias for full synthesis."""
    return await synthesize_full(text, lang_code, style, custom_description)


from parler_tts import ParlerTTSStreamer
import math

class OptimizedParlerTTSStreamer(ParlerTTSStreamer):
    """Custom streamer that uses a sliding window for audio decoding to avoid quadratic O(N^2) slowdown."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hop_length = math.floor(self.audio_encoder.config.sampling_rate / self.audio_encoder.config.frame_rate)
        # Sliding window size: at least 3x play_steps, capped to minimum of 30 tokens for delay pattern coverage
        self.window_tokens = max(self.play_steps * 3, 30)

    def put(self, value):
        batch_size = value.shape[0] // self.decoder.num_codebooks
        if batch_size > 1:
            raise ValueError("OptimizedParlerTTSStreamer only supports batch size 1")

        if self.token_cache is None:
            self.token_cache = value
        else:
            self.token_cache = torch.concatenate([self.token_cache, value[:, None]], dim=-1)

        total_tokens = self.token_cache.shape[-1]
        if total_tokens % self.play_steps == 0:
            start_token = max(0, total_tokens - self.window_tokens)
            token_window = self.token_cache[:, start_token:]
            audio_values = self.apply_delay_pattern_mask(token_window)
            
            window_start_sample = start_token * self.hop_length
            rel_start = self.to_yield - window_start_sample
            rel_end = len(audio_values) - self.stride

            rel_start = max(0, rel_start)
            rel_end = max(rel_start, rel_end)
            chunk = audio_values[rel_start:rel_end]
            if len(chunk):
                self.on_finalized_audio(chunk)
                self.to_yield += len(chunk)

    def end(self):
        """Flushes the final chunk using the sliding window."""
        if self.token_cache is not None:
            total_tokens = self.token_cache.shape[-1]
            start_token = max(0, total_tokens - self.window_tokens)
            token_window = self.token_cache[:, start_token:]
            audio_values = self.apply_delay_pattern_mask(token_window)
            
            window_start_sample = start_token * self.hop_length
            rel_start = self.to_yield - window_start_sample
            rel_start = max(0, rel_start)
            chunk = audio_values[rel_start:]
        else:
            chunk = np.zeros(0)

        self.on_finalized_audio(chunk, stream_end=True)


def _stream_synthesize_sync(text, lang_code, style="default", custom_description=None, play_steps=None):
    """Run streaming inference. Yields (WAV chunk bytes, sample_rate, is_final)."""
    description = custom_description or get_voice_description(lang_code, style)
    params = _generation_params(play_steps)
    play_steps = params["play_steps"]

    desc_inputs   = desc_tokenizer(description, return_tensors="pt").to(device)
    prompt_inputs = tokenizer(text, return_tensors="pt").to(device)

    streamer = OptimizedParlerTTSStreamer(
        model,
        device=device,
        play_steps=play_steps,
        timeout=STREAMER_TIMEOUT_SECONDS,
    )

    generation_kwargs = dict(
        input_ids=desc_inputs.input_ids,
        attention_mask=desc_inputs.attention_mask,
        prompt_input_ids=prompt_inputs.input_ids,
        prompt_attention_mask=prompt_inputs.attention_mask,
        streamer=streamer,
        do_sample=params["do_sample"],
    )

    generation_error = []

    def run_generate():
        try:
            with torch.inference_mode():
                model.generate(**generation_kwargs)
        except Exception as exc:
            generation_error.append(exc)
            try:
                streamer.on_finalized_audio(np.zeros(0, dtype=np.float32), stream_end=True)
            except Exception:
                pass

    thread = threading.Thread(target=run_generate, daemon=True)
    collected_chunks = []

    # No _infer_lock needed — _generation_semaphore (async) already serializes
    # concurrent generation calls. The threading.Lock was redundant and held
    # during I/O waits (queue.put) unnecessarily.
    thread.start()

    native_rate = model.config.sampling_rate
    sample_rate = native_rate

    while True:
        try:
            new_audio = next(streamer)
        except StopIteration:
            break
        except Empty as exc:
            raise TimeoutError("Timed out waiting for streaming audio") from exc

        if new_audio.shape[0] == 0:
            break

        audio_arr = new_audio.astype(np.float32)

        # Keep for combined caching
        collected_chunks.append(audio_arr)

        chunk_bytes, chunk_sample_rate = _audio_to_wav(audio_arr, native_rate)
        yield chunk_bytes, chunk_sample_rate, False

    thread.join()

    if generation_error:
        raise generation_error[0]

    # Compile the full audio for caching
    if collected_chunks:
        full_arr = np.concatenate(collected_chunks)
        full_bytes, full_sample_rate = _audio_to_wav(full_arr, native_rate)
        yield full_bytes, full_sample_rate, True


async def stream_audio_chunks(text, lang_code, style="default", custom_description=None, play_steps=None):
    """Async generator yielding (chunk_bytes, sample_rate, is_final) via a queue and background thread."""
    if TTS_PROVIDER == "bhashini":
        # Bhashini TTS is batch-only: synthesise the full utterance and yield it
        # as a single final chunk so the WebSocket protocol stays unchanged.
        wav_bytes, sample_rate = await asyncio.to_thread(_bhashini_synthesize_sync, text, lang_code, style)
        yield (wav_bytes, sample_rate, True)
        return

    loop = asyncio.get_running_loop()
    queue = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)

    def run_synthesis():
        def put_threadsafe(item):
            future = asyncio.run_coroutine_threadsafe(queue.put(item), loop)
            future.result()

        try:
            for chunk_bytes, sr, is_final in _stream_synthesize_sync(text, lang_code, style, custom_description, play_steps):
                put_threadsafe((chunk_bytes, sr, is_final))
            put_threadsafe(None)
        except Exception as e:
            log.error(f"Error in stream synthesis: {e}", exc_info=True)
            put_threadsafe(e)

    threading.Thread(target=run_synthesis, daemon=True).start()

    while True:
        item = await queue.get()
        if item is None:
            break
        if isinstance(item, Exception):
            raise item
        yield item


async def stream_synthesize(text, lang_code, style="default", custom_description=None, play_steps=None):
    """Backward-compatible alias for model-level streaming."""
    async for item in stream_audio_chunks(text, lang_code, style, custom_description, play_steps):
        yield item


def make_cache_key(
    text,
    lang_code,
    style,
    description=None,
    output_sample_rate=None,
    generation_params=None,
):
    payload = {
        "text": text,
        "lang": lang_code,
        "style": style,
        "description": description or "",
        "output_sample_rate": OUTPUT_SAMPLE_RATE if output_sample_rate is None else output_sample_rate,
        "model_id": MODEL_ID,
        "generation_params": generation_params or {},
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ─── Batch Worker ──────────────────────────────────────────────────────────────

async def _batch_worker():
    """Background coroutine — pulls jobs from queue and runs synthesis."""
    log.info("Batch worker started")
    while True:
        try:
            job_id = await _batch_queue.get()
        except asyncio.CancelledError:
            raise
        except Exception:
            log.error("[batch] Error getting from queue", exc_info=True)
            await asyncio.sleep(1)
            continue

        try:
            job = _batch_jobs.get(job_id)
            if not job or job.status != JobStatus.QUEUED:
                continue

            job.status = JobStatus.PROCESSING
            log.info(f"[batch] Processing job {job_id} ({job.language}/{job.style}, {len(job.text)} chars)")

            start = time.time()
            wav_bytes, sample_rate = await synthesize(job.text, job.language, job.style, job.description)
            latency = int((time.time() - start) * 1000)

            job.audio_b64 = base64.b64encode(wav_bytes).decode()
            job.sample_rate = sample_rate
            job.latency_ms = latency
            job.status = JobStatus.COMPLETED
            job.completed_at = time.time()

            log.info(f"[batch] Job {job_id} completed ({latency}ms)")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.error(f"[batch] Job {job_id} failed: {e}", exc_info=True)
            if job_id in _batch_jobs:
                _batch_jobs[job_id].status = JobStatus.FAILED
                _batch_jobs[job_id].error_message = "Synthesis failed"
                _batch_jobs[job_id].completed_at = time.time()
        finally:
            try:
                _batch_queue.task_done()
            except ValueError:
                pass


async def _batch_cleanup_loop():
    """Remove completed/failed jobs older than BATCH_JOB_TTL every 5 minutes."""
    while True:
        await asyncio.sleep(300)
        now = time.time()
        expired = [
            jid for jid, job in _batch_jobs.items()
            if job.completed_at and (now - job.completed_at) > BATCH_JOB_TTL
        ]
        for jid in expired:
            del _batch_jobs[jid]
        if expired:
            log.info(f"[batch] Cleaned up {len(expired)} expired jobs")


@dataclass
class RAGStreamSession:
    request_id: str
    language: str
    style: str
    description: Optional[str]
    include_full_audio: bool
    play_steps: int
    streaming_mode: str
    buffer: RAGSegmentBuffer = field(default_factory=RAGSegmentBuffer)
    queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=MAX_QUEUE_SIZE))
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    task: Optional[asyncio.Task] = None


async def _send_audio_chunk(websocket, request_id, chunk_bytes, sample_rate, chunk_index, cached=False, cached_full_audio=False):
    message = {
        "type": "audio_chunk",
        "request_id": request_id,
        "audio_b64": base64.b64encode(chunk_bytes).decode(),
        "sample_rate": sample_rate,
        "chunk_index": chunk_index,
        "cached": cached,
    }
    if cached_full_audio:
        message["cached_full_audio"] = True
    await websocket.send(json.dumps(message))


async def _send_error(websocket, request_id, message):
    await websocket.send(json.dumps({
        "type": "error",
        "request_id": request_id,
        "message": message,
    }))


async def _send_streaming_synthesis(
    websocket,
    request_id,
    text,
    lang_code,
    style,
    custom_desc=None,
    play_steps=None,
    include_full_audio=False,
    cancel_event=None,
    streaming_mode=None,
):
    global cache_hits, cache_total

    if streaming_mode is None:
        streaming_mode = os.getenv("VEXYL_TTS_STREAMING_MODE", "sentence").lower()

    if _is_english(lang_code):
        streaming_mode = "sentence"

    if streaming_mode == "sentence":
        sentences = split_into_sentences(text)
        if not sentences:
            end_msg = build_audio_end_message(
                request_id=request_id,
                total_chunks=0,
                first_chunk_ms=0,
                latency_ms=0,
                sample_rate=OUTPUT_SAMPLE_RATE or (24000 if _is_english(lang_code) else (model.config.sampling_rate if model else 0)),
                include_full_audio=include_full_audio,
                cancelled=bool(cancel_event and cancel_event.is_set()),
            )
            await websocket.send(json.dumps(end_msg))
            return

        chunk_idx = 0
        collected_wavs = []
        first_chunk_ms = None
        start = time.time()
        sample_rate = OUTPUT_SAMPLE_RATE or (24000 if _is_english(lang_code) else (model.config.sampling_rate if model else 0))

        for sentence in sentences:
            if cancel_event and cancel_event.is_set():
                break

            ck = make_cache_key(sentence, lang_code, style, custom_desc, OUTPUT_SAMPLE_RATE, {"mode": "sentence"})
            cached = audio_cache.get(ck)
            cache_total += 1

            if cached:
                cache_hits += 1
                wav_bytes = base64.b64decode(cached["b64"])
                sample_rate = cached["sr"]
                if first_chunk_ms is None:
                    first_chunk_ms = int((time.time() - start) * 1000)
                await _send_audio_chunk(
                    websocket,
                    request_id,
                    wav_bytes,
                    sample_rate,
                    chunk_idx,
                    cached=True,
                )
                chunk_idx += 1
                collected_wavs.append(wav_bytes)
            else:
                try:
                    if _is_english(lang_code):
                        if cancel_event and cancel_event.is_set():
                            break
                        wav_bytes, sample_rate = await synthesize_full(sentence, lang_code, style, custom_desc)
                    else:
                        async with _generation_semaphore:
                            if cancel_event and cancel_event.is_set():
                                break
                            wav_bytes, sample_rate = await synthesize_full(sentence, lang_code, style, custom_desc)
                except Exception as exc:
                    log.warning(f"[{request_id}] Skipping sentence synthesis error: {exc} | text='{sentence[:60]}'")
                    continue

                if first_chunk_ms is None:
                    first_chunk_ms = int((time.time() - start) * 1000)

                b64_audio = base64.b64encode(wav_bytes).decode()
                audio_cache.put(ck, {"b64": b64_audio, "sr": sample_rate})

                await _send_audio_chunk(
                    websocket,
                    request_id,
                    wav_bytes,
                    sample_rate,
                    chunk_idx,
                    cached=False,
                )
                chunk_idx += 1
                collected_wavs.append(wav_bytes)

        latency = int((time.time() - start) * 1000)
        if first_chunk_ms is None:
            first_chunk_ms = latency

        full_audio_b64 = ""
        if include_full_audio and collected_wavs:
            try:
                arrays = []
                for w_bytes in collected_wavs:
                    data, sr = sf.read(io.BytesIO(w_bytes))
                    arrays.append(data)
                if arrays:
                    full_arr = np.concatenate(arrays)
                    full_wav, sr = _audio_to_wav(full_arr, sr)
                    full_audio_b64 = base64.b64encode(full_wav).decode()
            except Exception as e:
                log.error(f"Error concatenating audio for request {request_id}: {e}", exc_info=True)

        end_msg = build_audio_end_message(
            request_id=request_id,
            total_chunks=chunk_idx,
            first_chunk_ms=first_chunk_ms,
            latency_ms=latency,
            sample_rate=sample_rate,
            full_audio_b64=full_audio_b64,
            include_full_audio=include_full_audio,
            cancelled=bool(cancel_event and cancel_event.is_set()),
        )
        await websocket.send(json.dumps(end_msg))

    else:
        # Legacy token streaming mode
        play_steps = clamp_play_steps(play_steps)
        generation_params = _generation_params(play_steps)
        ck = make_cache_key(text, lang_code, style, custom_desc, OUTPUT_SAMPLE_RATE, generation_params)
        cached = audio_cache.get(ck)

        cache_total += 1
        start = time.time()
        first_chunk_ms = None
        sample_rate = OUTPUT_SAMPLE_RATE or (model.config.sampling_rate if model else 0)

        if cached:
            cache_hits += 1
            await _send_audio_chunk(
                websocket,
                request_id,
                base64.b64decode(cached["b64"]),
                cached["sr"],
                0,
                cached=True,
                cached_full_audio=True,
            )
            end_msg = build_audio_end_message(
                request_id=request_id,
                total_chunks=1,
                first_chunk_ms=2,
                latency_ms=2,
                sample_rate=cached["sr"],
                full_audio_b64=cached["b64"],
                include_full_audio=include_full_audio,
                cancelled=bool(cancel_event and cancel_event.is_set()),
            )
            await websocket.send(json.dumps(end_msg))
            return

        chunk_idx = 0
        full_audio_b64 = ""

        async with _generation_semaphore:
            async for chunk_bytes, sample_rate, is_final in stream_audio_chunks(
                text,
                lang_code,
                style,
                custom_desc,
                play_steps,
            ):
                if cancel_event and cancel_event.is_set():
                    break  # Stop consuming — don't waste GPU on cancelled audio

                if not is_final:
                    if first_chunk_ms is None:
                        first_chunk_ms = int((time.time() - start) * 1000)
                    await _send_audio_chunk(websocket, request_id, chunk_bytes, sample_rate, chunk_idx, cached=False)
                    chunk_idx += 1
                else:
                    full_audio_b64 = base64.b64encode(chunk_bytes).decode()
                    audio_cache.put(ck, {"b64": full_audio_b64, "sr": sample_rate})

        latency = int((time.time() - start) * 1000)
        if first_chunk_ms is None:
            first_chunk_ms = latency

        end_msg = build_audio_end_message(
            request_id=request_id,
            total_chunks=chunk_idx,
            first_chunk_ms=first_chunk_ms,
            latency_ms=latency,
            sample_rate=sample_rate,
            full_audio_b64=full_audio_b64,
            include_full_audio=include_full_audio,
            cancelled=bool(cancel_event and cancel_event.is_set()),
        )
        await websocket.send(json.dumps(end_msg))


async def _send_full_synthesis(websocket, request_id, text, lang_code, style, custom_desc=None):
    global cache_hits, cache_total

    ck = make_cache_key(text, lang_code, style, custom_desc, OUTPUT_SAMPLE_RATE, {"mode": "full"})
    cached = audio_cache.get(ck)
    cache_total += 1

    if cached:
        cache_hits += 1
        log.info(f"[{request_id}] CACHE HIT ({cache_hits/cache_total*100:.0f}%) | '{text[:40]}'")
        await websocket.send(json.dumps({
            "type": "audio", "request_id": request_id,
            "audio_b64": cached["b64"],
            "sample_rate": cached["sr"],
            "cached": True, "latency_ms": 2
        }))
        return

    start = time.time()
    if _is_english(lang_code):
        wav_bytes, sample_rate = await synthesize_full(text, lang_code, style, custom_desc)
    else:
        async with _generation_semaphore:
            wav_bytes, sample_rate = await synthesize_full(text, lang_code, style, custom_desc)

    latency = int((time.time() - start) * 1000)
    b64audio = base64.b64encode(wav_bytes).decode()
    audio_cache.put(ck, {"b64": b64audio, "sr": sample_rate})

    log.info(f"[{request_id}] Synthesized {latency}ms | {lang_code}/{style} | '{text[:40]}'")
    await websocket.send(json.dumps({
        "type": "audio", "request_id": request_id,
        "audio_b64": b64audio,
        "sample_rate": sample_rate,
        "cached": False, "latency_ms": latency
    }))


async def stream_text_segments(websocket, session: RAGStreamSession):
    """Pipelined RAG sentence consumer.

    Overlaps synthesis of the current sentence with buffering of the next one
    by launching each synthesis as a task and awaiting the *previous* task only
    when the *next* segment is ready.  This eliminates the inter-sentence gap
    where previously the server would sit idle waiting for the next sentence
    boundary while the GPU was free.
    """
    pending_task: asyncio.Task | None = None

    async def _synth(segment_text: str):
        await _send_streaming_synthesis(
            websocket,
            session.request_id,
            segment_text,
            session.language,
            session.style,
            session.description,
            session.play_steps,
            session.include_full_audio,
            session.cancel_event,
            streaming_mode=session.streaming_mode,
        )

    while True:
        segment = await session.queue.get()
        try:
            if segment is None or session.cancel_event.is_set():
                # Drain: wait for the last in-flight synthesis to finish
                if pending_task:
                    try:
                        await pending_task
                    except Exception as exc:
                        if not session.cancel_event.is_set():
                            await _send_error(websocket, session.request_id, str(exc))
                break

            # Wait for previous segment synthesis to complete before sending
            # the next one (preserves ordering), but the *buffering* of the
            # next segment happened concurrently while the GPU was busy.
            if pending_task:
                try:
                    await pending_task
                except Exception as exc:
                    session.cancel_event.set()
                    session.buffer.cancel()
                    await _send_error(websocket, session.request_id, str(exc))
                    while not session.queue.empty():
                        try:
                            session.queue.get_nowait()
                            session.queue.task_done()
                        except asyncio.QueueEmpty:
                            break
                    break

            # Launch synthesis for this segment as a task so the event loop
            # can keep receiving stream_text_delta messages while we wait.
            pending_task = asyncio.create_task(_synth(segment))

        except Exception as exc:
            session.cancel_event.set()
            session.buffer.cancel()
            await _send_error(websocket, session.request_id, str(exc))
            while not session.queue.empty():
                try:
                    session.queue.get_nowait()
                    session.queue.task_done()
                except asyncio.QueueEmpty:
                    break
            break
        finally:
            session.queue.task_done()


# ─── WebSocket Handler ─────────────────────────────────────────────────────────
async def handle_connection(websocket):
    """
    Protocol (all JSON):
    Client → {"type":"synthesize","text":"...","lang":"ml-IN","style":"default","request_id":"x"}
    Server ← {"type":"audio","request_id":"x","audio_b64":"...","sample_rate":22050,"cached":bool,"latency_ms":N}

    Client → {"type":"get_stats"}
    Server ← {"type":"stats","cache_hits":N,"cache_total":N,"hit_rate":N}

    On connect:
    Server ← {"type":"ready","model":"indic-parler-tts","sample_rate":22050}
    """
    global active_connections
    remote = websocket.remote_address
    active_connections += 1
    rag_sessions = {}

    try:
        await websocket.send(json.dumps({
            "type":        "ready",
            "model":       "bhashini" if TTS_PROVIDER == "bhashini" else "indic-parler-tts",
            "sample_rate": (OUTPUT_SAMPLE_RATE or 22050) if TTS_PROVIDER == "bhashini" else model.config.sampling_rate,
            "languages":   list(LANG_MAP.keys()),
        }))
        log.info(f"New connection: {remote}")

        async for message in websocket:
            try:
                msg = json.loads(message)
            except json.JSONDecodeError:
                await websocket.send(json.dumps({"type": "error", "message": "Invalid JSON"}))
                continue

            msg_type   = msg.get("type")
            request_id = msg.get("request_id", f"req_{int(time.time())}")

            if msg_type == "synthesize":
                text        = msg.get("text", "").strip()
                lang_code   = msg.get("lang", "ml-IN")
                style       = msg.get("style", "default")
                custom_desc = msg.get("description")
                stream      = msg.get("stream", False)
                play_steps  = clamp_play_steps(msg.get("play_steps", STREAM_PLAY_STEPS))
                include_full_audio = bool(msg.get("include_full_audio", False))

                if not text:
                    await websocket.send(json.dumps({
                        "type": "error", "request_id": request_id,
                        "message": "Empty text"
                    }))
                    continue

                try:
                    if stream:
                        await _send_streaming_synthesis(
                            websocket,
                            request_id,
                            text,
                            lang_code,
                            style,
                            custom_desc,
                            play_steps,
                            include_full_audio,
                            streaming_mode=msg.get("streaming_mode"),
                        )
                    else:
                        await _send_full_synthesis(websocket, request_id, text, lang_code, style, custom_desc)
                except Exception as e:
                    log.error(f"[{request_id}] Synthesis failed: {e}", exc_info=True)
                    await _send_error(websocket, request_id, str(e))

            elif msg_type == "stream_text_start":
                if request_id in rag_sessions:
                    old = rag_sessions.pop(request_id)
                    old.buffer.cancel()
                    old.cancel_event.set()
                    if old.task:
                        old.task.cancel()

                session = RAGStreamSession(
                    request_id=request_id,
                    language=msg.get("lang", "ml-IN"),
                    style=msg.get("style", "default"),
                    description=msg.get("description"),
                    include_full_audio=bool(msg.get("include_full_audio", False)),
                    play_steps=clamp_play_steps(msg.get("play_steps", STREAM_PLAY_STEPS)),
                    streaming_mode=msg.get("streaming_mode"),
                )
                session.task = asyncio.create_task(stream_text_segments(websocket, session))
                rag_sessions[request_id] = session
                await websocket.send(json.dumps({
                    "type": "stream_text_ready",
                    "request_id": request_id,
                    "play_steps": session.play_steps,
                    "streaming_mode": session.streaming_mode or os.getenv("VEXYL_TTS_STREAMING_MODE", "sentence").lower(),
                }))

            elif msg_type == "stream_text_delta":
                session = rag_sessions.get(request_id)
                if not session:
                    await _send_error(websocket, request_id, "Unknown stream_text request_id")
                    continue

                for segment in session.buffer.push(msg.get("text", "")):
                    try:
                        session.queue.put_nowait(segment)
                    except asyncio.QueueFull:
                        await _send_error(websocket, request_id, "TTS segment queue is full")
                        session.cancel_event.set()
                        break

            elif msg_type == "stream_text_end":
                session = rag_sessions.get(request_id)
                if not session:
                    await _send_error(websocket, request_id, "Unknown stream_text request_id")
                    continue

                for segment in session.buffer.flush():
                    await session.queue.put(segment)
                await session.queue.put(None)
                await session.queue.join()
                rag_sessions.pop(request_id, None)
                await websocket.send(json.dumps({
                    "type": "stream_text_done",
                    "request_id": request_id,
                }))

            elif msg_type == "cancel":
                session = rag_sessions.pop(request_id, None)
                if session:
                    session.buffer.cancel()
                    session.cancel_event.set()
                    try:
                        session.queue.put_nowait(None)
                    except asyncio.QueueFull:
                        pass
                    if session.task:
                        session.task.cancel()
                await websocket.send(json.dumps({
                    "type": "cancelled",
                    "request_id": request_id,
                }))

            elif msg_type == "get_stats":
                await websocket.send(json.dumps({
                    "type": "stats",
                    "cache_size":  len(audio_cache),
                    "cache_hits":  cache_hits,
                    "cache_total": cache_total,
                    "hit_rate":    round(cache_hits / max(cache_total, 1) * 100, 1),
                    "device":      device,
                    "max_active_generations": MAX_ACTIVE_GENERATIONS,
                    "max_queue_size": MAX_QUEUE_SIZE,
                    "stream_play_steps": STREAM_PLAY_STEPS,
                    "streaming_mode": os.getenv("VEXYL_TTS_STREAMING_MODE", "sentence").lower(),
                }))

            elif msg_type == "ping":
                await websocket.send(json.dumps({"type": "pong"}))

    except websockets.exceptions.ConnectionClosed:
        log.info(f"Disconnected: {remote}")
    except Exception as e:
        log.error(f"Handler error: {e}", exc_info=True)
        try:
            await websocket.send(json.dumps({"type": "error", "message": "Internal server error"}))
        except Exception:
            pass
    finally:
        for session in rag_sessions.values():
            session.buffer.cancel()
            session.cancel_event.set()
            if session.task:
                session.task.cancel()
        active_connections -= 1


async def _limited_handler(websocket):
    """Wrap handle_connection with a semaphore to cap concurrent connections."""
    if _conn_semaphore.locked() and _conn_semaphore._value == 0:
        await websocket.close(1013, "Server at capacity")
        log.warning(f"Rejected connection from {websocket.remote_address} — at capacity ({MAX_CONNECTIONS})")
        return
    async with _conn_semaphore:
        await handle_connection(websocket)


# ─── Batch-Capable Connection ────────────────────────────────────────────────
# websockets 16.x rejects POST requests at the HTTP/1.1 parsing level before
# _process_request() is ever called.  We subclass ServerConnection and override
# data_received() to intercept POST requests at the transport level.

class BatchCapableConnection(ServerConnection):
    """ServerConnection subclass that intercepts HTTP POST for batch endpoints."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._post_buffer = b""
        self._is_post: Optional[bool] = None  # None = undetermined
        self._handled_as_http = False

    async def handshake(self, *args, **kwargs):
        """Override to suppress the EOF error when we already handled as HTTP.
        The race: handshake() starts awaiting protocol data, then data_received()
        intercepts POST/OPTIONS and closes the transport, causing an EOF here."""
        try:
            return await super().handshake(*args, **kwargs)
        except Exception:
            if self._handled_as_http:
                return  # suppress — we already sent an HTTP response
            raise

    def data_received(self, data: bytes) -> None:
        # First chunk: determine request type
        if self._is_post is None:
            self._post_buffer = data
            if data[:7] == b"OPTIONS":
                self._handled_as_http = True
                self._send_cors_preflight()
                return
            elif data[:4] == b"POST":
                self._is_post = True
                self._handled_as_http = True
                self._try_handle_post()
                return
            else:
                self._is_post = False
                super().data_received(data)
                return

        if self._is_post:
            # Cap buffer to prevent unbounded memory growth
            max_buffer = BATCH_MAX_BODY_SIZE + 64 * 1024
            if len(self._post_buffer) + len(data) > max_buffer:
                self._send_json_response(413, "Payload Too Large",
                                         {"error": "Request too large"})
                return
            self._post_buffer += data
            self._try_handle_post()
        else:
            super().data_received(data)

    def _try_handle_post(self):
        """Check if we have the full POST request, then handle it."""
        header_end = self._post_buffer.find(b"\r\n\r\n")
        if header_end == -1:
            return  # need more header data

        headers_section = self._post_buffer[:header_end]
        body_start = header_end + 4

        # Parse Content-Length (with validation)
        content_length = 0
        for line in headers_section.decode("utf-8", errors="replace").split("\r\n"):
            if line.lower().startswith("content-length:"):
                try:
                    content_length = int(line.split(":", 1)[1].strip())
                except (ValueError, IndexError):
                    self._send_json_response(400, "Bad Request",
                                             {"error": "Invalid Content-Length"})
                    return
                if content_length < 0 or content_length > BATCH_MAX_BODY_SIZE:
                    self._send_json_response(413, "Payload Too Large",
                                             {"error": "Content-Length exceeds limit"})
                    return
                break

        body_so_far = self._post_buffer[body_start:]
        if len(body_so_far) < content_length:
            return  # need more body data

        # We have the full request
        body = body_so_far[:content_length]
        headers_raw = headers_section.decode("utf-8", errors="replace")
        task = asyncio.ensure_future(self._handle_post(headers_raw, body))
        task.add_done_callback(self._post_task_done)

    def _post_task_done(self, task: asyncio.Task):
        """Callback for POST handler task — log unhandled exceptions."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc:
            log.error(f"[batch] Unhandled POST handler error: {exc}", exc_info=exc)

    async def _handle_post(self, headers_raw: str, body: bytes):
        """Route and handle the POST request."""
        try:
            lines = headers_raw.split("\r\n")
            request_line = lines[0]  # e.g. "POST /batch/synthesize HTTP/1.1"
            parts = request_line.split(" ", 2)
            path = parts[1] if len(parts) > 1 else "/"

            # Parse headers into dict
            headers = {}
            for line in lines[1:]:
                if ":" in line:
                    key, val = line.split(":", 1)
                    headers[key.strip().lower()] = val.strip()

            # API key check (timing-safe)
            if API_KEY:
                client_key = headers.get("x-api-key", "")
                if not hmac.compare_digest(client_key, API_KEY):
                    self._send_json_response(403, "Forbidden",
                                             {"error": "Invalid or missing API key"})
                    return

            if path == "/batch/synthesize":
                await self._handle_batch_synthesize(headers, body)
            else:
                self._send_json_response(404, "Not Found",
                                         {"error": f"Unknown endpoint: {path}"})
        except Exception as e:
            log.error(f"[batch] POST handler error: {e}", exc_info=True)
            self._send_json_response(500, "Internal Server Error",
                                     {"error": "Internal server error"})

    async def _handle_batch_synthesize(self, headers: dict, body: bytes):
        """Handle POST /batch/synthesize — accept JSON for async synthesis."""
        content_type = headers.get("content-type", "")

        if "application/json" not in content_type:
            self._send_json_response(400, "Bad Request",
                                     {"error": "Content-Type must be application/json"})
            return

        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            self._send_json_response(400, "Bad Request",
                                     {"error": "Invalid JSON body"})
            return

        text = payload.get("text", "").strip()
        if not text:
            self._send_json_response(400, "Bad Request",
                                     {"error": "Missing 'text' field"})
            return

        if len(text) > BATCH_MAX_TEXT_LENGTH:
            self._send_json_response(400, "Bad Request",
                                     {"error": f"Text too long ({len(text)} chars). Max {BATCH_MAX_TEXT_LENGTH}"})
            return

        language = payload.get("lang", "ml-IN")
        style = payload.get("style", "default")
        description = payload.get("description")

        # Check job limit
        pending_count = sum(1 for j in _batch_jobs.values()
                           if j.status in (JobStatus.QUEUED, JobStatus.PROCESSING))
        if pending_count >= BATCH_MAX_JOBS:
            self._send_json_response(429, "Too Many Requests",
                                     {"error": f"Too many pending jobs (max {BATCH_MAX_JOBS})"})
            return

        # Create job
        job_id = f"batch_{uuid.uuid4().hex[:16]}"
        job = BatchJob(
            job_id=job_id,
            status=JobStatus.QUEUED,
            text=text,
            language=language,
            style=style,
            created_at=time.time(),
            description=description,
        )
        _batch_jobs[job_id] = job
        await _batch_queue.put(job_id)

        log.info(f"[batch] Job {job_id} queued: {language}/{style}, {len(text)} chars")

        self._send_json_response(201, "Created", {
            "job_id": job_id,
            "status": "queued",
            "language": language,
            "style": style,
            "text_length": len(text),
        })

    def _send_cors_preflight(self):
        """Respond to an OPTIONS preflight request."""
        response = (
            "HTTP/1.1 204 No Content\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
            "Access-Control-Allow-Headers: Content-Type, X-API-Key\r\n"
            "Access-Control-Max-Age: 86400\r\n"
            "Content-Length: 0\r\n"
            "Connection: close\r\n"
            "\r\n"
        ).encode("utf-8")
        try:
            self.transport.write(response)
            self.transport.close()
        except Exception:
            pass

    def _send_json_response(self, status_code: int, status_text: str, body_dict: dict):
        """Write a raw HTTP JSON response to the transport and close."""
        body = json.dumps(body_dict).encode("utf-8")
        response = (
            f"HTTP/1.1 {status_code} {status_text}\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\n"
            f"Access-Control-Allow-Origin: *\r\n"
            f"Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
            f"Access-Control-Allow-Headers: Content-Type, X-API-Key\r\n"
            f"Connection: close\r\n"
            f"\r\n"
        ).encode("utf-8") + body
        try:
            self.transport.write(response)
            self.transport.close()
        except Exception:
            pass


# ─── CORS & HTTP Helpers ──────────────────────────────────────────────────────

_CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, X-API-Key",
}

def _json_response(connection, status: HTTPStatus, body_dict: dict):
    """Helper to build a JSON HTTP response via websockets' connection.respond()."""
    body = json.dumps(body_dict)
    response = connection.respond(status, body)
    response.headers["Content-Type"] = "application/json"
    for k, v in _CORS_HEADERS.items():
        response.headers[k] = v
    return response


def _process_request(connection, request):
    """Intercept HTTP requests before WebSocket upgrade.
    Serves /health, /batch/status/{id}, /batch/result/{id}.
    websockets 16.x API: (ServerConnection, Request) -> Response | None."""

    # ── Health check (no auth required) ──
    if request.path == "/health":
        queued = sum(1 for j in _batch_jobs.values() if j.status == JobStatus.QUEUED)
        return _json_response(connection, HTTPStatus.OK, {
            "status":             "ok",
            "model":              "indic-parler-tts",
            "device":             device,
            "cache_size":         len(audio_cache),
            "cache_capacity":     CACHE_SIZE,
            "cache_hit_rate":     round(cache_hits / max(cache_total, 1) * 100, 1),
            "active_connections": active_connections,
            "max_connections":    MAX_CONNECTIONS,
            "uptime_seconds":     round(time.time() - _server_start_time, 1),
            "batch_jobs_queued":  queued,
            "batch_jobs_total":   len(_batch_jobs),
        })

    # API key check — skip if no key configured
    if API_KEY:
        client_key = request.headers.get("X-API-Key", "")
        if not hmac.compare_digest(client_key, API_KEY):
            log.warning(f"Rejected connection — invalid or missing API key from {request.headers.get('Host', 'unknown')}")
            return connection.respond(HTTPStatus.FORBIDDEN, "Invalid or missing API key")

    # ── Batch status endpoint ──
    if request.path.startswith("/batch/status/"):
        job_id = request.path[len("/batch/status/"):]
        job = _batch_jobs.get(job_id)
        if not job:
            return _json_response(connection, HTTPStatus.NOT_FOUND,
                                  {"error": "Job not found", "job_id": job_id})

        result = {
            "job_id": job.job_id,
            "status": job.status.value,
            "language": job.language,
            "style": job.style,
            "text_length": len(job.text),
            "created_at": job.created_at,
        }
        if job.status == JobStatus.COMPLETED:
            result["audio_b64"] = job.audio_b64
            result["sample_rate"] = job.sample_rate
            result["latency_ms"] = job.latency_ms
            result["completed_at"] = job.completed_at
        elif job.status == JobStatus.FAILED:
            result["error_message"] = job.error_message
            result["completed_at"] = job.completed_at

        return _json_response(connection, HTTPStatus.OK, result)

    # ── Batch result endpoint ──
    if request.path.startswith("/batch/result/"):
        job_id = request.path[len("/batch/result/"):]
        job = _batch_jobs.get(job_id)
        if not job:
            return _json_response(connection, HTTPStatus.NOT_FOUND,
                                  {"error": "Job not found", "job_id": job_id})

        if job.status == JobStatus.COMPLETED:
            return _json_response(connection, HTTPStatus.OK, {
                "job_id": job.job_id,
                "status": "completed",
                "audio_b64": job.audio_b64,
                "sample_rate": job.sample_rate,
                "language": job.language,
                "style": job.style,
                "latency_ms": job.latency_ms,
            })
        elif job.status == JobStatus.FAILED:
            return _json_response(connection, HTTPStatus.OK, {
                "job_id": job.job_id,
                "status": "failed",
                "error_message": job.error_message,
            })
        else:
            # Still processing — 202 Accepted
            return _json_response(connection, HTTPStatus.ACCEPTED, {
                "job_id": job.job_id,
                "status": job.status.value,
                "language": job.language,
                "style": job.style,
            })

    # ── Fix headers mangled by reverse proxies (e.g. Cloudflare Tunnel) ──
    if request.headers.get("Sec-WebSocket-Key"):
        conn_values = [v.lower() for v in request.headers.get_all("Connection")]
        if not any("upgrade" in v for v in conn_values):
            log.info(f"Fixing Connection header mangled by reverse proxy (was: {request.headers.get('Connection')})")
            del request.headers["Connection"]
            request.headers["Connection"] = "Upgrade"

        upgrade_values = [v.lower() for v in request.headers.get_all("Upgrade")]
        if not any("websocket" in v for v in upgrade_values):
            log.info(f"Fixing Upgrade header mangled by reverse proxy (was: {request.headers.get('Upgrade')})")
            if "Upgrade" in request.headers:
                del request.headers["Upgrade"]
            request.headers["Upgrade"] = "websocket"

    return None


# ─── Main ──────────────────────────────────────────────────────────────────────
async def main():
    global _conn_semaphore, _generation_semaphore, _server_start_time, _batch_queue
    global _batch_worker_task, _batch_cleanup_task

    load_model()

    if TTS_PROVIDER == "bhashini":
        _init_bhashini()
    else:
        log.info("Running warm-up inference...")
        _synthesize_sync("Hello", "en-IN", "default")
        log.info("Warm-up complete")

    _conn_semaphore = asyncio.Semaphore(MAX_CONNECTIONS)
    _generation_semaphore = asyncio.Semaphore(MAX_ACTIVE_GENERATIONS)
    _server_start_time = time.time()

    # Initialize batch processing
    _batch_queue = asyncio.Queue()
    _batch_worker_task = asyncio.create_task(_batch_worker())
    _batch_cleanup_task = asyncio.create_task(_batch_cleanup_loop())

    log.info(f"Starting VEXYL-TTS WebSocket server on ws://{HOST}:{PORT}")

    stop_event = asyncio.Event()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda s=sig: _handle_signal(s, stop_event))

    async with websockets.serve(
        _limited_handler,
        HOST,
        PORT,
        max_size=512 * 1024,         # 512KB max message (text requests are small)
        ping_interval=30,
        ping_timeout=10,
        close_timeout=5,
        process_request=_process_request,
        create_connection=BatchCapableConnection,
    ) as server:
        log.info(f"VEXYL-TTS server ready | ws://{HOST}:{PORT} | max_conn={MAX_CONNECTIONS} | batch=enabled")
        await stop_event.wait()

        log.info("Shutting down... cancelling batch tasks")
        _batch_worker_task.cancel()
        _batch_cleanup_task.cancel()
        try:
            await _batch_worker_task
        except asyncio.CancelledError:
            pass
        try:
            await _batch_cleanup_task
        except asyncio.CancelledError:
            pass

        log.info("Closing active connections")
        server.close()
        await server.wait_closed()
        log.info("Server stopped cleanly")


def _handle_signal(sig, stop_event: asyncio.Event):
    log.info(f"Received {signal.Signals(sig).name}, initiating shutdown...")
    stop_event.set()


if __name__ == "__main__":
    asyncio.run(main())
