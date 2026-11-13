"""Local text-to-speech for the podcast: Piper voices -> one WAV -> MP3 (ffmpeg). Nothing leaves the Pi.

Pronunciation fixes live in config/pronunciation.yaml (written-form -> spoken-form); they are applied to the text sent to
the voice only, never to the transcript.
"""
from __future__ import annotations

import logging
import re
import subprocess
import tempfile
import wave
from pathlib import Path

import yaml

log = logging.getLogger(__name__)
CITE = re.compile(r"\s*\[i\d+\]")
PAUSE_S = 0.35                     # silence between speaker turns
SEGMENT_PAUSE_S = 0.9              # extra silence where the show moves from the news round-up to the deep dive


def load_lexicon(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text()) or {}
    return {str(k): str(v) for k, v in (data.get("replace") or {}).items()}


def speakable(text: str, lexicon: dict[str, str]) -> str:
    """Strip citation tags and markup, then apply the pronunciation lexicon (whole words, longest entry first)."""
    t = CITE.sub("", text).replace("**", "")
    for written in sorted(lexicon, key=len, reverse=True):
        t = re.sub(rf"(?<![\w-]){re.escape(written)}(?![\w-])", lexicon[written], t)
    return re.sub(r"\s{2,}", " ", t).strip()


def render(turns: list[dict], voices: dict[str, dict], voice_dir: Path, lexicon: dict[str, str], out_mp3: Path,
           bitrate: str = "64k") -> dict:
    """turns: [{speaker: 'A'|'B', text}]; voices: {'A': {model, length_scale}, 'B': {...}}. Returns {seconds, bytes}."""
    from piper import PiperVoice, SynthesisConfig
    loaded = {k: (PiperVoice.load(str(voice_dir / f"{v['model']}.onnx")), SynthesisConfig(length_scale=v.get("length_scale")))
              for k, v in voices.items()}
    out_mp3.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="lit-tts-") as tmp:
        wav_path = Path(tmp) / "episode.wav"
        samples, rate, last_seg = 0, None, None
        with wave.open(str(wav_path), "wb") as w:
            for t in turns:
                voice, cfg = loaded[t["speaker"]]
                text = speakable(t["text"], lexicon)
                if not text:
                    continue
                seg = t.get("segment", "news")
                if rate is not None and last_seg is not None and seg != last_seg:
                    w.writeframes(b"\0" * int(SEGMENT_PAUSE_S * rate) * 2)
                    samples += int(SEGMENT_PAUSE_S * rate)
                last_seg = seg
                for chunk in voice.synthesize(text, syn_config=cfg):
                    if rate is None:
                        rate = chunk.sample_rate
                        w.setnchannels(chunk.sample_channels)
                        w.setsampwidth(chunk.sample_width)
                        w.setframerate(rate)
                    elif chunk.sample_rate != rate:
                        raise RuntimeError("the two voices have different sample rates; pick voices of the same quality tier")
                    w.writeframes(chunk.audio_int16_bytes)
                    samples += len(chunk.audio_int16_bytes) // 2
                gap = int(PAUSE_S * rate)
                w.writeframes(b"\0" * gap * 2)
                samples += gap
        if rate is None:
            raise RuntimeError("nothing to speak")
        tmp_mp3 = out_mp3.with_suffix(".mp3.tmp")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav_path), "-ac", "1", "-b:a", bitrate, "-f", "mp3", str(tmp_mp3)],
                       check=True)
        tmp_mp3.replace(out_mp3)
    return {"seconds": round(samples / rate, 1), "bytes": out_mp3.stat().st_size}
