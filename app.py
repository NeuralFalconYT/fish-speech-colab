# Gradio app code modified from here:
# https://huggingface.co/spaces/artificialguybr/fish-s2-pro-zero

# I added automatic reference audio transcript generation using Whisper.
# However, due to GPU problems, I can't do anything right now.

# root_path = '/teamspace/studios/this_studio' # @param ['/content', '/root', '/kaggle/working','/teamspace/studios/this_studio']
# %cd $root_path/fish-speech-colab

import os
import sys
import subprocess
import traceback
import gradio as gr
import numpy as np
import librosa
import torch
import gc
from pathlib import Path
from huggingface_hub import snapshot_download
from faster_whisper import WhisperModel
from pathlib import Path

root_path = str(Path.cwd().parent)
 
# fish_speech_path = f"{root_path}/fish-speech-colab/fish-speech"
sys.path.append(fish_speech_path)

from fish_speech.models.text2semantic.inference import init_model, generate_long

# ==========================================
# 1. Model Initialization (Fish Speech TTS)
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
precision = torch.bfloat16

checkpoint_dir = f"{root_path}/fish-speech-colab/model"
config_path = f"{root_path}/fish-speech-colab/fish-speech/fish_speech/configs/modded_dac_vq.yaml"
llama_model, decode_one_token = init_model(
    checkpoint_path=checkpoint_dir,
    device=device,
    precision=precision,
    compile=False,
)

with torch.device(device):
    llama_model.setup_caches(
        max_batch_size=1,
        max_seq_len=llama_model.config.max_seq_len,
        dtype=next(llama_model.parameters()).dtype,
    )

def load_codec(codec_checkpoint_path, target_device, target_precision):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(config_path)
    codec = instantiate(cfg)

    state_dict = torch.load(codec_checkpoint_path, map_location="cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if any("generator" in k for k in state_dict):
        state_dict = {
            k.replace("generator.", ""): v
            for k, v in state_dict.items()
            if "generator." in k
        }

    codec.load_state_dict(state_dict, strict=False)
    codec.eval()
    codec.to(device=target_device, dtype=target_precision)
    return codec

codec_model = load_codec(os.path.join(checkpoint_dir, "codec.pth"), device, precision)

@torch.no_grad()
def encode_reference_audio(audio_path):
    wav_np, _ = librosa.load(audio_path, sr=codec_model.sample_rate, mono=True)
    wav = torch.from_numpy(wav_np).to(device)
    model_dtype = next(codec_model.parameters()).dtype
    audios = wav[None, None, :].to(dtype=model_dtype)
    audio_lengths = torch.tensor([wav.shape[0]], device=device, dtype=torch.long)
    indices, feature_lengths = codec_model.encode(audios, audio_lengths)
    return indices[0, :, : feature_lengths[0]]

@torch.no_grad()
def decode_codes_to_audio(merged_codes):
    audio = codec_model.from_indices(merged_codes[None])
    return audio[0, 0]

def estimate_duration(text):
    words = len(text.split())
    seconds = max(5, int(words * 0.4))
    return seconds


# ==========================================
# 2. Whisper Transcription Setup
# ==========================================
LANGUAGE_CODE = {
    'Akan': 'aka', 'Albanian': 'sq', 'Amharic': 'am', 'Arabic': 'ar', 'Armenian': 'hy',
    'Assamese': 'as', 'Azerbaijani': 'az', 'Basque': 'eu', 'Bashkir': 'ba', 'Bengali': 'bn',
    'Bosnian': 'bs', 'Bulgarian': 'bg', 'Burmese': 'my', 'Catalan': 'ca', 'Chinese': 'zh',
    'Croatian': 'hr', 'Czech': 'cs', 'Danish': 'da', 'Dutch': 'nl', 'English': 'en',
    'Estonian': 'et', 'Faroese': 'fo', 'Finnish': 'fi', 'French': 'fr', 'Galician': 'gl',
    'Georgian': 'ka', 'German': 'de', 'Greek': 'el', 'Gujarati': 'gu', 'Haitian Creole': 'ht',
    'Hausa': 'ha', 'Hebrew': 'he', 'Hindi': 'hi', 'Hungarian': 'hu', 'Icelandic': 'is',
    'Indonesian': 'id', 'Italian': 'it', 'Japanese': 'ja', 'Kannada': 'kn', 'Kazakh': 'kk',
    'Korean': 'ko', 'Kurdish': 'ckb', 'Kyrgyz': 'ky', 'Lao': 'lo', 'Lithuanian': 'lt',
    'Luxembourgish': 'lb', 'Macedonian': 'mk', 'Malay': 'ms', 'Malayalam': 'ml', 'Maltese': 'mt',
    'Maori': 'mi', 'Marathi': 'mr', 'Mongolian': 'mn', 'Nepali': 'ne', 'Norwegian': 'no',
    'Norwegian Nynorsk': 'nn', 'Pashto': 'ps', 'Persian': 'fa', 'Polish': 'pl', 'Portuguese': 'pt',
    'Punjabi': 'pa', 'Romanian': 'ro', 'Russian': 'ru', 'Serbian': 'sr', 'Sinhala': 'si',
    'Slovak': 'sk', 'Slovenian': 'sl', 'Somali': 'so', 'Spanish': 'es', 'Sundanese': 'su',
    'Swahili': 'sw', 'Swedish': 'sv', 'Tamil': 'ta', 'Telugu': 'te', 'Thai': 'th',
    'Turkish': 'tr', 'Ukrainian': 'uk', 'Urdu': 'ur', 'Uzbek': 'uz', 'Vietnamese': 'vi',
    'Welsh': 'cy', 'Yiddish': 'yi', 'Yoruba': 'yo', 'Zulu': 'zu'
}

def get_language_name(code):
    """Retrieves the full language name from its code."""
    for name, value in LANGUAGE_CODE.items():
        if value == code:
            return name
    return None

def transcribe_audio(audio_path, language="Automatic"):
    if not audio_path:
        return "", ""
        
    gr.Info("Transcribing audio...")
    device_w = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device_w == "cuda" else "int8"
    whisper_path = f"{root_path}/fish-speech-colab/faster-whisper-large-v3-turbo-ct2"
    
    # Initialize model
    model = WhisperModel(
        whisper_path,
        device=device_w,
        compute_type=compute_type
    )
    
    if language == "Automatic":
        segments, info = model.transcribe(audio_path)
    else:
        lang_code = LANGUAGE_CODE.get(language)
        segments, info = model.transcribe(audio_path, language=lang_code)
        
    detected_lang_code = info.language
    detected_language = get_language_name(detected_lang_code) or detected_lang_code
    transcript = " ".join([s.text for s in segments])

    # Unload Whisper model to free memory for Fish Speech
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return transcript.strip(), detected_language


# ==========================================
# 3. TTS Generation Inference
# ==========================================
def tts_inference(
    text,
    ref_audio,
    ref_text,
    max_new_tokens,
    chunk_length,
    top_p,
    repetition_penalty,
    temperature,
):
    try:
        if not text or not text.strip():
            raise gr.Error("Please enter some text to synthesize.")

        est = estimate_duration(text)
        gr.Info(f"Generating audio... estimated ~{est}s depending on text length.")

        prompt_tokens_list = None
        if ref_audio is not None and ref_text and ref_text.strip():
            prompt_tokens_list = [encode_reference_audio(ref_audio).cpu()]

        generator = generate_long(
            model=llama_model,
            device=device,
            decode_one_token=decode_one_token,
            text=text,
            num_samples=1,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=30,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            compile=False,
            iterative_prompt=True,
            chunk_length=chunk_length,
            prompt_text=[ref_text] if ref_text else None,
            prompt_tokens=prompt_tokens_list,
        )

        codes = []
        for response in generator:
            if response.action == "sample":
                codes.append(response.codes)
            elif response.action == "next":
                break

        if not codes:
            raise gr.Error("No audio was generated. Please check your input text.")

        merged_codes = codes[0] if len(codes) == 1 else torch.cat(codes, dim=1)
        merged_codes = merged_codes.to(device)

        audio_waveform = decode_codes_to_audio(merged_codes)
        audio_np = audio_waveform.cpu().float().numpy()
        audio_np = (audio_np * 32767).clip(-32768, 32767).astype(np.int16)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return (codec_model.sample_rate, audio_np)

    except gr.Error:
        raise
    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"Inference error: {str(e)}")


# ==========================================
# 4. Custom CSS & JS for Clickable Tags
# ==========================================
CUSTOM_CSS = """
.tag-container {
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 8px !important;
    margin-top: 5px !important;
    margin-bottom: 10px !important;
    border: none !important;
    background: transparent !important;
}
.tag-btn {
    min-width: fit-content !important;
    width: auto !important;
    height: 32px !important;
    font-size: 13px !important;
    background: #eef2ff !important;
    border: 1px solid #c7d2fe !important;
    color: #3730a3 !important;
    border-radius: 6px !important;
    padding: 0 10px !important;
    margin: 0 !important;
    box-shadow: none !important;
}
.tag-btn:hover {
    background: #c7d2fe !important;
    transform: translateY(-1px);
}
.gradio-container {
    font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
}
"""

INSERT_TAG_JS = """
(tag_val, current_text) => {
    current_text = current_text || "";
    const textarea = document.querySelector('#main_textbox textarea');
    if (!textarea) return current_text + " " + tag_val;
    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;
    let prefix = " ";
    let suffix = " ";
    if (start === 0) prefix = "";
    else if (current_text[start - 1] === ' ') prefix = "";
    if (end < current_text.length && current_text[end] === ' ') suffix = "";
    return current_text.slice(0, start) + prefix + tag_val + suffix + current_text.slice(end);
}
"""

TAGS = [
    "[pause]", "[emphasis]", "[laughing]", "[inhale]", "[chuckle]", "[tsk]",
    "[singing]", "[excited]", "[laughing tone]", "[interrupting]", "[chuckling]",
    "[excited tone]", "[volume up]", "[echo]", "[angry]", "[low volume]", "[sigh]",
    "[low voice]", "[whisper]", "[screaming]", "[shouting]", "[loud]", "[surprised]",
    "[short pause]", "[exhale]", "[delight]", "[panting]", "[audience laughter]",
    "[with strong accent]", "[volume down]", "[clearing throat]", "[sad]",
    "[moaning]", "[shocked]", "[professional broadcast tone]", "[whisper in small voice]",
]


# ==========================================
# 5. Gradio UI Blocks
# ==========================================
with gr.Blocks(title="Fish Audio S2 Pro", theme=gr.themes.Soft(),css=CUSTOM_CSS) as demo:

    gr.Markdown(
        f"""
        <div style="text-align:center;max-width:900px;margin:0 auto;padding:24px 0 8px">
            <h1 style="font-size:2.4rem;font-weight:800;color:#1E3A8A;margin-bottom:6px">
                🐟 Fish Audio S2 Pro
            </h1>
            <p style="font-size:1.05rem;color:#4B5563;margin-bottom:8px">
                State-of-the-Art Dual-Autoregressive Text-to-Speech &nbsp;·&nbsp;
                <a href="https://huggingface.co/fishaudio/s2-pro" target="_blank" style="color:#2563EB">Model Page ↗</a>
                &nbsp;·&nbsp;
                <a href="https://github.com/fishaudio/fish-speech" target="_blank" style="color:#2563EB">GitHub ↗</a>
            </p>
            <p style="font-size:0.95rem;color:#6B7280">
                80+ languages supported · Zero-shot voice cloning · 15,000+ inline emotion tags
            </p>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=5):
            gr.Markdown("### ✍️ Input Text")
            
            # Textbox element ID helps JS locate it
            text_input = gr.Textbox(
                show_label=False,
                placeholder="Type the text you want to synthesize.\nLanguage is auto-detected — write in any language.\nAdd emotion tags like [laugh] or [whisper in small voice] anywhere in the text.",
                lines=7,
                elem_id="main_textbox"
            )

            with gr.Accordion("🎙️ Voice Cloning [ Optional ]", open=False):
                gr.Markdown(
                    "Upload a clean **5–10 second** audio clip. "
                    "The text will be **automatically transcribed**, but you can review and edit it if needed. "
                    "Fish-speech supports generating more languages than the transcriber."
                )
                
                with gr.Row():
                    ref_audio = gr.Audio(label="Reference Audio", type="filepath")
                    
                    with gr.Column():
                        whisper_lang = gr.Dropdown(
                            label="Transcription Language",
                            choices=["Automatic"] + list(LANGUAGE_CODE.keys()),
                            value="Automatic",
                            info="Select language to help Whisper transcribe, or leave as Automatic."
                        )
                        detected_lang_ui = gr.Textbox(
                            label="Detected Audio Language", 
                            interactive=False,
                            placeholder="Language detected by Whisper..."
                        )

                ref_text = gr.Textbox(
                    label="Reference Audio Transcription (Editable)",
                    placeholder="Transcription of the reference audio will appear here automatically. Feel free to edit it manually.",
                    lines=3
                )

            with gr.Accordion("⚙️ Advanced Settings", open=False):
                with gr.Row():
                    max_new_tokens = gr.Slider(0, 2048, 1024, step=8, label="Max New Tokens (0 = auto)")
                    chunk_length = gr.Slider(100, 400, 200, step=8, label="Chunk Length")
                with gr.Row():
                    top_p = gr.Slider(0.1, 1.0, 0.7, step=0.01, label="Top-P")
                    repetition_penalty = gr.Slider(0.9, 2.0, 1.2, step=0.01, label="Repetition Penalty")
                    temperature = gr.Slider(0.1, 1.0, 0.7, step=0.01, label="Temperature")

            generate_btn = gr.Button("🚀 Generate Audio", variant="primary", size="lg")

        # ====== RIGHT COLUMN (Moved tags here) ======
        with gr.Column(scale=4):
            gr.Markdown("### 🎧 Result")
            audio_output = gr.Audio(
                label="Generated Audio",
                type="numpy",
                interactive=False,
                autoplay=True,
            )

            gr.Markdown(
                """
                <div style="background:#EFF6FF;padding:16px;border-radius:10px;margin-top:16px;margin-bottom:8px;">
                    <h4 style="margin:0 0 8px;color:#1D4ED8">🏷️ Supported Emotion Tags</h4>
                    <p style="font-size:0.85rem;color:#374151;margin:0">
                        15,000+ unique tags supported. Click a tag below to insert it at your cursor position, or type free-form descriptions yourself.
                    </p>
                </div>
                """
            )

            # Interactive Tag Buttons are now located under the Audio Output
            with gr.Row(elem_classes=["tag-container"]):
                for tag in TAGS:
                    btn = gr.Button(value=tag, elem_classes=["tag-btn"])
                    btn.click(
                        fn=None,
                        inputs=[btn, text_input],
                        outputs=[text_input],
                        js=INSERT_TAG_JS
                    )

    # Event trigger: automatically transcribe when audio is uploaded or language dropdown is changed
    ref_audio.change(
        fn=transcribe_audio,
        inputs=[ref_audio, whisper_lang],
        outputs=[ref_text, detected_lang_ui]
    )
    whisper_lang.change(
        fn=transcribe_audio,
        inputs=[ref_audio, whisper_lang],
        outputs=[ref_text, detected_lang_ui]
    )

    gr.Markdown(
        """
        <div style="background:#F0FDF4;padding:16px;border-radius:10px;margin-top:8px">
            <h4 style="margin:0 0 8px;color:#166534">🌍 Supported Languages</h4>
            <p style="font-size:0.9rem;color:#374151;margin:0">
                <strong>Tier 1:</strong> Japanese · English · Chinese &nbsp;|&nbsp;
                <strong>Tier 2:</strong> Korean · Spanish · Portuguese · Arabic · Russian · French · German<br>
                <strong>Also supported:</strong> sv, it, tr, no, nl, cy, eu, ca, da, gl, ta, hu, fi, pl, et, hi,
                la, ur, th, vi, jw, bn, yo, sl, cs, sw, nn, he, ms, uk, id, kk, bg, lv, my, tl, sk, ne, fa,
                af, el, bo, hr, ro, sn, mi, yi, am, be, km, is, az, sd, br, sq, ps, mn, ht, ml, sr, sa, te,
                ka, bs, pa, lt, kn, si, hy, mr, as, gu, fo, and more.
                Language is <strong>auto-detected</strong> from the input text — no configuration needed.
            </p>
        </div>
        """
    )

    generate_btn.click(
        fn=tts_inference,
        inputs=[text_input, ref_audio, ref_text, max_new_tokens, chunk_length, top_p, repetition_penalty, temperature],
        outputs=[audio_output],
    )

# if __name__ == "__main__":
#     demo.launch(share=True, debug=True)

import click
@click.command()
@click.option("--debug", is_flag=True, default=False, help="Enable debug mode.")
@click.option("--share", is_flag=True, default=False, help="Enable sharing of the interface.")
def run_demo(share,debug):
    global demo
    demo.queue(max_size=10).launch(share=share,debug=debug)
if __name__ == "__main__":
    run_demo()
