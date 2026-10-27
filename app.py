import re
import os
import torch
import gradio as gr
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoProcessor, AutoModelForCTC, pipeline
from pyctcdecode import BeamSearchDecoderCTC

MODEL_ID = "snapwre/hohe-asr-amharic"

# Download the full model (including LM files if present)
print("Downloading model...")
model_path = snapshot_download(repo_id=MODEL_ID)

# Try to load language model decoder from the lm/ folder
decoder = None
lm_dir = os.path.join(model_path, "lm")
if os.path.isdir(lm_dir):
    try:
        print(f"Loading KenLM decoder from {lm_dir}...")
        decoder = BeamSearchDecoderCTC.load_from_dir(lm_dir)
        print("Language model decoder loaded successfully.")
    except Exception as e:
        print(f"Could not load LM decoder: {e}. Falling back to greedy decoding.")
        decoder = None
else:
    print("No lm/ folder found. Using standard greedy decoding.")

# Load processor and model
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForCTC.from_pretrained(MODEL_ID)

# Create ASR pipeline
asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    decoder=decoder,
    device=0 if torch.cuda.is_available() else -1,
)

def transcribe(audio_path):
    if not audio_path:
        return ""

    result = asr_pipeline(audio_path)
    text = result.get("text", "") if isinstance(result, dict) else str(result)

    # Strip [AMH] prefix tag or any bracketed tokens
    cleaned_text = re.sub(r"^\s*\[AMH\]\s*", "", text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r"\[.*?\]", "", cleaned_text).strip()

    return cleaned_text

# Gradio Interface
demo = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(type="filepath", label="Upload Audio (Amharic)"),
    outputs=gr.Textbox(label="Amharic Transcript"),
    title="Hohe Amharic ASR",
    description="Speech-to-text for Amharic using snapwre/hohe-asr-amharic. Ready for n8n integration via /call/predict.",
    api_name="predict",
)

if __name__ == "__main__":
    demo.launch()
