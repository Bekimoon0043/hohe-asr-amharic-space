---
title: Hohe Amharic ASR
emoji: 🎙️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
license: apache-2.0
---

# Hohe Amharic ASR Space

Gradio Space for [snapwre/hohe-asr-amharic](https://huggingface.co/snapwre/hohe-asr-amharic).

## Features
- Automatic Speech Recognition for Amharic
- Optional KenLM language model decoding via pyctcdecode
- Clean API endpoint (`/call/predict`) ready for n8n / HTTP clients

## How to use with n8n

**POST** `https://<your-username>-hohe-asr-amharic-space.hf.space/call/predict`

```json
{
  "data": [
    {
      "path": "data:audio/wav;base64,<BASE64_AUDIO>",
      "orig_name": "input_audio.wav"
    }
  ]
}
```

Then poll the returned `event_id` with a GET request.

See the original setup notes for full n8n configuration.
