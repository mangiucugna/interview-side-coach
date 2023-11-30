import gradio as gr
from transformers import pipeline
import numpy as np
import torch

gpu_available = torch.cuda.is_available()  # Nvidia GPU
mps_available = (
    hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
)  # Apple Metal (M1/M2/M3)
xpu_available = hasattr(torch, "xpu") and torch.xpu.is_available()  # Intel XPU

transcriber = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-base.en",
    device="cuda"
    if gpu_available
    else "xpu"
    if xpu_available
    else "mps"
    if mps_available
    else "cpu",
)


def process_orca(transcription, memory, job_title):
    import json_repair
    import requests

    output = ""
    m = ""
    for t in memory:
        m += f"<|im_start|>user {t[0]}<|im_end|><|im_start|>assistant {t[0]}<|im_end|>"
    prompt = """
<|im_start|>system
You are being interviewed for the position of {job_title}. Please answer the question concisely. The question posed is a transcription so it might contain typos or artifacts, take that into account
<|im_end|>
{m}
<|im_start|>user
{transcription}<|im_end|>
<|im_start|>assistant
""".format(
        job_title=job_title, m=m, transcription=transcription
    )
    data = {"model": "mistral-openorca:latest", "prompt": prompt}
    try:
        response = requests.post(
            url="http://127.0.0.1:11434/api/generate",
            headers={"Content-Type": "application/json"},
            json=data,
            stream=True,
            timeout=30,
        )
    except Exception as e:
        print(e)
        return "--There was an error calling Ollama--"
    for chunk in response.text.split("\n"):
        chunk = json_repair.loads(chunk)
        if isinstance(chunk, dict):
            output += chunk.get("response") or ""
            if e := chunk.get("error"):
                print(e)

    return output


def process_oai(transcription, memory, job_title, api_key):
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    model = "gpt-3.5-turbo"
    m = []
    for t in memory:
        m.append({"role": "user", "content": t[0]})
        m.append({"role": "assistant", "content": t[1]})
    messages = []
    messages.append(
        {
            "role": "system",
            "content": f"You are being interviewed for the position of {job_title}. Please answer the question concisely. The question posed is a transcription so it might contain typos or artifacts, take that into account",
        }
    )
    messages.extend(m)
    messages.append({"role": "user", "content": transcription})

    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content


def transcribe(cookie, new_chunk, job_title, api_key):
    sr, y = new_chunk
    y = y.astype(np.float32)
    if isinstance(
        y[0], np.ndarray
    ):  # If we get a stereo signal, let's average those waves
        y = np.mean(y, axis=1)
    abs_y = np.abs(y)
    avg_y = np.mean(np.abs(y))
    if avg_y < 100:
        if cookie["transcription"] != "":
            if api_key:
                answer = process_oai(
                    cookie["transcription"], cookie["memory"], job_title, api_key
                )
            else:
                answer = process_orca(
                    cookie["transcription"], cookie["memory"], job_title
                )
            cookie["answer"] += "🤖 " + answer + "\n"

            cookie["memory"].append((cookie["transcription"], answer))
            cookie["memory"] = cookie["memory"][-20:]

            cookie["hear"] += "🎙️ " + cookie["transcription"] + "\n"
            cookie["transcription"] = ""
            cookie["stream"] = None
    else:
        y /= np.max(abs_y)
        if cookie["stream"] is not None:
            cookie["stream"] = np.concatenate([cookie["stream"], y])
        else:
            cookie["stream"] = y

        cookie["transcription"] = transcriber(
            {"sampling_rate": sr, "raw": cookie["stream"]}
        )["text"]

    return cookie, cookie["hear"] + cookie["transcription"], cookie["answer"]


with gr.Blocks() as demo:
    cookie = gr.State(
        {"stream": None, "hear": "", "answer": "", "transcription": "", "memory": []}
    )
    with gr.Row():
        api_key = gr.Textbox(value="", type="password", label="OpenAI API key")
        job_title = gr.Textbox(
            placeholder="Job title you are interviewed for..", label="Job Title"
        )
        audio = gr.Audio(sources=["microphone"], streaming=True)
    with gr.Row():
        hear = gr.Textbox(placeholder="Processing...", value="", label="What I hear")
    with gr.Row():
        answer = gr.Textbox(placeholder="Processing...", value="", label="My answer")
    audio.stream(
        transcribe,
        inputs=[cookie, audio, job_title, api_key],
        outputs=[cookie, hear, answer],
        show_progress=False,
    )

if __name__ == "__main__":
    demo.launch(show_error=True, share=True)
