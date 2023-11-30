# Interview Side Coach

A project to have an AI helping you out during an interview, especially those idiotic ones that just ask quizzes

## Installation

### 1. Clone this repository

```
git clone https://github.com/mangiucugna/interview-side-coach.git
```

### 2. Install requirements

Create a venv and install requirements

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. LLM backend

You have two choices, either run locally on your machine or use GPT-3.5 and spend moneyz

#### 3a. Run locally
Visit https://github.com/jmorganca/ollama to install Ollama then run
```
ollama pull mistral-openorca
```
and
```
ollama serve
```

to start the server

#### 3b. Create a valid OpenAI API key

If you'd like to use GPT and have an OpenAI account visit https://platform.openai.com to create an API key.

Then type the API key in the web UI

### 4. Install prerequisites

Install FFMPEG: https://ffmpeg.org/download.html

For Mac (using brew)

```
brew install ffmpeg
```

### 5. Launch the backend
Open another terminal window

```
source .venv/bin/activate
python app.py
```
if you are using Ollama do not forget to run
```
ollama serve
```
in another window

### 6. Open the UI on your phone
The backend will output something like
```
Running on local URL:  http://127.0.0.1:7860
Running on public URL: https://abgghsfhgjk1jhk33.gradio.live
```
Open the public URL with your phone and make sure it's close enough to your computer speakers.

Compile the required fields and press "Record"

### 7. Profit!!!
