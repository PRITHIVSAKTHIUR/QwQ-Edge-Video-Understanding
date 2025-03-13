# QwQ Edge - Multimodal Chatbot with Text, Image, and Video Processing

**QwQ Edge** is a multimodal chatbot that integrates text, image, and video processing capabilities. It leverages state-of-the-art models like **Qwen2VL**, **Stable Diffusion XL**, and **FastThink** for text generation, image generation, and video analysis. The chatbot supports text-to-speech (TTS) functionality, image generation, and video summarization, making it a versatile tool for various applications.

## Features

1. **Text Generation**:
   - Powered by the `FastThink-0.5B-Tiny` model for efficient text generation.
   - Supports conversational history and context-aware responses.

2. **Image Generation**:
   - Uses **Stable Diffusion XL** for high-quality image generation.
   - Customizable parameters like seed, resolution, guidance scale, and more.

3. **Video Processing**:
   - Utilizes **Qwen2VL** for video analysis and summarization.
   - Downsamples videos to 10 frames and processes them for insights.

4. **Text-to-Speech (TTS)**:
   - Supports two TTS voices (`en-US-JennyNeural` and `en-US-GuyNeural`).
   - Converts generated text into speech for an interactive experience.

5. **Multimodal Input**:
   - Accepts text, images, and videos as input.
   - Processes multimodal inputs seamlessly using **Qwen2VL**.

6. **Gradio Interface**:
   - Provides an intuitive and interactive web interface for users.
   - Supports queuing, progress tracking, and example-based interactions.

---

## Installation

To set up the project locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/QwQ-Edge.git
   cd QwQ-Edge
   ```

2. **Install Dependencies**:
   Ensure you have Python 3.8+ installed. Then, install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Environment Variables**:
   Create a `.env` file in the root directory and add the following variables:
   ```plaintext
   MODEL_VAL_PATH=SG161222/RealVisXL_V4.0_Lightning
   MAX_IMAGE_SIZE=4096
   USE_TORCH_COMPILE=0
   ENABLE_CPU_OFFLOAD=0
   BATCH_SIZE=1
   ```

4. **Run the Application**:
   Start the Gradio interface:
   ```bash
   python app.py
   ```
   The application will launch locally, and you can access it via the provided URL.

---

## Usage

### 1. **Text Generation**
   - Enter text in the input box and press `Enter`.
   - The chatbot will generate a response based on the input.

### 2. **Image Generation**
   - Use the `@image` command followed by a prompt to generate images.
   - Example: `@image A futuristic cityscape at night`.

### 3. **Video Processing**
   - Use the `@qwen2vl-video` command followed by a prompt and upload a video file.
   - Example: `@qwen2vl-video Describe the video`.

### 4. **Text-to-Speech (TTS)**
   - Use the `@tts1` or `@tts2` command followed by text to generate speech.
   - Example: `@tts1 Who is Nikola Tesla?`.

### 5. **Multimodal Input**
   - Upload images or videos along with text queries for combined processing.
   - Example: `Extract text from the image` + upload an image.

---

## Examples

Here are some example inputs you can try:

1. **Text Generation**:
   - `Python Program for Array Rotation`

2. **Image Generation**:
   - `@image Chocolate dripping from a donut`

3. **Video Processing**:
   - `@qwen2vl-video Summarize the event in video` + upload a video file.

4. **Text-to-Speech**:
   - `@tts2 What causes rainbows to form?`

5. **Multimodal Input**:
   - `Extract JSON from the image` + upload an image.

---

## Models Used

1. **FastThink-0.5B-Tiny**:
   - A lightweight text generation model for efficient conversational AI.

2. **Qwen2VL-OCR-2B-Instruct**:
   - A multimodal model for text, image, and video processing.

3. **Stable Diffusion XL**:
   - A state-of-the-art image generation model.

4. **Edge TTS**:
   - A text-to-speech engine for generating speech from text.

---

## Configuration

You can customize the following parameters in the `.env` file:

- `MODEL_VAL_PATH`: Path to the Stable Diffusion XL model.
- `MAX_IMAGE_SIZE`: Maximum size for generated images.
- `USE_TORCH_COMPILE`: Enable/disable model compilation for speedup.
- `ENABLE_CPU_OFFLOAD`: Enable/disable CPU offloading for memory optimization.
- `BATCH_SIZE`: Number of images to generate in a batch.
