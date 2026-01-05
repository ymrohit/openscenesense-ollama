# OpenSceneSense Ollama

**OpenSceneSense Ollama** is a powerful Python package that brings advanced video analysis capabilities using Ollama's local models. By leveraging local AI models, this package offers frame analysis, audio transcription, dynamic frame selection, and comprehensive video summaries without relying on cloud-based APIs.

## Table of Contents

1. [🚀 Why OpenSceneSense Ollama?](#-why-openscenesense-ollama)
2. [🌟 Features](#-features)
3. [📦 Installation](#-installation)
4. [🛠️ Usage](#-usage)
5. [⚙️ Configuration Options](#-configuration-options)
6. [🖥️ CLI](#-cli)
7. [🧪 Playground Demo](#-playground-demo)
8. [🧱 Structured Outputs](#-structured-outputs)
9. [🎯 Customizing Prompts](#-customizing-prompts)
10. [📈 Applications](#-applications)
11. [🛠️ Contributing](#-contributing)
12. [📄 License](#-license)
13. [📄 Additional Resources](Docs/prompts.md)

## 🚀 Why OpenSceneSense Ollama?

OpenSceneSense Ollama brings the power of video analysis to your local machine. By using Ollama's models, you can:

- Run everything locally without depending on external APIs
- Maintain data privacy by processing videos on your own hardware
- Avoid usage costs associated with cloud-based solutions
- Customize and fine-tune models for your specific needs
- Process videos without internet connectivity

## 🌟 Features

- **📸 Local Frame Analysis:** Analyze visual elements using Ollama's vision models
- **🎙️ Whisper Audio Transcription:** Transcribe audio using local Whisper models
- **🔄 Dynamic Frame Selection:** Automatically select the most relevant frames
- **📝 Comprehensive Summaries:** Generate cohesive summaries integrating visual and audio elements
- **🛠️ Customizable Prompts:** Tailor the analysis process with custom prompts
- **📊 Metadata Extraction:** Extract valuable video metadata

## 📦 Installation

### Prerequisites

- Python 3.10+
- FFmpeg
- Ollama installed and running
- NVIDIA GPU (recommended)
- CUDA 12.1 or later (for GPU support)

### Install Required Dependencies

First, install PyTorch with CUDA 12.1 support:

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Install Transformers and other required packages:
```bash
pip install transformers
```

### Installing FFmpeg


#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install ffmpeg
```

#### macOS
```bash
brew install ffmpeg
```

#### Windows
1. Download FFmpeg from [ffmpeg.org/download.html](https://ffmpeg.org/download.html)
2. Extract and add to PATH

### Install OpenSceneSense Ollama

```bash
pip install openscenesense-ollama
```

## 🛠️ Usage

Here's a complete example showing how to use OpenSceneSense Ollama:

```python
from openscenesense_ollama.models import AnalysisPrompts
from openscenesense_ollama.transcriber import WhisperTranscriber
from openscenesense_ollama.analyzer import OllamaVideoAnalyzer
from openscenesense_ollama.frame_selectors import DynamicFrameSelector
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Initialize Whisper transcriber
transcriber = WhisperTranscriber(
    model_name="openai/whisper-small"
)

# Custom prompts for analysis
custom_prompts = AnalysisPrompts(
    frame_analysis="Analyze this frame focusing on visible elements, actions, and their relationship with any audio.",
    detailed_summary="""Create a comprehensive narrative that cohesively integrates visual and audio elements into a single story or summary from this 
    {duration:.1f}-second video:\n\nVideo Timeline:\n{timeline}\n\nAudio Transcript:\n{transcript}""",
    brief_summary="""Based on this {duration:.1f}-second video timeline and audio transcript:\n{timeline}\n\n{transcript}\n
    Provide a concise cohesive short summary combining the key visual and audio elements."""
)

# Initialize analyzer
analyzer = OllamaVideoAnalyzer(
    frame_analysis_model="ministral-3:latest",
    summary_model="ministral-3:latest",
    min_frames=10,
    max_frames=64,
    frames_per_minute=10.0,
    frame_selector=DynamicFrameSelector(),
    audio_transcriber=transcriber,
    prompts=custom_prompts,
    log_level=logging.INFO
)

# Analyze video
video_path = "your_video.mp4"
results = analyzer.analyze_video(video_path)

# Print results
print("\nBrief Summary:")
print(results['brief_summary'])

print("\nDetailed Summary:")
print(results['summary'])

print("\nVideo Timeline:")
print(results['timeline'])

print("\nMetadata:")
for key, value in results['metadata'].items():
    print(f"{key}: {value}")
```
## ⚙️ Configuration Options

The `OllamaVideoAnalyzer` class offers extensive configuration options to customize the analysis process:

### Basic Configuration

- **frame_analysis_model** (str, default="ministral-3:latest")
  - The Ollama model to use for analyzing individual frames
  - Common options: "ministral-3:latest", "llava", "minicpm-v", "bakllava"
  - Choose models with vision capabilities for best results

- **summary_model** (str, default="ministral-3:latest")
  - The Ollama model used for generating video summaries
  - Common options: "ministral-3:latest", "llama3.2", "mistral"
  - Text-focused models work best for summarization

- **host** (str, default="http://localhost:11434")
  - The URL where your Ollama instance is running
  - Modify if running Ollama on a different port or remote server

### Frame Selection Parameters

- **min_frames** (int, default=8)
  - Minimum number of frames to analyze
  - Lower values result in faster analysis but might miss details
  - Recommended range: 6-12 for short videos

- **max_frames** (int, default=64)
  - Maximum number of frames to analyze
  - Higher values provide more detailed analysis but increase processing time
  - Consider your hardware capabilities when adjusting this

- **frames_per_minute** (float, default=4.0)
  - Target rate of frame extraction
  - Higher values capture more temporal detail
  - Balance between detail and processing time
  - Recommended ranges:
    - 2-4 fps: Simple videos with minimal action
    - 4-8 fps: Standard content
    - 8+ fps: Fast-paced or complex scenes

### Component Configuration

- **frame_selector** (Optional[FrameSelector], default=None)
  - Custom frame selection strategy
  - Defaults to dynamic selection if None
  - Available built-in selectors:
    - `DynamicFrameSelector`: Adapts to scene changes
    - `UniformFrameSelector`: Evenly spaced frames
    - `AllFrameSelector`: Selects every frame

- **audio_transcriber** (Optional[AudioTranscriber], default=None)
  - Component for handling audio transcription
  - Defaults to no audio processing if None
  - Common options:
    ```python
    WhisperTranscriber(
        model_name="openai/whisper-small",
        device="cuda"  # or "cpu"
    )
    ```

- **prompts** (Optional[AnalysisPrompts], default=None)
  - Customized prompts for different analysis stages
  - Defaults to standard prompts if None
  - Customize using the `AnalysisPrompts` class

### Advanced Options

- **custom_frame_processor** (Optional[Callable[[Frame], Dict]], default=None)
  - Custom function for processing individual frames
  - Allows integration of additional analysis tools
  - Must accept a Frame object and return a dictionary
  - Example:
    ```python
    def custom_processor(frame: Frame) -> Dict:
        return {
            "timestamp": frame.timestamp,
            "custom_data": your_analysis(frame.image)
        }
    ```

- **log_level** (int, default=logging.INFO)
  - Controls verbosity of logging output
  - Common levels:
    - `logging.DEBUG`: Detailed debugging information
    - `logging.INFO`: General operational information
    - `logging.WARNING`: Warning messages only
    - `logging.ERROR`: Error messages only
    
- **request_timeout** (float, default=120.0)
  - Timeout in seconds for Ollama API requests
  - Useful for long-running model responses or slow hardware

- **request_retries** (int, default=3)
  - Number of retry attempts for timeouts or transient API errors
  - Total attempts = `request_retries + 1`

- **request_backoff** (float, default=1.0)
  - Exponential backoff in seconds between retries

### Example Configuration

Here's an example of a fully configured analyzer with custom settings:

```python
analyzer = OllamaVideoAnalyzer(
    frame_analysis_model="llava",
    summary_model="ministral-3:latest",
    host="http://localhost:11434",
    min_frames=12,
    max_frames=48,
    frames_per_minute=6.0,
    frame_selector=DynamicFrameSelector(
        threshold=0.3
    ),
    audio_transcriber=WhisperTranscriber(
        model_name="openai/whisper-small",
        device="cuda"
    ),
    prompts=AnalysisPrompts(
        frame_analysis="Detailed frame analysis prompt...",
        detailed_summary="Custom summary template...",
        brief_summary="Brief summary template..."
    ),
    custom_frame_processor=your_custom_processor,
    log_level=logging.DEBUG,
    request_timeout=120.0,
    request_retries=3,
    request_backoff=1.0
)
```
## 🖥️ CLI

Analyze videos from the command line and output JSON results:

```bash
openscenesense-ollama /path/to/video.mp4 --audio --output result.json --cache-dir .cache
```

Common options:
- `--frame-selector dynamic|uniform|all`
- `--frame-model` and `--summary-model`
- `--timeout` for Ollama API requests
- `--retries` and `--retry-backoff` for retry behavior
- `--prompts-file` for custom prompt templates
- `--cache-dir` to reuse results for the same inputs
- `--structured-output` to emit nested structured JSON
- `--schema` to print the structured JSON schema

## 🧪 Playground Demo

Use the playground script for deeper tuning and debugging on your own videos:

```bash
.venv/bin/python Examples/PlaygroundDemo.py /path/to/video.mp4 --audio --print-json
```

Useful flags include `--segment-duration`, `--beam-size`, `--temperature`, and `--no-collapse-repetitions`.

## 🧱 Structured Outputs

For typed results, use `analyze_video_structured` and convert to JSON when needed:

```python
from openscenesense_ollama.analyzer import OllamaVideoAnalyzer

analyzer = OllamaVideoAnalyzer()
result = analyzer.analyze_video_structured("your_video.mp4")

print(result.summary.brief)
print(result.warnings)

legacy_dict = result.to_legacy_dict()
structured_dict = result.to_dict()
```

`analyze_video` still returns the legacy dictionary shape for backward compatibility.

To retrieve a JSON schema:

```python
from openscenesense_ollama.models import analysis_result_schema

schema = analysis_result_schema()
```

Or via CLI:

```bash
openscenesense-ollama --schema
```

Schema file in repo: `Docs/analysis_result.schema.json`

## 🎯 Customizing Prompts

OpenSceneSense Ollama allows you to customize prompts for different types of analyses. The `AnalysisPrompts` class accepts the following parameters:

- **frame_analysis:** Guide the model's focus during frame analysis
- **detailed_summary:** Template for comprehensive video summaries
- **brief_summary:** Template for concise summaries

Available template tags:
- `{duration}`: Video duration in seconds
- `{timeline}`: Generated timeline of events
- `{transcript}`: Audio transcript

## 📈 Applications

OpenSceneSense Ollama is ideal for:

- **Content Creation:** Automatically generate video descriptions and summaries
- **Education:** Analyze educational content and create study materials
- **Research:** Build datasets for computer vision research
- **Local Content Moderation:** Monitor video content while maintaining privacy
- **Offline Analysis:** Process sensitive videos without internet connectivity

## 🛠️ Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/YourFeature`
3. Commit changes: `git commit -m "Add YourFeature"`
4. Push to branch: `git push origin feature/YourFeature`
5. Submit a pull request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
