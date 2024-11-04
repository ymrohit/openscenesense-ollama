from openscenesense_ollama.models import AnalysisPrompts
from openscenesense_ollama.transcriber import WhisperTranscriber
from openscenesense_ollama.analyzer import OllamaVideoAnalyzer
from openscenesense_ollama.frame_selectors import DynamicFrameSelector
import logging
def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)

    try:
        # Initialize Whisper transcriber
        transcriber = WhisperTranscriber(
            model_name="openai/whisper-tiny",
            #chunk_length_s=30,
            #batch_size=8
        )

        # Custom prompts for analysis
        custom_prompts = AnalysisPrompts(
            frame_analysis="Analyze this frame focusing on visible elements, actions, and their relationship with any audio.",
            detailed_summary="""*DO NOT SEPARATE/MENTION AUDIO/VIDEO Specifically, MAKE IT COHESIVE AND NATURAL* Create a comprehensive narrative that cohesively integrates visual and audio elements into a single story or summary from this 
            {duration:.1f}-second video:\n\nVideo Timeline:\n{timeline}\n\nAudio Transcript:\n{transcript}""",
            brief_summary="""Based on this {duration:.1f}-second video timeline and audio transcript:\n{timeline}\n\n{transcript}\n
            Provide a concise cohesive short summary combining the key visual and audio elements, this should be easy to read and understand the entire context of the video *DO NOT SEPARATE/MENTION AUDIO/VIDEO Specifically, MAKE IT COHESIVE AND NATURAL* """
        )

        # Initialize analyzer with all components
        analyzer = OllamaVideoAnalyzer(
            frame_analysis_model="minicpm-v",
            summary_model="llama3.2",
            min_frames=10,
            max_frames=64,
            frames_per_minute=10.0,
            frame_selector=DynamicFrameSelector(),
            audio_transcriber=transcriber,
            prompts=custom_prompts,
            log_level=logging.INFO
        )

        # Analyze video
        video_path = "pizza.mp4"  # Replace with your video path
        results = analyzer.analyze_video(video_path)

        # Print results
        print("\nBrief Summary:")
        print("-" * 50)
        print(results['brief_summary'])

        print("\nDetailed Summary:")
        print("-" * 50)
        print(results['summary'])

        print("\nVideo Timeline with Audio:")
        print("-" * 50)
        print(results['timeline'])

        print("\nAudio Transcript:")
        print("-" * 50)
        print(results['transcript'])

        print("\nMetadata:")
        print("-" * 50)
        for key, value in results['metadata'].items():
            print(f"{key}: {value}")

    except Exception as e:
        logger.error(f"Error in main: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()