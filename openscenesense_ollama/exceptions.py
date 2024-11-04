# Custom exceptions for better error handling
class VideoAnalysisError(Exception):
    """Base exception class for video analysis errors"""
    pass

class VideoLoadError(VideoAnalysisError):
    """Raised when there are issues loading or accessing the video file"""
    pass

class TranscriptionError(VideoAnalysisError):
    """Raised when audio transcription fails"""
    pass

class FrameExtractionError(VideoAnalysisError):
    """Raised when frame extraction fails"""
    pass

class ModelInferenceError(VideoAnalysisError):
    """Raised when AI model inference fails"""
    pass

class APIError(VideoAnalysisError):
    """Raised when API calls fail"""
    pass
