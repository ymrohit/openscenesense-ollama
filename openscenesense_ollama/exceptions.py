class OpenSceneSenseError(Exception):
    """Base exception for expected OpenSceneSense Ollama failures."""


class ConfigurationError(OpenSceneSenseError, ValueError):
    pass


class MissingDependencyError(OpenSceneSenseError, ImportError):
    pass


class VideoLoadError(OpenSceneSenseError):
    pass


class VideoMetadataError(OpenSceneSenseError):
    pass


class AudioExtractionError(OpenSceneSenseError):
    pass


class TranscriptionError(OpenSceneSenseError):
    pass


class ProviderConnectionError(OpenSceneSenseError):
    pass


class AuthenticationError(OpenSceneSenseError):
    pass


class RateLimitError(OpenSceneSenseError):
    pass


class ModelNotFoundError(OpenSceneSenseError):
    pass


class ModelCapabilityError(OpenSceneSenseError):
    pass


class ResponseValidationError(OpenSceneSenseError):
    pass


# v1.1 compatibility names.
VideoAnalysisError = OpenSceneSenseError
FrameExtractionError = VideoLoadError
ModelInferenceError = ResponseValidationError
APIError = ProviderConnectionError
