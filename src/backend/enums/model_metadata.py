from enum import Enum


class ArtifactType(str, Enum):
    PYTORCH = "pytorch"
    TORCHSCRIPT = "torchscript"


class ValidationVerdict(str, Enum):
    GOOD = "good"
    REVIEW = "review"
    UNRELIABLE = "unreliable"


class UncertaintyLevel(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
