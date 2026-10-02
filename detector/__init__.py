"""Language Detection package: English, Hindi, Gujarati, Hinglish, Gujlish + 34 other languages."""

from .detector import LANGUAGE_INFO, LanguageDetector, detect, get_detector

__all__ = ["LanguageDetector", "detect", "get_detector", "LANGUAGE_INFO"]
__version__ = "1.0.0"
