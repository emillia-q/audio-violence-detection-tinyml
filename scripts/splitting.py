from pathlib import Path


def get_source_name(file_path: str) -> str:
    """Extracts the original filename from a segmented audio file path.
    Example: 'path/to/yell.wav_seg12.wav' -> 'yell.wav'.
    """
    base_name = Path(file_path).name
    return base_name.split("_seg")[0]
