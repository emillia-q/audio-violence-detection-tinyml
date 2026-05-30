import librosa
import numpy as np
from pathlib import Path

def _print_to_c(y: np.ndarray) -> None:
    REPO_ROOT = Path(__file__).resolve().parent.parent
    OUTPUT_HEADER = REPO_ROOT / 'firmware' / 'golden_input.h'
    OUTPUT_HEADER.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_HEADER.open("w", encoding="utf-8") as f:
        f.write("#pragma once\n\n")
        f.write("const float golden_input[32000] = {\n")
        lines = [
            ", ".join(f"{v:.8f}f" for v in y[i : i + 8])
            for i in range(0, len(y), 8)
        ]
        f.write(",\n".join(lines))
        f.write("\n};\n")

def extract_digital() -> None:        
    filename = librosa.ex('trumpet')
    target_sr = 16000
    target_size = 32000
    n_mfcc = 13

    y, sr = librosa.load(filename, sr=target_sr, duration=2.0)
    _print_to_c(y)
    y = librosa.util.fix_length(y, size=target_size)
    y = librosa.util.normalize(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    ref = mfcc.T.flatten().astype(np.float32)  
    print(ref)
    print(f"y shape: {y.shape}, ref shape: {ref.shape}")

if __name__ == "__main__":
    extract_digital()