import librosa
import numpy as np
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent

def _export_to_c_header(y: np.ndarray) -> None:
    """
    Exports the raw, unnormalized input audio signal into a C++ header file 
    as a constant float array. This header serves as the 'golden_input' 
    reference for on-device hardware DSP pipeline validation.
    """
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

def generate_sanity_references(filename: str, target_sr: int, target_size: int, n_mfcc: int) -> None: 
    """
    Loads a reference audio sample, exports it to a C++ header and generates 
    baseline regression files (normalized audio and flattened MFCC features). 
    These generated text files are used to verify the mathematical accuracy 
    of the embedded DSP engine against the Librosa framework.
    """       
    filename = librosa.ex(filename)

    y, sr = librosa.load(filename, sr=target_sr, duration=2.0)
    _export_to_c_header(y)
    y = librosa.util.fix_length(y, size=target_size)
    y = librosa.util.normalize(y)

    # NORMALIZED
    OUTPUT_NORMALIZED = REPO_ROOT / "sanity" / "golden_y_normalized.txt"
    np.savetxt(OUTPUT_NORMALIZED, y, fmt="%.8f")

    # MFCC computed
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    ref = mfcc.T.flatten().astype(np.float32)  
    OUTPUT_MFCC = REPO_ROOT / "sanity" / "golden_mfcc_reference.txt"
    np.savetxt(OUTPUT_MFCC, ref, fmt="%.8f")
    

if __name__ == "__main__":
    generate_sanity_references('trumpet', 16000, 32000, 13)