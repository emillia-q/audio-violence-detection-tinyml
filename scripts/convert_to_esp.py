from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VERSION_FILE = REPO_ROOT / "latest_version_tflite.txt"
MODEL_BASE_DIR = REPO_ROOT / "models" / "tflite"
OUTPUT_HEADER = REPO_ROOT / "firmware" / "model_data.h"

version = VERSION_FILE.read_text(encoding="utf-8").strip()
model_path = MODEL_BASE_DIR / version / "audio_detection_model.tflite"

if not model_path.is_file():
    raise FileNotFoundError(
        f"Model not found: {model_path}\n"
        f"Expected TFLite at models/tflite/<version>/audio_detection_model.tflite "
        f"(version from {VERSION_FILE.name}: {version!r})"
    )

tflite_model = model_path.read_bytes()

OUTPUT_HEADER.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT_HEADER.open("w", encoding="utf-8") as f:
    f.write("#pragma once\n\n")
    f.write(f"// Model size: {len(tflite_model)} bytes\n")
    f.write(f"const unsigned int model_data_len = {len(tflite_model)};\n\n")
    f.write("const unsigned char model_data[] __attribute__((aligned(16))) = {\n")

    hex_lines = [
        ", ".join(f"0x{b:02x}" for b in tflite_model[i : i + 12])
        for i in range(0, len(tflite_model), 12)
    ]
    f.write(",\n".join(hex_lines))
    f.write("\n};\n")

print(f"Wrote {OUTPUT_HEADER} ({len(tflite_model)} bytes) from {model_path.relative_to(REPO_ROOT)}")
