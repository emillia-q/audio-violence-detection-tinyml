from pathlib import Path
from typing import Union
import librosa
import soundfile as sf

def process_audio_dataset(raw_dir: Union[str, Path], processed_dir: Union[str, Path], duration: int = 2, step: int = 1, target_sr: int = 16000) ->None:
    """
    Processes raw audio files: resamples, normalizes and cuts into fragments.
    """

    raw_root=Path(raw_dir)
    processed_root=Path(processed_dir)
    
    # Create the parent folder if it doesn't exist
    processed_root.mkdir(parents=True, exist_ok=True)

    # Iterate through subfolders
    for index, category_dir in enumerate(raw_root.iterdir()):
        if not category_dir.is_dir():
            continue

        # Create the corresponding folder in processed/
        processed_cat_name = f"{index}_{category_dir.name}"
        output_dir = processed_root / processed_cat_name
        output_dir.mkdir(exist_ok=True)

        # Recursively walk through nested subfolders (e.g. ambient/tv/*.wav)
        for file_path in category_dir.rglob("*"):
            if not file_path.is_file():
                continue
            # Skip files that have already been processed
            check_file = output_dir / f"{file_path.name}_seg0.wav"
            if check_file.exists():
                continue

            if file_path.suffix.lower() in ['.wav', '.mp3', '.mp4', '.m4a', '.ogg']:
                try:
                    y, sr = librosa.load(str(file_path), sr=target_sr)
                    # Peak normalization ensures consistent amplitude across different recording devices
                    y = librosa.util.normalize(y)

                    # Calculate total samples for a fixed-size buffer
                    samples_per_segment = duration * target_sr
                    samples_per_step = step * target_sr

                    # The +1 prevents getting 0 segments when the file length exactly matches the window size.
                    num_segments = (len(y) - samples_per_segment) // samples_per_step + 1 

                    for i in range(num_segments):
                        start = i * samples_per_step
                        end = start + samples_per_segment
                        segment = y[start:end]

                        out_filename = f"{file_path.name}_seg{i}.wav"
                        out_path = output_dir / out_filename

                        sf.write(out_path, segment, target_sr, subtype='PCM_16')

                except Exception as e:
                    print(f"Failed to process {file_path.name}: {e}")

if __name__ == "__main__":
    process_audio_dataset('data/raw', 'data/processed')