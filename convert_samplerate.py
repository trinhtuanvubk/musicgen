import librosa
from scipy.io import wavfile
import json
import os
# 16k upscale to 32k
# json_raw_data = "./zalo23/train/train.json"
# output_dir = "./zalo23/train/audio_wav/"
# with open(json_raw_data) as f:
#     data = json.load(f)

# for audio_name, text in data.items():
#     print(audio_name)
#     audio_path = os.path.join("./zalo23/train/audio", audio_name)
    
#     y, sr = librosa.load(audio_path, sr=16000)
#     y_upscaled = librosa.resample(y, orig_sr=sr, target_sr=32000)
#     # soundfile.write(os.path.join(output_dir, audio_name), y_upscaled, 32000)
#     # librosa.output.write_wav("upscaled_audio.wav", y_upscaled, 32000)
#     wavfile.write(os.path.join(output_dir, f"{audio_name[:-4]}.wav"), 32000, y_upscaled)

# 32k downscale to 16k

output_dir = "./output/mp3/"
output_16k_dir = "./output/mp3_16k/"
os.makedirs(output_16k_dir, exist_ok=True)
mp3_file = os.listdir(output_dir)

for filename in mp3_file:
    audio_path = os.path.join(output_dir, filename)
    
    y, sr = librosa.load(audio_path, sr=32000)
    y_downscaled = librosa.resample(y, orig_sr=sr, target_sr=16000)
    # soundfile.write(os.path.join(output_dir, audio_name), y_upscaled, 32000)
    # librosa.output.write_wav("upscaled_audio.wav", y_upscaled, 32000)
    wavfile.write(os.path.join(output_16k_dir, filename), 16000, y_downscaled)