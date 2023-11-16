import os
import json
import random
import wave

# make sure the .jsonl has a place to go
os.makedirs("./zalo_data/train", exist_ok=True)
os.makedirs("./zalo_data/eval", exist_ok=True)

json_raw_data = "./zalo23/train/train.json"
train_manifest_path = "./zalo_data/train/data.jsonl"
eval_manifest_path = "./zalo_data/eval/data.jsonl"

dataset_len = 0
train_len = 0
eval_len = 0

use_existing_json = False

train_file = open(train_manifest_path, 'w')
eval_file = open(eval_manifest_path, 'w')

with open(json_raw_data) as f:
    data = json.load(f)

for audio_name, text in data.items():
    audio_path = os.path.join("./zalo23/train/audio_wav", f"{audio_name[:-4]}.wav")
    print(audio_path)
    if os.path.exists(audio_path):
        dataset_len += 1

        if use_existing_json:
            # json_filepath = os.path.splitext(filename)[0] + ".json"
            # if os.path.exists(json_filepath):
            #     with open(json_filepath, 'r') as json_file:
            #         entry = json.load(json_file)
            #         entry["path"] = os.path.join(dataset_folder, filename)
            # else:
            #     print(f'error loading json: could not find {json_filepath}')
            print("i dont implement this fucking code")
        else:

            # empty fields for now, alter as needed to match your metadata.
            # all this does is make sure each file loads and trains semi-unconditionally
            import librosa
            y, sr = librosa.load(audio_path, sr=16000)
            length = librosa.get_duration(y=y, sr=16000)

            entry = {
                "key": "",
                "artist": "",
                "sample_rate": 32000,
                "file_extension": "wav",
                "description": text,
                "keywords": "",
                "duration": length,
                "bpm": "",
                "genre": "",
                "title": "",
                "name": "",
                "instrument": "",
                "moods": [],
                "path": audio_path,
            }

        if random.random() < 0.85:
            train_len += 1
            train_file.write(json.dumps(entry) + '\n')
        else:
            eval_len += 1
            eval_file.write(json.dumps(entry) + '\n')

train_file.close()
eval_file.close()

print(f'dataset length: {dataset_len} audio clips')
print(f'train length: {train_len} audio clips')
print(f'eval length: {eval_len} audio clips')