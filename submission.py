import os
import json
import torch
import scipy
from pydub import AudioSegment
# from transformers import MusicgenForConditionalGeneration
# from transformers import AutoProcessor

from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
device = "cuda:0" if torch.cuda.is_available() else "cpu"

def convert_wav_to_mp3(path, output_path):
    wav_audio = AudioSegment.from_file(path, format="wav")
    wav_audio.export(output_path, format="mp3")

def infer(text):
    if isinstance(text, str):
        text = [text]
    # inputs = processor(
    #         text=text,
    #         padding=True,
    #         return_tensors="pt")
    
    # audio_values = model.generate(**inputs.to(device),
    #                               do_sample=True,
    #                               guidance_scale=3,
    #                               max_new_tokens=512)
    # descriptions = ["The recording features a widely spread electric piano melody, followed by synth pad chords. It sounds emotional and passionate."]
    wav = model.generate(text)
    
    return wav[0]

# # Load from local
# processor = AutoProcessor.from_pretrained("./model_repository/musicgen_small/processor", local_files_only=True)

# model = MusicgenForConditionalGeneration.from_pretrained("./model_repository/musicgen_small/model", local_files_only=True)
# model = model.to(device)


# audio_length_in_s = 256 / model.config.audio_encoder.frame_rate
# # set maxlength
# model.generation_config.max_length = 500
# sampling_rate = model.config.audio_encoder.sampling_rate

model = MusicGen.get_pretrained('./checkpoints/my_mugen_lm')
model.set_generation_params(duration=10)  

json_path = "./zalo23/test/public.json"
with open(json_path) as json_file:
    data = json.load(json_file)
    print(data)

for filepath, text in data.items():
    print(filepath)
    print(text)
    print("===========================")
    # path = os.path.join("./output", "wav", filepath)
    output_path = os.path.join("./output", "mp3", filepath)
    os.makedirs(os.path.join("./output", "mp3"), exist_ok=True)

    audio_values = infer(text)
    # scipy.io.wavfile.write(path, rate=sampling_rate, data=audio_values[0, 0].cpu().numpy())
    # convert_wav_to_mp3(path, output_path)
    outpath = audio_write(output_path[:-4], 
                          audio_values, 
                          model.sample_rate, 
                          format='mp3',
                          strategy="loudness", 
                          loudness_compressor=True)
