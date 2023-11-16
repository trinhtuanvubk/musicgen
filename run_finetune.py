
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from scipy.io.wavfile import write
model = MusicGen.get_pretrained('./checkpoints/my_mugen_lm')
model.set_generation_params(duration=10)  # generate 8 seconds.
# wav = model.generate_unconditional(4)    # generates 4 unconditional audio samples
descriptions = ["The recording features a widely spread electric piano melody, followed by synth pad chords. It sounds emotional and passionate."]
wav = model.generate(descriptions)  # generates 3 samples.
print(wav.shape)
# melody, sr = torchaudio.load('./assets/bach.mp3')
# # generates using the melody from the given audio and the provided descriptions.
# wav = model.generate_with_chroma(descriptions, melody[None].expand(3, -1, -1), sr)
# wav_ = wav[0].cpu().detach().numpy()
# print(wav_.shape)
# write("./hihi.wav", 16000, wav_)
for idx, one_wav in enumerate(wav):
    # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
    a = audio_write(f'{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)
    print(a)
