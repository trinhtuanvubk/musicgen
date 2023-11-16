# from audiocraft.models import MusicGen
# from audiocraft.solvers import MusicGenSolver

# solver = MusicGenSolver.get_eval_solver_from_sig('60625e70')
# solver.model.cfg = solver.cfg
# musicgen = MusicGen(name='mymusicgen', compression_model=solver.compression_model, lm=solver.model)

from audiocraft.utils import export
from audiocraft import train
import os
xp = train.main.get_xp_from_sig('968434f2')
os.makedirs("./checkpoints/my_mugen_lm/", exist_ok=True)
export.export_lm(xp.folder / 'checkpoint.th', './checkpoints/my_mugen_lm/state_dict.bin')
# You also need to bundle the EnCodec model you used !!
## Case 1) you trained your own
# xp_encodec = train.main.get_xp_from_sig('')
# export.export_encodec(xp_encodec.folder / 'checkpoint.th', '/checkpoints/my_audio_lm/compression_state_dict.bin')
## Case 2) you used a pretrained model. Give the name you used without the //pretrained/ prefix.
## This will actually not dump the actual model, simply a pointer to the right model to download.
export.export_pretrained_compression_model('facebook/encodec_32khz', './checkpoints/my_mugen_lm/compression_state_dict.bin')