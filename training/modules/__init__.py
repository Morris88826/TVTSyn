from training.modules.speaker_module import SpeakerModule
from training.modules.pasrwithquant import PasrWithQuantModule
from training.modules.resynthesiswithquantandtvtimbre import ResynthesisWithQuantAndTVTimbreModule
from training.modules.supervisedwav2wav import SupervisedWav2WavModule
from training.modules.supervisedwav2wavexcitation import SupervisedWav2WavExcitationModule


__all__ = [
    "SpeakerModule",
    "PasrWithQuantModule",
    "ResynthesisWithQuantAndTVTimbreModule",
    "SupervisedWav2WavModule",
    "SupervisedWav2WavExcitationModule",
]

# Lightning CLI checkpoints before the layout migration stored this class path.
import sys
sys.modules.setdefault("modules", sys.modules[__name__])
# Stage-1 checkpoints predate the PasrWithQuantModule rename. Lightning 2.6.6+
# resolves this exact name among already imported classes before instantiation.
MimiPasrV2WithQuantModule = PasrWithQuantModule
