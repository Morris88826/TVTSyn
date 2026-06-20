from src.modules.speaker_module import SpeakerModule
from src.modules.pasrwithquant import PasrWithQuantModule
from src.modules.resynthesiswithquantandtvtimbre import ResynthesisWithQuantAndTVTimbreModule
from src.modules.supervisedwav2wav import SupervisedWav2WavModule
from src.modules.supervisedwav2wavexcitation import SupervisedWav2WavExcitationModule


__all__ = [
    "SpeakerModule",
    "PasrWithQuantModule",
    "ResynthesisWithQuantAndTVTimbreModule",
    "SupervisedWav2WavModule",
    "SupervisedWav2WavExcitationModule",
]
