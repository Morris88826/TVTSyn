from src.modules.speaker_module import SpeakerModule

from src.modules.pasr_module import PasrModule
from src.modules.pasrwithbottleneck import PasrWithBottleneckModule
from src.modules.pasrwithquant import PasrWithQuantModule

from src.modules.resynthesiswithquant import ResynthesisWithQuantModule
from src.modules.resynthesiswithbottleneck import ResynthesisWithBottleneckModule

from src.modules.resynthesiswithlogitsandtvtimbre import ResynthesisWithLogitsAndTVTimbreModule
from src.modules.resynthesiswithquantandtvtimbre import ResynthesisWithQuantAndTVTimbreModule
from src.modules.resynthesiswithbottleneckandtvtimbre import ResynthesisWithBottleneckAndTVTimbreModule


__all__ = [
    "SpeakerModule",

    "PasrModule",
    "PasrWithBottleneckModule",
    "PasrWithQuantModule",

    "ResynthesisWithQuantModule",
    "ResynthesisWithBottleneckModule",

    "ResynthesisWithLogitsAndTVTimbreModule",
    "ResynthesisWithQuantAndTVTimbreModule",
    "ResynthesisWithBottleneckAndTVTimbreModule",
]
