import pytest
import torch
from tvtsyn.models.inference import TVTSynModel


def test_explicit_checkpoint_mapping():
    class Tiny(torch.nn.Module):
        load_training_state = TVTSynModel.load_training_state
        def __init__(self):
            super().__init__()
            self.encoder=torch.nn.Linear(2,2)
            self.decoder=torch.nn.Linear(2,1)
    model=Tiny()
    state={('encoder.net.'+k[len('encoder.'):] if k.startswith('encoder.') else k):v.clone()
           for k,v in model.state_dict().items()}
    state['discriminator.weight']=torch.ones(1)
    assert model.load_training_state(state)==['discriminator.weight']
    state.pop('encoder.net.weight')
    with pytest.raises(RuntimeError,match='Missing key'): model.load_training_state(state)
