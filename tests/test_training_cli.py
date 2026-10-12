"""Subcommand hooks must not assume a fit configuration."""
from types import SimpleNamespace
import pytest
from training.utils.cli import CustomLightningCLI

@pytest.mark.parametrize('command', ['fit', 'validate', 'test', 'predict'])
def test_weights_hook_accepts_selected_subcommand(command):
    cli = SimpleNamespace(subcommand=command, config={command: {}})
    CustomLightningCLI.after_instantiate_classes(cli)

def test_historical_encoder_class_is_imported():
    import sys
    from training.modules import PasrWithQuantModule
    assert sys.modules['modules'].MimiPasrV2WithQuantModule is PasrWithQuantModule
