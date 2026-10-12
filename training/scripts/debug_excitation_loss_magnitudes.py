#!/usr/bin/env python3
"""Print one-batch loss magnitudes for supervised excitation training."""

from __future__ import annotations

import argparse
from dataclasses import fields

import torch
import yaml

from tvtsyn.components.data_classes import (
    DecSeanetNCArgs,
    EncoderFrontendArgs,
    HuBERTLossArgs,
    SeanetArgs,
    SupervisedWav2WavLambdas,
    TransformerArgs,
    WaveformDecoderArgs,
)
from training.data_modules.parallel_audio import ParallelAudioDataModule
from training.modules.supervisedwav2wavexcitation import SupervisedWav2WavExcitationModule


DATACLASSES = {
    "seanet_args": SeanetArgs,
    "encoder_frontend_args": EncoderFrontendArgs,
    "transformer_args": TransformerArgs,
    "decoder_seanet_args": DecSeanetNCArgs,
    "waveform_decoder_args": WaveformDecoderArgs,
    "lambdas": SupervisedWav2WavLambdas,
    "hubert_loss_args": HuBERTLossArgs,
}


def _dataclass_from_dict(cls, values):
    allowed = {field.name for field in fields(cls)}
    return cls(**{key: value for key, value in values.items() if key in allowed})


def instantiate_model(config: dict) -> SupervisedWav2WavExcitationModule:
    init_args = dict(config["model"]["init_args"])
    for name, cls in DATACLASSES.items():
        if name in init_args and isinstance(init_args[name], dict):
            init_args[name] = _dataclass_from_dict(cls, init_args[name])
    return SupervisedWav2WavExcitationModule(**init_args)


def instantiate_data(config: dict, batch_size: int, num_workers: int):
    init_args = dict(config["data"]["init_args"])
    init_args["batch_size"] = batch_size
    init_args["num_workers"] = num_workers
    init_args["pin_memory"] = False
    return ParallelAudioDataModule(**init_args)


def move_batch(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {
            key: move_batch(value, device)
            if isinstance(value, (torch.Tensor, dict, list, tuple))
            else value
            for key, value in batch.items()
        }
    if isinstance(batch, tuple):
        return tuple(move_batch(value, device) for value in batch)
    if isinstance(batch, list):
        return [move_batch(value, device) for value in batch]
    return batch


def load_generator_weights(model, path: str, exclude_modules: list[str]):
    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    model_state = model.state_dict()
    exclude_prefixes = tuple(f"{module}." for module in exclude_modules)
    filtered = {}
    skipped_shape = []
    excluded = 0
    for key, value in state_dict.items():
        if exclude_prefixes and key.startswith(exclude_prefixes):
            excluded += 1
            continue
        if key in model_state and model_state[key].shape != value.shape:
            skipped_shape.append((key, tuple(model_state[key].shape), tuple(value.shape)))
            continue
        filtered[key] = value
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(f"loaded_checkpoint: {path}")
    print(f"excluded_keys: {excluded}")
    print(f"missing_keys: {len(missing)}")
    print(f"unexpected_keys: {len(unexpected)}")
    print(f"shape_mismatches: {len(skipped_shape)}")
    for key, model_shape, ckpt_shape in skipped_shape[:20]:
        print(f"shape_mismatch: {key}: model={model_shape} ckpt={ckpt_shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--exclude_load_modules", nargs="*", default=[])
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--use_target_prosody", action="store_true")
    args = parser.parse_args()

    with open(args.config) as handle:
        config = yaml.safe_load(handle)

    device = torch.device(args.device)
    model = instantiate_model(config).to(device)
    if args.checkpoint:
        load_generator_weights(model, args.checkpoint, args.exclude_load_modules)
    model.eval()

    data = instantiate_data(config, args.batch_size, args.num_workers)
    data.setup("fit")
    batch = move_batch(next(iter(data.train_dataloader())), device)
    with torch.no_grad():
        reconstruction, target, _, names, excitation, aux = model.model_step(
            batch,
            use_target_prosody=args.use_target_prosody,
        )
        losses = model.generator_objective(reconstruction, target, excitation, aux)

    weights = model.loss_weights
    print(f"batch_files: {list(names)[: min(len(names), 4)]}")
    print("loss_name,raw,weight,weighted")
    for name in sorted(weights):
        raw = float(losses[name].detach().cpu())
        weight = float(weights[name])
        print(f"{name},{raw:.10g},{weight:.10g},{raw * weight:.10g}")
    print(f"total_loss,{float(losses['total_loss'].detach().cpu()):.10g},1,{float(losses['total_loss'].detach().cpu()):.10g}")
    if "voiced_frames" in losses:
        print(f"voiced_frames,{float(losses['voiced_frames'].detach().cpu()):.10g},1,{float(losses['voiced_frames'].detach().cpu()):.10g}")


if __name__ == "__main__":
    main()
