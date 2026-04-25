from __future__ import annotations

import argparse

import torch

from phyfreqssm import build_datamodule, build_model, build_output_dir, evaluate_model, load_config, save_json


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PhyFreqSSM from a saved checkpoint.")
    parser.add_argument("--config", type=str, default="configs/phyfreqssm_ip.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["val", "test", "all"])
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    seed = args.seed or config["train"]["seed"]
    output_dir = build_output_dir(config, int(seed))
    data_bundle = build_datamodule(config, int(seed), output_dir=output_dir)
    config["model"]["in_channels"] = data_bundle.in_channels
    config["model"]["num_classes"] = data_bundle.num_classes
    model = build_model(config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])

    device_name = config["train"]["device"]
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    model.to(device)
    loader = {"val": data_bundle.val_loader, "test": data_bundle.test_loader, "all": data_bundle.all_loader}[args.split]
    split_mode = str(config["dataset"].get("split_mode", "random_ratio")).lower()
    legacy_map_style = split_mode != "spatial_block"
    metrics = evaluate_model(
        model,
        loader,
        device,
        data_bundle.num_classes,
        full_shape=data_bundle.labels.shape,
        output_dir=output_dir,
        split_name=args.split,
        dataset_name=config["dataset"]["name"],
        map_loader=data_bundle.all_loader,
        legacy_map_style=legacy_map_style,
    )
    save_json(metrics, output_dir / f"{args.split}_metrics.json")


if __name__ == "__main__":
    main()
