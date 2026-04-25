from __future__ import annotations

import argparse
import math
from statistics import mean, pstdev

from phyfreqssm import Trainer, build_datamodule, build_model, build_output_dir, load_config, save_config, save_json, seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="Train PhyFreqSSM.")
    parser.add_argument("--config", type=str, default="configs/phyfreqssm_ip.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--all-seeds", action="store_true")
    return parser.parse_args()


def compute_summary_stats(values: list[float], with_ci: bool) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "ci95": 0.0}
    std = pstdev(values) if len(values) > 1 else 0.0
    ci95 = 1.96 * std / math.sqrt(len(values)) if with_ci and values else 0.0
    return {"mean": mean(values), "std": std, "ci95": ci95}


def main():
    args = parse_args()
    config = load_config(args.config)
    seeds = config["train"]["seeds"] if args.all_seeds else [args.seed or config["train"]["seed"]]
    aggregated = []
    with_ci = bool(config["train"].get("report_confidence_interval", True))

    for seed in seeds:
        print(f"[Seed] Starting seed {int(seed)} / total {len(seeds)}", flush=True)
        seed_everything(int(seed), deterministic=bool(config["train"]["deterministic"]))
        output_dir = build_output_dir(config, int(seed))
        data_bundle = build_datamodule(config, int(seed), output_dir=output_dir)
        config["model"]["in_channels"] = data_bundle.in_channels
        config["model"]["num_classes"] = data_bundle.num_classes
        model = build_model(config)
        save_config(config, output_dir / "config.yaml")
        trainer = Trainer(config, model, data_bundle, output_dir)
        artifacts = trainer.fit()
        aggregated.append(
            {
                "seed": int(seed),
                "OA": artifacts.test_metrics["OA"],
                "AA": artifacts.test_metrics["AA"],
                "Kappa": artifacts.test_metrics["Kappa"],
                "Macro-F1": artifacts.test_metrics["Macro-F1"],
                "MIoU": artifacts.test_metrics["MIoU"],
                "best_epoch": artifacts.best_epoch,
            }
        )
        print(
            f"[Seed] Finished seed {int(seed)} "
            f"test_OA={artifacts.test_metrics['OA']:.4f} "
            f"test_AA={artifacts.test_metrics['AA']:.4f} "
            f"test_MacroF1={artifacts.test_metrics['Macro-F1']:.4f}",
            flush=True,
        )

    if len(aggregated) > 1:
        metric_names = ["OA", "AA", "Kappa", "Macro-F1", "MIoU"]
        summary = {
            "num_seeds": len(aggregated),
            "metric_for_best": config["train"].get("metric_for_best", "oa"),
            "metric_weights": config["train"].get("metric_weights", {"oa": 1.0, "macro_f1": 0.0}),
            "report_confidence_interval": with_ci,
            "per_seed": aggregated,
        }
        for metric_name in metric_names:
            stats = compute_summary_stats([float(item[metric_name]) for item in aggregated], with_ci=with_ci)
            summary[f"{metric_name}_mean"] = stats["mean"]
            summary[f"{metric_name}_std"] = stats["std"]
            if with_ci:
                summary[f"{metric_name}_ci95"] = stats["ci95"]
        best_epoch_stats = compute_summary_stats([float(item["best_epoch"]) for item in aggregated], with_ci=False)
        summary["best_epoch_mean"] = best_epoch_stats["mean"]
        summary["best_epoch_std"] = best_epoch_stats["std"]
        save_json(summary, build_output_dir(config, int(seeds[0])).parent / "seed_summary.json")


if __name__ == "__main__":
    main()
