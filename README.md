# PhyFreqSSM

Open-source code for the main PhyFreqSSM model used in our paper.

## Current Scope

This repository currently contains the runnable main-model path only:

- `train.py`: training entrypoint
- `eval.py`: evaluation entrypoint
- `infer.py`: convenience wrapper for full-scene inference
- `configs/`: paper-final configs for the four datasets
- `phyfreqssm/`: model, data, and utility package

The repository has been organized into a cleaner open-source structure focused on the main model.

## Main Model Command

```bash
python train.py --config configs/phyfreqssm_ip.yaml --all-seeds
```

## Evaluation Command

```bash
python eval.py --config configs/phyfreqssm_ip.yaml --checkpoint checkpoints/best_checkpoint.pt --seed 3407 --split test
```
