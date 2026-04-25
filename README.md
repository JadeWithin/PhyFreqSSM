# PhyFreqSSM

Official-style open-source layout for the main PhyFreqSSM model used in the paper.

## Current scope

This folder currently contains the runnable main-model path only:

- `train.py`: training entrypoint
- `eval.py`: evaluation entrypoint
- `infer.py`: convenience wrapper for full-scene inference
- `configs/`: paper-final configs for the four datasets
- `phyfreqssm/`: model, data, and utility package

The codebase is being migrated from the paper workspace into a cleaner open-source structure. For now, some internals are still routed through `phyfreqssm/core.py` while the public API already follows the new naming.

## Main model command

```bash
python train.py --config configs/phyfreqssm_ip.yaml --all-seeds
```

## Evaluation command

```bash
python eval.py --config configs/phyfreqssm_ip.yaml --checkpoint checkpoints/best_checkpoint.pt --seed 3407 --split test
```
