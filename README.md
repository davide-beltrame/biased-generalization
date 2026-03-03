# Biased Generalization in Diffusion Models

Code for the numerical experiments in the paper [Biased Generalization in Diffusion Models](https://arxiv.org/abs/2602.-----) by J. Garnier-Brun, L. Biggio, D. Beltrame, M. Mézard, and L. Saglietti.

Hierarchical-data (controlled setting) experiments live in `scripts/` (experiment files) and `modules/` (core logic).

CelebA experiments are in the `wdmdm/` submodule (fork of [Bonnaire et al.](https://github.com/tbonnair/Why-Diffusion-Models-Don-t-Memorize)).

Unless otherwise provided, all default arguments and hardcoded constants in the scripts (such as seed = 0) are configured to reproduce the experimental setup detailed in the paper.

## Setup

**HPC (GPU):**
```bash
module load miniconda3

cd /path/to/biased-generalization

bash setup_env.sh --gpu    # creates conda env gpu_env_dl with CUDA 12.1
pip install -e .
```

**Local (CPU only):**
```bash
bash setup_env.sh --cpu
pip install -e .
```

The editable install (`-e .`) makes the `modules` package importable everywhere.

## Data generation

Controlled setting data is generated once and reused by all experiments:

```bash
cd modules
python gen_filtered_hierarchical_data_wforbidden.py
# saves data/labeled_data_restrictedfixed_6_4_1.0_4_0.npy  (q=6, l=4, σ=1, q_eff=4)
```

By default, the script generates the dataset used for all hierarchical-data experiments in the paper.

## Training

### Transformer for controlled setting

Run from `scripts/`.  Three (size, epochs) settings cover the paper figures:

```bash
# n=5k, 20k iterations
python train_transformer.py --seed 0 --reduced-length 5000 --pick-i-for-training 0 \
    --batch-size 512 --n-iter 20000 --use-cross-entropy-loss

# n=12k, 30k iterations
python train_transformer.py --seed 0 --reduced-length 12000 --pick-i-for-training 0 \
    --batch-size 512 --n-iter 30000 --use-cross-entropy-loss

# n=70k, 35k iterations
python train_transformer.py --seed 0 --reduced-length 70000 --pick-i-for-training 0 \
    --batch-size 512 --n-iter 35000 --use-cross-entropy-loss
```

For multi-seed experiments (NN divergence, loss decomposition), repeat with
`--seed 0 ... 14` and `--pick-i-for-training 0 ... 14`.
Checkpoints are saved under `results_transformer_pick_<i>_for_training/`.

### CelebA U-Net

See `wdmdm/README.md`. In brief, from `wdmdm/Experiments/src/Training`:

```bash
python run_Unet.py -n 1024 -i 0 -s 32 -LR 0.0001 -O Adam -W 32 -t -1 --index 0 -se 0
```

Train 15 independent splits (`-i 0 ... 14`) for the sample-split analysis.

## Figure reproduction

Every script below is run from `scripts/`. Each script's docstring contains the
full CLI invocation. Here we give the minimal form.
Default values match the paper figures, thus most flags can be omitted.

### Hierarchical-data figures

| Figure | Script | Minimal command |
|--------|--------|-----------------|
| Fig. 1(b), Fig. 6 | `sequential_learning.py` | `python sequential_learning.py --train-size 12000 --same-dataset 0` |
| Fig. 3(b) | `score_divergence_along_t.py` | `python score_divergence_along_t.py --same-dataset 0` |
| Fig. 4(a), App. 9–10 (left) | `nn_divergence.py` | `python nn_divergence.py` |
| Fig. 4(b) | `loss_decomposition.py` | `python loss_decomposition.py --seeds "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14" --fixed-t-ratio 0.3 --plots` |
| Fig. 5, App. 9–10 (right) | `uturn_overlap.py` | `python uturn_overlap.py` |
| App. Fig. 11 | `uturn_overlap_random.py` | `python uturn_overlap_random.py` |

All scripts accept `--output-root` (default `../plots/`) to set the output directory.
Model paths (`--base-path`, `--paths-gen`, etc.) must point to the trained
checkpoint trees; see each script's docstring for the exact flags.

### CelebA figures

| Figure | Script (in `wdmdm/Experiments/src/`) | Command |
|--------|---------------------------------------|---------|
| Fig. 1(a) left | `Generation/loss_compute.py` | `python loss_compute.py` |
| Fig. 1(a) right, App. 7 | `Generation/sample_split_inference.py` | `python sample_split_inference.py --n 1024 --size_gen 2000 --T 1000` |
| Fig. 2 | `Generation/compare_scores.py` | `python compare_scores.py --n 1024 --T 1000 --num_samples 5000 --times "200,500,800"` |

## Citation

```bibtex
#TODO
```