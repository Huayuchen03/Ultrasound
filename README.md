# Ultrasound Beamforming Optimization with Learned Weights (TensorFlow)

This repo contains:
- A **simulated ultrasound dataset generator** (publicly shareable; no patient data).
- Classical beamforming baselines: **Delay-and-Sum (DAS)** and **Adaptive Coherence Factor (CF)**.
- An optional **MVDR (Capon) beamformer** (slower; included for completeness).
- A **TensorFlow** model that predicts **data-dependent receive weights** (per-pixel) to improve image quality while keeping inference fast.
- Scripts to compute **resolution** (FWHM) and **contrast/CNR**, and to export figures/tables for an IEEE-style report.

> **Note**: The simulation is intentionally simplified (2D, point-scatterer model, monostatic pulse-echo). It is sufficient for a course project focused on algorithm comparison and computational tradeoffs.

---

## 1. Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you want CPU-only TensorFlow:

```bash
pip install tensorflow-cpu==2.15.0
```

If you encounter errors such as:

```
ImportError: attempted relative import with no known parent package
```

ensure you run scripts with the module flag so Python treats `src/` as a
package:

```bash
python -m src.run_params_demo
```
This invocation works regardless of your current working directory and avoids
the relative-import problem that occurs when calling `python src/run_params_demo.py`.

---

## 2. Quick start (generate data + beamform baselines)

```bash
python -m src.run_baselines --out_dir results --seed 0
```

This produces:
- `results/bmode_das.png`
- `results/bmode_cf.png`
- `results/metrics.csv`

---

## 2.1 Real-data example (public dataset)

If your rubric requires a **real** (non-simulated) example, this repo includes a
best-effort loader + plane-wave DAS/CF beamformer for **HDF5/UFF** acquisitions.

1) Download a *public* UFF/HDF5 channel dataset (e.g., an USTB/PICMUS UFF file).
2) Run:

```bash
python -m src.run_real_uff \
  --file_path /path/to/your_public_dataset.uff \
  --out_dir results_real \
  --nx 128 --nz 256 \
  --x_min_m -0.020 --x_max_m 0.020 \
  --z_min_m 0.005  --z_max_m 0.060 \
  --max_waves 11
```

Outputs:
- `results_real/real_bmode_das.png`
- `results_real/real_bmode_cf.png`
- `results_real/real_metrics.csv`
- `results_real/real_data_meta.json`

If the loader cannot infer probe geometry or sampling frequency, pass
`--pitch_m` and/or `--fs_hz` explicitly. You can also inspect the file structure
with:

```bash
python -m src.run_real_uff --file_path /path/to/file.uff --out_dir results_real --dump_hdf5
```

---

## 3. Train the neural network (TensorFlow)

The network learns to predict receive weights **w** from the delayed channel snapshot **x**.
You can choose the **teacher** target:
- `cf` (fast adaptive teacher; stable)
- `mvdr` (stronger but slower teacher)

### Step A: build a training set (snapshots + teacher outputs)
```bash
python -m src.build_dataset --out_npz data/train.npz --teacher cf --num_phantoms 20 --samples_per_phantom 8000 --seed 0
python -m src.build_dataset --out_npz data/val.npz   --teacher cf --num_phantoms 5  --samples_per_phantom 8000 --seed 1
```

### Step B: train
```bash
python -m src.train_tf --train_npz data/train.npz --val_npz data/val.npz --out_dir models/cf_teacher --epochs 15
```

### Step C: run inference (NN beamforming) + evaluation
```bash
python -m src.eval_all --model_dir models/cf_teacher --out_dir results_nn --seed 2
```

Outputs:
- `results_nn/bmode_nn.png`
- `results_nn/metrics.csv` (DAS vs CF vs NN and optionally MVDR)

---

## 4. Report (LaTeX)

Report sources are in `report/`.
Compile with:

```bash
cd report
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

---

## 5. Project structure

- `src/simulate_phantom.py`: scatterer phantom generation (speckle + cyst + point targets)
- `src/simulate_rf.py`: RF channel simulation (pulse-echo, linear array)
- `src/beamform.py`: DAS + envelope detection + log compression
- `src/adaptive.py`: coherence factor (CF) adaptive weighting
- `src/mvdr.py`: MVDR/Capon beamformer (optional, slow)
- `src/nn_model_tf.py`: TensorFlow model (predict weights, form weighted sum)
- `src/metrics.py`: FWHM, CR, CNR and runtime helpers
- `src/run_baselines.py`: baseline experiment + figure export
- `src/build_dataset.py`: export snapshots and teacher labels to `.npz`
- `src/train_tf.py`: training script
- `src/eval_all.py`: evaluate DAS / CF / NN (+ MVDR optional)

---

## 6. Reproducibility

- All scripts take `--seed`.
- Results are saved to disk as images + CSV.
# Ultrasound
