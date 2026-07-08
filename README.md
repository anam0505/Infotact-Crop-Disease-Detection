# AgriVision — Crop Disease Detection (Computer Vision)

Deep-learning image classifier that detects crop leaf diseases from photos,
with a Streamlit UI for farmers/agronomists.

## Project structure

```
.
├── app.py                        # Streamlit frontend (UI only)
├── requirements.txt
├── .gitignore
├── src/
│   ├── data_preprocessing.py     # Week 1: scan, split (train/val/test), resize, normalize, augment
│   ├── train_baseline_cnn.py     # Week 2: custom CNN from scratch
│   ├── train_transfer_learning.py# Week 3: MobileNetV2 / ResNet50 transfer learning
│   ├── evaluate.py                # Week 4: confusion matrix + precision/recall report
│   └── predict.py                 # Shared inference logic (used by app.py AND CLI)
├── models/                       # Trained .keras weights land here (gitignored — see below)
└── notebooks/                    # Optional EDA notebooks
```

"Backend" in this project = everything in `src/` (data pipeline + model +
inference). "Frontend" = `app.py`. There's no separate server — Streamlit's
script *is* the whole app, calling straight into `src/predict.py`.

## 1. Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Download the PlantVillage dataset and place it so it looks like:

```
PlantVillage/
├── Tomato_Early_Blight/
├── Tomato_Healthy/
└── Tomato_Late_Blight/
```

## 2. Weekly workflow

**Week 1 — preprocessing**
```bash
python -m src.data_preprocessing
```
This scans `PlantVillage/`, prints your train/val/test counts, and confirms
the pipeline resizes to 224×224 and normalizes correctly.

**Week 2 — baseline CNN**
```bash
python -m src.train_baseline_cnn --data_dir PlantVillage/ --epochs 25
```
Saves the best checkpoint to `models/baseline_cnn.keras`.

**Week 3 — transfer learning**
```bash
python -m src.train_transfer_learning --data_dir PlantVillage/ --backbone mobilenet --epochs 15
```
Saves to `models/optimized_mobilenet.keras` — this is the file `app.py`
looks for first.

**Week 4 — evaluation**
```bash
python -m src.evaluate --model_path models/optimized_mobilenet.keras --data_dir PlantVillage/
```
Prints precision/recall/F1 per class and saves a confusion matrix PNG to
`outputs/confusion_matrix.png`.

## 3. Run the app

```bash
streamlit run app.py
```

## 4. Git workflow (VS Code)

The `.gitignore` in this repo replaces your old `gitignore.txt` — that file
had **unresolved merge-conflict markers** in it, which is why Git wasn't
behaving. Delete `gitignore.txt` from the repo once you've added this one.

```bash
git add .
git commit -m "Fix broken .gitignore, add full data pipeline, training scripts, and shared inference module"
git push origin main
```

If your team leader assigned per-person branches, work on your own branch
and open a PR instead of pushing straight to `main`:

```bash
git checkout -b feature/data-pipeline
git add .
git commit -m "Add preprocessing pipeline and baseline CNN training script"
git push origin feature/data-pipeline
```

**About model files:** `.keras` files are excluded from Git by default
(they can get large). For a student project that's usually fine — just
don't commit them. If your team leader wants the trained weights in the
repo for the mid-review, either:
- remove the `models/*.keras` line from `.gitignore`, or
- use [Git LFS](https://git-lfs.com/) for anything over ~50MB.

## 5. Deployment

Once you have a trained model in `models/`, the easiest free options are:

- **Streamlit Community Cloud** (simplest): push the repo to GitHub, go to
  share.streamlit.io, point it at `app.py`. It reads `requirements.txt`
  automatically. Only catch: your `.keras` file needs to be in the repo (or
  pulled from cloud storage on startup) since the free tier has no
  persistent external storage step built in.
- **Hugging Face Spaces** (Streamlit SDK): good if the model file is large —
  Spaces handles bigger repos comfortably and has its own Git LFS support.
- **Render / Railway**: use if you want a normal always-on web service
  instead of Streamlit's managed hosting.

For a student mid-review, Streamlit Community Cloud is the fastest path —
free, no server config, deploys straight from your GitHub repo.