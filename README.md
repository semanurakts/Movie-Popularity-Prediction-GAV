# Movie Popularity Prediction via GA-V Classifier
### Based on: Faisal et al. (2022) — *"Prediction of Movie Quality via Adaptive Voting Classifier"*
### IEEE Access | DOI: 10.1109/ACCESS.2022.3195228

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Requirements](#3-requirements)
4. [Dataset Setup](#4-dataset-setup)
5. [Installation](#5-installation)
6. [Running the Notebook](#6-running-the-notebook)
7. [Expected Outputs](#7-expected-outputs)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Project Overview

This project implements and extends the **Genetic Algorithm Voting (GA-V) Classifier** proposed by Faisal et al. (2022) for predicting movie quality/popularity. The implementation uses a merged **TMDb + IMDb** dataset and addresses several methodological issues from the original paper, including target leakage prevention, proper class-imbalance handling, and a fully functional genetic algorithm with crossover and mutation operators.

**Key components:**
- Content-based movie recommendation system (TF-IDF + Cosine Distance)
- GA-V Classifier with adaptive crossover and mutation (Eq.4–5 from paper)
- 10-fold cross-validation by movie genre category
- Target audience analysis using real IMDb age-group voting data
- Final comparison against the original CNN baseline

---

## 2. Repository Structure

```
project/
│
├── Movie_Popularity_RS_Fourth_Progressive.ipynb   ← Main notebook (run this)
│                                                  ← Place all 4 CSV files here
|── tmdb_5000_movies.csv                           
├── tmdb_5000_credits.csv
├── IMDb_ratings.csv
|── IMDb_movies.csv  
|                                                  ← Auto-generated after running
├── gav_results.csv                                ← 10-fold CV results per genre
└── gav_model.pkl                                  ← Saved trained model pipeline
```

> **Note:** The notebook expects all four CSV files to be in the **same directory** as the `.ipynb` file, or adjust the file paths in Cell 4 accordingly.

---

## 3. Requirements

### Python Version

```
Python 3.8 or higher (recommended: 3.10+)
```

### Required Libraries

| Library | Version | Purpose |
|---|---|---|
| `pandas` | ≥ 1.3.0 | Data loading and manipulation |
| `numpy` | ≥ 1.21.0 | Numerical operations |
| `scikit-learn` | ≥ 1.0.0 | ML models, preprocessing, CV |
| `matplotlib` | ≥ 3.4.0 | Visualizations |
| `seaborn` | ≥ 0.11.0 | Statistical plots |
| `joblib` | ≥ 1.0.0 | Model serialization |

All of these are included in the `pip install` command in Step 5.

---

## 4. Dataset Setup

This project requires **four CSV files**. Download each from the sources below and place them all in the same folder as the notebook.

### Dataset 1 & 2 — TMDb 5000 Movies

**Source:** Kaggle — *"TMDB 5000 Movie Dataset"*
**URL:** https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

Download and extract. You will get two files:

| File | Description |
|---|---|
| `tmdb_5000_movies.csv` | Movie metadata (title, genres, keywords, budget, revenue, etc.) |
| `tmdb_5000_credits.csv` | Cast and crew information per movie |

### Dataset 3 & 4 — IMDb Ratings

**Source:** Kaggle — *"IMDb movies extensive dataset"*
**URL:** https://www.kaggle.com/datasets/simhyunsu/imdbextensivedataset

Download and extract. You need two specific files:

| File | Description |
|---|---|
| `IMDb_ratings.csv` | Per-movie ratings, vote counts, age-group breakdowns |
| `IMDb_movies.csv` | Movie titles and IMDb IDs (used for merging) |

> **Important:** After downloading, verify that all four file names exactly match those listed above. The notebook reads them with these exact names.

### After Download — Verify Files

Open a terminal in the project folder and run:

```bash
python3 -c "
import pandas as pd
for f in ['tmdb_5000_movies.csv','tmdb_5000_credits.csv','IMDb_ratings.csv','IMDb_movies.csv']:
    df = pd.read_csv(f, nrows=1)
    print(f'{f}: OK ({df.shape[1]} columns)')
"
```

Expected output:
```
tmdb_5000_movies.csv:   OK (20 columns)
tmdb_5000_credits.csv:  OK (4 columns)
IMDb_ratings.csv:       OK (49 columns)
IMDb_movies.csv:        OK (22 columns)
```

---

## 5. Installation

### Option A — Google Colab (Recommended)

1. Open [Google Colab](https://colab.research.google.com)
2. Upload the notebook file: `File → Upload notebook`
3. Upload all four CSV files using the Files panel (left sidebar → folder icon → upload)
4. Run the first cell — all standard libraries are pre-installed in Colab

If any library is missing, add this cell at the very top of the notebook and run it first:

```python
!pip install scikit-learn pandas numpy matplotlib seaborn joblib
```

### Option B — Local Environment (Jupyter Notebook)

**Step 1:** Verify Python version

```bash
python3 --version
# Must be 3.8 or higher
```

**Step 2:** Create a virtual environment (strongly recommended)

```bash
# Create
python3 -m venv gav_env

# Activate — macOS / Linux
source gav_env/bin/activate

# Activate — Windows
gav_env\Scripts\activate
```

**Step 3:** Install all required libraries

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib jupyter
```

Or install from a requirements file (create `requirements.txt` with the content below, then install):

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
joblib>=1.0.0
jupyter>=1.0.0
```

```bash
pip install -r requirements.txt
```

**Step 4:** Verify installation

```bash
python3 -c "
import pandas, numpy, sklearn, matplotlib, seaborn, joblib
print('pandas     :', pandas.__version__)
print('numpy      :', numpy.__version__)
print('scikit-learn:', sklearn.__version__)
print('matplotlib :', matplotlib.__version__)
print('seaborn    :', seaborn.__version__)
print('All OK!')
"
```

**Step 5:** Launch Jupyter Notebook

```bash
jupyter notebook
```

This will open a browser window. Navigate to the project folder and open `Movie_Popularity_RS_Final.ipynb`.

### Option C — VS Code with Jupyter Extension

1. Install [VS Code](https://code.visualstudio.com/)
2. Install the **Python** and **Jupyter** extensions from the Extensions panel
3. Open the project folder in VS Code
4. Open the `.ipynb` file — VS Code will handle notebook rendering natively
5. Select your Python interpreter (bottom-left corner) — choose the virtual environment from Option B if you created one
6. Run cells with the ▶ button or `Shift+Enter`

---

## 6. Running the Notebook

### Step-by-Step Execution

The notebook is organized into **32 cells**. Run them **in order from top to bottom**.

| Cells | Section | Description |
|---|---|---|
| 1–2 | Setup | Import libraries, set random seed |
| 3 | Banner | Print project header |
| 4 | Data Loading | Load and merge TMDb + IMDb datasets |
| 5–7 | EDA | Visualize rating distribution, class distribution, top genres |
| 8 | Data Prep | Train/test split, class distribution summary |
| 9 | GA-V Definition | Define the `GAVotingClassifier` class |
| 10 | Training | Fit GA-V on training data, evaluate on test set |
| 11 | Heatmap | Classification report heatmap |
| 12 | GA Weights | Display per-class weights found by GA |
| 13 | Genre CV | 10-fold cross-validation per movie genre |
| 14 | Save | Export results CSV and trained model |
| 15 | Recommendations | Content-based film recommendations (TF-IDF) |
| 16 | Distribution | Train/test distribution pie and bar charts |
| 17 | Class Weights | Imbalance handling visualization |
| 18–19 | Feature Analysis | KDE plots and correlation heatmap |
| 20–22 | CV Training | 5-fold CV training history |
| 23 | Learning Curves | Accuracy and error curves |
| 24–25 | Final Eval | Test accuracy, confusion matrices |
| 26–28 | Model Ranking | Comparison of all base models vs GA-V |
| 29–31 | Audience | Target audience demographic analysis |
| 32 | Paper Comparison | GA-V vs original CNN from the paper |

### Running All Cells at Once

In Jupyter Notebook:
```
Kernel → Restart & Run All
```

In Google Colab:
```
Runtime → Run all
```

> **Estimated runtime:** 15–30 minutes on a standard laptop (the GA optimization and 5-fold CV loop are the most time-consuming steps).

---

## 7. Expected Outputs

After successful execution, the following will be produced:

### Console Output (key sections)

```
=========================================================================
Faisal et al. (2022) - Prediction of Movie Quality via GA-V Application
=========================================================================

[INFO] Train Set: 1646 movies, Test Set: 412 movies
   High-quality (1): 931  Low-quality (0): 1127

[GA-V] Phase 1: Training Basic Models...
[GA-V] Phase 2: GA Optimization...
       -> Generation 1/50 | Best Fitness: 0.6683
       -> Generation 10/50 | Best Fitness: 0.6683
       ...

🎯 ADAPTIVE GA-VOTING CLASSIFIER TEST RESULTS
              precision    recall  f1-score   support
Low-Quality      0.650     0.780     0.710       226
High-Quality     0.650     0.480     0.550       186

🚀 PROPOSED GA-DRIVEN ENSEMBLE ACCURACY: 0.6481
```

### Saved Files

| File | Contents |
|---|---|
| `gav_results.csv` | 10-fold CV accuracy, precision, recall, F1 per genre |
| `gav_model.pkl` | Complete trained pipeline (scaler + GA-V model) |

### Visualizations (displayed inline)

| # | Plot |
|---|---|
| 1 | IMDb Rating Distribution histogram |
| 2 | Movie Quality Binary Class Distribution |
| 3 | Top-10 Most Common Genres |
| 4 | Classification Report Heatmap |
| 5 | Train vs Test distribution pie charts |
| 6 | Train vs Test absolute count bar chart |
| 7 | Class Weights bar chart |
| 8 | Feature KDE distributions by quality class |
| 9 | Feature correlation with target |
| 10 | 5-fold CV Accuracy curve |
| 11 | 5-fold CV Error rate curve |
| 12 | Confusion Matrix (counts + normalized) |
| 13 | Model Ranking (trend + bar chart) |
| 14 | Target Audience bar chart |
| 15 | Target Audience radar chart |
| 16 | GA-V vs Original CNN comparison |

---

## 8. Troubleshooting

### `FileNotFoundError: tmdb_5000_movies.csv`

The CSV files are not in the correct location. Make sure all four files are in the **same folder** as the notebook. If you are using Colab, upload them via the Files panel before running Cell 4.

### `ModuleNotFoundError: No module named 'sklearn'`

Run the installation command again:
```bash
pip install scikit-learn
```
If using Colab, add `!pip install scikit-learn` as a new first cell.

### Kernel Dies / Out of Memory

The 5-fold CV loop in Cell 22 trains multiple GA-V instances sequentially, which is memory intensive. Try:
- Reducing `population_size` from 50 to 20 in Cell 22
- Reducing `generations` from 20 to 10 in Cell 22
- Restarting the kernel and running again (`Kernel → Restart & Run All`)

### Plots Not Showing (Local Jupyter)

Add the following line to Cell 1, after the imports:

```python
%matplotlib inline
```

### `on_bad_lines` Warning (older pandas)

If you see a warning about `on_bad_lines` not being recognized, replace in Cell 4:
```python
# Change this line:
tmdb_credits = pd.read_csv('tmdb_5000_credits.csv', on_bad_lines='skip', engine='python')

# To this (for pandas < 1.3):
tmdb_credits = pd.read_csv('tmdb_5000_credits.csv', error_bad_lines=False, engine='python')
```

### Very Long Runtime (> 1 hour)

The default settings use `population_size=100, generations=50` for the main GA-V and `population_size=50, generations=20` for each fold in the 5-fold CV loop. For a faster test run, temporarily reduce these in Cells 9 and 22:

```python
# Faster test settings (lower accuracy)
GAVotingClassifier(estimators=base_models, population_size=20, generations=10)
```

---

## Notes

- The random seed is fixed at `42` throughout the entire notebook for full reproducibility. Every run with the same data will produce identical results.
- The notebook uses an 80/20 stratified train/test split, ensuring both classes are proportionally represented in both sets.
- All cross-validation is performed strictly on the training set. The test set is only used for final evaluation.

---

*Implementation based on: Faisal, M. S., Rizwan, A., Iqbal, K., Fasihuddin, H., Banjar, A., & Daud, A. (2022). Prediction of Movie Quality via Adaptive Voting Classifier. IEEE Access, 10, 81581–81596.*
