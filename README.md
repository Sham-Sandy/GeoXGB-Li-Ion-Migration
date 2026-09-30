# GeoXGB-Li-Ion-Migration
Geometry-based machine learning framework for Li-ion migration barrier prediction
GeoXGB: Li-Ion Migration-Barrier Prediction
Physically Interpretable Geometric Descriptors for Li-Ion Migration-Barrier Prediction
This repository contains the Python implementation and analysis workflow for the study “Physically Interpretable Geometric Descriptors for Li-Ion Migration-Barrier Prediction.”
The work develops GeoXGB, an XGBoost regression framework for predicting Li-ion migration barriers from physically interpretable geometry-based descriptors. The framework is intended as a rapid prescreening approach and does not replace DFT-NEB calculations.
Dataset
The workflow uses the official nebDFT2k migration-barrier dataset containing 1,681 migration events.
Split	Events
Training	1,220
Validation	241
Independent test	220
Total	1,681
The target variable is `em_dft`. The official train/validation/test partition is retained throughout the main workflow.
36 Geometry Descriptors
GeoXGB uses 36 physically interpretable descriptors:
Descriptor group	Number
Global structural descriptors	5
Local Li-environment descriptors	12
Li-distribution descriptors	4
Migration-path descriptors	15
Total	36
The descriptors describe structural extent, local Li-host environment, Li distribution, migration distance, pathway geometry, bottleneck clearance, and coordination changes.
GeoXGB Model
The final XGBoost configuration is:
Parameter	Value
`n_estimators`	1200
`max_depth`	5
`learning_rate`	0.02
`min_child_weight`	4
`subsample`	0.85
`colsample_bytree`	0.75
`reg_alpha`	0.10
`reg_lambda`	2.0
Objective	`reg:absoluteerror`
Evaluation metric	MAE
Tree method	histogram
Random seed	42
Reported Test Performance
On the independent test set:
Metric	Value
MAE	0.277 eV
RMSE	0.434 eV
R²	0.729
The 15 migration-path/bottleneck descriptors alone give a test MAE of approximately 0.289 eV.
Repository Workflow
```text
nebDFT2k dataset
       |
       v
01_descriptor_generation.py
       |
       v
36 geometry descriptors
       |
       v
02_benchmark_model.py
       |
       +--> Mean baseline
       +--> Ridge
       +--> Random Forest
       +--> ExtraTrees
       +--> HistGradientBoosting
       +--> XGBoost / GeoXGB
       |
       v
03_analysis_and_publication_figures.py
       |
       +--> Error analysis
       +--> Permutation importance
       +--> Bootstrap confidence intervals
       +--> Conformal uncertainty
       +--> Learning curve
       +--> Publication figures
```
Main Scripts
`01_descriptor_generation.py`
Prepares the nebDFT2k index and generates the geometry descriptors. It verifies the 1,681 events and the official 1,220/241/220 split, processes migration-event structure files, identifies the mobile ion, and calculates local, distribution, and migration-path descriptors.
`02_benchmark_model.py`
Benchmarks:
Mean baseline
Ridge
Random Forest
ExtraTrees
HistGradientBoosting
XGBoost (GeoXGB)
The script uses training-only median imputation, trains the models, evaluates train/validation/test performance, and saves model files and prediction tables.
`03_analysis_and_publication_figures.py`
Generates the manuscript analysis, including:
Test predictions
Error analysis by barrier range
Worst test events
Permutation importance
Bootstrap 95% confidence intervals
Conformal uncertainty
Learning curve
Test parity plot
Residual plot
Absolute-error plot
Feature-importance plot
Uncertainty-calibration plot
Learning-curve figure
`new_pred.py`
Predicts migration barriers for a new crystal structure. It identifies mobile ions, generates candidate Li-to-Li hops from periodic crystal geometry, constructs a seven-point straight-line candidate path, calculates the required descriptors, and applies the trained GeoXGB model.
For a new static structure, the migration path is geometry-inferred rather than a DFT-NEB trajectory. Predictions should therefore be treated as screening estimates.
`olivine.py`
Performs a family-specific analysis for Olivine-type structures using the same 36 descriptors and GeoXGB configuration.
Recommended Repository Structure
```text
GeoXGB/
├── README.md
├── requirements.txt
├── 01_descriptor_generation.py
├── 02_benchmark_model.py
├── 03_analysis_and_publication_figures.py
├── new_pred.py
├── olivine.py
│
├── data/
│   └── nebDFT2k/
│
├── processed/
│   ├── descriptors.csv
│   ├── official_1681_dataset.csv
│   ├── official_descriptors.csv
│   └── feature_list.txt
│
├── models/
│   ├── GeoXGB.pkl
│   ├── selected_model.pkl
│   ├── train_only_medians.pkl
│   └── feature_list.txt
│
├── results/
├── tables/
├── figures/
└── olivine_GeoXGB/
```
Installation
Python 3.11 or later is recommended.
```bash
python -m venv venv
```
Windows:
```bash
venv\Scripts\activate
```
Linux/macOS:
```bash
source venv/bin/activate
```
Install dependencies:
```bash
pip install -r requirements.txt
```
The workflow uses packages including NumPy, Pandas, Scikit-learn, XGBoost, ASE, and Matplotlib.
Running the Workflow
1. Generate descriptors
```bash
python 01_descriptor_generation.py
```
2. Benchmark models
```bash
python 02_benchmark_model.py
```
3. Generate analysis and figures
```bash
python 03_analysis_and_publication_figures.py
```
4. Predict a new structure
```bash
python new_pred.py path/to/structure.cif
```
Analysis and Uncertainty
The repository includes:
Permutation importance using the validation set
Bootstrap resampling with 10,000 iterations for MAE, RMSE, and R² confidence intervals
Conformal prediction using validation residuals
Nominal 80%, 90%, and 95% prediction intervals
Learning-curve analysis
Reproducibility
The main workflow uses random seed 42. Preprocessing statistics are obtained from the training subset and then applied unchanged to validation and test data. The independent test set is reserved for final evaluation.
Scientific Scope and Limitation
Migration barriers are pathway-dependent energetic quantities. GeoXGB predicts barriers from geometric information and does not explicitly represent the complete potential-energy surface or transition-state energetics.
The model therefore may not capture all effects associated with:
Electronic structure
Bonding changes
Lattice relaxation
Transition-state energetics
Other energetic effects not explicitly represented by the descriptors
GeoXGB is a rapid prescreening tool and does not replace DFT-NEB calculations.
Data Availability
The nebDFT2k dataset should be obtained from its original public source and used according to its original license. The original dataset publication should be cited when the dataset is used.
Code Availability
GitHub repository:
`https://github.com/YOUR-USERNAME/YOUR-REPOSITORY`
Replace the placeholder above with the final repository URL before journal submission.
A permanent Zenodo archive and DOI can be added after the GitHub repository is connected to Zenodo.
Contact
Shamili G.  
Department of Chemistry  
SRM Institute of Science and Technology  
Tamil Nadu, India
