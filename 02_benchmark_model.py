
from pathlib import Path
import pickle
import json
import numpy as np
import pandas as pd

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "processed"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
TABLES = ROOT / "tables"

for p in [RESULTS, MODELS, TABLES]:
    p.mkdir(parents=True, exist_ok=True)

DATA = PROCESSED / "official_descriptors.csv"
FEATURE_FILE = PROCESSED / "feature_list.txt"

SEED = 42

# Official nebDFT2k split
N_TRAIN = 1220
N_VAL = 241
N_TEST = 220


def met(y, p):
    return {
        "MAE_eV": mean_absolute_error(y, p),
        "RMSE_eV": np.sqrt(mean_squared_error(y, p)),
        "R2": r2_score(y, p),
    }


def main():

    # --------------------------------------------------------
    # 1. Load data
    # --------------------------------------------------------
    if not DATA.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA}")

    if not FEATURE_FILE.exists():
        raise FileNotFoundError(f"Feature list not found: {FEATURE_FILE}")

    d = pd.read_csv(DATA)

    if len(d) != 1681:
        raise RuntimeError(
            f"Expected 1681 rows, found {len(d)}"
        )

    FEATURES = [
        x.strip()
        for x in FEATURE_FILE.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]

    if len(FEATURES) != 36:
        raise RuntimeError(
            f"Expected exactly 36 descriptors, found {len(FEATURES)}"
        )

    required_columns = set(FEATURES) | {
        "split",
        "em_dft",
        "material_id",
        "edge_id",
    }

    missing = required_columns - set(d.columns)

    if missing:
        raise RuntimeError(
            f"Missing required columns: {sorted(missing)}"
        )

    # --------------------------------------------------------
    # 2. Official split
    # --------------------------------------------------------
    tr = d[d["split"] == "train"].copy()
    va = d[d["split"] == "val"].copy()
    te = d[d["split"] == "test"].copy()

    if (len(tr), len(va), len(te)) != (
        N_TRAIN,
        N_VAL,
        N_TEST,
    ):
        raise RuntimeError(
            "Official split mismatch: "
            f"train={len(tr)}, val={len(va)}, test={len(te)}"
        )

    print("\nOfficial split:")
    print(f"  Train      : {len(tr)}")
    print(f"  Validation : {len(va)}")
    print(f"  Test       : {len(te)}")
    print(f"  Features   : {len(FEATURES)}")

    # --------------------------------------------------------
    # 3. Feature matrices
    # --------------------------------------------------------
    Xtr = tr[FEATURES].replace([np.inf, -np.inf], np.nan)
    Xv = va[FEATURES].replace([np.inf, -np.inf], np.nan)
    Xt = te[FEATURES].replace([np.inf, -np.inf], np.nan)

    # TRAIN-ONLY imputation.
    # Validation and test never contribute to imputation statistics.
    med = Xtr.median().fillna(0)

    Xtr = Xtr.fillna(med)
    Xv = Xv.fillna(med)
    Xt = Xt.fillna(med)

  
    medians = med.to_dict()

    with open(MODELS / "train_only_medians.pkl", "wb") as f:
        pickle.dump(medians, f)

    (MODELS / "feature_list.txt").write_text(
        "\n".join(FEATURES) + "\n",
        encoding="utf-8",
    )

    ytr = tr["em_dft"].to_numpy(dtype=float)
    yv = va["em_dft"].to_numpy(dtype=float)
    yt = te["em_dft"].to_numpy(dtype=float)

    # --------------------------------------------------------
    # 4. Define benchmark models
    # --------------------------------------------------------
    models = {

        "Mean_baseline": DummyRegressor(
            strategy="mean"
        ),

        "Ridge": Ridge(
            alpha=10.0
        ),

        "RandomForest": RandomForestRegressor(
            n_estimators=1000,
            max_features=0.70,
            min_samples_leaf=1,
            random_state=SEED,
            n_jobs=-1,
        ),

        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=1200,
            max_features=0.65,
            min_samples_leaf=2,
            random_state=SEED,
            n_jobs=-1,
        ),

        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=600,
            learning_rate=0.04,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=SEED,
        ),

        # Final model: GeoXGB
        "XGBoost": XGBRegressor(
            n_estimators=1200,
            max_depth=5,
            learning_rate=0.02,
            min_child_weight=4,
            subsample=0.85,
            colsample_bytree=0.75,
            reg_alpha=0.10,
            reg_lambda=2.0,
            objective="reg:absoluteerror",
            eval_metric="mae",
            tree_method="hist",
            random_state=SEED,
            n_jobs=-1,
        ),
    }

    # --------------------------------------------------------
    # 5. Train and save EVERY model
    # --------------------------------------------------------
    rows = []
    predictions = {}

    print("\nTraining models...\n")

    for name, model in models.items():

        print(f"Training {name}...")

        model.fit(Xtr, ytr)

        ptr = model.predict(Xtr)
        pv = model.predict(Xv)
        pt = model.predict(Xt)

        mtr = met(ytr, ptr)
        mv = met(yv, pv)
        mt = met(yt, pt)

        rows.append({
            "model": name,
            "n_features": len(FEATURES),

            "train_MAE_eV": mtr["MAE_eV"],
            "train_RMSE_eV": mtr["RMSE_eV"],
            "train_R2": mtr["R2"],

            "val_MAE_eV": mv["MAE_eV"],
            "val_RMSE_eV": mv["RMSE_eV"],
            "val_R2": mv["R2"],

            "test_MAE_eV": mt["MAE_eV"],
            "test_RMSE_eV": mt["RMSE_eV"],
            "test_R2": mt["R2"],
        })

        predictions[name] = {
            "model": model,
            "train": ptr,
            "val": pv,
            "test": pt,
        }

        # ----------------------------------------------------
        # Save each model independently
        # ----------------------------------------------------
        model_filename = f"{name}.pkl"

        with open(MODELS / model_filename, "wb") as f:
            pickle.dump(model, f)

        print(
            f"{name:24s} | "
            f"VAL MAE={mv['MAE_eV']:.6f} | "
            f"TEST MAE={mt['MAE_eV']:.6f}"
        )

    # --------------------------------------------------------
    # 6. Benchmark table
    # --------------------------------------------------------
    result = pd.DataFrame(rows)

  
    result["validation_rank"] = (
        result["val_MAE_eV"]
        .rank(method="min", ascending=True)
        .astype(int)
    )

    
    result["final_model"] = result["model"].eq("XGBoost")

    result = result.sort_values(
        "val_MAE_eV"
    ).reset_index(drop=True)

    result.to_csv(
        TABLES / "benchmark.csv",
        index=False
    )

    # --------------------------------------------------------
    # 7. Explicitly select XGBoost as GeoXGB
    # --------------------------------------------------------
    final_name = "XGBoost"

    final_model = predictions[final_name]["model"]
    final_ptr = predictions[final_name]["train"]
    final_pv = predictions[final_name]["val"]
    final_pt = predictions[final_name]["test"]

    final_row = result[
        result["model"] == final_name
    ].iloc[0]

    for filename in [
        "GeoXGB.pkl",
        "selected_model.pkl",
    ]:
        with open(MODELS / filename, "wb") as f:
            pickle.dump(final_model, f)

    # --------------------------------------------------------
    # 8. Save predictions for final XGBoost model
    # --------------------------------------------------------
    pred_train = pd.DataFrame({
        "material_id": tr["material_id"].values,
        "edge_id": tr["edge_id"].values,
        "split": "train",
        "y_true_eV": ytr,
        "prediction_eV": final_ptr,
    })

    pred_val = pd.DataFrame({
        "material_id": va["material_id"].values,
        "edge_id": va["edge_id"].values,
        "split": "val",
        "y_true_eV": yv,
        "prediction_eV": final_pv,
    })

    pred_test = pd.DataFrame({
        "material_id": te["material_id"].values,
        "edge_id": te["edge_id"].values,
        "split": "test",
        "y_true_eV": yt,
        "prediction_eV": final_pt,
    })

    pred = pd.concat(
        [pred_train, pred_val, pred_test],
        ignore_index=True
    )

    pred["residual_eV"] = (
        pred["prediction_eV"] - pred["y_true_eV"]
    )

    pred["abs_error_eV"] = np.abs(
        pred["residual_eV"]
    )

    pred.to_csv(
        RESULTS / "GeoXGB_predictions_all.csv",
        index=False
    )

    pred_val.to_csv(
        RESULTS / "GeoXGB_predictions_val.csv",
        index=False
    )

    pred_test.to_csv(
        RESULTS / "GeoXGB_predictions_test.csv",
        index=False
    )

    # Also retain a generic predictions_test.csv for compatibility.
    pred_test.to_csv(
        RESULTS / "predictions_test.csv",
        index=False
    )

    # --------------------------------------------------------
    # 9. Save final-model metadata
    # --------------------------------------------------------
    metadata = {
        "selected_model": "XGBoost",
        "model_name": "GeoXGB",
        "selection_basis": "predefined_final_model",
        "test_set_used_for_model_selection": False,

        "n_features": len(FEATURES),
        "features": FEATURES,

        "train_size": N_TRAIN,
        "validation_size": N_VAL,
        "test_size": N_TEST,

        "validation_MAE_eV": float(
            final_row["val_MAE_eV"]
        ),
        "validation_RMSE_eV": float(
            final_row["val_RMSE_eV"]
        ),
        "validation_R2": float(
            final_row["val_R2"]
        ),

        "test_MAE_eV": float(
            final_row["test_MAE_eV"]
        ),
        "test_RMSE_eV": float(
            final_row["test_RMSE_eV"]
        ),
        "test_R2": float(
            final_row["test_R2"]
        ),


        "xgboost_parameters": {
            "n_estimators": 1200,
            "max_depth": 5,
            "learning_rate": 0.02,
            "min_child_weight": 4,
            "subsample": 0.85,
            "colsample_bytree": 0.75,
            "reg_alpha": 0.10,
            "reg_lambda": 2.0,
            "objective": "reg:absoluteerror",
            "eval_metric": "mae",
            "tree_method": "hist",
            "random_state": SEED,
        },

        "saved_models": [
            "Mean_baseline.pkl",
            "Ridge.pkl",
            "RandomForest.pkl",
            "ExtraTrees.pkl",
            "HistGradientBoosting.pkl",
            "XGBoost.pkl",
            "GeoXGB.pkl",
            "selected_model.pkl",
        ],

        "preprocessing": {
            "missing_value_strategy": "median",
            "median_fit_on": "training_split_only",
            "infinite_values_replaced_with_nan": True,
        },
    }

    (RESULTS / "selected_model.json").write_text(
        json.dumps(
            metadata,
            indent=2
        ),
        encoding="utf-8",
    )

 
    (RESULTS / "benchmark_metadata.json").write_text(
        json.dumps(
            {
                "dataset": "official nebDFT2k",
                "total_events": 1681,
                "train": N_TRAIN,
                "validation": N_VAL,
                "test": N_TEST,
                "n_features": len(FEATURES),
                "final_model": "GeoXGB",
                "final_estimator": "XGBRegressor",
                "selection_basis": "predefined_final_model",
                "test_used_for_selection": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # 10. Print final summary
    # --------------------------------------------------------
    print("\n" + "=" * 72)
    print("FINAL MODEL: GeoXGB")
    print("=" * 72)

    print(
        f"Validation MAE : "
        f"{final_row['val_MAE_eV']:.6f} eV"
    )
    print(
        f"Test MAE       : "
        f"{final_row['test_MAE_eV']:.6f} eV"
    )
    print(
        f"Test RMSE      : "
        f"{final_row['test_RMSE_eV']:.6f} eV"
    )
    print(
        f"Test R2        : "
        f"{final_row['test_R2']:.6f}"
    )

    print("\nSaved models:")
    for filename in [
        "Mean_baseline.pkl",
        "Ridge.pkl",
        "RandomForest.pkl",
        "ExtraTrees.pkl",
        "HistGradientBoosting.pkl",
        "XGBoost.pkl",
        "GeoXGB.pkl",
        "selected_model.pkl",
        "train_only_medians.pkl",
    ]:
        print(f"  models/{filename}")

    print("\nSaved benchmark:")
    print("  tables/benchmark.csv")

    print("\nSaved final-model metadata:")
    print("  results/selected_model.json")

    print("\nDone.")


if __name__ == "__main__":
    main()
