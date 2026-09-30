from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from xgboost import XGBRegressor


warnings.filterwarnings("ignore")


# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

PROCESSED = ROOT / "processed"

DATA_FILE = (
    PROCESSED
    / "descriptors.csv"
)

FEATURE_FILE = (
    PROCESSED
    / "feature_list.txt"
)

FAMILY_FILE = (
    ROOT
    / "data"
    / "nebDFT2k"
    / "structure_classification_V3"
    / "structure_family_classified_V3.csv"
)

OUTPUT_DIR = ROOT / "olivine_GeoXGB"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. SETTINGS
# ============================================================

SEED = 42

TARGET = "em_dft"

FAMILY_NAME = "Olivine-type"

EXPECTED_TOTAL_EVENTS = 1681

EXPECTED_SPLITS = {
    "train": 1220,
    "val": 241,
    "test": 220,
}

# ------------------------------------------------------------
# Same GeoXGB configuration as the main model
# ------------------------------------------------------------

XGB_PARAMS = {
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
    "n_jobs": -1,
}


# ============================================================
# 3. HEADER
# ============================================================

def print_header():

    print()
    print("=" * 78)
    print("Family-specific retraining experiment")
    print("=" * 78)

    print(f"Dataset  : official nebDFT2k")
    print(f"Target   : {TARGET}")
    print("Features : 36 geometry descriptors")
    print(f"Family   : {FAMILY_NAME}")
    print(f"Seed     : {SEED}")

    print("=" * 78)
    print()


# ============================================================
# 4. LOAD FEATURE LIST
# ============================================================

def load_feature_list():

    print("=" * 78)
    print("1. LOADING GEOXGB FEATURE LIST")
    print("=" * 78)

    print()
    print(f"Feature file:")
    print(f"  {FEATURE_FILE}")
    print()

    if not FEATURE_FILE.exists():

        raise FileNotFoundError(
            "\nCould not find GeoXGB feature list:\n"
            f"  {FEATURE_FILE}\n"
        )

    features = [
        line.strip()
        for line in FEATURE_FILE.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    print(f"Features found: {len(features)}")

    if len(features) != 36:

        raise ValueError(
            "\nExpected exactly 36 GeoXGB features, "
            f"but found {len(features)}."
        )

    print()
    print("Verified 36-feature GeoXGB input.")
    print()

    for i, feature in enumerate(features, start=1):
        print(f"{i:2d}. {feature}")

    print()

    return features


# ============================================================
# 5. LOAD DESCRIPTOR DATASET
# ============================================================

def load_descriptor_dataset():

    print("=" * 78)
    print("2. LOADING GEOMETRY DESCRIPTOR DATASET")
    print("=" * 78)

    print()
    print(f"Dataset:")
    print(f"  {DATA_FILE}")
    print()

    if not DATA_FILE.exists():

        raise FileNotFoundError(
            "\nCould not find descriptor dataset:\n"
            f"  {DATA_FILE}\n"
        )

    df = pd.read_csv(DATA_FILE)

    print(f"Rows    : {len(df)}")
    print(f"Columns : {len(df.columns)}")
    print()

    if len(df) != EXPECTED_TOTAL_EVENTS:

        raise ValueError(
            "\nUnexpected number of events.\n"
            f"Expected : {EXPECTED_TOTAL_EVENTS}\n"
            f"Found    : {len(df)}"
        )

    required_columns = [
        "material_id",
        "edge_id",
        TARGET,
        "_split",
        "split",
    ]

    missing = [
        c for c in required_columns
        if c not in df.columns
    ]

    if missing:

        raise ValueError(
            "\nMissing required columns:\n"
            + "\n".join(f"  {x}" for x in missing)
        )

    print("Required columns verified.")

    return df


# ============================================================
# 6. VERIFY FEATURE COLUMNS
# ============================================================

def verify_features(df, features):

    print()
    print("=" * 78)
    print("3. VERIFYING 36 GEOXGB FEATURES")
    print("=" * 78)
    print()

    missing = [
        feature
        for feature in features
        if feature not in df.columns
    ]

    if missing:

        raise ValueError(
            "\nThe following GeoXGB features are missing "
            "from the descriptor dataset:\n"
            + "\n".join(f"  {x}" for x in missing)
        )

    print("All 36 features are present in the dataset.")

    print()

    print("Feature groups:")
    print("  Global structural descriptors : 5")
    print("  Local Li-environment          : 12")
    print("  Li-distribution               : 4")
    print("  Migration-path descriptors    : 15")
    print("  --------------------------------")
    print("  Total                         : 36")

    print()


# ============================================================
# 7. VERIFY OFFICIAL SPLIT
# ============================================================

def verify_official_split(df):

    print("=" * 78)
    print("4. VERIFYING OFFICIAL nebDFT2k SPLIT")
    print("=" * 78)
    print()

    df["split"] = (
        df["split"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    split_counts = (
        df["split"]
        .value_counts()
        .to_dict()
    )

    print("Observed split:")
    for split_name in ["train", "val", "test"]:

        count = split_counts.get(
            split_name,
            0,
        )

        expected = EXPECTED_SPLITS[split_name]

        print(
            f"  {split_name:5s} : "
            f"{count:4d} "
            f"(expected {expected})"
        )

        if count != expected:

            raise ValueError(
                f"\nOfficial split mismatch for '{split_name}'. "
                f"Expected {expected}, found {count}."
            )

    print()
    print("Official split verified.")
    print()


# ============================================================
# 8. CREATE EVENT KEY
# ============================================================

def add_event_key(df):

    df = df.copy()

    df["event_key"] = (
        df["material_id"].astype(str)
        + "::"
        + df["edge_id"].astype(str)
    )

    if df["event_key"].duplicated().any():

        duplicated = (
            df.loc[
                df["event_key"].duplicated(
                    keep=False
                ),
                "event_key",
            ]
            .unique()
        )

        raise ValueError(
            "\nDuplicate event_key detected.\n"
            f"Examples: {duplicated[:10]}"
        )

    return df


# ============================================================
# 9. LOAD V3 FAMILY CLASSIFICATION
# ============================================================

def load_family_classification():

    print("=" * 78)
    print("5. LOADING V3 STRUCTURE-FAMILY CLASSIFICATION")
    print("=" * 78)
    print()

    print("Family file:")
    print(f"  {FAMILY_FILE}")
    print()

    if not FAMILY_FILE.exists():

        raise FileNotFoundError(
            "\nCould not find V3 family classification file:\n"
            f"  {FAMILY_FILE}\n"
        )

    family = pd.read_csv(FAMILY_FILE)

    print(f"Rows    : {len(family)}")
    print(f"Columns : {len(family.columns)}")
    print()

    required = [
        "material_id",
        "edge_id",
    ]

    missing = [
        c
        for c in required
        if c not in family.columns
    ]

    if missing:

        raise ValueError(
            "\nMissing required family columns:\n"
            + "\n".join(f"  {x}" for x in missing)
        )

    # --------------------------------------------------------
    # The V3 file contains duplicate family-related columns
    # because information from multiple sources was merged.
    #
    # We explicitly use the FINAL 'structure_family' column.
    # --------------------------------------------------------

    if "structure_family" not in family.columns:

        raise ValueError(
            "\nV3 classification file does not contain "
            "'structure_family'."
        )

    print("Available structure families:")

    family_counts = (
        family["structure_family"]
        .astype(str)
        .value_counts()
    )

    for name, count in family_counts.items():

        print(
            f"  {name:40s} : {count:5d}"
        )

    print()

    # --------------------------------------------------------
    # Keep only columns needed by this experiment
    # --------------------------------------------------------

    keep_columns = [
        "material_id",
        "edge_id",
        "structure_family",
    ]

    # Use the final classification confidence if available.
    if "classification_confidence_y" in family.columns:

        keep_columns.append(
            "classification_confidence_y"
        )

    elif "classification_confidence" in family.columns:

        keep_columns.append(
            "classification_confidence"
        )

    # Useful classification score if available.
    if "classification_score" in family.columns:

        keep_columns.append(
            "classification_score"
        )

    family = family[keep_columns].copy()

    # --------------------------------------------------------
    # Check event uniqueness
    # --------------------------------------------------------

    family["event_key"] = (
        family["material_id"].astype(str)
        + "::"
        + family["edge_id"].astype(str)
    )

    duplicated = family["event_key"].duplicated(
        keep=False
    )

    if duplicated.any():

        duplicate_count = duplicated.sum()

        print(
            "WARNING:"
            f" {duplicate_count} duplicate family event rows detected."
        )

        # Keep first classification for duplicated event keys.
        family = family.drop_duplicates(
            subset="event_key",
            keep="first",
        )

    print(
        f"Unique classified events: {len(family)}"
    )

    print()

    return family


# ============================================================
# 10. MERGE FAMILY CLASSIFICATION
# ============================================================

def merge_family_information(
    df,
    family,
):

    print("=" * 78)
    print("6. MERGING FAMILY CLASSIFICATION")
    print("=" * 78)
    print()

    merged = df.merge(
        family,
        on="event_key",
        how="left",
        validate="one_to_one",
        suffixes=("", "_family"),
    )

    missing_family = (
        merged["structure_family"]
        .isna()
        .sum()
    )

    print(
        f"Events without family classification : "
        f"{missing_family}"
    )

    if missing_family > 0:

        missing_rows = merged.loc[
            merged["structure_family"].isna(),
            [
                "material_id",
                "edge_id",
            ],
        ]

        print()
        print("Examples:")
        print(
            missing_rows.head(10).to_string(
                index=False
            )
        )

        raise ValueError(
            "\nSome nebDFT2k events could not be matched "
            "to the V3 family classification."
        )

    print()
    print("Family classification successfully merged.")
    print()

    return merged


# ============================================================
# 11. SELECT OLIVINE EVENTS
# ============================================================

def select_olivine_events(df):

    print("=" * 78)
    print("7. SELECTING OLIVINE-TYPE EVENTS")
    print("=" * 78)
    print()

    olivine = df.loc[
        df["structure_family"].astype(str).str.strip()
        == FAMILY_NAME
    ].copy()

    if len(olivine) == 0:

        raise ValueError(
            "\nNo Olivine-type events were found."
        )

    print(
        f"Olivine-type events : {len(olivine)}"
    )

    print(
        f"Unique materials    : "
        f"{olivine['material_id'].nunique()}"
    )

    print()

    print("Split distribution:")

    split_counts = (
        olivine["split"]
        .value_counts()
    )

    for split_name in [
        "train",
        "val",
        "test",
    ]:

        print(
            f"  {split_name:5s} : "
            f"{split_counts.get(split_name, 0):4d}"
        )

    print()

    # --------------------------------------------------------
    # Confidence distribution
    # --------------------------------------------------------

    confidence_column = None

    if "classification_confidence_y" in olivine.columns:

        confidence_column = (
            "classification_confidence_y"
        )

    elif "classification_confidence" in olivine.columns:

        confidence_column = (
            "classification_confidence"
        )

    if confidence_column is not None:

        print(
            "Olivine classification confidence:"
        )

        confidence_counts = (
            olivine[confidence_column]
            .astype(str)
            .value_counts()
        )

        for confidence, count in (
            confidence_counts.items()
        ):

            print(
                f"  {confidence:10s} : "
                f"{count:4d}"
            )

        print()

    return olivine


# ============================================================
# 12. CHECK MATERIAL OVERLAP
# ============================================================

def check_material_overlap(df):

    print("=" * 78)
    print("8. CHECKING MATERIAL OVERLAP")
    print("=" * 78)
    print()

    train = df.loc[
        df["split"] == "train"
    ]

    val = df.loc[
        df["split"] == "val"
    ]

    test = df.loc[
        df["split"] == "test"
    ]

    train_materials = set(
        train["material_id"]
    )

    val_materials = set(
        val["material_id"]
    )

    test_materials = set(
        test["material_id"]
    )

    train_val = (
        train_materials
        & val_materials
    )

    train_test = (
        train_materials
        & test_materials
    )

    val_test = (
        val_materials
        & test_materials
    )

    print(
        f"Train materials : "
        f"{len(train_materials)}"
    )

    print(
        f"Val materials   : "
        f"{len(val_materials)}"
    )

    print(
        f"Test materials  : "
        f"{len(test_materials)}"
    )

    print()

    print(
        f"Train ∩ Val     : {len(train_val)}"
    )

    print(
        f"Train ∩ Test    : {len(train_test)}"
    )

    print(
        f"Val ∩ Test      : {len(val_test)}"
    )

    print()

    if train_val:
        print(
            "WARNING: material overlap exists "
            "between train and validation."
        )

    if train_test:
        print(
            "WARNING: material overlap exists "
            "between train and test."
        )

    if val_test:
        print(
            "WARNING: material overlap exists "
            "between validation and test."
        )

    if not train_val and not train_test and not val_test:

        print(
            "Material-level split integrity verified."
        )

    print()

    return {
        "train_materials": len(train_materials),
        "val_materials": len(val_materials),
        "test_materials": len(test_materials),
        "train_val_overlap": len(train_val),
        "train_test_overlap": len(train_test),
        "val_test_overlap": len(val_test),
    }


# ============================================================
# 13. PREPARE X AND y
# ============================================================

def prepare_data(
    olivine,
    features,
):

    print("=" * 78)
    print("9. PREPARING MODEL INPUT")
    print("=" * 78)
    print()

    train = olivine.loc[
        olivine["split"] == "train"
    ].copy()

    val = olivine.loc[
        olivine["split"] == "val"
    ].copy()

    test = olivine.loc[
        olivine["split"] == "test"
    ].copy()

    print(
        f"Train events : {len(train)}"
    )

    print(
        f"Val events   : {len(val)}"
    )

    print(
        f"Test events  : {len(test)}"
    )

    print()

    if len(train) == 0:
        raise ValueError(
            "No Olivine training events."
        )

    if len(val) == 0:
        raise ValueError(
            "No Olivine validation events."
        )

    if len(test) == 0:
        raise ValueError(
            "No Olivine test events."
        )

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    for split_name, split_df in [
        ("train", train),
        ("val", val),
        ("test", test),
    ]:

        if split_df[TARGET].isna().any():

            raise ValueError(
                f"Missing target values in "
                f"{split_name} split."
            )

    X_train = train[features].copy()
    X_val = val[features].copy()
    X_test = test[features].copy()

    y_train = (
        train[TARGET]
        .astype(float)
        .to_numpy()
    )

    y_val = (
        val[TARGET]
        .astype(float)
        .to_numpy()
    )

    y_test = (
        test[TARGET]
        .astype(float)
        .to_numpy()
    )

    # --------------------------------------------------------
    # Train-only median imputation
    # --------------------------------------------------------

    imputer = SimpleImputer(
        strategy="median"
    )

    X_train_imp = imputer.fit_transform(
        X_train
    )

    X_val_imp = imputer.transform(
        X_val
    )

    X_test_imp = imputer.transform(
        X_test
    )

    print(
        "Train-only median imputation applied."
    )

    print()

    print(
        f"X_train shape : {X_train_imp.shape}"
    )

    print(
        f"X_val shape   : {X_val_imp.shape}"
    )

    print(
        f"X_test shape  : {X_test_imp.shape}"
    )

    print()

    return (
        train,
        val,
        test,
        X_train_imp,
        X_val_imp,
        X_test_imp,
        y_train,
        y_val,
        y_test,
        imputer,
    )


# ============================================================
# 14. TRAIN MODEL
# ============================================================

def train_model(
    X_train,
    y_train,
    X_val,
    y_val,
):

    print("=" * 78)
    print("10. TRAINING OLIVINE-SPECIFIC GEOXGB")
    print("=" * 78)
    print()

    print("XGBoost parameters:")

    for key, value in XGB_PARAMS.items():

        print(
            f"  {key:20s} : {value}"
        )

    print()

    model = XGBRegressor(
        **XGB_PARAMS
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[
            (
                X_train,
                y_train,
            ),
            (
                X_val,
                y_val,
            ),
        ],
        verbose=False,
    )

    print()
    print("Training completed.")
    print()

    return model


# ============================================================
# 15. EVALUATION
# ============================================================

def evaluate_predictions(
    y_true,
    y_pred,
):

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    r2 = r2_score(
        y_true,
        y_pred,
    )

    return {
        "MAE_eV": float(mae),
        "RMSE_eV": float(rmse),
        "R2": float(r2),
    }


# ============================================================
# 16. EVALUATE MODEL
# ============================================================

def evaluate_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
):

    print("=" * 78)
    print("11. MODEL PERFORMANCE")
    print("=" * 78)
    print()

    train_pred = model.predict(
        X_train
    )

    val_pred = model.predict(
        X_val
    )

    test_pred = model.predict(
        X_test
    )

    train_metrics = evaluate_predictions(
        y_train,
        train_pred,
    )

    val_metrics = evaluate_predictions(
        y_val,
        val_pred,
    )

    test_metrics = evaluate_predictions(
        y_test,
        test_pred,
    )

    print(
        f"{'Split':<10}"
        f"{'MAE (eV)':>15}"
        f"{'RMSE (eV)':>15}"
        f"{'R²':>12}"
    )

    print("-" * 52)

    print(
        f"{'Train':<10}"
        f"{train_metrics['MAE_eV']:>15.6f}"
        f"{train_metrics['RMSE_eV']:>15.6f}"
        f"{train_metrics['R2']:>12.6f}"
    )

    print(
        f"{'Validation':<10}"
        f"{val_metrics['MAE_eV']:>15.6f}"
        f"{val_metrics['RMSE_eV']:>15.6f}"
        f"{val_metrics['R2']:>12.6f}"
    )

    print(
        f"{'Test':<10}"
        f"{test_metrics['MAE_eV']:>15.6f}"
        f"{test_metrics['RMSE_eV']:>15.6f}"
        f"{test_metrics['R2']:>12.6f}"
    )

    print()

    print("=" * 78)
    print("OLIVINE TEST RESULT")
    print("=" * 78)

    print(
        f"Test MAE  : "
        f"{test_metrics['MAE_eV']:.6f} eV"
    )

    print(
        f"Test RMSE : "
        f"{test_metrics['RMSE_eV']:.6f} eV"
    )

    print(
        f"Test R²   : "
        f"{test_metrics['R2']:.6f}"
    )

    print()

    predictions = {
        "train": train_pred,
        "val": val_pred,
        "test": test_pred,
    }

    metrics = {
        "train": train_metrics,
        "validation": val_metrics,
        "test": test_metrics,
    }

    return (
        predictions,
        metrics,
    )


# ============================================================
# 17. SAVE PREDICTIONS
# ============================================================

def save_predictions(
    train,
    val,
    test,
    predictions,
):

    print("=" * 78)
    print("12. SAVING PREDICTIONS")
    print("=" * 78)
    print()

    frames = []

    for split_name, split_df in [
        ("train", train),
        ("val", val),
        ("test", test),
    ]:

        tmp = split_df[
            [
                "material_id",
                "edge_id",
                "event_key",
                "split",
                "structure_family",
                TARGET,
            ]
        ].copy()

        tmp["prediction_eV"] = predictions[
            split_name
        ]

        tmp["error_eV"] = (
            tmp["prediction_eV"]
            - tmp[TARGET]
        )

        tmp["absolute_error_eV"] = (
            tmp["error_eV"]
            .abs()
        )

        frames.append(tmp)

    prediction_df = pd.concat(
        frames,
        ignore_index=True,
    )

    output_file = (
        OUTPUT_DIR
        / "olivine_GeoXGB_predictions.csv"
    )

    prediction_df.to_csv(
        output_file,
        index=False,
    )

    print(
        f"Saved:\n  {output_file}"
    )

    print()

    return prediction_df


# ============================================================
# 18. FEATURE IMPORTANCE
# ============================================================

def save_feature_importance(
    model,
    features,
):

    print("=" * 78)
    print("13. FEATURE IMPORTANCE")
    print("=" * 78)
    print()

    importance = model.feature_importances_

    importance_df = pd.DataFrame(
        {
            "feature": features,
            "importance": importance,
        }
    )

    importance_df = (
        importance_df
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    output_file = (
        OUTPUT_DIR
        / "olivine_feature_importance.csv"
    )

    importance_df.to_csv(
        output_file,
        index=False,
    )

    print(
        importance_df.to_string(
            index=False
        )
    )

    print()

    print(
        f"Saved:\n  {output_file}"
    )

    print()

    return importance_df


# ============================================================
# 19. SAVE OLIVINE DATASET
# ============================================================

def save_olivine_dataset(
    olivine,
):

    output_file = (
        OUTPUT_DIR
        / "olivine_events_used.csv"
    )

    olivine.to_csv(
        output_file,
        index=False,
    )

    print(
        f"Saved Olivine dataset:\n"
        f"  {output_file}"
    )

    print()


# ============================================================
# 20. SAVE MODEL
# ============================================================

def save_model(
    model,
):

    output_file = (
        OUTPUT_DIR
        / "Olivine_GeoXGB.pkl"
    )

    import joblib

    joblib.dump(
        model,
        output_file,
    )

    print(
        f"Saved model:\n"
        f"  {output_file}"
    )

    print()

    return output_file


# ============================================================
# 21. SAVE IMPUTER
# ============================================================

def save_imputer(
    imputer,
):

    output_file = (
        OUTPUT_DIR
        / "Olivine_GeoXGB_imputer.pkl"
    )

    import joblib

    joblib.dump(
        imputer,
        output_file,
    )

    print(
        f"Saved imputer:\n"
        f"  {output_file}"
    )

    print()

    return output_file


# ============================================================
# 22. SAVE RESULTS JSON
# ============================================================

def save_results(
    features,
    olivine,
    metrics,
    overlap_info,
    importance_df,
):

    print("=" * 78)
    print("14. SAVING EXPERIMENT SUMMARY")
    print("=" * 78)
    print()

    confidence_summary = {}

    confidence_column = None

    if "classification_confidence_y" in olivine.columns:

        confidence_column = (
            "classification_confidence_y"
        )

    elif "classification_confidence" in olivine.columns:

        confidence_column = (
            "classification_confidence"
        )

    if confidence_column is not None:

        confidence_summary = (
            olivine[confidence_column]
            .astype(str)
            .value_counts()
            .to_dict()
        )

    split_summary = (
        olivine["split"]
        .value_counts()
        .to_dict()
    )

    results = {

        "experiment": {
            "name": (
                "Olivine-specific GeoXGB "
                "retraining"
            ),
            "dataset": "official nebDFT2k",
            "family": FAMILY_NAME,
            "target": TARGET,
            "seed": SEED,
        },

        "dataset": {
            "total_official_events":
                EXPECTED_TOTAL_EVENTS,

            "olivine_events":
                int(len(olivine)),

            "olivine_materials":
                int(
                    olivine[
                        "material_id"
                    ].nunique()
                ),

            "split_counts": {
                str(k): int(v)
                for k, v
                in split_summary.items()
            },

            "classification_confidence":
                {
                    str(k): int(v)
                    for k, v
                    in confidence_summary.items()
                },
        },

        "features": {
            "n_features":
                len(features),

            "feature_names":
                features,
        },

        "model": {
            key: (
                float(value)
                if isinstance(
                    value,
                    (np.float32, np.float64),
                )
                else value
            )
            for key, value
            in XGB_PARAMS.items()
        },

        "metrics": metrics,

        "material_overlap":
            overlap_info,

        "top_features": [
            {
                "feature": str(row["feature"]),
                "importance":
                    float(row["importance"]),
            }
            for _, row
            in importance_df.head(36).iterrows()
        ],
    }

    output_file = (
        OUTPUT_DIR
        / "olivine_GeoXGB_results.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
        )

    print(
        f"Saved:\n  {output_file}"
    )

    print()


# ============================================================
# 23. SAVE EXPERIMENT README
# ============================================================

def save_readme(
    features,
    olivine,
    metrics,
):

    output_file = (
        OUTPUT_DIR
        / "experiment_summary.txt"
    )

    train_count = int(
        (olivine["split"] == "train").sum()
    )

    val_count = int(
        (olivine["split"] == "val").sum()
    )

    test_count = int(
        (olivine["split"] == "test").sum()
    )

    test_metrics = metrics["test"]

    text = f"""
OLIVINE-SPECIFIC GEOXGB RETRAINING
===================================

Dataset
-------
Official nebDFT2k

Official total events
---------------------
1681

Family
------
Olivine-type

Olivine events
--------------
{len(olivine)}

Unique Olivine materials
------------------------
{olivine["material_id"].nunique()}

Official split
--------------
Train : {train_count}
Val   : {val_count}
Test  : {test_count}

Target
------
em_dft

Features
--------
36 geometry descriptors

Feature list
------------
"""

    for i, feature in enumerate(
        features,
        start=1,
    ):

        text += (
            f"{i:02d}. {feature}\n"
        )

    text += f"""

Model
-----
XGBoost GeoXGB configuration

n_estimators      = 1200
max_depth         = 5
learning_rate     = 0.02
min_child_weight  = 4
subsample         = 0.85
colsample_bytree  = 0.75
reg_alpha         = 0.10
reg_lambda        = 2.0
objective         = reg:absoluteerror
eval_metric       = mae
tree_method       = hist
random_state      = 42

Test performance
----------------
MAE  = {test_metrics["MAE_eV"]:.6f} eV
RMSE = {test_metrics["RMSE_eV"]:.6f} eV
R2   = {test_metrics["R2"]:.6f}

Input restrictions
------------------
The model uses only the 36 geometry descriptors.

No relaxed structure.
No NEB energy.
No BVSE feature.
No migration barrier target leakage.
No additional family-specific model features.

Purpose
-------
This experiment evaluates whether retraining the same GeoXGB
model specifically on Olivine-type migration events changes
predictive performance relative to the all-material GeoXGB model.
"""

    output_file.write_text(
        text.strip() + "\n",
        encoding="utf-8",
    )

    print(
        f"Saved:\n  {output_file}"
    )

    print()


# ============================================================
# 24. MAIN
# ============================================================

def main():

    print_header()

    # --------------------------------------------------------
    # Load exact 36-feature definition
    # --------------------------------------------------------

    features = load_feature_list()

    # --------------------------------------------------------
    # Load 1681-event descriptor dataset
    # --------------------------------------------------------

    df = load_descriptor_dataset()

    # --------------------------------------------------------
    # Verify exact feature columns
    # --------------------------------------------------------

    verify_features(
        df,
        features,
    )

    # --------------------------------------------------------
    # Verify official split
    # --------------------------------------------------------

    verify_official_split(
        df
    )

    # --------------------------------------------------------
    # Create event key
    # --------------------------------------------------------

    df = add_event_key(
        df
    )

    # --------------------------------------------------------
    # Load V3 family classification
    # --------------------------------------------------------

    family = load_family_classification()

    # --------------------------------------------------------
    # Merge family classification
    # --------------------------------------------------------

    df = merge_family_information(
        df,
        family,
    )

    # --------------------------------------------------------
    # Select Olivine events
    # --------------------------------------------------------

    olivine = select_olivine_events(
        df
    )

    # --------------------------------------------------------
    # Material overlap
    # --------------------------------------------------------

    overlap_info = check_material_overlap(
        olivine
    )

    # --------------------------------------------------------
    # Save the exact dataset used
    # --------------------------------------------------------

    save_olivine_dataset(
        olivine
    )

    # --------------------------------------------------------
    # Prepare model matrices
    # --------------------------------------------------------

    (
        train,
        val,
        test,
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        imputer,
    ) = prepare_data(
        olivine,
        features,
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model = train_model(
        X_train,
        y_train,
        X_val,
        y_val,
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    (
        predictions,
        metrics,
    ) = evaluate_model(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    save_predictions(
        train,
        val,
        test,
        predictions,
    )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    importance_df = save_feature_importance(
        model,
        features,
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    save_model(
        model
    )

    # --------------------------------------------------------
    # Save imputer
    # --------------------------------------------------------

    save_imputer(
        imputer
    )

    # --------------------------------------------------------
    # Save JSON result
    # --------------------------------------------------------

    save_results(
        features,
        olivine,
        metrics,
        overlap_info,
        importance_df,
    )

    # --------------------------------------------------------
    # Save text summary
    # --------------------------------------------------------

    save_readme(
        features,
        olivine,
        metrics,
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("=" * 78)
    print("OLIVINE-SPECIFIC GEOXGB EXPERIMENT COMPLETE")
    print("=" * 78)
    print()

    print(
        f"Olivine events : {len(olivine)}"
    )

    print(
        f"Olivine materials : "
        f"{olivine['material_id'].nunique()}"
    )

    print()

    print(
        "Final test performance:"
    )

    print(
        f"  MAE  : "
        f"{metrics['test']['MAE_eV']:.6f} eV"
    )

    print(
        f"  RMSE : "
        f"{metrics['test']['RMSE_eV']:.6f} eV"
    )

    print(
        f"  R²   : "
        f"{metrics['test']['R2']:.6f}"
    )

    print()

    print(
        f"All outputs saved to:\n"
        f"  {OUTPUT_DIR}"
    )

    print()
    print("=" * 78)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()