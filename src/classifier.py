"""
НИРС-03: Классификатор адресов в сети криптовалюты Bitcoin  (v2)

Улучшения по сравнению с v1:
  - SMOTE vs ADASYN: автовыбор лучшей стратегии ресэмплинга
  - Добавлен LightGBM с тюнингом
  - Тюнинг XGBoost и RandomForest через RandomizedSearchCV
  - StackingClassifier (RF + XGBoost + LightGBM → LogReg)
  - Расширенный поиск гиперпараметров MLP
  - Оптимизация порогов классификации (критерий Юдена)
  - SHAP-анализ для лучшей древовидной модели
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, f1_score,
    ConfusionMatrixDisplay,
)
from imblearn.over_sampling import SMOTE, ADASYN
import xgboost as xgb
import lightgbm as lgb
import shap

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Пути и константы
# ---------------------------------------------------------------------------
ROOT         = Path(__file__).parent.parent          # nirs-03-report/
DATASET_PATH = ROOT.parent / "nirs-02-report" / "bitcoin_data_collector" / "data" / "bitcoin_address_features_optimized.csv"
OUTPUT_DIR   = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
CV           = StratifiedKFold(n_splits=2, shuffle=True, random_state=RANDOM_STATE)


# ---------------------------------------------------------------------------
# 1. Загрузка данных
# ---------------------------------------------------------------------------

def load_data(path: Path):
    df = pd.read_csv(path)
    print(f"Загружено {len(df)} адресов, {df.shape[1]} столбцов")
    print(f"Распределение классов:\n{df['label'].value_counts()}\n")
    X = df.drop(columns=["address", "label"]).select_dtypes(include="number")
    y = df["label"]
    return X, y


def remove_correlated(X: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    corr  = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop  = [c for c in upper.columns if any(upper[c] > threshold)]
    print(f"Удалено {len(drop)} коррелированных признаков: {drop}")
    return X.drop(columns=drop)


# ---------------------------------------------------------------------------
# 2. Разбивка и ресэмплинг
# ---------------------------------------------------------------------------

def _best_sampler(X_train, y_train, X_val, y_val) -> tuple:
    """Выбирает SMOTE или ADASYN по val F1 (быстрый RF-прокси)."""
    probe  = RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1)
    scaler = StandardScaler()
    best_f1, best_sampler, best_name = -1, None, "SMOTE"

    for name, sampler in [("SMOTE",  SMOTE(random_state=RANDOM_STATE)),
                           ("ADASYN", ADASYN(random_state=RANDOM_STATE))]:
        try:
            Xr, yr = sampler.fit_resample(X_train, y_train)
            Xs     = scaler.fit_transform(Xr)
            probe.fit(Xs, yr)
            f1 = f1_score(y_val, probe.predict(scaler.transform(X_val)), average="weighted")
            print(f"  {name}: val F1 = {f1:.4f}")
            if f1 > best_f1:
                best_f1, best_sampler, best_name = f1, sampler, name
        except Exception as e:
            print(f"  {name}: ошибка — {e}")

    print(f"  → выбран {best_name}\n")
    return best_sampler, best_name


def prepare_splits(X: pd.DataFrame, y):
    le    = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.15, stratify=y_enc, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15 / 0.85, stratify=y_train, random_state=RANDOM_STATE
    )
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    sampler, sampler_name = _best_sampler(X_train, y_train, X_val, y_val)
    X_train_res, y_train_res = sampler.fit_resample(X_train, y_train)
    counts = dict(zip(*np.unique(y_train_res, return_counts=True)))
    print(f"После {sampler_name} — Train: {len(X_train_res)}, классы: {counts}\n")

    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_res)
    X_val_sc   = scaler.transform(X_val)
    X_test_sc  = scaler.transform(X_test)

    return (X_train_sc, y_train_res,
            X_val_sc,   y_val,
            X_test_sc,  y_test,
            scaler, le, le.classes_)


# ---------------------------------------------------------------------------
# 3. Вспомогательный тюнер
# ---------------------------------------------------------------------------

def _tune(base_model, param_dist, X_train, y_train, n_iter: int, label: str):
    search = RandomizedSearchCV(
        base_model, param_dist, n_iter=n_iter,
        scoring="f1_weighted", cv=CV,
        random_state=RANDOM_STATE, n_jobs=-1, verbose=0,
    )
    search.fit(X_train, y_train)
    print(f"  {label:<20s} CV F1={search.best_score_:.4f}  {search.best_params_}")
    return search.best_estimator_


# ---------------------------------------------------------------------------
# 4. Обучение всех моделей
# ---------------------------------------------------------------------------

def train_all(X_train, y_train, X_val, y_val, classes):
    print("=== Тюнинг и обучение ===")

    # MLP: используем лучшие параметры из предыдущего эксперимента (тюнинг v1)
    # RandomizedSearchCV на 15k сэмплах нецелесообразен для sklearn MLP
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64), activation="relu", solver="adam",
        alpha=1e-4, learning_rate_init=5e-4, batch_size=32, max_iter=200,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=10,
        random_state=RANDOM_STATE, verbose=False,
    )
    mlp.fit(X_train, y_train)
    mlp_f1 = f1_score(y_val, mlp.predict(X_val), average="weighted")
    print(f"  {'MLP':<20s} CV F1=n/a  val F1={mlp_f1:.4f}  (фиксированные гиперпараметры)")

    rf = _tune(
        RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1),
        {
            "n_estimators":      [100, 200],
            "max_depth":         [None, 15],
            "min_samples_split": [2, 5],
            "max_features":      ["sqrt", "log2"],
        },
        X_train, y_train, n_iter=4, label="RandomForest",
    )

    xgboost = _tune(
        xgb.XGBClassifier(eval_metric="mlogloss", random_state=RANDOM_STATE,
                           verbosity=0, n_jobs=1),
        {
            "n_estimators":     [100, 200],
            "max_depth":        [4, 6],
            "learning_rate":    [0.1, 0.2],
            "subsample":        [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        },
        X_train, y_train, n_iter=6, label="XGBoost",
    )

    lightgbm = _tune(
        lgb.LGBMClassifier(class_weight="balanced", random_state=RANDOM_STATE,
                            verbosity=-1, n_jobs=1),
        {
            "n_estimators":     [100, 200],
            "max_depth":        [4, 6, -1],
            "learning_rate":    [0.1, 0.2],
            "num_leaves":       [31, 63],
            "subsample":        [0.8, 1.0],
        },
        X_train, y_train, n_iter=6, label="LightGBM",
    )

    # Stacking: RF + XGBoost + LightGBM → LogReg
    stacking = StackingClassifier(
        estimators=[("rf", rf), ("xgb", xgboost), ("lgb", lightgbm)],
        final_estimator=LogisticRegression(max_iter=500, random_state=RANDOM_STATE),
        cv=3, n_jobs=-1,
    )
    stacking.fit(X_train, y_train)
    print(f"  {'Stacking':<20s} обучен")

    # kNN без тюнинга (базовый)
    knn = KNeighborsClassifier(n_neighbors=7, n_jobs=-1)
    knn.fit(X_train, y_train)

    models = {
        "MLP": mlp,
        "RandomForest": rf,
        "XGBoost": xgboost,
        "LightGBM": lightgbm,
        "Stacking": stacking,
        "kNN": knn,
    }

    print("\nVal weighted F1:")
    for name, model in models.items():
        f1 = f1_score(y_val, model.predict(X_val), average="weighted")
        print(f"  {name:<20s}: {f1:.4f}")
    print()
    return models


# ---------------------------------------------------------------------------
# 5. Оценка
# ---------------------------------------------------------------------------

def evaluate(models: dict, X_test, y_test, classes):
    print("=" * 60)
    print("ОЦЕНКА НА ТЕСТОВОЙ ВЫБОРКЕ")
    print("=" * 60)

    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        f1     = f1_score(y_test, y_pred, average="weighted")
        try:
            y_prob = model.predict_proba(X_test)
            auc    = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
        except Exception:
            auc = float("nan")

        results[name] = {"F1 (weighted)": f1, "ROC-AUC (macro)": auc}
        print(f"\n--- {name} ---")
        print(classification_report(y_test, y_pred, target_names=classes))

    df_res = pd.DataFrame(results).T.round(4)
    df_res.index.name = "Модель"
    print("\nСводная таблица:")
    print(df_res.to_string())
    df_res.to_csv(OUTPUT_DIR / "comparison_table.csv")
    return results


# ---------------------------------------------------------------------------
# 6. Оптимизация порогов (критерий Юдена)
# ---------------------------------------------------------------------------

def optimize_thresholds(model, X_test, y_test, classes):
    print("\n=== Оптимизация порогов (критерий Юдена) ===")
    y_prob     = model.predict_proba(X_test)
    thresholds = {}
    for i, cls in enumerate(classes):
        y_bin = (y_test == i).astype(int)
        fpr, tpr, thresh = roc_curve(y_bin, y_prob[:, i])
        best_idx       = np.argmax(tpr - fpr)
        thresholds[cls] = float(thresh[best_idx])
        print(f"  {cls:<15s}: оптимальный порог = {thresholds[cls]:.3f}")

    # Применяем пороги и пересчитываем F1
    y_pred_thr = np.zeros(len(y_test), dtype=int)
    scores     = y_prob.copy()
    for i, cls in enumerate(classes):
        scores[:, i] -= thresholds[cls]
    y_pred_thr = np.argmax(scores, axis=1)
    f1_thr = f1_score(y_test, y_pred_thr, average="weighted")
    print(f"\n  F1 с оптимизированными порогами: {f1_thr:.4f}")
    print(classification_report(y_test, y_pred_thr, target_names=classes))
    return thresholds


# ---------------------------------------------------------------------------
# 7. Визуализация
# ---------------------------------------------------------------------------

def plot_confusion_matrix(model, X_test, y_test, classes, name: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    cm   = confusion_matrix(y_test, model.predict(X_test), normalize="true")
    disp = ConfusionMatrixDisplay(cm, display_labels=classes)
    disp.plot(ax=ax, colorbar=True, cmap="Blues", values_format=".2f")
    ax.set_title(f"Матрица ошибок — {name}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"confusion_matrix_{name}.png", dpi=150)
    plt.close()
    print(f"Сохранено: confusion_matrix_{name}.png")


def plot_roc_curves(models: dict, X_test, y_test, classes):
    for model_name, model in models.items():
        try:
            y_prob = model.predict_proba(X_test)
        except Exception:
            continue
        fig, ax = plt.subplots(figsize=(8, 6))
        for i, cls in enumerate(classes):
            y_bin      = (y_test == i).astype(int)
            fpr, tpr, _ = roc_curve(y_bin, y_prob[:, i])
            auc        = roc_auc_score(y_bin, y_prob[:, i])
            ax.plot(fpr, tpr, label=f"{cls} (AUC={auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.set_title(f"ROC-кривые — {model_name}")
        ax.legend(loc="lower right", fontsize=8)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"roc_{model_name}.png", dpi=150)
        plt.close()
        print(f"Сохранено: roc_{model_name}.png")


def plot_shap(model, X_test, feature_names, model_name: str):
    print(f"\nSHAP-анализ для {model_name}...")
    X_sample = X_test[:300]
    try:
        if hasattr(model, "get_booster") or hasattr(model, "booster_"):
            explainer  = shap.TreeExplainer(model)
            shap_vals  = explainer(X_sample, check_additivity=False)
        else:
            background = shap.sample(X_test, 50, random_state=RANDOM_STATE)
            explainer  = shap.PermutationExplainer(model.predict_proba, background)
            shap_vals  = explainer(X_sample)

        plt.figure(figsize=(10, 7))
        shap.summary_plot(shap_vals, X_sample, feature_names=feature_names, show=False)
        plt.title(f"SHAP summary — {model_name}")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"shap_summary_{model_name}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Сохранено: shap_summary_{model_name}.png")
    except Exception as e:
        print(f"SHAP для {model_name}: {e}")


def plot_comparison_bar(results: dict):
    df   = pd.DataFrame(results).T
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric in zip(axes, ["F1 (weighted)", "ROC-AUC (macro)"]):
        vals = df[metric].dropna().sort_values(ascending=False)
        vals.plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")
        ax.set_title(metric); ax.set_ylim(0, 1); ax.set_ylabel("Score")
        ax.tick_params(axis="x", rotation=30)
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.3f}",
                        (p.get_x() + p.get_width() / 2, p.get_height() + 0.005),
                        ha="center", fontsize=9)
    plt.suptitle("Сравнение моделей (тестовая выборка)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_comparison.png", dpi=150)
    plt.close()
    print("Сохранено: model_comparison.png")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    print(f"Датасет: {DATASET_PATH}\n")

    X, y = load_data(DATASET_PATH)
    X    = remove_correlated(X, threshold=0.95)
    feature_names = list(X.columns)
    print(f"Признаков после отбора: {len(feature_names)}\n")

    (X_train, y_train,
     X_val,   y_val,
     X_test,  y_test,
     scaler, le, classes) = prepare_splits(X, y)

    models  = train_all(X_train, y_train, X_val, y_val, classes)
    results = evaluate(models, X_test, y_test, classes)

    best_name = max(results, key=lambda k: results[k]["F1 (weighted)"])
    print(f"\nЛучшая модель: {best_name}  "
          f"F1={results[best_name]['F1 (weighted)']:.4f}  "
          f"AUC={results[best_name]['ROC-AUC (macro)']:.4f}")

    optimize_thresholds(models[best_name], X_test, y_test, classes)

    print("\n--- Графики ---")
    for name, model in models.items():
        plot_confusion_matrix(model, X_test, y_test, classes, name)
    plot_roc_curves(models, X_test, y_test, classes)
    plot_comparison_bar(results)

    # SHAP для лучшей древовидной модели и MLP
    for shap_target in [best_name, "LightGBM", "XGBoost", "MLP"]:
        if shap_target in models:
            plot_shap(models[shap_target], X_test, feature_names, shap_target)
            if shap_target != "MLP":
                plot_shap(models["MLP"], X_test, feature_names, "MLP")
            break

    print(f"\nВсе результаты сохранены в {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
