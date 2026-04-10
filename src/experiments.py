import time
import pandas as pd

from src.ml_model import generate_training_data, train_model, evaluate_model
from src.genetic_algorithm import (
    run_genetic_algorithm,
    run_genetic_algorithm_with_ml_guidance
)


def run_single_experiment(
    instance_id,
    weights,
    values,
    capacity,
    pop_size=100,
    generations=100,
    n_training_samples=1000,
    seed=42
):
    """
    Lance une expérience complète :
    - entraîne le modèle ML
    - exécute GA seul
    - exécute GA + ML
    - retourne les résultats
    """
    X_train, y_train = generate_training_data(
        weights=weights,
        values=values,
        capacity=capacity,
        n_samples=n_training_samples,
        seed=seed
    )

    model = train_model(X_train, y_train)
    ml_metrics = evaluate_model(model, X_train, y_train)

    start_ga = time.perf_counter()
    _, ga_score, ga_history = run_genetic_algorithm(
        weights=weights,
        values=values,
        capacity=capacity,
        pop_size=pop_size,
        generations=generations,
        seed=seed
    )
    end_ga = time.perf_counter()

    start_ga_ml = time.perf_counter()
    _, ga_ml_score, ga_ml_history = run_genetic_algorithm_with_ml_guidance(
        weights=weights,
        values=values,
        capacity=capacity,
        model=model,
        pop_size=pop_size,
        generations=generations,
        seed=seed
    )
    end_ga_ml = time.perf_counter()

    result = {
        "instance_id": instance_id,
        "n_items": len(weights),
        "capacity": capacity,
        "ga_score": ga_score,
        "ga_time_sec": end_ga - start_ga,
        "ga_ml_score": ga_ml_score,
        "ga_ml_time_sec": end_ga_ml - start_ga_ml,
        "ml_mse": ml_metrics["mse"],
        "ml_r2": ml_metrics["r2"],
        "ga_improvement": ga_ml_score - ga_score
    }

    histories = {
        "ga_history": ga_history,
        "ga_ml_history": ga_ml_history
    }

    return result, histories


def run_all_experiments(
    instances,
    pop_size=100,
    generations=100,
    n_training_samples=1000,
    seed=42
):
    """
    Exécute toutes les expériences sur toutes les instances.
    """
    results = []
    all_histories = {}

    for instance in instances:
        result, histories = run_single_experiment(
            instance_id=instance["instance_id"],
            weights=instance["weights"],
            values=instance["values"],
            capacity=instance["capacity"],
            pop_size=pop_size,
            generations=generations,
            n_training_samples=n_training_samples,
            seed=seed
        )

        results.append(result)
        all_histories[instance["instance_id"]] = histories

    return results, all_histories


def save_results(results, filepath="results/comparison_results.csv"):
    """
    Sauvegarde les résultats sous forme CSV.
    """
    df = pd.DataFrame(results)
    df.to_csv(filepath, index=False)
    return df


def save_results_excel(results, filepath="results/comparison_results.xlsx"):
    """
    Sauvegarde les résultats sous forme Excel.
    """
    df = pd.DataFrame(results)
    df.to_excel(filepath, index=False)
    return df


def summarize_results(results):
    """
    Retourne un résumé global.
    """
    df = pd.DataFrame(results)

    summary = {
        "nb_instances": len(df),
        "ga_score_mean": df["ga_score"].mean(),
        "ga_ml_score_mean": df["ga_ml_score"].mean(),
        "ga_time_mean": df["ga_time_sec"].mean(),
        "ga_ml_time_mean": df["ga_ml_time_sec"].mean(),
        "mean_improvement": df["ga_improvement"].mean(),
        "nb_better_with_ml": int((df["ga_ml_score"] > df["ga_score"]).sum()),
        "nb_equal": int((df["ga_ml_score"] == df["ga_score"]).sum()),
        "nb_worse_with_ml": int((df["ga_ml_score"] < df["ga_score"]).sum()),
    }

    return summary