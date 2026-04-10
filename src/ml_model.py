import random
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from src.genetic_algorithm import create_individual, repair_solution, fitness


def generate_training_data(
    weights,
    values,
    capacity,
    n_samples=1000,
    seed=42
):
    """
    Génère des exemples (solutions binaires, fitness) pour entraîner le modèle ML.
    """
    random.seed(seed)
    n_items = len(weights)

    X = []
    y = []

    for _ in range(n_samples):
        sol = create_individual(n_items)
        sol = repair_solution(sol, weights, values, capacity)
        score = fitness(sol, weights, values, capacity)

        X.append(sol)
        y.append(score)

    return X, y


def train_model(X, y, n_estimators=100, random_state=42):
    """
    Entraîne un Random Forest Regressor.
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state
    )
    model.fit(X, y)
    return model


def evaluate_model(model, X, y):
    """
    Évalue le modèle sur les données données.
    """
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    r2 = r2_score(y, predictions)

    return {
        "mse": mse,
        "r2": r2
    }


def save_training_data(X, y, filepath="data/train_data.csv"):
    """
    Sauvegarde les données d'entraînement.
    """
    rows = []
    for sol, score in zip(X, y):
        row = {f"x_{i}": bit for i, bit in enumerate(sol)}
        row["fitness"] = score
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)
    return df