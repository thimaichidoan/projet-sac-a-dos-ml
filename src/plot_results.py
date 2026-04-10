import os
import matplotlib.pyplot as plt
import pandas as pd


def plot_scores_comparison(results_csv="results/comparison_results.csv", output_path="results/scores_comparison.png"):
    """
    Graphique comparaison scores GA vs GA+ML.
    """
    df = pd.read_csv(results_csv)

    plt.figure(figsize=(10, 6))
    plt.plot(df["instance_id"], df["ga_score"], marker="o", label="GA seul")
    plt.plot(df["instance_id"], df["ga_ml_score"], marker="s", label="GA + ML")
    plt.xlabel("Instance")
    plt.ylabel("Score")
    plt.title("Comparaison des scores : GA seul vs GA + ML")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_times_comparison(results_csv="results/comparison_results.csv", output_path="results/times_comparison.png"):
    """
    Graphique comparaison temps GA vs GA+ML.
    """
    df = pd.read_csv(results_csv)

    plt.figure(figsize=(10, 6))
    plt.plot(df["instance_id"], df["ga_time_sec"], marker="o", label="GA seul")
    plt.plot(df["instance_id"], df["ga_ml_time_sec"], marker="s", label="GA + ML")
    plt.xlabel("Instance")
    plt.ylabel("Temps (s)")
    plt.title("Comparaison des temps : GA seul vs GA + ML")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_improvement_histogram(results_csv="results/comparison_results.csv", output_path="results/improvement_histogram.png"):
    """
    Histogramme de l'amélioration apportée par le ML.
    """
    df = pd.read_csv(results_csv)

    plt.figure(figsize=(10, 6))
    plt.hist(df["ga_improvement"], bins=15)
    plt.xlabel("Amélioration (GA+ML - GA)")
    plt.ylabel("Fréquence")
    plt.title("Histogramme des améliorations apportées par le ML")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_histories(histories_dict, output_dir="results/histories"):
    """
    Sauvegarde un graphique d'évolution par instance.
    """
    os.makedirs(output_dir, exist_ok=True)

    for instance_id, histories in histories_dict.items():
        plt.figure(figsize=(10, 6))
        plt.plot(histories["ga_history"], label="GA seul")
        plt.plot(histories["ga_ml_history"], label="GA + ML")
        plt.xlabel("Génération")
        plt.ylabel("Meilleur score cumulé")
        plt.title(f"Évolution du score - Instance {instance_id}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/history_instance_{instance_id}.png")
        plt.close()