import os

from src.generate_data import generate_multiple_instances, save_instances_to_csv
from src.experiments import (
    run_all_experiments,
    save_results,
    save_results_excel,
    summarize_results
)
from src.plot_results import (
    plot_scores_comparison,
    plot_times_comparison,
    plot_improvement_histogram,
    plot_histories
)


def ensure_directories():
    os.makedirs("data", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/histories", exist_ok=True)


def main():
    ensure_directories()

    print("1) Génération des instances...")
    instances = generate_multiple_instances(
        num_instances=5,       # tu peux augmenter à 10 ou 20
        n_items_list=[20, 50], # tu peux ajouter 100
        seed=42
    )
    save_instances_to_csv(instances, filepath="data/instances.csv")
    print(f"{len(instances)} instances générées et sauvegardées dans data/instances.csv")

    print("\n2) Lancement des expériences...")
    results, histories = run_all_experiments(
        instances=instances,
        pop_size=80,
        generations=80,
        n_training_samples=800,
        seed=42
    )

    print("\n3) Sauvegarde des résultats...")
    save_results(results, filepath="results/comparison_results.csv")
    save_results_excel(results, filepath="results/comparison_results.xlsx")

    print("\n4) Génération des graphiques...")
    plot_scores_comparison()
    plot_times_comparison()
    plot_improvement_histogram()
    plot_histories(histories)

    print("\n5) Résumé global...")
    summary = summarize_results(results)

    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nProjet exécuté avec succès.")
    print("Fichiers créés :")
    print("- data/instances.csv")
    print("- results/comparison_results.csv")
    print("- results/comparison_results.xlsx")
    print("- results/scores_comparison.png")
    print("- results/times_comparison.png")
    print("- results/improvement_histogram.png")
    print("- results/histories/")


if __name__ == "__main__":
    main()