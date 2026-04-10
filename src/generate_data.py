import random
import pandas as pd


def generate_instance(n=20, weight_min=1, weight_max=15, value_min=10, value_max=100):
    """
    Génère une instance aléatoire du problème du sac à dos.
    Retourne :
        weights : liste des poids
        values : liste des valeurs
        capacity : capacité du sac
    """
    weights = [random.randint(weight_min, weight_max) for _ in range(n)]
    values = [random.randint(value_min, value_max) for _ in range(n)]

    # capacité choisie pour ne pas pouvoir prendre tous les objets
    capacity = int(sum(weights) * 0.4)

    return weights, values, capacity


def generate_multiple_instances(
    num_instances=20,
    n_items_list=None,
    seed=42
):
    """
    Génère plusieurs instances et les retourne sous forme de liste de dictionnaires.
    """
    if n_items_list is None:
        n_items_list = [20, 50, 100]

    random.seed(seed)

    instances = []
    instance_id = 1

    for n_items in n_items_list:
        for _ in range(num_instances):
            weights, values, capacity = generate_instance(n=n_items)
            instances.append(
                {
                    "instance_id": instance_id,
                    "n_items": n_items,
                    "weights": weights,
                    "values": values,
                    "capacity": capacity,
                }
            )
            instance_id += 1

    return instances


def save_instances_to_csv(instances, filepath="data/instances.csv"):
    """
    Sauvegarde les instances dans un CSV.
    """
    df = pd.DataFrame(instances)
    df.to_csv(filepath, index=False)
    return df