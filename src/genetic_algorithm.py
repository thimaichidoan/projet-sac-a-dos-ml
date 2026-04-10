import random
from typing import List, Tuple


def fitness(solution: List[int], weights: List[int], values: List[int], capacity: int) -> int:
    """
    Calcule la fitness d'une solution.
    Si la contrainte de capacité est violée, retourne 0.
    """
    total_weight = sum(w * x for w, x in zip(weights, solution))
    total_value = sum(v * x for v, x in zip(values, solution))

    if total_weight > capacity:
        return 0

    return total_value


def repair_solution(solution: List[int], weights: List[int], values: List[int], capacity: int) -> List[int]:
    """
    Répare une solution infaisable en retirant des objets
    jusqu'à respecter la capacité.
    On retire en priorité les objets avec le plus faible ratio valeur/poids.
    """
    repaired = solution[:]

    def current_weight(sol):
        return sum(w * x for w, x in zip(weights, sol))

    selected_items = [i for i, bit in enumerate(repaired) if bit == 1]
    selected_items.sort(key=lambda i: values[i] / weights[i])

    while current_weight(repaired) > capacity and selected_items:
        i = selected_items.pop(0)
        repaired[i] = 0

    return repaired


def create_individual(n_items: int) -> List[int]:
    """
    Crée un individu binaire aléatoire.
    """
    return [random.randint(0, 1) for _ in range(n_items)]


def create_population(pop_size: int, n_items: int) -> List[List[int]]:
    """
    Crée une population initiale.
    """
    return [create_individual(n_items) for _ in range(pop_size)]


def tournament_selection(
    population: List[List[int]],
    weights: List[int],
    values: List[int],
    capacity: int,
    k: int = 3
) -> List[int]:
    """
    Sélection par tournoi.
    """
    candidates = random.sample(population, k)
    candidates.sort(key=lambda ind: fitness(ind, weights, values, capacity), reverse=True)
    return candidates[0][:]


def crossover(parent1: List[int], parent2: List[int], crossover_rate: float = 0.8) -> Tuple[List[int], List[int]]:
    """
    Croisement à un point.
    """
    if random.random() > crossover_rate:
        return parent1[:], parent2[:]

    point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]

    return child1, child2


def mutate(individual: List[int], mutation_rate: float = 0.02) -> List[int]:
    """
    Mutation bit-flip.
    """
    mutated = individual[:]
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            mutated[i] = 1 - mutated[i]
    return mutated


def run_genetic_algorithm(
    weights: List[int],
    values: List[int],
    capacity: int,
    pop_size: int = 100,
    generations: int = 100,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.02,
    elitism: bool = True,
    seed: int = None
):
    """
    Exécute un algorithme génétique classique pour le sac à dos.
    Retourne :
        best_solution, best_score, history
    """
    if seed is not None:
        random.seed(seed)

    n_items = len(weights)
    population = create_population(pop_size, n_items)

    # Réparer la population initiale
    population = [repair_solution(ind, weights, values, capacity) for ind in population]

    best_solution = None
    best_score = -1
    history = []

    for _ in range(generations):
        population_scores = [fitness(ind, weights, values, capacity) for ind in population]

        gen_best_score = max(population_scores)
        gen_best_index = population_scores.index(gen_best_score)
        gen_best_solution = population[gen_best_index][:]

        if gen_best_score > best_score:
            best_score = gen_best_score
            best_solution = gen_best_solution[:]

        history.append(best_score)

        new_population = []

        # Élitisime : garder le meilleur
        if elitism:
            new_population.append(best_solution[:])

        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, weights, values, capacity)
            parent2 = tournament_selection(population, weights, values, capacity)

            child1, child2 = crossover(parent1, parent2, crossover_rate)
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)

            child1 = repair_solution(child1, weights, values, capacity)
            child2 = repair_solution(child2, weights, values, capacity)

            new_population.append(child1)
            if len(new_population) < pop_size:
                new_population.append(child2)

        population = new_population

    return best_solution, best_score, history


def run_genetic_algorithm_with_ml_guidance(
    weights: List[int],
    values: List[int],
    capacity: int,
    model,
    pop_size: int = 100,
    generations: int = 100,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.02,
    elitism: bool = True,
    seed: int = None
):
    """
    Variante GA + ML :
    le modèle ML prédit la qualité des individus et aide à guider la sélection.
    Ici, on combine score réel + score prédit pour favoriser les individus prometteurs.
    """
    if seed is not None:
        random.seed(seed)

    n_items = len(weights)
    population = create_population(pop_size, n_items)
    population = [repair_solution(ind, weights, values, capacity) for ind in population]

    best_solution = None
    best_score = -1
    history = []

    for _ in range(generations):
        real_scores = [fitness(ind, weights, values, capacity) for ind in population]
        predicted_scores = model.predict(population)

        combined_scores = []
        for real_s, pred_s in zip(real_scores, predicted_scores):
            combined_scores.append(0.7 * real_s + 0.3 * pred_s)

        best_real_score = max(real_scores)
        best_real_index = real_scores.index(best_real_score)
        best_real_solution = population[best_real_index][:]

        if best_real_score > best_score:
            best_score = best_real_score
            best_solution = best_real_solution[:]

        history.append(best_score)

        ranked_population = [
            ind for _, ind in sorted(
                zip(combined_scores, population),
                key=lambda x: x[0],
                reverse=True
            )
        ]

        new_population = []

        if elitism:
            new_population.append(best_solution[:])

        top_pool_size = max(10, pop_size // 2)
        selection_pool = ranked_population[:top_pool_size]

        while len(new_population) < pop_size:
            parent1 = random.choice(selection_pool)
            parent2 = random.choice(selection_pool)

            child1, child2 = crossover(parent1, parent2, crossover_rate)
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)

            child1 = repair_solution(child1, weights, values, capacity)
            child2 = repair_solution(child2, weights, values, capacity)

            new_population.append(child1)
            if len(new_population) < pop_size:
                new_population.append(child2)

        population = new_population

    return best_solution, best_score, history