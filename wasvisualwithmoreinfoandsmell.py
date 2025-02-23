import numpy as np
import random
import matplotlib.pyplot as plt

# Fitness function based on proximity to food sources
def fitness_function(wolf, food_sources):
    distances = [np.linalg.norm(wolf - food) for food in food_sources]
    return -min(distances)

# Smell accuracy function
def smell_accuracy(distance, max_range):
    # Decrease accuracy as distance increases
    if distance > max_range:
        return 0
    return 1 - (distance / max_range)

# Wolf Search Algorithm with smell ability, moving entities, and best fitness figure
def wolf_search_algorithm_with_smell_and_fitness_plot(
        dimensions, population_size, generations, visual_range, smell_range, step_size, threat_prob):
    boundary_min = -20
    boundary_max = 20

    wolves = np.random.uniform(boundary_min, boundary_max, (population_size, dimensions))
    num_dangerous_animals = 15
    num_food_sources = 5
    dangerous_animals = np.random.uniform(boundary_min, boundary_max, (num_dangerous_animals, dimensions))
    food_sources = np.random.uniform(boundary_min, boundary_max, (num_food_sources, dimensions))

    best_fitness_history = []
    average_fitness_history = []
    fitness_offset = None

    # Visualization setup
    fig1, ax1 = plt.subplots(figsize=(8, 8))
    ax1.set_xlim(boundary_min - 2, boundary_max + 2)
    ax1.set_ylim(boundary_min - 2, boundary_max + 2)
    wolves_scatter = ax1.scatter(wolves[:, 0], wolves[:, 1], c='blue', label='Wolves')
    dangerous_scatter = ax1.scatter(dangerous_animals[:, 0], dangerous_animals[:, 1], c='red', label='Dangerous Animals')
    food_scatter = ax1.scatter(food_sources[:, 0], food_sources[:, 1], c='green', label='Food Sources')
    best_fit_scatter = ax1.scatter([], [], c='gold', s=100, label='Best Fit', edgecolors='black', zorder=5)
    ax1.legend()

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.set_title("Best and Average Fitness Over Generations")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Fitness (Normalized)")
    ax2.set_xlim(0, generations)
    fitness_line, = ax2.plot([], [], 'r-', label="Best Fitness")
    avg_fitness_line, = ax2.plot([], [], 'b-', label="Average Fitness")
    ax2.legend()

    plt.ion()  # Turn on interactive mode

    for gen in range(generations):
        new_positions = []
        best_fitness = -np.inf
        total_fitness = 0  # Accumulate fitness for average calculation
        best_fit_wolf = None

        # Move food sources and dangerous animals
        food_sources += np.random.uniform(-1, 1, food_sources.shape)
        dangerous_animals += np.random.uniform(-1, 1, dangerous_animals.shape)

        # Enforce boundaries for moving entities
        food_sources = np.clip(food_sources, boundary_min, boundary_max)
        dangerous_animals = np.clip(dangerous_animals, boundary_min, boundary_max)

        for i, wolf in enumerate(wolves):
            escape = False
            smelled_food = False
            nearest_food = None

            # Smell dangerous animals
            for danger in dangerous_animals:
                distance = np.linalg.norm(wolf - danger)
                if distance < smell_range and smell_accuracy(distance, smell_range) > random.random():
                    escape = True
                    wolf += np.random.uniform(-visual_range, visual_range, dimensions)
                    break

            if not escape:
                # Smell food sources
                for food in food_sources:
                    distance = np.linalg.norm(wolf - food)
                    if distance < smell_range and smell_accuracy(distance, smell_range) > random.random():
                        smelled_food = True
                        nearest_food = food
                        break

                if smelled_food:
                    wolf = wolf + step_size * (nearest_food - wolf) / np.linalg.norm(nearest_food - wolf)

                # If not moved by smell, use visual range
                elif not smelled_food:
                    distances_to_food = [np.linalg.norm(wolf - food) for food in food_sources]
                    nearest_food = food_sources[np.argmin(distances_to_food)]
                    if np.linalg.norm(wolf - nearest_food) < visual_range:
                        wolf = wolf + step_size * (nearest_food - wolf) / np.linalg.norm(nearest_food - wolf)

                # Random Brownian motion
                if random.random() < threat_prob:
                    wolf += np.random.uniform(-step_size, step_size, dimensions)

            # Enforce boundaries for wolves
            wolf = np.clip(wolf, boundary_min, boundary_max)
            new_positions.append(wolf)

            # Update best fitness
            current_fitness = fitness_function(wolf, food_sources)
            total_fitness += current_fitness  # Update total fitness
            if current_fitness > best_fitness:
                best_fitness = current_fitness
                best_fit_wolf = wolf

        wolves = np.array(new_positions)

        # Fitness tracking
        if fitness_offset is None or best_fitness < fitness_offset:
            fitness_offset = best_fitness
        normalized_fitness = best_fitness - fitness_offset
        avg_normalized_fitness = (total_fitness / population_size) - fitness_offset
        best_fitness_history.append(normalized_fitness)
        average_fitness_history.append(avg_normalized_fitness)

        # Update visualizations
        wolves_scatter.set_offsets(wolves)
        food_scatter.set_offsets(food_sources)
        dangerous_scatter.set_offsets(dangerous_animals)
        best_fit_scatter.set_offsets([best_fit_wolf])
        ax1.set_title(f"Wolf Search Algorithm - Generation {gen + 1}")

        fitness_line.set_data(range(len(best_fitness_history)), best_fitness_history)
        avg_fitness_line.set_data(range(len(average_fitness_history)), average_fitness_history)
        min_y_value = min(min(best_fitness_history), min(average_fitness_history)) - 1
        max_y_value = max(max(best_fitness_history), max(average_fitness_history)) + 1
        ax2.set_ylim(min_y_value, max_y_value)
        ax2.set_xlim(0, len(best_fitness_history))

        plt.pause(0.1)

        print(f"Generation {gen + 1}: Best fitness (normalized) = {normalized_fitness}, "
              f"Average fitness (normalized) = {avg_normalized_fitness}")

    plt.ioff()
    plt.show()

    return best_fit_wolf, best_fitness

# Parameters
dimensions = 2
population_size = 5
generations = 500
visual_range = 2.0
smell_range = 5.0
step_size = 1.0
threat_prob = 0.2

# Run the enhanced Wolf Search Algorithm with fitness plot
best_solution, best_fitness = wolf_search_algorithm_with_smell_and_fitness_plot(
    dimensions, population_size, generations, visual_range, smell_range, step_size, threat_prob
)

print("Best Solution (closest to food source):", best_solution)
print("Best Fitness:", best_fitness)
