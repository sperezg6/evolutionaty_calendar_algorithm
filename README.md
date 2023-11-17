# Premier League Schedule/ Evolutionary Algorithm

This Python tool is designed to create and evaluate schedules for Premier League football teams, optimizing for fairness and travel distance. It uses evolutionary algorithms to generate schedules where teams play each other once at home and once away, with attention to reducing consecutive home or away games and balancing the number of games played on different days of the week.

## Features:

- **Team and City Data**: Lists Premier League teams and associates them with their cities.
- **Distance Matrix**: Contains the distances between the cities of the teams to calculate travel penalties.
- **Schedule Generation**: Produces an initial population of schedules with a specified number of rounds.
- **Schedule Evaluation**: Scores schedules based on home/away balance, consecutive games, and city game distribution.
- **Evolutionary Operations**: Includes functions for crossover, mutation, parent selection, and population evolution to improve schedules over generations.

## How to Use:

1. Set the parameters for the evolutionary algorithm, such as population size, number of generations, and mutation rate.
2. Generate an initial population of schedules with the `generate_initial_population()` function.
3. Evaluate the initial population using the `evaluate()` function.
4. Evolve the population using the `evolve_v2()` function.
5. Print the best schedule and team statistics using `pretty_print_schedule_step_by_step()` and `print_team_stats()` functions.

## Custom Functions:

- `generate_initial_population(pop_size, teams)`: Generates an initial set of schedules.
- `evaluate(schedule, cities, distances)`: Assigns a score to a schedule based on various criteria.
- `evolve_v2(evaluated_population, num_parents, num_generations, evaluation_function)`: Evolves the population over a number of generations.
- `pretty_print_schedule_step_by_step(schedule)`: Prints the schedule round by round.
- `print_team_stats(schedule, cities, distances)`: Prints statistics for each team in the schedule.

## Example Workflow:

```python
# Adjustable algorithm parameters
population_size = 100
num_generations = 10
num_parents = 20
mutation_rate = 0.02

# Create an initial population for a 38-round season and evaluate it
population_38j = generate_initial_population(population_size, teams)
print("Initial Premier League 2023 Schedule:")
print_team_stats(population_38j[0], cities, distances)

# Evolve the population and display the best schedule
evolved_population = evolve_v2(evaluated_population_sorted_38j_days_corrected, num_parents, num_generations, evaluate)

# Print the best schedule and team statistics
print("\nBest Schedule:")
pretty_print_schedule_step_by_step(evolved_population[0][0])
print("\nTeam Statistics:")
print_team_stats(evolved_population[0][0], cities, distances)
```

Note: This tool assumes a fair level of understanding of Python programming and evolutionary algorithms. Adjustments to the code may be necessary to tailor the schedule to specific constraints or preferences.
