import time
import random

# Lista de equipos de la Premier League
teams = ["Arsenal", "Brentford", "Chelsea", "Crystal Palace", "Fulham", "Tottenham Hotspur", "West Ham United",
         "Aston Villa", "Bournemouth", "Brighton", "Burnley", "Liverpool", "Everton", "Luton Town", "Manchester City",
         "Manchester United", "Newcastle United", "Nottingham Forrest", "Sheffield United", "Wolverhampton Wanderers"]


# Diccionario que asocia equipos con sus ciudades
cities = {
    "Londres": ["Arsenal", "Brentford", "Chelsea", "Crystal Palace", "Fulham", "Tottenham Hotspur", "West Ham United"],
    "Liverpool": ["Liverpool", "Everton"],
    "Manchester": ["Manchester City", "Manchester United"],
    "Wolverhampton": ["Wolverhampton Wanderers"],
    "Bournemouth": ["Bournemouth"],
    "Brentford": ["Brentford"],
    "Burnley": ["Burnley"],
    "Luton" : ["Luton Town"],
    "Newcastle": ["Newcastle United"],
    "Nottinghamshire": ["Nottingham Forrest"],
    "Sheffield": ["Sheffield United"],
    "Brighton y Hove": ["Brighton"],
    "Birmingham": ["Aston Villa"],

}

# Diccionario que contiene las distancias entre las ciudades de los equipos
distancias = {
    "Londres": {
        "Londres": 0,
        "Birmingham": 202,
        "Bournemouth": 158,
        "Brighton y Hove": 76,
        "Burnley": 357,
        "Liverpool": 346,
        "Luton": 54,
        "Manchester": 330,
        "Newcastle": 398,
        "Nottinghamshire": 206,
        "Sheffield": 266,
        "Wolverhampton": 225,
    },

    "Birmingham": {
        "Londres": 202,
        "Birmingham": 0,
        "Bournemouth": 234,
        "Brighton y Hove": 228,
        "Burnley": 145,
        "Liverpool": 126,
        "Luton": 150,
        "Manchester": 113,
        "Newcastle": 300,
        "Nottinghamshire": 80,
        "Sheffield": 122,
        "Wolverhampton": 30,
    },
    "Bournemouth": {
        "Londres": 158,
        "Birmingham": 234,
        "Bournemouth": 0,
        "Brighton y Hove": 170,
        "Burnley": 341,
        "Liverpool": 362,
        "Luton": 202,
        "Manchester": 352,
        "Newcastle": 486,
        "Nottinghamshire": 312,
        "Sheffield": 319,
        "Wolverhampton": 280,
    },
    "Brighton y Hove": {
        "Londres": 76,
        "Birmingham": 228,
        "Bournemouth": 170,
        "Brighton y Hove": 0,
        "Burnley": 340,
        "Liverpool": 364,
        "Luton": 131,
        "Manchester": 344,
        "Newcastle": 450,
        "Nottinghamshire": 238,
        "Sheffield": 288,
        "Wolverhampton": 248,
    },
    "Burnley": {
        "Londres": 357,
        "Birmingham": 145,
        "Bournemouth": 341,
        "Brighton y Hove": 340,
        "Burnley": 0,
        "Liverpool": 79,
        "Luton": 304,
        "Manchester": 54,
        "Newcastle": 154,
        "Nottinghamshire": 137,
        "Sheffield": 69,
        "Wolverhampton": 134,
    },
    "Liverpool": {
        "Londres": 346,
        "Birmingham": 126,
        "Bournemouth": 362,
        "Brighton y Hove": 364,
        "Burnley": 79,
        "Liverpool": 0,
        "Luton": 228,
        "Manchester": 55,
        "Newcastle": 212,
        "Nottinghamshire": 177,
        "Sheffield": 102,
        "Wolverhampton": 146,
    },
    "Luton": {
        "Londres": 54,
        "Birmingham": 150,
        "Bournemouth": 202,
        "Brighton y Hove": 131,
        "Burnley": 304,
        "Liverpool": 228,
        "Luton": 0,
        "Manchester": 176,
        "Newcastle": 284,
        "Nottinghamshire": 119,
        "Sheffield": 170,
        "Wolverhampton": 162,
    },
    "Manchester": {
        "Londres": 330,
        "Birmingham": 113,
        "Bournemouth": 352,
        "Brighton y Hove": 344,
        "Burnley": 54,
        "Liverpool": 55,
        "Luton": 176,
        "Manchester": 0,
        "Newcastle": 169,
        "Nottinghamshire": 113,
        "Sheffield": 52,
        "Wolverhampton": 98,
    },
    "Newcastle": {
        "Londres": 398,
        "Birmingham": 300,
        "Bournemouth": 486,
        "Brighton y Hove": 450,
        "Burnley": 154,
        "Liverpool": 212,
        "Luton": 284,
        "Manchester": 169,
        "Newcastle": 0,
        "Nottinghamshire": 210,
        "Sheffield": 148,
        "Wolverhampton": 219,
    },
    "Nottinghamshire": {
        "Londres": 206,
        "Birmingham": 80,
        "Bournemouth": 312,
        "Brighton y Hove": 238,
        "Burnley": 137,
        "Liverpool": 177,
        "Luton": 119,
        "Manchester": 113,
        "Newcastle": 210,
        "Nottinghamshire": 0,
        "Sheffield": 50,
        "Wolverhampton": 110,
    },


    "Sheffield": {
        "Londres": 266,
        "Birmingham": 122,
        "Bournemouth": 319,
        "Brighton y Hove": 288,
        "Burnley": 69,
        "Liverpool": 102,
        "Luton": 170,
        "Manchester": 52,
        "Newcastle": 148,
        "Nottinghamshire": 50,
        "Sheffield": 0,
        "Wolverhampton": 61,
    },
    "Wolverhampton": {
        "Londres": 225,
        "Birmingham": 30,
        "Bournemouth": 280,
        "Brighton y Hove": 248,
        "Burnley": 134,
        "Liverpool": 146,
        "Luton": 162,
        "Manchester": 98,
        "Newcastle": 219,
        "Nottinghamshire": 110,
        "Sheffield": 61,
        "Wolverhampton": 0,
    },
}


def generate_initial_population(pop_size, teams):
    # Genera una población inicial de calendarios
    population = []
    num_teams = len(teams)
    num_rounds = num_teams - 1  # Cada equipo juega contra cada otro equipo una vez

    # Rotación de equipos para la primera mitad de la temporada
    for _ in range(pop_size):
        schedule = []
        for round_num in range(num_rounds):
            round_matches = []

            # El equipo en la posición 0 no se mueve, es el "pivote"
            for j in range(num_teams // 2):
                home = teams[j]
                away = teams[-j - 1]
                if home != away:  # Asegurarse de que no sea el mismo equipo
                    round_matches.append((home, away))

            # Asignar partidos a los días de la semana
            assigned_matches = assign_matches_to_days(round_matches)
            schedule.append(assigned_matches)

            # Rotar la lista de equipos, excepto el primero (pivote)
            teams.insert(1, teams.pop())  # Mueve el último al inicio (justo después del pivote)

        # Inversión de partidos para la segunda mitad de la temporada
        for round_matches_dict in schedule[:num_rounds]:
            new_round = {day: [(away, home) for home, away in matches] for day, matches in round_matches_dict.items()}
            schedule.append(new_round)

        population.append(schedule)

    return population



def procreate_sammy(cal1, cal2, mutation_rate=0.02):
    # Función de cruce y mutación para crear nuevos calendarios
    num_genes = len(cal1)
    son_cal = [{} for _ in range(num_genes)]

    # Iterar sobre cada jornada
    for i in range(num_genes):
        # Combinar partidos para cada día, excepto viernes y lunes
        for day in ["Sábado", "Domingo"]:
            matches_day1 = set(cal1[i].get(day, []))
            matches_day2 = set(cal2[i].get(day, []))
            combined_matches = list(matches_day1.union(matches_day2))

            # Agregar mutación aleatoria
            if random.random() < mutation_rate:
                son_cal[i][day] = combined_matches
            else:
                # Conservar la intersección de los partidos si es posible
                intersection = matches_day1.intersection(matches_day2)
                if intersection:
                    son_cal[i][day] = list(intersection)
                else:
                    son_cal[i][day] = combined_matches

        # Asegurarse de que solo haya un partido los viernes y los lunes
        for day in ["Viernes", "Lunes"]:
            combined_matches = set()
            if day in cal1[i]:
                combined_matches.update(cal1[i][day])
            if day in cal2[i]:
                combined_matches.update(cal2[i][day])

            if combined_matches:
                # Seleccionar solo un partido al azar para Viernes y Lunes
                son_cal[i][day] = [random.choice(list(combined_matches))]

    return son_cal


# Función de evaluación
def evaluate(schedule,cities,distancias):
    
    """Evalúa un calendario que asigna partidos a días."""
    # Inicializar el score
    score = 0
    
    # 1. Todos los equipos deben jugar exactamente una vez en casa y una vez fuera contra cada otro equipo
    matches = {}
    for team in teams:
        matches[team] = {"home": [], "away": []}
    
    for round_matches_dict in schedule:
        for day, matches_for_day in round_matches_dict.items():
            for match in matches_for_day:
                home, away = match
                matches[home]["home"].append(away)
                matches[away]["away"].append(home)
    
    for team, opponents in matches.items():
        score += abs(len(opponents["home"]) - len(opponents["away"]))  # Desequilibrio en casa/fuera
        score += abs(len(opponents["home"]) - (len(teams) - 1) // 2)  # No se enfrenta a todos los equipos en casa
        score += abs(len(opponents["away"]) - (len(teams) - 1) // 2)  # No se enfrenta a todos los equipos fuera
    
    # 2. Tratar de alternar partidos en casa y fuera
    for i in range(1, len(schedule)):
        for day in schedule[i]:
            for j in range(len(schedule[i][day])):
                if j < len(schedule[i-1][day]) and (schedule[i][day][j][0] == schedule[i-1][day][j][0] or schedule[i][day][j][1] == schedule[i-1][day][j][1]):
                    score += 1
    
    # 3. En las ciudades con múltiples equipos, asegurar que en cada jornada haya un partido en esa ciudad
    for city, city_teams in cities.items():
        if len(city_teams) > 1:
            for round_matches_dict in schedule:
                home_teams_in_city = []
                for day, matches_for_day in round_matches_dict.items():
                    home_teams_in_city.extend([match[0] for match in matches_for_day if match[0] in city_teams])
                if len(home_teams_in_city) < 1:  # Ningún equipo de la ciudad juega en casa
                    score += 1
    
    # Nuevo criterio: Equilibrio de partidos en casa/fuera a lo largo de la temporada
    for team in teams:
        last_home_away = None
        consecutive_home_away = 0
        for round_matches_dict in schedule:
            for day, matches_for_day in round_matches_dict.items():
                for match in matches_for_day:
                    if team in match:
                        home_away = "home" if match[0] == team else "away"
                        if last_home_away == home_away:
                            consecutive_home_away += 1
                            if consecutive_home_away > 2:  # Más de dos partidos consecutivos en casa/fuera
                                score += 1
                        else:
                            consecutive_home_away = 0
                        last_home_away = home_away

    # Nuevo criterio: Equilibrio de partidos en viernes/lunes a lo largo de la temporada
    for team in teams:
        last_day = None
        consecutive_days = 0
        for round_matches_dict in schedule:
            for day, matches_for_day in round_matches_dict.items():
                if day in ["Viernes", "Lunes"]:
                    if last_day == day:
                        consecutive_days += 1
                        if consecutive_days > 2:  # Más de dos partidos consecutivos en viernes/lunes
                            score += 1
                    else:
                        consecutive_days = 0
                    last_day = day

    
    travel_penalty = calculate_travel_distance(schedule, distancias, cities)
    
    # Ajustar la puntuación para incluir la penalización por viaje
    score += travel_penalty



    
    return score



def select_parents(evaluated_population, num_parents):
    """Selecciona los padres mediante selección por torneo."""
    parents = []
    tournament_size = 5
    for _ in range(num_parents):
        tournament = random.sample(evaluated_population, tournament_size)
        winner = min(tournament, key=lambda x: x[1])
        parents.append(winner[0])
    return parents




def crossover(parent1, parent2):
    """Realiza el cruce entre dos padres para producir dos hijos."""
    split = len(parent1) // 2
    child1 = parent1[:split] + parent2[split:]
    child2 = parent2[:split] + parent1[split:]
    return child1, child2

def mutate(child):
    """Realiza una mutación aleatoria en un hijo."""
    if random.random() < 0.1:  # Probabilidad de mutación
        rnd1 = random.randint(0, len(child)-1)
        rnd2 = random.randint(0, len(child)-1)
        child[rnd1], child[rnd2] = child[rnd2], child[rnd1]
    return child


def evolve_v2(evaluated_population, num_parents, num_generations, evaluation_function):
    """Evoluciona la población utilizando procreate_sammy."""
    for _ in range(num_generations):
        parents = select_parents(evaluated_population, num_parents)
        new_population = []
        for i in range(0, len(parents), 2):
            parent1 = parents[i]
            parent2 = parents[i+1] if i+1 < len(parents) else parents[0]
            child = procreate_sammy(parent1, parent2)
            new_population.append(child)

        evaluated_population = [(schedule, evaluation_function(schedule,cities,distancias)) for schedule in new_population]
        evaluated_population_sorted = sorted(evaluated_population, key=lambda x: x[1])

    return evaluated_population_sorted


def assign_matches_to_days(round_matches):
     # Asigna partidos a días específicos en una jornada
    days = ["Viernes", "Sábado", "Domingo", "Lunes"]
    matches_per_day = {
        "Viernes": 1,
        "Sábado": (len(round_matches) - 2) // 2,
        "Domingo": (len(round_matches) - 2) // 2,
        "Lunes": 1
    }
    
    assigned_matches = {}
    match_index = 0
    for day in days:
        assigned_matches[day] = []
        for _ in range(matches_per_day[day]):
            if match_index < len(round_matches):
                assigned_matches[day].append(round_matches[match_index])
                match_index += 1
    return assigned_matches


def team_schedule(team, schedule):
    """Devuelve el calendario de un equipo."""
    team_schedule = []
    for round_matches_dict in schedule:
        for day, matches in round_matches_dict.items():
            for match in matches:
                if match[0] == team or match[1] == team:
                    team_schedule.append((day, match))
    return team_schedule


def calculate_travel_distance(schedule, distancias, cities):
    # Calcula la distancia total de viaje para todos los equipos en el calendario
    total_distance = 0
    for round_matches_dict in schedule:
        for day, matches_for_day in round_matches_dict.items():
            for home, away in matches_for_day:
                home_city = next(city for city, teams in cities.items() if home in teams)
                away_city = next(city for city, teams in cities.items() if away in teams)
                distance = distancias[home_city][away_city]
                total_distance += distance
    return total_distance



def pretty_print_schedule_step_by_step(schedule):
    """Imprime el calendario jornada por jornada en un orden específico."""
    days_order = ["Viernes", "Sábado", "Domingo", "Lunes"]

    for i, round_matches_dict in enumerate(schedule, 1):
        print(f"Jornada {i}:")
        for day in days_order:
            if day in round_matches_dict:
                day_matches = ', '.join(f"{home} vs {away}" for home, away in round_matches_dict[day])
                print(f"  {day}: {day_matches}")
        print("-" * 40)


def print_team_stats(schedule, cities, distancias):
    # Imprime el calendario jornada por jornada
    # Inicializar contadores y sumas de distancia
    home_away_count = {team: {"home": 0, "away": 0} for team in teams}
    total_distances = {team: 0 for team in teams}
    day_count = {team: {"Viernes": 0, "Sábado": 0, "Domingo": 0, "Lunes": 0} for team in teams}

    # Contar partidos en casa, a distancia, y por día
    for round_matches_dict in schedule:
        for day, matches_for_day in round_matches_dict.items():
            for home, away in matches_for_day:
                home_away_count[home]["home"] += 1
                home_away_count[away]["away"] += 1
                day_count[home][day] += 1
                day_count[away][day] += 1

                home_city = next(city for city, teams in cities.items() if home in teams)
                away_city = next(city for city, teams in cities.items() if away in teams)
                distance = distancias[home_city][away_city]
                total_distances[away] += distance

    # Imprimir estadísticas para cada equipo
    for team in teams:
        away_count = home_away_count[team]["away"]
        avg_distance = total_distances[team] / away_count if away_count > 0 else 0
        print(f"Equipo: {team}")
        print(f"  Partidos totales: {home_away_count[team]['home'] + away_count}")
        print(f"  Partidos en casa: {home_away_count[team]['home']}")
        print(f"  Partidos fuera: {away_count}")
        print(f"  Distancia promedio viajada: {avg_distance:.2f} km")
        print(f"  Partidos en Viernes: {day_count[team]['Viernes']}")
        print(f"  Partidos en Sábado: {day_count[team]['Sábado']}")
        print(f"  Partidos en Domingo: {day_count[team]['Domingo']}")
        print(f"  Partidos en Lunes: {day_count[team]['Lunes']}")
        print("-" * 40)




# Parámetros ajustables del algoritmo evolutivo
population_size = 100  # Tamaño de la población
num_generations = 10  # Número de generaciones
num_parents = 20  # Número de padres
mutation_rate = 0.02  # Tasa de mutación


# Crear una población inicial con 38 jornadas y evaluarla
population_38j = generate_initial_population(population_size,teams)
print("Calendario inicial Premier League 2023:")
print_team_stats(population_38j[0], cities, distancias)

# Evaluar nuevamente con la función de evaluación adaptada
evaluated_population_38j_days_corrected = [(schedule, evaluate(schedule,cities,distancias)) for schedule in population_38j]
evaluated_population_sorted_38j_days_corrected = sorted(evaluated_population_38j_days_corrected, key=lambda x: x[1])

# Evolucionar la población y mostrar el mejor calendario
evolved_population = evolve_v2(evaluated_population_sorted_38j_days_corrected, num_parents, num_generations,evaluate)       


time.sleep(2)
#Imprimir el calendario con días y partidos
print("\nMejor calendario:")
# Imprimir el calendario jornada por jornada
pretty_print_schedule_step_by_step(evolved_population[0][0])
print("\nEstadísticas de los equipos:")
# Imprimir estadísticas de los equipos
print_team_stats(evolved_population[0][0], cities, distancias)