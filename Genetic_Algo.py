import random
import math
import copy
from collections import defaultdict
import pandas as pd

# Define constants
POPULATION_SIZE = 500
GENERATIONS = 100
MUTATION_RATE = 0.01
TIME_SLOTS = ["10 AM", "11 AM", "12 PM", "1 PM", "2 PM", "3 PM"]

ROOMS = {
    "Slater 003": 45,
    "Roman 216": 30,
    "Loft 206": 75,
    "Roman 201": 50,
    "Loft 310": 108,
    "Beach 201": 60,
    "Beach 301": 75,
    "Logos 325": 450,
    "Frank 119": 60
}

FACILITATORS = ["Lock", "Glen", "Banks", "Richards", "Shaw", "Singer", "Uther", "Tyler", "Numen", "Zeldin"]

ACTIVITIES = {
    "SLA100A": {"enroll": 50, "preferred": ["Glen", "Lock", "Banks", "Zeldin"], "others": ["Numen", "Richards"]},
    "SLA100B": {"enroll": 50, "preferred": ["Glen", "Lock", "Banks", "Zeldin"], "others": ["Numen", "Richards"]},
    "SLA191A": {"enroll": 50, "preferred": ["Glen", "Lock", "Banks", "Zeldin"], "others": ["Numen", "Richards"]},
    "SLA191B": {"enroll": 50, "preferred": ["Glen", "Lock", "Banks", "Zeldin"], "others": ["Numen", "Richards"]},
    "SLA201": {"enroll": 50, "preferred": ["Glen", "Banks", "Zeldin", "Shaw"], "others": ["Numen", "Richards", "Singer"]},
    "SLA291": {"enroll": 50, "preferred": ["Lock", "Banks", "Zeldin", "Singer"], "others": ["Numen", "Richards", "Shaw", "Tyler"]},
    "SLA303": {"enroll": 60, "preferred": ["Glen", "Zeldin", "Banks"], "others": ["Numen", "Singer", "Shaw"]},
    "SLA304": {"enroll": 25, "preferred": ["Glen", "Banks", "Tyler"], "others": ["Numen", "Singer", "Shaw", "Richards", "Uther", "Zeldin"]},
    "SLA394": {"enroll": 20, "preferred": ["Tyler", "Singer"], "others": ["Richards", "Zeldin"]},
    "SLA449": {"enroll": 60, "preferred": ["Tyler", "Singer", "Shaw"], "others": ["Zeldin", "Uther"]},
    "SLA451": {"enroll": 100, "preferred": ["Tyler", "Singer", "Shaw"], "others": ["Zeldin", "Uther", "Richards", "Banks"]},
}

# Representation: {activity: (room, time, facilitator)}
def generate_random_schedule():
    schedule = {}
    for activity in ACTIVITIES:
        room = random.choice(list(ROOMS.keys()))
        time = random.choice(TIME_SLOTS)
        facilitator = random.choice(FACILITATORS)
        schedule[activity] = (room, time, facilitator)
    return schedule

def compute_fitness(schedule):
    score = 100.0
    time_facilitator = defaultdict(set)
    time_room = defaultdict(set)

    for activity, (room, time, facilitator) in schedule.items():
        enroll = ACTIVITIES[activity]["enroll"]

        if enroll > ROOMS[room]:
            score -= 5

        if facilitator in time_facilitator[time]:
            score -= 10
        else:
            time_facilitator[time].add(facilitator)

        if room in time_room[time]:
            score -= 5
        else:
            time_room[time].add(room)

        if facilitator in ACTIVITIES[activity]["preferred"]:
            score += 10
        elif facilitator in ACTIVITIES[activity]["others"]:
            score += 3
        else:
            score -= 1

    return max(score, 0)

def softmax_selection(population, fitness_scores):
    exp_scores = [math.exp(f) for f in fitness_scores]
    total = sum(exp_scores)
    probs = [s / total for s in exp_scores]
    return random.choices(population, weights=probs, k=2)

def crossover(parent1, parent2):
    child = {}
    for activity in ACTIVITIES:
        child[activity] = parent1[activity] if random.random() < 0.5 else parent2[activity]
    return child

def mutate(schedule):
    for activity in schedule:
        if random.random() < MUTATION_RATE:
            room = random.choice(list(ROOMS.keys()))
            time = random.choice(TIME_SLOTS)
            facilitator = random.choice(FACILITATORS)
            schedule[activity] = (room, time, facilitator)

def write_schedule_table(schedule):
    data = [
        {
            "Activity": act,
            "Room": room,
            "Time": time,
            "Facilitator": fac
        }
        for act, (room, time, fac) in sorted(schedule.items())
    ]
    df = pd.DataFrame(data)
    df.to_csv("final_schedule_table.csv", index=False)
    print("Tabular schedule written to final_schedule_table.csv")

def print_optional_schedule(schedule):
    print("\nOptional View of Schedule:")
    organized = defaultdict(list)
    for activity, (room, time, facilitator) in schedule.items():
        organized[time].append((activity, room, facilitator))

    for time in TIME_SLOTS:
        print(f"\nTime Slot: {time}")
        for activity, room, facilitator in organized[time]:
            print(f"  {activity}: Room={room}, Facilitator={facilitator}")

def run_evolution():
    population = [generate_random_schedule() for _ in range(POPULATION_SIZE)]
    fitness_history = []
    generation = 0

    while generation < GENERATIONS:
        fitness_scores = [compute_fitness(s) for s in population]
        avg_fitness = sum(fitness_scores) / POPULATION_SIZE
        fitness_history.append(avg_fitness)

        if generation >= 2:
            improvement = (fitness_history[-1] - fitness_history[-2]) / (fitness_history[-2] + 1e-6)
            if improvement < 0.01:
                break

        new_population = []
        for _ in range(POPULATION_SIZE):
            p1, p2 = softmax_selection(population, fitness_scores)
            child = crossover(p1, p2)
            mutate(child)
            new_population.append(child)

        population = new_population
        generation += 1

    best = max(population, key=compute_fitness)
    with open("final_schedule.txt", "w") as f:
        for activity, (room, time, facilitator) in best.items():
            f.write(f"{activity}: Room={room}, Time={time}, Facilitator={facilitator}\n")
    print("Best schedule written to final_schedule.txt")

    write_schedule_table(best)
    print_optional_schedule(best)

if __name__ == "__main__":
    run_evolution()
