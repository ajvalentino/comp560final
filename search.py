import heapq
from data_creation import merged_df

# Parameters
number_of_meals = 5
calorie_goal = 310
required_ingredients = set(['chicken', 'lettuce', 'onion'])
available_recipes = merged_df.head(10000)

# Weights for scoring
w1, w2, w3 = 1, 1, 1

# Precompute average number of new ingredients across all recipes
def compute_avg_new_ingredients(recipes):
    total = 0
    for ings in recipes['ingredient_names']:
        total += len(set(ings))
    return total / len(recipes)

avg_new_ings_global = compute_avg_new_ingredients(available_recipes)

# State class
class State:
    def __init__(self, meal_plan, ingredients_used, total_calories):
        self.meal_plan = meal_plan
        self.ingredients_used = set(ingredients_used)
        self.total_calories = total_calories
        self.ids = frozenset(r['id'] for r in meal_plan)  # used for visited

    def calories_per_meal(self):
        return self.total_calories / len(self.meal_plan) if self.meal_plan else 0

    def __lt__(self, other):
        return False  # unused for A* tie-breaking but required by heapq

# Goal test
def is_goal(state, number_of_meals, calorie_goal, required_ingredients):
    return (
        len(state.meal_plan) == number_of_meals and
        abs(state.calories_per_meal() - calorie_goal) < 50 and
        required_ingredients.issubset(state.ingredients_used)
    )

# Heuristic function
def heuristic(state):
    required_remaining = len(required_ingredients - state.ingredients_used)
    calorie_diff = abs(state.calories_per_meal() - calorie_goal)
    overlap_bonus = len(state.ingredients_used & required_ingredients)
    return (
        w1 * avg_new_ings_global +
        w2 * calorie_diff +
        w3 * required_remaining -
        overlap_bonus
    )

# Cost function
def path_cost(state):
    calorie_diff = abs(state.calories_per_meal() - calorie_goal)
    return (
        w1 * len(state.ingredients_used) +
        w2 * calorie_diff
    )

# Successor generation with pruning
def get_successors(state):
    successors = []
    used_ids = state.ids
    for row in available_recipes.itertuples(index=False):
        if row.id in used_ids:
            continue

        # Prune if adding this recipe pushes calories way too far off
        if state.total_calories + row.calorie_count > calorie_goal * number_of_meals * 1.25:
            continue

        # Prune if this recipe has no required ingredients at all (unless already fulfilled)
        new_ingredients = state.ingredients_used | set(row.ingredient_names)
        if not required_ingredients.issubset(new_ingredients):
            if not (set(row.ingredient_names) & required_ingredients):
                continue

        new_meal_plan = state.meal_plan + [{
            'id': row.id,
            'name': row.name,
            'calorie_count': row.calorie_count,
            'ingredients': row.ingredient_names
        }]
        new_calories = state.total_calories + row.calorie_count

        successors.append(State(new_meal_plan, new_ingredients, new_calories))
    return successors

# Pure A* search (no iterative deepening)
def astar():
    initial_state = State([], set(), 0)
    frontier = [(path_cost(initial_state) + heuristic(initial_state), initial_state)]
    visited = set()

    while frontier:
        _, current = heapq.heappop(frontier)

        if is_goal(current, number_of_meals, calorie_goal, required_ingredients):
            return current.meal_plan

        if current.ids in visited:
            continue
        visited.add(current.ids)

        for succ in get_successors(current):
            f = path_cost(succ) + heuristic(succ)
            heapq.heappush(frontier, (f, succ))
    return None

print("Total recipes with at least one required ingredient:",
      sum(1 for row in available_recipes.itertuples(index=False)
          if set(row.ingredient_names) & required_ingredients))

# Run it
meal_plan = astar()
if meal_plan:
    for recipe in meal_plan:
        print(f"ID: {recipe['id']}, Recipe Name: {recipe['name']}, Calories: {recipe['calorie_count']}, Ingredients: {recipe['ingredients']}")
else:
    print("No valid meal plan found.")
