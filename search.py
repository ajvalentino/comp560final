import heapq
from data_creation import merged_df

# Parameters (just examples for now)
number_of_meals = 4
calorie_goal = 310
required_ingredients = set(['sausage', 'hoagie roll'])
# available_recipes = merged_df
available_recipes = merged_df.head(200)

# Weight tuning - adjust according to goals (doesn't have to be between 0 and 1)
w1, w2, w3 = 1, 1, 1

# current code penalizes recipes that add too many new ingredients but doesn't explicitly reward recipes with ingredient overlap

# State class
class State:
    def __init__(self, meal_plan, ingredients_used, total_calories):
        self.meal_plan = meal_plan
        self.ingredients_used = set(ingredients_used)  # using sets for faster computation
        self.total_calories = total_calories

    def calories_per_meal(self):
        return self.total_calories / len(self.meal_plan) if self.meal_plan else 0

    def __lt__(self, other):
        return True  # Required for heapq
    

def is_goal(state, number_of_meals, calorie_goal, required_ingredients):
    return (len(state.meal_plan) == number_of_meals and
            abs(state.calories_per_meal() - calorie_goal) < 50 and # set calorie tolerance threshold to something else?
            required_ingredients.issubset(state.ingredients_used))

# Heuristic (h)
def heuristic(state):
    used = state.ingredients_used
    required_remaining = len(required_ingredients - used)
    calorie_diff = abs(state.calories_per_meal() - calorie_goal)

    # Estimate additional ingredients needed in next recipe
    if len(state.meal_plan) < number_of_meals:
        # Faster: loop through ings instead of apply method with lambda
        total_new_ings = 0
        for ings in available_recipes['ingredient_names']:
            total_new_ings += len(set(ings) - used)
        avg_new_ings = total_new_ings / len(available_recipes)
    else:
        avg_new_ings = 0

    return w1 * avg_new_ings + w2 * calorie_diff + w3 * required_remaining

# Cost so far (g)
def path_cost(state):
    calorie_diff = abs(state.calories_per_meal() - calorie_goal)
    return w1 * len(state.ingredients_used) + w2 * calorie_diff

# Successor generation
def get_successors(state):
    successors = []
    used_ids = {r['id'] for r in state.meal_plan}
    for row in available_recipes.itertuples(index=False):
        if row.id in used_ids:
            continue
        new_meal_plan = state.meal_plan + [{
            'id': row.id,
            'name': row.name,
            'calorie_count': row.calorie_count,
            'ingredients': row.ingredient_names
        }]
        new_ingredients = state.ingredients_used | set(row.ingredient_names)
        new_calories = state.total_calories + row.calorie_count
        successors.append(State(new_meal_plan, new_ingredients, new_calories))
    return successors

# A* Search with Iterative Deepening
def astar_iterative_deepening():
    depth_limit = 1
    while True:
        # print(f"Trying depth limit: {depth_limit}")
        result = astar_limited(depth_limit)
        if result is not None:
            return result
        depth_limit += 1

def astar_limited(max_depth):
    initial_state = State([], set(), 0)
    frontier = [(path_cost(initial_state) + heuristic(initial_state), initial_state)]
    visited = set()

    while frontier:
        _, current = heapq.heappop(frontier)
        if is_goal(current, number_of_meals, calorie_goal, required_ingredients):
            return current.meal_plan
        if len(current.meal_plan) >= max_depth:
            continue # skips if cheapest state popped from the frontier is too deep
        for succ in get_successors(current):
            key = tuple(sorted(r['id'] for r in succ.meal_plan))
            if key not in visited:
                visited.add(key)
                f = path_cost(succ) + heuristic(succ)
                heapq.heappush(frontier, (f, succ))
    return None

meal_plan = astar_iterative_deepening()
for recipe in meal_plan:
    print(f"ID: {recipe['id']}, Recipe Name: {recipe['name']}, Calories: {recipe['calorie_count']}") # get recipe links?