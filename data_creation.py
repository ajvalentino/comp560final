import sys
import pandas as pd
import pickle
import ast

# --- Workaround for the pickle error ---
# Import the module where the pickle expects Int64Index to be.
import pandas.core.indexes.base as base_indexes

# If "Int64Index" isn't defined, define a dummy subclass of pd.Index.
if not hasattr(base_indexes, 'Int64Index'):
    class DummyInt64Index(pd.Index):
        pass
    base_indexes.Int64Index = DummyInt64Index

# Alias the numeric indexes module so that the pickle loader finds the expected module.
sys.modules['pandas.core.indexes.numeric'] = base_indexes

# --- Step 1: Load Data Files from the Current Folder ---
pp_recipes = pd.read_csv('/Users/graceku/COMP-560-final-personal/PP_recipes.csv')
raw_recipes = pd.read_csv('/Users/graceku/COMP-560-final-personal/RAW_recipes.csv')

with open('/Users/graceku/COMP-560-final-personal/ingr_map.pkl', 'rb') as f:
    ingr_map = pickle.load(f)

# --- Step 2: Map Ingredient IDs to Names Using "ingredient_ids" Column ---
ingredient_id_to_name = {i: name for i, name in enumerate(ingr_map['replaced'])}
def map_ingredients(ingredient_ids_str):
    ingredient_ids = ast.literal_eval(ingredient_ids_str)
    ingredient_names = [ingredient_id_to_name.get(int(ing), str(ing)) for ing in ingredient_ids]
    return ingredient_names

pp_recipes['ingredient_names'] = pp_recipes['ingredient_ids'].apply(map_ingredients)

# --- Step 3: Extract Calorie Count from the "nutrition" Column ---
def extract_calories(nutrition_str):
    try:
        # Convert the string representation of a list into a Python list.
        nutrition_list = ast.literal_eval(nutrition_str)
        # If it's a list with at least one element, use the first element as the calorie count.
        if isinstance(nutrition_list, list) and len(nutrition_list) > 0:
            return float(nutrition_list[0])
        else:
            return None
    except Exception:
        return None

raw_recipes['calorie_count'] = raw_recipes['nutrition'].apply(extract_calories)

# --- Step 4: Merge the DataFrames ---
# Merge PP_recipes and RAW_recipes on the "id" column using an inner join.
merged_df = pd.merge(pp_recipes[['id', 'ingredient_ids', 'ingredient_names']], raw_recipes[['name', 'id', 'calorie_count']], on='id', how='inner')

# --- Step 5: Save the Merged Data to a CSV in the Current Folder ---
merged_df.to_csv('/Users/graceku/comp560final/merged_recipes.csv', index=False)