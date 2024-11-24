import pandas as pd

import json

import random

from typing import List, Dict

import argparse

import requests



# Templates multilingues pour les instructions et questions

templates = {

    "en": {

        "instruction": "Check if this ingredient is compatible with the specified diet. If not compatible or uncertain, explain why.",

        "input": "Is {ingredient} compatible with a {diet} diet? Diet Description: {diet_description}",

        "diet_names": {

            "vegan": "vegan",

            "vegetarian": "vegetarian", 

            "halal": "halal",

            "gluten_free": "gluten free"

        }

    },

    "fr": {

        "instruction": "Vérifiez si cet ingrédient est compatible avec le régime alimentaire spécifié. Si non compatible ou incertain, expliquez pourquoi.",

        "input": "{ingredient} compatible avec régime {diet}?",

        "diet_names": {

            "vegan": "végétalien",

            "vegetarian": "végétarien",

            "halal": "halal", 

            "gluten_free": "sans gluten"

        }

    },

    "ar": {

        "instruction": "تحقق مما إذا كان هذا المكون متوافقًا مع النظام الغذائي المحدد. إذا لم يكن متوافقًا أو غير مؤكد، اشرح السبب.",

        "input": "{ingredient} متوافق مع نظام {diet}؟",

        "diet_names": {

            "vegan": "نباتي",

            "vegetarian": "نباتي",

            "halal": "حلال",

            "gluten_free": "خالي من الغلوتين"

        }

    }

}



# Réponses localisées

responses = {

    "en": {

        "1": "yes",

        "0": "no, because {ingredient} {reason}",

        "2": "not sure, {reason}"

    },

    "fr": {

        "yes": "oui",

        "no": "non, car {ingredient} {reason}",

        "not_sure": "pas certain, {reason}"

    },

    "ar": {

        "yes": "نعم",

        "no": "لا، لأن {ingredient} {reason}",

        "not_sure": "غير متأكد، {reason}"

    }

}



# Statuts des ingrédients avec gestion des cas spéciaux

STATUS_MAPPING = {

    '0': "no",

    '1': "yes", 

    '2': "not sure",

    '111': "not sure",  # Ajout du statut 111

    'null': "not sure", # Gestion des valeurs null

    None: "not sure"    # Gestion des None

}



def get_status(status) -> str:

    """

    Fonction utilitaire pour gérer tous les cas de statuts possibles

    """

    # Convertir en string si ce n'est pas None

    status_str = str(status) if status is not None else None

    

    # Retourner le statut mappé ou "not sure" par défaut

    return STATUS_MAPPING.get(status_str, "not sure")



# Raisons par statut et régime

REASONS = {

    "vegan": {

        '0': {  # Changé de 0 à '0'

            "meat": "is derived from animals",

            "fish": "is derived from animals",

            "seafood": "is derived from marine animals",

            "dairy": "is derived from animal milk",

            "eggs": "is derived from animals",

            "honey": "is produced by bees",

            "default": "contains animal products"

        },

        '2': "requires verification of production methods"  # Changé de 2 à '2'

    },

    "vegetarian": {

        '0': {  # Changé de 0 à '0'

            "meat": "contains animal flesh",

            "fish": "contains animal flesh",

            "seafood": "contains animal flesh",

            "default": "contains dead animal parts"

        },

        '2': "may contain animal-derived ingredients"  # Changé de 2 à '2'

    },

    "halal": {

        '0': {  # Changé de 0 à '0'

            "pork": "is derived from pork",

            "alcohol": "contains alcohol",

            "default": "does not meet halal requirements"

        },

        '2': "requires halal certification verification"  # Changé de 2 à '2'

    },

    "gluten_free": {

        '0': {  # Changé de 0 à '0'

            "wheat": "contains gluten from wheat",

            "barley": "contains gluten from barley", 

            "rye": "contains gluten from rye",

            "default": "contains gluten"

        },

        '2': "may contain traces of gluten"  # Changé de 2 à '2'

    }

}



def load_ingredients_from_api(api_data: List[Dict]) -> Dict:

    """

    Convertit les données de l'API en dictionnaire structuré

    Ne garde que les ingrédients avec des régimes validés manuellement par l'utilisateur

    """

    ingredients_db = {}

    regime_counts = {

        "vegan": {"total": 0, "status": {"0": 0, "1": 0, "2": 0, "111": 0, "null": 0}},

        "vegetarian": {"total": 0, "status": {"0": 0, "1": 0, "2": 0, "111": 0, "null": 0}},

        "halal": {"total": 0, "status": {"0": 0, "1": 0, "2": 0, "111": 0, "null": 0}},

        "gluten_free": {"total": 0, "status": {"0": 0, "1": 0, "2": 0, "111": 0, "null": 0}}

    }

    

    for ingredient in api_data:

        name = ingredient["name"]

        ingredient_id = ingredient.get("_id")

        translations = ingredient.get("translations", [])

        

        has_validated_regime = False

        regimes = {}

        

        for regime in ingredient.get("regimes", []):

            regime_type = regime["type"]

            if regime_type in regime_counts:

                user_validation = regime.get("userValidation", False)

                match_validation = regime.get("matchValidation")

                

                if user_validation and (match_validation is None or match_validation is False):

                    has_validated_regime = True

                    status = regime.get("status")

                    status_str = str(status) if status is not None else "null"

                    

                    regimes[regime_type] = {

                        "status": status_str,

                        "validated": True

                    }

                    

                    regime_counts[regime_type]["total"] += 1

                    

                    if status_str in regime_counts[regime_type]["status"]:

                        regime_counts[regime_type]["status"][status_str] += 1

                    else:

                        regime_counts[regime_type]["status"][status_str] = 1

        

        if has_validated_regime:

            ingredients_db[name] = {

                "id": ingredient_id,

                "translations": translations,

                "regimes": regimes

            }

    

    print("\nStatistiques détaillées de validation:")

    print(f"Nombre total d'ingrédients dans l'API: {len(api_data)}")

    print(f"Nombre d'ingrédients avec au moins un régime validé manuellement: {len(ingredients_db)}")

    

    print("\nStatistiques par régime alimentaire:")

    for regime, stats in regime_counts.items():

        print(f"\n{regime.upper()}:")

        print(f"Total d'ingrédients validés: {stats['total']}")

        if stats['total'] > 0:

            print("Distribution des statuts:")

            for status, count in stats["status"].items():

                if count > 0:  # N'afficher que les statuts qui ont des occurrences

                    status_label = {

                        "0": "Non compatible",

                        "1": "Compatible",

                        "2": "Incertain",

                        "111": "Incertain (111)",

                        "null": "Incertain (null)"

                    }.get(status, f"Autre ({status})")

                    percentage = (count / stats['total']) * 100

                    print(f"  {status_label}: {count} ({percentage:.1f}%)")

    

    return ingredients_db



def get_reason(ingredient: str, diet: str, status: str, ingredients_db: Dict) -> str:

    """

    Génère une explication pour le statut d'un ingrédient

    """

    if status not in ['0', '2']:  # Modifié pour utiliser des strings

        return ""

        

    if diet not in REASONS:

        return ""

        

    reasons = REASONS[diet]

    

    if status == '2':  # Modifié pour utiliser des strings

        return reasons['2']

        

    # Pour status 0 (interdit)

    for key, reason in reasons['0'].items():  # Modifié pour utiliser des strings

        if key in ingredient.lower():

            return reason

    return reasons['0']["default"]



def load_diets_config():

    """Load diet descriptions from diets.json"""

    try:

        with open('diets.json', 'r', encoding='utf-8') as f:

            diets = json.load(f)

            return {diet["key"]: diet for diet in diets}

    except Exception as e:

        print(f"Error loading diets.json: {e}")

        return {}



def generate_dataset(ingredients_db: Dict, num_samples: int = None) -> List[Dict]:

    """

    Génère le dataset d'entraînement en anglais

    

    Args:

        ingredients_db: Dictionnaire des ingrédients

        num_samples: Nombre d'échantillons à générer. Si None, traite tous les ingrédients

    """
    diets_config = load_diets_config()
    dataset = []

    diets = ["vegan", "vegetarian", "halal", "gluten_free"]

    ingredients = list(ingredients_db.keys())

    

    if num_samples is None:

        for ingredient in ingredients:

            for diet in diets:

                ingredient_info = ingredients_db[ingredient]

                regime_info = ingredient_info["regimes"].get(diet, {"status": None, "validated": False})

                

                status = regime_info["status"]

                ingredient_name = ingredient

                ingredient_id = ingredient_info["id"]

                

                # Ne garder que les statuts valides

                if status not in ['0', '1', '2']:

                    continue

                

                if status == '1':  # OK

                    response = responses["en"]["1"]

                elif status == '0':  # Not OK

                    reason = get_reason(ingredient, diet, status, ingredients_db)

                    response = responses["en"]["0"].format(

                        ingredient=ingredient_name,

                        reason=reason

                    )

                elif status == '2':  # Uncertain

                    reason = "requires further verification"

                    response = responses["en"]["2"].format(reason=reason)



                entry = {

                    "instruction": templates["en"]["instruction"],

                    "input": templates["en"]["input"].format(

                        ingredient=ingredient_name,

                        diet=templates["en"]["diet_names"][diet],
                        
                        diet_description=diets_config[diet]["taskcmd"] if diet in diets_config else ""

                    ),

                    "output": response,

                    "status": status,  # Ajout du statut numérique

                    "language": "en",

                    "ingredient_id": ingredient_id,

                    "diet": diet

                }

                dataset.append(entry)

    else:

        while len(dataset) < num_samples:

            ingredient = random.choice(ingredients)

            diet = random.choice(diets)

            

            ingredient_info = ingredients_db[ingredient]

            regime_info = ingredient_info["regimes"].get(diet, {"status": None, "validated": False})

            

            status = regime_info["status"]

            ingredient_name = ingredient

            ingredient_id = ingredient_info["id"]

            

            # Ne garder que les statuts valides

            if status not in ['0', '1', '2']:

                continue

                

            if status == '1':  # OK

                response = responses["en"]["1"]

            elif status == '0':  # Not OK

                reason = get_reason(ingredient, diet, status, ingredients_db)

                response = responses["en"]["0"].format(

                    ingredient=ingredient_name,

                    reason=reason

                )

            elif status == '2':  # Uncertain

                reason = "requires further verification"

                response = responses["en"]["2"].format(reason=reason)



            entry = {

                "instruction": templates["en"]["instruction"],

                "input": templates["en"]["input"].format(

                    ingredient=ingredient_name,

                    diet=templates["en"]["diet_names"][diet]

                ),

                "output": response,

                "status": status,  # Ajout du statut numérique

                "language": "en",

                "ingredient_id": ingredient_id,

                "diet": diet

            }

            dataset.append(entry)

    

    return dataset



def translate_diet(diet: str, lang: str) -> str:

    """

    Traduit le nom du régime alimentaire

    """

    translations = {

        "fr": {

            "vegan": "végétalien",

            "vegetarian": "végétarien",

            "halal": "halal",

            "gluten_free": "sans gluten"

        },

        "ar": {

            "vegan": "نباتي",

            "vegetarian": "نباتي",

            "halal": "حلال",

            "gluten_free": "خالي من الغلوتين"

        }

    }

    return translations.get(lang, {}).get(diet, diet)



def translate_reason(reason: str, lang: str) -> str:

    """

    Traduit les raisons

    """

    if lang == "fr":

        translations = {

            "is derived from animals": "provient d'animaux",

            "contains animal flesh": "contient de la chair animale",

            "contains gluten": "contient du gluten",

            "requires verification": "nécessite une vérification",

            # Ajoutez d'autres traductions selon besoin

        }

        for eng, fr in translations.items():

            if eng in reason:

                return reason.replace(eng, fr)

    elif lang == "ar":

        translations = {

            "is derived from animals": "مشتق من الحيوانات",

            "contains animal flesh": "يحتوي على لحم حيواني",

            "contains gluten": "يحتوي على الغلوتين",

            "requires verification": "يتطلب التحقق",

            # Ajoutez d'autres traductions selon besoin

        }

        for eng, ar in translations.items():

            if eng in reason:

                return reason.replace(eng, ar)

    return reason



def save_dataset(dataset: List[Dict], output_prefix: str = 'training_data') -> Dict[str, pd.DataFrame]:

    """

    Sauvegarde le dataset au format JSONL dans des fichiers séparés par langue

    

    Args:

        dataset: Liste des données

        output_prefix: Préfixe pour les noms de fichiers (ex: 'training_data')

    

    Returns:

        Dict des DataFrames par langue

    """

    # Séparer les données par langue

    data_by_lang = {

        "en": [],

        "fr": [],

        "ar": []

    }

    

    for item in dataset:

        lang = item["language"]

        if lang in data_by_lang:

            data_by_lang[lang].append(item)

    

    # Créer un DataFrame pour chaque langue et sauvegarder

    dataframes = {}

    for lang, items in data_by_lang.items():

        if items:  # Ne créer un fichier que si nous avons des données pour cette langue

            output_file = f"{output_prefix}_{lang}.jsonl"

            with open(output_file, 'w', encoding='utf-8') as f:

                for item in items:

                    json.dump({

                        "instruction": item["instruction"],

                        "input": item["input"],

                        "output": item["output"]

                    }, f, ensure_ascii=False)

                    f.write('\n')

            

            dataframes[lang] = pd.DataFrame(items)

            print(f"Fichier {output_file} créé avec {len(items)} exemples")

    

    return dataframes



if __name__ == "__main__":

    # Configuration des arguments

    parser = argparse.ArgumentParser(description='Générer un dataset pour l\'entraînement du modèle')

    parser.add_argument('--num_samples', type=int, default=None, 

                      help='Nombre d\'exemples à générer. Si non spécifié, traite tous les ingrédients')

    parser.add_argument('--output_prefix', type=str, default='training_data',

                      help='Préfixe pour les fichiers de sortie')

    

    args = parser.parse_args()



    print("Récupération des données depuis l'API...")

    try:

        response = requests.get("https://api.liviz.app/ingredients")

        response.raise_for_status()

        api_data = response.json()

    except requests.RequestException as e:

        print(f"Erreur lors de la récupération des données de l'API: {e}")

        exit(1)

    

    print("Conversion des données...")

    ingredients_db = load_ingredients_from_api(api_data)

    

    if args.num_samples is None:

        print(f"Génération d'exemples pour tous les ingrédients ({len(ingredients_db)} ingrédients)...")

    else:

        print(f"Génération de {args.num_samples} exemples aléatoires...")

    

    dataset = generate_dataset(ingredients_db, num_samples=args.num_samples)

    

    print("Sauvegarde des datasets...")

    dataframes = save_dataset(dataset, args.output_prefix) 
