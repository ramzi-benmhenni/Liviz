import pandas as pd
import json
import random
from typing import List, Dict
import argparse

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
    """
    ingredients_db = {}
    
    for ingredient in api_data:
        name = ingredient["name"]
        # Garder les traductions dans leur format original
        translations = ingredient.get("translations", [])
        
        regimes = {}
        for regime in ingredient.get("regimes", []):
            regime_type = regime["type"]
            if regime_type in ["vegan", "vegetarian", "halal", "gluten_free"]:
                status = regime.get("status")
                status_str = str(status) if status is not None else None
                
                regimes[regime_type] = {
                    "status": status_str,
                    "validated": regime.get("userValidation", False)
                }
        
        ingredients_db[name] = {
            "translations": translations,  # Liste de dictionnaires avec 'language' et 'label'
            "regimes": regimes
        }
    
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

def generate_dataset(ingredients_db: Dict, num_samples: int = 1000) -> List[Dict]:
    """
    Génère le dataset d'entraînement multilingue
    """
    dataset = []
    diets = ["vegan", "vegetarian", "halal", "gluten_free"]
    
    # Templates multilingues pour les instructions et questions
    templates = {
        "en": {
            "instruction": "Check if this ingredient is compatible with the specified diet. If not compatible or uncertain, explain why.",
            "input": "Is {ingredient} compatible with a {diet} diet?",
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
            "yes": "yes",
            "no": "no, because {ingredient} {reason}",
            "not_sure": "not sure, {reason}"
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

    ingredients = list(ingredients_db.keys())
    
    for _ in range(num_samples):
        ingredient = random.choice(ingredients)
        diet = random.choice(diets)
        
        ingredient_info = ingredients_db[ingredient]
        regime_info = ingredient_info["regimes"].get(diet, {"status": None, "validated": False})
        translations = ingredient_info["translations"]
        
        status = regime_info["status"]
        
        # Vérifier les traductions disponibles
        available_translations = {t["language"].lower(): t["label"] for t in translations}
        
        # Générer uniquement pour l'anglais et les langues disponibles
        languages_to_generate = ["en"]  # Toujours inclure l'anglais
        
        # Ajouter les autres langues seulement si la traduction existe
        if "french" in available_translations:
            languages_to_generate.append("fr")
        if "arabic" in available_translations:
            languages_to_generate.append("ar")
        
        for lang in languages_to_generate:
            # Obtenir le nom de l'ingrédient dans la bonne langue
            if lang == "en":
                ingredient_name = ingredient
            elif lang == "fr" and "french" in available_translations:
                ingredient_name = available_translations["french"]
            elif lang == "ar" and "arabic" in available_translations:
                ingredient_name = available_translations["arabic"]
            else:
                continue  # Sauter cette langue si pas de traduction
            
            # Formater la réponse selon le statut et la langue
            if status == '1':
                response = responses[lang]["yes"]
            elif status == '0':
                reason = get_reason(ingredient, diet, status, ingredients_db)
                response = responses[lang]["no"].format(
                    ingredient=ingredient_name,
                    reason=translate_reason(reason, lang)
                )
            else:
                reason = "requires further verification"
                response = responses[lang]["not_sure"].format(reason=translate_reason(reason, lang))

            entry = {
                "instruction": templates[lang]["instruction"],
                "input": templates[lang]["input"].format(
                    ingredient=ingredient_name,
                    diet=templates[lang]["diet_names"][diet]
                ),
                "output": response,
                "language": lang
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

def save_dataset(dataset: List[Dict], output_file: str = 'training_data.jsonl') -> pd.DataFrame:
    """
    Sauvegarde le dataset au format JSONL dans le format Alpaca
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in dataset:
            # Format Alpaca direct
            json.dump({
                "instruction": item["instruction"],
                "input": item["input"],
                "output": item["output"]
            }, f, ensure_ascii=False)
            f.write('\n')
    
    return pd.DataFrame(dataset)

if __name__ == "__main__":
    # Configuration des arguments
    parser = argparse.ArgumentParser(description='Générer un dataset pour l\'entraînement du modèle')
    parser.add_argument('--num_samples', type=int, default=1000, 
                      help='Nombre d\'exemples à générer')
    parser.add_argument('--output', type=str, default='training_data.jsonl',
                      help='Chemin du fichier de sortie')
    parser.add_argument('--api_data', type=str, default='api_data.json',
                      help='Chemin du fichier de données API')
    args = parser.parse_args()

    print("Chargement des données de l'API...")
    with open(args.api_data, 'r', encoding='utf-8') as f:
        api_data = json.load(f)
    
    print("Conversion des données...")
    ingredients_db = load_ingredients_from_api(api_data)
    
    print(f"Génération de {args.num_samples} exemples...")
    dataset = generate_dataset(ingredients_db, num_samples=args.num_samples)
    
    print(f"Sauvegarde dans {args.output}...")
    df = save_dataset(dataset, args.output)
    
    print("\nStatistiques du dataset:")
    print(f"Nombre total d'exemples: {len(df)}")
    
    # Distribution par langue
    languages = df['language'].value_counts()
    print("\nDistribution par langue:")
    for lang, count in languages.items():
        print(f"{lang}: {count} exemples ({count/len(df)*100:.1f}%)")
    
    # Distribution des réponses
    responses = df['output'].value_counts()
    print("\nDistribution des réponses:")
    for response, count in responses.items():
        print(f"{response}: {count} exemples ({count/len(df)*100:.1f}%)")
    
    # Exemples pour chaque langue
    print("\nExemples de données générées:")
    for lang in ["en", "fr", "ar"]:
        print(f"\n{lang.upper()} exemple:")
        example = df[df['language'] == lang].iloc[0]
        print("Instruction:", example['instruction'])
        print("Input:", example['input'])
        print("Output:", example['output'])
    
    # Vérification du format pour Llama
    print("\nVérification du format Llama...")
    with open(args.output, 'r', encoding='utf-8') as f:
        first_line = json.loads(f.readline())
        if "text" in first_line:
            print("✓ Format correct pour l'entraînement Llama")
            print("\nExemple de format:")
            print(first_line["text"])
        else:
            print("⚠ Format incorrect - vérifiez la structure du fichier") 