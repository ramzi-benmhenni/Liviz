import json
import os
import google.generativeai as genai
from typing import Dict, List
from openai import OpenAI
from groq import Groq
from tqdm import tqdm
from config import API_KEYS
import asyncio
import time

# Configuration des clés API
OPENAI_API_KEY = API_KEYS["OPENAI_API_KEY"]
GOOGLE_API_KEY = API_KEYS["GOOGLE_API_KEY"]  # commenté
GROQ_API_KEY = API_KEYS["GROQ_API_KEY"]

# Délai entre les requêtes (en secondes)
REQUEST_DELAY = 3   

def load_diets_config():
    """Load diet descriptions from diets.json"""
    try:
        with open('diets.json', 'r', encoding='utf-8') as f:
            diets = json.load(f)
            return {diet["key"]: diet["description"] for diet in diets}
    except Exception as e:
        print(f"Error loading diets.json: {e}")
        return {}

class ModelTester:
    def __init__(self, models_config_path="models_config.json"):
        # Load diets configuration
        self.diets_config = load_diets_config()
        
        # Charger la configuration des modèles
        with open(models_config_path, 'r') as f:
            self.models_config = json.load(f)
        
        # Filtrer les modèles activés
        self.enabled_models = {k: v for k, v in self.models_config.items() if v.get("enabled", True)}
        
        # Initialiser les clients
        self.clients = {}
        self.initialize_clients()
        
        # Pour stocker les résultats
        self.results = {
            "total_tested": 0,
            "agreement": 0,
            "detailed_results": [],
            "agreement_analysis": {
                "total_agreements": 0,
                "correct_agreements": 0,
                "agreement_cases": []
            }
        }
        
        # Initialiser les résultats pour chaque modèle activé
        for model_key in self.enabled_models:
            self.results[model_key] = {
                "correct": 0,
                "total": 0,
                "responses": {"yes": 0, "no": 0, "not_sure": 0},
                "model_name": self.enabled_models[model_key]["model_name"],
                "total_tokens": {"input": 0, "output": 0}
            }
        
        # Pour gérer les délais entre requêtes
        self.last_request_time = {
            config["rate_limit_key"]: 0 for config in self.enabled_models.values()
        }

        self.max_retries = 3
        self.retry_delay = 10

    def initialize_clients(self):
        """Initialise les clients API pour les modèles activés"""
        for model_config in self.enabled_models.values():
            if model_config["type"] == "openai" and "openai" not in self.clients:
                self.clients["openai"] = OpenAI(api_key=API_KEYS[model_config["client_key"]])
            elif model_config["type"] == "gemini" and "gemini" not in self.clients:
                genai.configure(api_key=API_KEYS[model_config["client_key"]])
                self.clients["gemini"] = genai.GenerativeModel('gemini-pro')
            elif model_config["type"] == "groq" and "groq" not in self.clients:
                self.clients["groq"] = Groq(api_key=API_KEYS[model_config["client_key"]])

    async def wait_for_rate_limit(self, model: str):
        """Attend le délai nécessaire entre les requêtes"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time[model]
        
        if time_since_last_request < REQUEST_DELAY:
            wait_time = REQUEST_DELAY - time_since_last_request
            await asyncio.sleep(wait_time)
        
        self.last_request_time[model] = time.time()

    async def retry_query(self, model: str, query_func, instruction: str, input_text: str, model_name: str) -> dict:
        """Modifié pour gérer correctement les retours"""
        for attempt in range(self.max_retries):
            try:
                response = await query_func(instruction, input_text, model_name)
                if response["response"] != "error":
                    return response
                
                print(f"\n{model.upper()} Error (attempt {attempt + 1}/{self.max_retries})")
                if attempt < self.max_retries - 1:
                    print(f"Waiting {self.retry_delay} seconds before retrying...")
                    await asyncio.sleep(self.retry_delay)
                
            except Exception as e:
                print(f"\n{model.upper()} Error (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    print(f"Waiting {self.retry_delay} seconds before retrying...")
                    await asyncio.sleep(self.retry_delay)
        
        return {"response": "error", "model": model_name, "tokens": {"input": 0, "output": 0}}

    async def query_gpt(self, instruction: str, input_text: str, model: str) -> dict:
        """Modified to include diet description"""
        await self.wait_for_rate_limit("gpt")
        try:
            # Extract diet from input text (assuming format contains "diet?")
            diet_name = input_text.split("diet?")[0].strip().split()[-1]
            diet_description = self.diets_config.get(diet_name, "")
            
            # Add diet description to the prompt
            prompt = (
                f"{instruction}\n"
                f"Diet Description: {diet_description}\n"
                f"{input_text}\n"
                "Reply with 'yes', 'no', or 'not sure'."
            )
            
            response = self.clients["openai"].chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            return {
                "response": self.clean_response(response.choices[0].message.content),
                "model": response.model,
                "tokens": {
                    "input": response.usage.prompt_tokens,
                    "output": response.usage.completion_tokens
                }
            }
        except Exception as e:
            print(f"OpenAI Error: {str(e)}")
            return {"response": "error", "model": model, "tokens": {"input": 0, "output": 0}}

    async def query_gemini(self, instruction: str, input_text: str, model_name: str) -> dict:
        """Modified to include diet description"""
        await self.wait_for_rate_limit("gemini")
        try:
            # Extract diet from input text
            diet_name = input_text.split("diet?")[0].strip().split()[-1]
            diet_description = self.diets_config.get(diet_name, "")
            
            # Add diet description to the prompt
            prompt = (
                f"{instruction}\n"
                f"Diet Description: {diet_description}\n"
                f"{input_text}\n"
                "Reply with 'yes', 'no', or 'not sure'."
            )
            
            response = self.clients["gemini"].generate_content(prompt)
            return {
                "response": self.clean_response(response.text),
                "model": "gemini-pro",
                "tokens": {"input": 0, "output": 0}
            }
        except Exception as e:
            print(f"Gemini Error: {str(e)}")
            return {"response": "error", "model": model_name, "tokens": {"input": 0, "output": 0}}

    async def query_groq(self, instruction: str, input_text: str, model_name: str) -> dict:
        """Modified to include diet description"""
        await self.wait_for_rate_limit("groq")
        try:
            # Extract diet from input text
            diet_name = input_text.split("diet?")[0].strip().split()[-1]
            diet_description = self.diets_config.get(diet_name, "")
            
            # Add diet description to the prompt
            prompt = (
                f"{instruction}\n"
                f"Diet Description: {diet_description}\n"
                f"{input_text}\n"
                "Reply with 'yes', 'no', or 'not sure'."
            )
            
            response = self.clients["groq"].chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            return {
                "response": self.clean_response(response.choices[0].message.content),
                "model": response.model,
                "tokens": {
                    "input": response.usage.prompt_tokens,
                    "output": response.usage.completion_tokens
                }
            }
        except Exception as e:
            print(f"Groq Error: {str(e)}")
            return {"response": "error", "model": model_name, "tokens": {"input": 0, "output": 0}}

    def clean_response(self, response: str) -> str:
        """Normalise la réponse en 'yes', 'no' ou 'not sure'"""
        response = response.lower().strip()
        if "yes" in response:
            return "yes"
        elif "no" in response:
            return "no"
        return "not_sure"

    def check_agreement(self, responses: Dict[str, dict]) -> bool:
        """Vérifie si tous les modèles sont d'accord"""
        valid_responses = [r["response"] for r in responses.values() if r["response"] != "error"]
        return len(valid_responses) > 1 and len(set(valid_responses)) == 1

    async def test_single_example(self, example: Dict) -> None:
        expected = example["output"].lower().strip()
        self.results["total_tested"] += 1

        responses = {}
        for model_key, model_config in self.enabled_models.items():
            responses[model_key] = await self.retry_query(
                model_config["rate_limit_key"],
                self.get_query_function(model_config["type"]),
                example["instruction"],
                example["input"],
                model_config["model_name"]
            )

        # Stocker les résultats détaillés
        detailed_result = {
            "question": example["input"],
            "instruction": example["instruction"],
            "true_answer": expected,
            "model_responses": {k: v["response"] for k, v in responses.items()}  # Stocke uniquement la réponse
        }
        self.results["detailed_results"].append(detailed_result)

        # Analyser l'accord entre les modèles
        if self.check_agreement(responses):
            self.results["agreement"] += 1
            unanimous_response = next(r["response"] for r in responses.values() if r["response"] != "error")
            self.results["agreement_analysis"]["total_agreements"] += 1
            
            if unanimous_response == expected:
                self.results["agreement_analysis"]["correct_agreements"] += 1
            
            self.results["agreement_analysis"]["agreement_cases"].append({
                "question": example["input"],
                "instruction": example["instruction"],
                "unanimous_response": unanimous_response,
                "true_answer": expected,
                "is_correct": unanimous_response == expected
            })

        # Mettre à jour les statistiques par modèle
        for model_key, response_data in responses.items():
            if response_data["response"] != "error":
                self.results[model_key]["total"] += 1
                self.results[model_key]["responses"][response_data["response"]] += 1
                self.results[model_key]["total_tokens"]["input"] += response_data["tokens"]["input"]
                self.results[model_key]["total_tokens"]["output"] += response_data["tokens"]["output"]
                if response_data["response"] in ["yes", "no"] and response_data["response"] == expected:
                    self.results[model_key]["correct"] += 1

    def get_query_function(self, model_type: str):
        """Retourne la fonction de requête appropriée pour le type de modèle"""
        query_functions = {
            "openai": self.query_gpt,
            "gemini": self.query_gemini,
            "groq": self.query_groq
        }
        return query_functions.get(model_type)

    async def run_tests(self, dataset_path: str):
        """Exécute les tests sur tout le dataset"""
        with open(dataset_path, 'r', encoding='utf-8') as f:
            examples = [json.loads(line) for line in f]
        
        # Filtrer pour ne garder que les exemples avec des réponses yes/no
        examples = [ex for ex in examples if any(answer in ex["output"].lower() for answer in ["yes", "no"])]
        
        print(f"Testing {len(examples)} examples with clear yes/no answers...")
        for example in tqdm(examples):
            await self.test_single_example(example)

    def save_results(self, output_file: str):
        """Sauvegarde les résultats dans un fichier JSON"""
        # Calculer les statistiques d'accord
        agreement_stats = self.results["agreement_analysis"]
        if agreement_stats["total_agreements"] > 0:
            agreement_stats["agreement_accuracy"] = (agreement_stats["correct_agreements"] / agreement_stats["total_agreements"]) * 100
            agreement_stats["agreement_percentage"] = (agreement_stats["total_agreements"] / self.results["total_tested"]) * 100
        else:
            agreement_stats["agreement_accuracy"] = 0
            agreement_stats["agreement_percentage"] = 0

        # Sauvegarder en JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

    def print_results(self):
        """Affiche les résultats des tests"""
        print("\n=== Résultats des Tests ===")
        print(f"Nombre total d'exemples testés: {self.results['total_tested']}")
        print("\nRésultats par modèle:")
        print("-" * 50)
        
        for model_key in self.enabled_models:
            total = self.results[model_key]["total"]
            if total > 0:
                accuracy = (self.results[model_key]["correct"] / total) * 100
                print(f"\n{model_key.upper()}:")
                print(f"Modèle utilisé: {self.results[model_key]['model_name']}")
                print(f"Précision: {accuracy:.1f}%")
                print(f"Nombre total de réponses: {total}")
                print(f"Tokens totaux: Input: {self.results[model_key]['total_tokens']['input']}, Output: {self.results[model_key]['total_tokens']['output']}")
                print("Distribution des réponses:")
                for response_type, count in self.results[model_key]["responses"].items():
                    percentage = (count / total) * 100 if total > 0 else 0
                    print(f"  - {response_type}: {count} ({percentage:.1f}%)")
                print("-" * 50)

        # Analyse des accords unanimes
        unanimous_agreements = []
        unanimous_correct = 0
        
        for result in self.results["detailed_results"]:
            responses = result["model_responses"]
            # Vérifier si tous les modèles sont d'accord
            if len(set(responses.values())) == 1 and "error" not in responses.values():
                unanimous_response = list(responses.values())[0]
                is_correct = unanimous_response == result["true_answer"]
                unanimous_agreements.append({
                    "question": result["question"],
                    "unanimous_response": unanimous_response,
                    "true_answer": result["true_answer"],
                    "is_correct": is_correct
                })
                if is_correct:
                    unanimous_correct += 1

        # Afficher les statistiques des accords unanimes
        print("\n=== Analyse des Accords Unanimes ===")
        print(f"Nombre total d'accords unanimes: {len(unanimous_agreements)}")
        if unanimous_agreements:
            accuracy = (unanimous_correct / len(unanimous_agreements)) * 100
            print(f"Précision des accords unanimes: {accuracy:.1f}%")
            
            print("\nDétail des accords unanimes incorrects:")
            for agreement in unanimous_agreements:
                if not agreement["is_correct"]:
                    print(f"\nQuestion: {agreement['question']}")
                    print(f"Réponse unanime: {agreement['unanimous_response']}")
                    print(f"Vraie réponse: {agreement['true_answer']}")

        # Taux d'accord général
        if self.results["total_tested"] > 0:
            agreement_rate = (self.results["agreement"] / self.results["total_tested"]) * 100
            print(f"\nTaux d'accord entre les modèles: {agreement_rate:.1f}%")
            print(f"(Sur {self.results['agreement']} exemples)")

if __name__ == "__main__":
    import asyncio
    
    async def main():
        print(f"Délai entre les requêtes: {REQUEST_DELAY} secondes")
        start_time = time.time()
        
        tester = ModelTester()
        await tester.run_tests("training_data_en_filtered.jsonl")
        
        # Sauvegarder les résultats en JSON
        tester.save_results("detailed_results.json")
        
        # Afficher les statistiques globales
        tester.print_results()
        
        end_time = time.time()
        total_time = end_time - start_time
        print(f"\nTemps total d'exécution: {total_time:.2f} secondes")
        print(f"Les résultats ont été sauvegardés dans 'detailed_results.json'")

    asyncio.run(main()) 