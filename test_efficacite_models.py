import json
import os
from typing import Dict, List
# import google.generativeai as genai  # commenté
from openai import OpenAI
from groq import Groq
from tqdm import tqdm
from config import API_KEYS
import asyncio
import time

# Configuration des clés API
OPENAI_API_KEY = API_KEYS["OPENAI_API_KEY"]
# GOOGLE_API_KEY = API_KEYS["GOOGLE_API_KEY"]  # commenté
GROQ_API_KEY = API_KEYS["GROQ_API_KEY"]

# Délai entre les requêtes (en secondes)
REQUEST_DELAY = 2

class ModelTester:
    def __init__(self):
        # Initialisation des clients API
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
        # Commenter Gemini
        # genai.configure(api_key=GOOGLE_API_KEY)
        # self.gemini_model = genai.GenerativeModel('gemini-pro')
        self.groq_client = Groq(api_key=GROQ_API_KEY)
        
        # Pour stocker les résultats
        self.results = {
            "gpt": {"correct": 0, "total": 0, "responses": {"yes": 0, "no": 0, "not_sure": 0}},
            # "gemini": {"correct": 0, "total": 0, "responses": {"yes": 0, "no": 0, "not_sure": 0}},  # commenté
            "groq": {"correct": 0, "total": 0, "responses": {"yes": 0, "no": 0, "not_sure": 0}},
            "agreement": 0,
            "total_tested": 0,
            "detailed_results": [],
            "agreement_analysis": {
                "total_agreements": 0,
                "correct_agreements": 0,
                "agreement_cases": []
            }
        }
        
        # Pour gérer les délais entre requêtes
        self.last_request_time = {
            "gpt": 0,
            # "gemini": 0,  # commenté
            "groq": 2
        }

        # Ajout des paramètres de retry
        self.max_retries = 3
        self.retry_delay = 10  # délai de 10 secondes entre les retries

    async def wait_for_rate_limit(self, model: str):
        """Attend le délai nécessaire entre les requêtes"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time[model]
        
        if time_since_last_request < REQUEST_DELAY:
            wait_time = REQUEST_DELAY - time_since_last_request
            await asyncio.sleep(wait_time)
        
        self.last_request_time[model] = time.time()

    async def retry_query(self, model: str, query_func, instruction: str, input_text: str) -> str:
        """
        Réessaie une requête en cas d'erreur avec un délai d'attente
        """
        for attempt in range(self.max_retries):
            try:
                response = await query_func(instruction, input_text)
                if response != "error":
                    return response
                
                print(f"\n{model.upper()} Error (attempt {attempt + 1}/{self.max_retries})")
                if attempt < self.max_retries - 1:  # Ne pas attendre après la dernière tentative
                    print(f"Waiting {self.retry_delay} seconds before retrying...")
                    await asyncio.sleep(self.retry_delay)
                
            except Exception as e:
                print(f"\n{model.upper()} Error (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    print(f"Waiting {self.retry_delay} seconds before retrying...")
                    await asyncio.sleep(self.retry_delay)
        
        return "error"

    async def query_gpt(self, instruction: str, input_text: str) -> str:
        await self.wait_for_rate_limit("gpt")
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{
                    "role": "user",
                    "content": f"{instruction}\n{input_text}\nReply with 'yes', 'no', or 'not sure'."
                }],
                temperature=0
            )
            return self.clean_response(response.choices[0].message.content)
        except Exception as e:
            return "error"

    async def query_gemini(self, instruction: str, input_text: str) -> str:
        await self.wait_for_rate_limit("gemini")
        try:
            response = self.gemini_model.generate_content(
                f"{instruction}\n{input_text}\nReply with 'yes', 'no', or 'not sure'."
            )
            return self.clean_response(response.text)
        except Exception as e:
            return "error"

    async def query_groq(self, instruction: str, input_text: str) -> str:
        await self.wait_for_rate_limit("groq")
        try:
            response = self.groq_client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[{
                    "role": "user",
                    "content": f"{instruction}\n{input_text}\nReply with 'yes', 'no', or 'not sure'."
                }],
                temperature=0
            )
            return self.clean_response(response.choices[0].message.content)
        except Exception as e:
            return "error"

    def clean_response(self, response: str) -> str:
        """Normalise la réponse en 'yes', 'no' ou 'not sure'"""
        response = response.lower().strip()
        if "yes" in response:
            return "yes"
        elif "no" in response:
            return "no"
        return "not_sure"

    def check_agreement(self, responses: Dict[str, str]) -> bool:
        """Vérifie si tous les modèles sont d'accord"""
        valid_responses = [r for r in responses.values() if r != "error"]
        return len(valid_responses) > 1 and len(set(valid_responses)) == 1

    async def test_single_example(self, example: Dict) -> None:
        """Teste un seul exemple sur tous les modèles"""
        expected = example["output"].lower().strip()
        
        self.results["total_tested"] += 1

        responses = {
            "gpt": await self.retry_query("gpt", self.query_gpt, example["instruction"], example["input"]),
            # "gemini": await self.retry_query("gemini", self.query_gemini, example["instruction"], example["input"]),  # commenté
            "groq": await self.retry_query("groq", self.query_groq, example["instruction"], example["input"])
        }

        # Stocker les résultats détaillés
        detailed_result = {
            "question": example["input"],
            "instruction": example["instruction"],
            "true_answer": expected,
            "model_responses": responses
        }
        self.results["detailed_results"].append(detailed_result)

        # Analyser l'accord entre les modèles
        if self.check_agreement(responses):
            self.results["agreement"] += 1
            unanimous_response = next(r for r in responses.values() if r != "error")
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
        for model, response in responses.items():
            if response != "error":
                self.results[model]["total"] += 1
                self.results[model]["responses"][response] += 1
                if response in ["yes", "no"] and response == expected:
                    self.results[model]["correct"] += 1

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
        
        for model in ["gpt", "groq"]:
            total = self.results[model]["total"]
            if total > 0:
                accuracy = (self.results[model]["correct"] / total) * 100
                print(f"\n{model.upper()}:")
                print(f"Précision: {accuracy:.1f}%")
                print(f"Nombre total de réponses: {total}")
                print("Distribution des réponses:")
                for response_type, count in self.results[model]["responses"].items():
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