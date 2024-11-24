import json
from typing import List, Dict
from pathlib import Path
import argparse

def clean_response(output: str) -> str:
    """
    Nettoie et normalise la réponse
    
    Args:
        output: La réponse à nettoyer
        
    Returns:
        'yes', 'no' ou None
    """
    output = output.lower().strip()
    
    # Cas simple pour "yes"
    if output == "yes":
        return "yes"
    
    # Ignorer les "not sure"
    if "not sure" in output:
        return None
    
    # Cas pour "no" avec explication
    if output.startswith("no"):
        return "no"
        
    return None

def filter_yes_no_answers(input_file: str, output_file: str, max_yes: int = 0, max_no: int = 0) -> None:
    """
    Filtre le fichier JSONL pour ne garder que les exemples avec des réponses 'yes' ou 'no'
    
    Args:
        input_file: Chemin du fichier d'entrée
        output_file: Chemin du fichier de sortie
        max_yes: Nombre maximum de réponses 'yes' à garder (0 = illimité)
        max_no: Nombre maximum de réponses 'no' à garder (0 = illimité)
    """
    filtered_data: List[Dict] = []
    total_count = 0
    yes_count = 0
    no_count = 0
    
    yes_entries = []
    no_entries = []
    
    # Lecture du fichier d'entrée
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            total_count += 1
            entry = json.loads(line)
            
            # Nettoyer et vérifier la réponse
            clean_output = clean_response(entry["output"])
            if clean_output:
                # Créer une nouvelle entrée avec la réponse simplifiée
                new_entry = entry.copy()
                new_entry["output"] = clean_output
                
                # Séparer les entrées yes et no
                if clean_output == "yes":
                    yes_entries.append(new_entry)
                else:
                    no_entries.append(new_entry)
    
    # Limiter le nombre d'entrées si nécessaire
    if max_yes > 0:
        yes_entries = yes_entries[:max_yes]
    if max_no > 0:
        no_entries = no_entries[:max_no]
    
    # Combiner les entrées
    filtered_data = yes_entries + no_entries
    yes_count = len(yes_entries)
    no_count = len(no_entries)
    
    # Sauvegarde des données filtrées
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in filtered_data:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')
    
    # Afficher les statistiques
    print(f"Statistiques de filtrage:")
    print(f"Total d'exemples dans le fichier original: {total_count}")
    print(f"Exemples conservés: {len(filtered_data)}")
    print(f"  - Réponses 'yes': {yes_count}")
    print(f"  - Réponses 'no': {no_count}")
    print(f"Pourcentage conservé: {(len(filtered_data)/total_count)*100:.1f}%")
    
    # Afficher quelques exemples
    print("\nExemples de données conservées:")
    if yes_entries:
        print("\nExemple de 'yes':")
        print(f"Question: {yes_entries[0]['input']}")
        print(f"Réponse: {yes_entries[0]['output']}")
    
    if no_entries:
        print("\nExemple de 'no':")
        print(f"Question: {no_entries[0]['input']}")
        print(f"Réponse: {no_entries[0]['output']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Filtre les questions avec réponses yes/no')
    parser.add_argument('--input', default="training_data_en.jsonl", help='Fichier d\'entrée')
    parser.add_argument('--output', default="training_data_en_filtered.jsonl", help='Fichier de sortie')
    parser.add_argument('--max_yes', type=int, default=0, help='Nombre maximum de réponses yes (0 = illimité)')
    parser.add_argument('--max_no', type=int, default=0, help='Nombre maximum de réponses no (0 = illimité)')
    
    args = parser.parse_args()
    
    # Vérifier si le fichier d'entrée existe
    if not Path(args.input).exists():
        print(f"Erreur: Le fichier {args.input} n'existe pas!")
        exit(1)
    
    # Filtrer les données
    filter_yes_no_answers(args.input, args.output, args.max_yes, args.max_no)
    
    # Confirmer la création du nouveau fichier
    if Path(args.output).exists():
        print(f"\nFichier filtré créé avec succès: {args.output}") 