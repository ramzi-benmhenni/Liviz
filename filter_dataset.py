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
        '0' (not ok), '1' (ok) ou '2' (uncertain)
    """
    output = output.lower().strip()
    
    # Cas pour "yes" (OK)
    if output == "yes":
        return "1"
    
    # Cas pour "not sure" (Uncertain)
    if output.startswith("not sure"):
        return "2"
    
    # Cas pour "no" (Not OK)
    if output.startswith("no"):
        return "0"
        
    return None

def filter_yes_no_answers(input_file: str, output_file: str, max_yes: int = 0, max_no: int = 0, max_not_sure: int = 0) -> None:
    """
    Filtre le fichier JSONL pour ne garder que les exemples avec des réponses 'yes', 'no' ou 'not sure'
    
    Args:
        input_file: Chemin du fichier d'entrée
        output_file: Chemin du fichier de sortie
        max_yes: Nombre maximum de réponses 'yes' à garder (0 = illimité)
        max_no: Nombre maximum de réponses 'no' à garder (0 = illimité)
        max_not_sure: Nombre maximum de réponses 'not sure' à garder (0 = illimité)
    """
    filtered_data: List[Dict] = []
    total_count = 0
    yes_count = 0
    no_count = 0
    not_sure_count = 0
    
    yes_entries = []
    no_entries = []
    not_sure_entries = []
    
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
                
                # Séparer les entrées selon leur type
                if clean_output == "1":
                    yes_entries.append(new_entry)
                elif clean_output == "0":
                    no_entries.append(new_entry)
                else:  # not sure
                    not_sure_entries.append(new_entry)
    
    # Limiter le nombre d'entrées si nécessaire
    if max_yes > 0:
        yes_entries = yes_entries[:max_yes]
    if max_no > 0:
        no_entries = no_entries[:max_no]
    if max_not_sure > 0:
        not_sure_entries = not_sure_entries[:max_not_sure]
    
    # Combiner les entrées
    filtered_data = yes_entries + no_entries + not_sure_entries
    yes_count = len(yes_entries)
    no_count = len(no_entries)
    not_sure_count = len(not_sure_entries)
    
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
    print(f"  - Réponses 'not sure': {not_sure_count}")
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
        
    if not_sure_entries:
        print("\nExemple de 'not sure':")
        print(f"Question: {not_sure_entries[0]['input']}")
        print(f"Réponse: {not_sure_entries[0]['output']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Filtre les questions avec réponses yes/no/not sure')
    parser.add_argument('--input', default="training_data_en.jsonl", help='Fichier d\'entrée')
    parser.add_argument('--output', default="training_data_en_filtered.jsonl", help='Fichier de sortie')
    parser.add_argument('--max_yes', type=int, default=0, help='Nombre maximum de réponses yes (0 = illimité)')
    parser.add_argument('--max_no', type=int, default=0, help='Nombre maximum de réponses no (0 = illimité)')
    parser.add_argument('--max_not_sure', type=int, default=0, help='Nombre maximum de réponses not sure (0 = illimité)')
    
    args = parser.parse_args()
    
    # Vérifier si le fichier d'entrée existe
    if not Path(args.input).exists():
        print(f"Erreur: Le fichier {args.input} n'existe pas!")
        exit(1)
    
    # Filtrer les données
    filter_yes_no_answers(args.input, args.output, args.max_yes, args.max_no, args.max_not_sure)
    
    # Confirmer la création du nouveau fichier
    if Path(args.output).exists():
        print(f"\nFichier filtré créé avec succès: {args.output}") 