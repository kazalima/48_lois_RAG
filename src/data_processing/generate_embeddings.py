from sentence_transformers import SentenceTransformer
import json
import os
import yaml
import math

def generate_embeddings(input_path, output_dir, model_name="all-MiniLM-L6-v2", num_parts=6):
    """
    Génère des embeddings pour le texte brut et les sauvegarde dans plusieurs fichiers JSON.

    Args:
        input_path (str): Chemin vers le fichier texte brut.
        output_dir (str): Répertoire où sauvegarder les fichiers JSON.
        model_name (str): Nom du modèle sentence-transformers.
        num_parts (int): Nombre de fichiers JSON à générer.
    """
    # Vérifier si le fichier texte existe
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Le fichier {input_path} n'existe pas.")

    # Charger le texte
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Diviser le texte en phrases
    segments = text.split('. ')
    segments = [s.strip() for s in segments if s.strip()]

    # Calculer la taille de chaque partie
    total_segments = len(segments)
    segments_per_part = math.ceil(total_segments / num_parts)

    # Charger le modèle
    model = SentenceTransformer(model_name)

    # Générer les embeddings
    embeddings = model.encode(segments, show_progress_bar=True)

    # Créer le répertoire de sortie
    os.makedirs(output_dir, exist_ok=True)

    # Diviser et sauvegarder les embeddings
    for part in range(num_parts):
        start_idx = part * segments_per_part
        end_idx = min((part + 1) * segments_per_part, total_segments)
        part_segments = segments[start_idx:end_idx]
        part_embeddings = embeddings[start_idx:end_idx]

        # Créer les données pour cette partie
        data = [{"text": segment, "embedding": embedding.tolist()}
                for segment, embedding in zip(part_segments, part_embeddings)]

        # Sauvegarder dans un fichier JSON
        output_path = os.path.join(output_dir, f"embeddings_part_{part + 1}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"Embeddings pour la partie {part + 1} sauvegardés dans {output_path}")

    print(f"Total de {total_segments} segments répartis en {num_parts} fichiers.")

if __name__ == "__main__":
    # Charger la configuration
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    input_path = cfg["data"]["text"]
    output_dir = cfg["data"]["embeddings_dir"]
    model_name = cfg["models"]["embedding"]
    num_parts = cfg["retrieval"]["num_embedding_parts"]
    generate_embeddings(input_path, output_dir, model_name, num_parts)
