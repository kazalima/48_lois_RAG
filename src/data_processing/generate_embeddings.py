from sentence_transformers import SentenceTransformer
import json
import os
import yaml
import math
import re

def split_into_sentences(text):
    # Découpage simple par ponctuation forte
    sentences = re.split(r'(?<=[.?!])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def group_sentences(sentences, group_size=4):
    grouped = []
    for i in range(0, len(sentences), group_size):
        chunk = ' '.join(sentences[i:i+group_size])
        grouped.append(chunk)
    return grouped

def generate_embeddings(input_path, output_dir, model_name="all-MiniLM-L6-v2", num_parts=6):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Le fichier {input_path} n'existe pas.")

    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Étapes de segmentation améliorées
    sentences = split_into_sentences(text)
    segments = group_sentences(sentences, group_size=4)

    total_segments = len(segments)
    segments_per_part = math.ceil(total_segments / num_parts)

    model = SentenceTransformer(model_name)
    embeddings = model.encode(segments, show_progress_bar=True)

    os.makedirs(output_dir, exist_ok=True)

    for part in range(num_parts):
        start_idx = part * segments_per_part
        end_idx = min((part + 1) * segments_per_part, total_segments)
        part_segments = segments[start_idx:end_idx]
        part_embeddings = embeddings[start_idx:end_idx]

        data = [{"text": segment, "embedding": embedding.tolist()}
                for segment, embedding in zip(part_segments, part_embeddings)]

        output_path = os.path.join(output_dir, f"embeddings_part_{part + 1}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"✅ Embeddings partie {part + 1} sauvegardés dans {output_path}")

    print(f"✅ {total_segments} segments générés et répartis en {num_parts} fichiers.")

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    input_path = cfg["data"]["text"]
    output_dir = cfg["data"]["embeddings_dir"]
    model_name = cfg["models"]["embedding"]
    num_parts = cfg["retrieval"]["num_embedding_parts"]
    generate_embeddings(input_path, output_dir, model_name, num_parts)
