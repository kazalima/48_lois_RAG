import yaml
import PyPDF2
import re
import os

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def clean_text(text):
    # Supprimer les sauts de ligne excessifs et les espaces
    text = re.sub(r'\s+', ' ', text)

    # Supprimer les mentions d'auteur, éditeur, etc.
    text = re.sub(r'(Robert Greene|Les 48 lois du pouvoir)', '', text, flags=re.IGNORECASE)

    # Supprimer les numéros de page (ex : "Page 32" ou "page 32")
    text = re.sub(r'page\s*\d+', '', text, flags=re.IGNORECASE)

    # Supprimer les lignes composées uniquement de chiffres (ex : numéros de sections)
    text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)

    # Corriger les retours à la ligne manquants entre les titres des lois
    text = re.sub(r'(LOI\s+\d+)', r'\n\1\n', text, flags=re.IGNORECASE)

    # Supprimer les caractères non ASCII (caractères exotiques ou invisibles)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    # Nettoyage final
    text = text.strip()

    return text

def extract_text_from_pdf(config):
    pdf_path = config["data"]["pdf"]
    text_path = config["data"]["text"]

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF non trouvé à {pdf_path}")

    full_text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"

    cleaned_text = clean_text(full_text)

    os.makedirs(os.path.dirname(text_path), exist_ok=True)
    with open(text_path, "w", encoding="utf-8") as file:
        file.write(cleaned_text)

    print(f"✅ Texte nettoyé extrait et sauvegardé dans {text_path}")
    return cleaned_text

if __name__ == "__main__":
    config = load_config()
    extract_text_from_pdf(config)
