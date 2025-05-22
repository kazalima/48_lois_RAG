import yaml
import PyPDF2
import re
import os

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = re.sub(r'page \d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'Les 48 lois du pouvoir|Robert Greene', '', text, flags=re.IGNORECASE)
    return text.strip()

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

    return cleaned_text

if __name__ == "__main__":
    config = load_config()
    cleaned_text = extract_text_from_pdf(config)
    print(f"Texte extrait et sauvegardé dans {config['data']['text']}")
