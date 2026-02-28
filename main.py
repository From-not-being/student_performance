from transformers import pipeline
import csv

# Modell laden (BART von Facebook via Hugging Face)
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

path = "/content/drive/MyDrive/Colab Notebooks/student_performance.csv"

def run_student_transformer(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i > 4: break # Testlauf für 5 Zeilen

            # Text-Repräsentation der Tabellenzeile
            # Korrigierte Spaltennamen: 'Hours Studied' und 'Sleep Hours'
            context = f"Student with {row['Hours Studied']}h study and {row['Sleep Hours']}h sleep."

            # Klassifizierung
            res = classifier(context, candidate_labels=["high performance", "low performance"])

            print(f"User {i}: {context}")
            print(f"KI-Einschätzung: {res['labels'][0]} ({res['scores'][0]:.2%})\n")

run_student_transformer(path)
