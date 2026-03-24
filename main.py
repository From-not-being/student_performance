import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt
from tqdm import tqdm

# Pfad zu deiner CSV
path = "/content/drive/MyDrive/Colab Notebooks/student_performance.csv"

def transformer_ablation_study_google_based(file_path):
    # 1. Daten laden
    df = pd.read_csv(file_path)
    
    # 2. Modell: DistilBERT (Basiert auf Google BERT, optimiert von Hugging Face/Typeform)
    # Nicht von Microsoft, nicht von Facebook.
    # 'device=0' nutzt die GPU für maximale Geschwindigkeit bei allen Daten
    model_name = "typeform/distilbert-base-uncased-mnli"
    classifier = pipeline("zero-shot-classification", model=model_name, device=0)

    impact_scores = {'Hours Studied': 0, 'Sample Question Papers Practiced': 0, 'Sleep Hours': 0}

    print(f"Nutze Google-basiertes Modell: {model_name}")
    print(f"Analysiere den kompletten Datensatz ({len(df)} Zeilen)...")

    # 3. Iteration über alle Daten
    for index, row in tqdm(df.iterrows(), total=len(df)):
        # Baseline Kontext
        base_ctx = (f"Student studied {row['Hours Studied']} hours, "
                    f"did {row['Sample Question Papers Practiced']} papers, "
                    f"slept {row['Sleep Hours']} hours.")
        
        # KI-Vorhersage für das volle Paket
        base_score = classifier(base_ctx, candidate_labels=["high performance"])['scores'][0]

        # Ablation 1: Stunden entfernen
        ctx_no_hours = f"Did {row['Sample Question Papers Practiced']} papers, slept {row['Sleep Hours']} hours."
        score_no_hours = classifier(ctx_no_hours, candidate_labels=["high performance"])['scores'][0]
        impact_scores['Hours Studied'] += abs(base_score - score_no_hours)

        # Ablation 2: Übungen entfernen
        ctx_no_papers = f"Studied {row['Hours Studied']} hours, slept {row['Sleep Hours']} hours."
        score_no_papers = classifier(ctx_no_papers, candidate_labels=["high performance"])['scores'][0]
        impact_scores['Sample Question Papers Practiced'] += abs(base_score - score_no_papers)

        # Ablation 3: Schlaf entfernen
        ctx_no_sleep = f"Studied {row['Hours Studied']} hours, did {row['Sample Question Papers Practiced']} papers."
        score_no_sleep = classifier(ctx_no_sleep, candidate_labels=["high performance"])['scores'][0]
        impact_scores['Sleep Hours'] += abs(base_score - score_no_sleep)

    # 4. Ergebnisse in Prozent umrechnen
    total_impact = sum(impact_scores.values())
    features = list(impact_scores.keys())
    prozente = [(impact_scores[f] / total_impact) * 100 for f in features]

    # 5. Visualisierung mit Matplotlib
    plt.figure(figsize=(12, 6))
    # Wir nutzen ein kühleres Farbschema (Google-ähnlich)
    colors = ['#4285F4', '#34A853', '#FBBC05'] 
    
    bars = plt.bar(features, prozente, color=colors, edgecolor='black', alpha=0.9)

    plt.title(f'Feature Importance Analyse\n(Modell: {model_name})', fontsize=14, pad=20)
    plt.ylabel('Einfluss auf die Entscheidung (%)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Prozentwerte oben auf die Balken schreiben
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f'{yval:.2f}%', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.show()

# Ausführen
transformer_ablation_study_google_based(path)
