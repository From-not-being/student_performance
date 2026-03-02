import pandas as pd
from transformers import pipeline

student_performance_path = "/content/drive/MyDrive/Colab Notebooks/student_performance.csv"
df_student_performance = pd.read_csv(student_performance_path)

display(df_student_performance.head(3))



# Pfad zu deiner CSV
path = "/content/drive/MyDrive/Colab Notebooks/student_performance.csv"

def transformer_ablation_study(file_path):
    # CSV laden (Pandas nur als reiner Daten-Lieferant)
    df = pd.read_csv(file_path)

    # Transformer laden (Zero-Shot)
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    # Wir nehmen die ersten 5 Schüler für den Test (spart Rechenzeit)
    sample_df = df.head(5)

    # Hier speichern wir, wie stark das Fehlen einer Variable die KI verwirrt
    impact_scores = {'Hours Studied': 0, 'Sample Question Papers Practiced': 0, 'Sleep Hours': 0}

    print("Starte Transformer-Analyse...\n")

    for index, row in sample_df.iterrows():
        # 1. Der komplette Kontext (Baseline)
        base_ctx = (f"Student studied {row['Hours Studied']} hours, "
                    f"did {row['Sample Question Papers Practiced']} practice papers, "
                    f"slept {row['Sleep Hours']} hours.")

        base_score = classifier(base_ctx, candidate_labels=["high performance"])['scores'][0]

        # 2. Test: Was passiert, wenn wir "Hours Studied" weglassen?
        ctx_no_hours = f"Student did {row['Sample Question Papers Practiced']} practice papers, slept {row['Sleep Hours']} hours."
        score_no_hours = classifier(ctx_no_hours, candidate_labels=["high performance"])['scores'][0]
        impact_scores['Hours Studied'] += abs(base_score - score_no_hours)

        # 3. Test: Was passiert, wenn wir "Practice Papers" weglassen?
        ctx_no_papers = f"Student studied {row['Hours Studied']} hours, slept {row['Sleep Hours']} hours."
        score_no_papers = classifier(ctx_no_papers, candidate_labels=["high performance"])['scores'][0]
        impact_scores['Sample Question Papers Practiced'] += abs(base_score - score_no_papers)

        # 4. Test: Was passiert, wenn wir "Sleep Hours" weglassen?
        ctx_no_sleep = f"Student studied {row['Hours Studied']} hours, did {row['Sample Question Papers Practiced']} practice papers."
        score_no_sleep = classifier(ctx_no_sleep, candidate_labels=["high performance"])['scores'][0]
        impact_scores['Sleep Hours'] += abs(base_score - score_no_sleep)

    # 5. Die Differenzen in saubere Prozentwerte umrechnen
    total_impact = sum(impact_scores.values())

    print("=== Prozentuale Wirkung der Variablen auf die Note ===")
    if total_impact > 0:
        for feature in impact_scores:
            prozent = (impact_scores[feature] / total_impact) * 100
            print(f"-> {feature}: {prozent:.2f} %")

transformer_ablation_study(path)
