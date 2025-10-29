# Future of Fragrance — Propensity Modeling (FHNW Machine Learning Projekt)

![CI](https://github.com/kendricscoles/ml-future-fragrance/actions/workflows/ci.yml/badge.svg)

## Projektbeschreibung

Dieses Projekt wurde im Rahmen des Moduls *Machine Learning* im Studiengang *Business Artificial Intelligence* an der FHNW entwickelt.  
Ziel ist es, die Kaufwahrscheinlichkeit (Propensity) für Parfümprodukte auf Basis synthetischer E-Commerce-Daten zu prognostizieren.  
Das Projekt zeigt, wie Machine Learning im Marketing eingesetzt werden kann, um potenzielle Käufer zu identifizieren und Kampagnen effizienter zu gestalten.

---

## Projektumfang

- Generierung synthetischer Kundendaten  
- Modelltraining mit **XGBoost** und Preprocessing-Pipeline  
- Evaluation mit **ROC-AUC**, **PR-AUC** und **Lift-Werten**  
- Modellinterpretation mit **SHAP-Werten**  
- Fairness-Analyse nach Altersgruppen  
- Vollständige Reproduzierbarkeit über **GitHub Actions (CI)**

---

## Repository Structure

```

.github/workflows/      # CI pipeline (smoke test on each push)
artifacts/              # Placeholder model outputs (for grading)
├── champion_model.pkl
├── metrics.json
├── predictions.csv
data/                   # Synthetic dataset
├── fragrance_data.csv
reports/                # Evaluation results & figures
├── fairness_age_group.csv
├── lift_by_decile.csv
├── metrics_summary.csv
├── shap_top_features.csv
└── figures/
    ├── lift_curve.png
    ├── pr_curve.png
    ├── roc_curve.png
    ├── shap_dependence_top.png
    ├── shap_summary_bar.png
    └── shap_summary_beeswarm.png
slides/                 
src/                    # Source code for data prep, training, evaluation
├── __init__.py
├── config.py
├── data_prep.py
├── evaluate.py
├── explain.py
├── export_targets.py
├── generate_pngs.py
├── make_predictions.py
├── metrics.py
├── score.py
├── train_tune.py
└── train.py
config.yaml             # Central configuration (seed, paths, model params)
requirements.txt       
Makefile              
LICENSE               
README.md              
```
> **Hinweis:**  
> Die Dateien in den Verzeichnissen `/artifacts` und `/reports/figures` sind **Beispiel-Platzhalter**, die die **Ausgabestruktur für die Bewertung** demonstrieren.

> Sie werden **automatisch beim Ausführen der Pipeline** generiert, bleiben aber **aus Gründen der Transparenz** im Repository erhalten.

---

## Ausführung

Die Pipeline kann in jeder Python-3.11-Umgebung ausgeführt werden:

```bash
# Virtuelle Umgebung aktivieren
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install pytest

# Daten generieren
python src/data_prep.py --rows 800 --seed 42 --out data/fragrance_data.csv

# Modell trainieren
python src/train.py --data data/fragrance_data.csv --out_dir artifacts --estimator xgb

# Vorhersagen erzeugen
python src/make_predictions.py

# Evaluation und Reports
python src/evaluate.py --pred artifacts/predictions.csv --data data/fragrance_data.csv --outdir reports

# PNG-Plots erzeugen (ROC, PR, Lift, SHAP)
python src/generate_pngs.py

---

## Run Modes

### Local (full)
```bash
python src/data_prep.py --rows 20000 --seed 42 --out data/fragrance_data.csv
python src/train.py --data data/fragrance_data.csv --out_dir artifacts --estimator xgb
python src/make_predictions.py --data data/fragrance_data.csv --model artifacts/champion_model.pkl --out artifacts/predictions.csv
python src/evaluate.py --pred artifacts/predictions.csv --data data/fragrance_data.csv --outdir reports
python src/generate_pngs.py --pred artifacts/predictions.csv --outdir reports/figures
```

---

## Ergebnisse und Metriken

| Metrik | Wert | Erklärung |
|--------|------|-----------|
| **ROC-AUC** | 0.8997 | Misst die Trennschärfe des Modells zwischen Käufern und Nicht-Käufern. 1.0 bedeutet perfekte Trennung, 0.5 entspricht Zufall. |
| **PR-AUC** | 0.8547 | Zeigt die Modellgüte bei unausgeglichenen Klassen. Fokus auf Präzision und Recall (wichtiger bei seltenen Käufen). |
| **Lift @ 10%** | 4.12 × Baseline | Käuferanteil in den obersten 10 % der Prognosen ist 4,12 × höher als im Gesamtdurchschnitt. |
| **Lift @ 20%** | 3.88 × Baseline | Käuferanteil in den obersten 20 % der Prognosen. |
| **Lift @ 30%** | 2.76 × Baseline | Käuferanteil in den obersten 30 % der Prognosen. |

Die wichtigsten Merkmale laut SHAP-Analyse:

1. days_since_last_purchase — Zeit seit dem letzten Kauf  
2. avg_price_viewed — Durchschnittlicher Preis der angesehenen Produkte  
3. add_to_cart_30d — Anzahl „Add to Cart“-Aktionen in 30 Tagen  
4. orders_12m — Anzahl Bestellungen im letzten Jahr  
5. views_7d — Seitenaufrufe in den letzten 7 Tagen

---

### Baselines & Kalibrierung
- **Dummy Classifier (AUC ≈ 0.50)** → Referenz-Zufallsniveau  
- **Logistische Regression (AUC ≈ 0.78)** → Klassische Baseline  
- **XGBoost (Finalmodell, AUC ≈ 0.90)** → +0.12 Verbesserung gegenüber der Baseline  
- **Kalibrierung:** Modellwahrscheinlichkeiten mit Brier Score ≈ 0.17 getestet → Gut kalibriert.

---

## Visualisierungen

- **ROC-Kurve** – Zeigt das Verhältnis von True-Positive-Rate zu False-Positive-Rate  
  <img width="1024" height="768" alt="roc_curve" src="https://github.com/user-attachments/assets/4b3f0b76-e9a4-4492-9fd4-bd5b662589d5" />

- **Precision-Recall-Kurve** – Zeigt die Genauigkeit (Precision) im Verhältnis zur Abdeckung (Recall)  
  <img width="1024" height="768" alt="pr_curve" src="https://github.com/user-attachments/assets/08871840-44dd-4ff0-84c1-efc601359074" />

- **Lift-Kurve** – Visualisiert die Effektivität der Zielgruppenansprache  
  <img width="1024" height="768" alt="lift_curve" src="https://github.com/user-attachments/assets/21dae666-f2fd-4cae-a3c7-64fb508d2409" />

- **SHAP Beeswarm Plot** – Zeigt den Einfluss einzelner Features auf Modellvorhersagen  
  <img width="1559" height="1640" alt="shap_summary_beeswarm" src="https://github.com/user-attachments/assets/9892eeca-588e-4f3c-a016-9c856a73a8d5" />

- **SHAP Bar Plot** – Rangiert die wichtigsten Einflussfaktoren nach Gesamtbedeutung  
  <img width="1580" height="1640" alt="shap_summary_bar" src="https://github.com/user-attachments/assets/cf9601e0-27c7-490c-b086-2f0312043fa5" />

- **SHAP Dependence Plot** – Zeigt den Effekt eines spezifischen Merkmals auf die Modellvorhersage  
  <img width="1179" height="980" alt="shap_dependence_top" src="https://github.com/user-attachments/assets/36eba174-4c76-4811-bb03-d5fbe8b9e488" />

(Siehe `reports/figures/` für alle PNG-Dateien.)

---

## Reproduzierbarkeit

- Abhängigkeiten: `requirements.txt`  
- Automatisierte Tests: `.github/workflows/ci.yml` (GitHub Actions)  
- Alle Ergebnisse sind vollständig reproduzierbar, sofern die Skripte in der angegebenen Reihenfolge ausgeführt werden.

---


### Continuous Integration
Dieses Repository führt bei jedem Push eine automatisierte **Smoke-Test-Pipeline** aus.  
Der Workflow installiert alle Abhängigkeiten, generiert **synthetische Daten**, trainiert das Modell und bewertet die Performance **End-to-End**.

Siehe die Workflow-Datei: [`ci.yml`](.github/workflows/ci.yml)


---

## Ethische Hinweise

Dieses Projekt verwendet ausschliesslich synthetische Daten und enthält keine echten Kundendaten.  
Alle Resultate dienen ausschliesslich Demonstrations- und Lehrzwecken und enthalten keine personenbezogenen Informationen.
