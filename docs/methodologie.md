# Methodologie: Parfum-Kaufneigungsmodell

## 1. Feature-Engineering-Begründung

### Verhaltensmerkmale
| Feature | Beschreibung | Begründung |
|---------|-------------|------------|
| `views_7d` | Seitenaufrufe der letzten 7 Tage | Aktuelle Interaktion zeigt aktives Interesse |
| `add_to_cart_30d` | Warenkorbzugänge in 30 Tagen | Starkes Kaufabsichtssignal |
| `orders_12m` | Bestellungen der letzten 12 Monate | Basis des Kaufverhaltens |
| `avg_price_viewed` | Durchschnittspreis angesehener Produkte | Preissensitivitätsindikator |
| `brand_diversity` | Anzahl unterschiedlicher Marken | Exploration vs. fokussiertes Einkaufen |
| `days_since_last_purchase` | Tage seit letztem Kauf | Kundenlebenszyklusphase |
| `campaign_clicks` | Klicks auf E-Mail-/Werbekampagnen | Marketingreaktivität |

### Demografische Merkmale
| Feature | Beschreibung | Begründung |
|---------|-------------|------------|
| `age_group` | Altersgruppe des Kunden | Kaufmuster variieren nach Alter |
| `region` | Geografische Region | Regionale Präferenzunterschiede |

## 2. Modellauswahl-Begründung

### Warum XGBoost?

1. **Verarbeitet gemischte Feature-Typen**: Numerische und kategorische Features ohne aufwendige Vorverarbeitung
2. **Robust bei Klassenungleichgewicht**: Parameter `scale_pos_weight` adressiert seltene Käufe
3. **Interpretierbar**: Feature-Importance und SHAP-Werte ermöglichen Geschäftsverständnis
4. **Effizient**: Schnelles Training für iterative Experimente
5. **State of the Art**: Bewährte Leistung bei tabellarischen Klassifikationsaufgaben

### Hyperparameter
```yaml
n_estimators: 600
learning_rate: 0.05
max_depth: 5
subsample: 0.9
colsample_bytree: 0.9
```

## 3. Evaluationsstrategie

### Train/Test-Split
- **Methode**: Stratifizierter 75%/25%-Split unter Beibehaltung der Klassenverteilung
- **Random State**: 42 für Reproduzierbarkeit

### Primäre Metriken

| Metrik | Zweck |
|--------|-------|
| **ROC AUC** | Ranking-Fähigkeit über alle Schwellenwerte |
| **PR AUC** | Leistung bei positiver Klasse (seltenes Ereignis) |
| **Lift @ 10%** | Geschäftswert: Verbesserung gegenüber zufälligem Targeting |

### Statistische Robustheit
- **Bootstrap-Konfidenzintervalle**: 1000 Bootstrap-Stichproben für 95%-CI
- **Kreuzvalidierung**: 5-fache StratifiedKFold für Stabilitätsbewertung

## 4. Fairness-Umfang und Limitationen

### Umfang
- **Geschütztes Merkmal**: Altersgruppe (18-24, 25-34, 35-44, 45-54, 55+)
- **Berechnete Metriken**: Selection Rate, TPR, FPR, PPV pro Gruppe
- **Fairness-Definitionen**: Demographic Parity, Equalized Odds

### Limitationen

1. **Einzelnes geschütztes Merkmal**: Nur Alter wird analysiert; Geschlecht, Einkommen, Standort nicht berücksichtigt
2. **Marketing-Kontext**: Dies ist keine risikoreiche Entscheidung (Kredit, Einstellung, Gesundheit)
3. **Synthetische Daten**: Fairness-Resultate können bei echten Kundendaten abweichen
4. **Schwellenwert-Sensitivität**: Fairness-Gaps variieren je nach Klassifikationsschwellenwert

### Fairness-Genauigkeits-Tradeoff
Die Visualisierung `fairness_tradeoff.png` zeigt, wie sich Fairness-Gaps bei variierendem Schwellenwert ändern. Geschäftliche Stakeholder sollten dies bei der Auswahl operativer Schwellenwerte berücksichtigen.

## 5. Reproduzierbarkeit

### Umgebung
- Python 3.12
- Abhängigkeiten: `requirements.txt`
- Random Seed: 42 (gesetzt via Umgebungsvariable `PYTHONHASHSEED`)

### Reproduktionsschritte
```bash
# Virtuelle Umgebung erstellen
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Vollständige Pipeline ausführen
make run-all

# Mit Kreuzvalidierung ausführen
python src/train.py --data data/fragrance_data.csv --out_dir artifacts --cv-folds 5
```
