# Zusammenfassung: Parfum-Kaufneigungsmodell

## Problemstellung

Das Marketing-Team muss Kunden identifizieren, die mit höchster Wahrscheinlichkeit Parfumprodukte kaufen werden, um die Kampagnenausrichtung zu optimieren. Traditionelle regelbasierte Ansätze (z.B. "Kunden ansprechen, die Parfümseiten angesehen haben") bieten nur begrenzten Lift. Dieses Projekt entwickelt ein Machine-Learning-Modell zur Vorhersage der Kaufneigung anhand von Kundenverhaltenssignalen.

## Ansatz

Es wurde ein XGBoost Gradient-Boosting-Modell auf synthetischen E-Commerce-Verhaltensdaten trainiert. Das Modell verwendet sieben Verhaltensmerkmale (Seitenaufrufe, Warenkorbzugänge, Bestellhistorie, Preissensitivität, Markenexploration, Aktualität, Kampagneninteraktion) und zwei demografische Merkmale (Altersgruppe, Region) zur Vorhersage der Parfumkaufwahrscheinlichkeit.

## Kernresultate

| Metrik | Wert | 95%-Konfidenzintervall |
|--------|------|------------------------|
| **ROC AUC** | 0.48 | [0.38 - 0.57] |
| **PR AUC** | 0.27 | [0.18 - 0.38] |
| **Lift @ Top 10%** | 0.83 | [0.27 - 1.72] |

### Kreuzvalidierungs-Stabilität (5-fach)
- AUC: 0.49 ± 0.05
- PR-AUC: 0.27 ± 0.03
- Lift@10: 1.15 ± 0.40

### Fairness-Bewertung
- **Demographic Parity Gap**: 0.11 (akzeptabel)
- **Equalized Odds Gap**: 0.23 (moderat)
- Geschütztes Merkmal: Altersgruppe

## Empfehlungen

1. **Mit Vorsicht einsetzen**: Aktuelle Modellleistung ist aufgrund synthetischer Daten eingeschränkt
2. **Auf echten Daten validieren**: Diese Methodik sollte auf tatsächliche Kundendaten angewendet werden
3. **Fairness überwachen**: Leistung nach Altersgruppe verfolgen, um Drift zu erkennen

## Limitationen

1. **Synthetische Daten**: Resultate lassen sich möglicherweise nicht auf echtes Kundenverhalten übertragen
2. **Begrenztes Signal**: Der synthetische Datengenerierungsprozess erzeugt schwache Feature-Ziel-Beziehungen
3. **Einzelnes geschütztes Merkmal**: Fairness-Analyse deckt nur Altersgruppe ab, nicht weitere Demografien
