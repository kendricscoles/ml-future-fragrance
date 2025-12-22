# Offenlegung der KI-Werkzeug-Nutzung (FHNW-Konformität)

## Projekt: ML Future Fragrance - Marketing-Neigungsmodell
**Autor**: Kendric Scoles  
**Studiengang**: Bachelor Business AI, FHNW Olten  
**Modul**: Machine Learning  
**Datum**: Dezember 2025

---

## Verwendete KI-Werkzeuge

### Entwicklungsunterstützung
- **Werkzeug**: GitHub Copilot / Claude (Anthropic)
- **Zweck**: Unterstützung bei der Code-Generierung für Vorlagencode, Debugging und Grammatikkorrekturen

### Nutzungsübersicht

| Aufgabe | KI-Beteiligung | Menschliche Verifikation |
|---------|----------------|-------------------------|
| Datengenerierung (`data_prep.py`) | Unterstützt | Geprüft & modifiziert |
| Modelltraining (`train.py`) | Unterstützt | Geprüft & getestet |
| Evaluationspipeline | Unterstützt | Outputs verifiziert |
| Fairness-Analyse | Unterstützt | Methodik validiert |
| Dokumentation | Grammatik & Wortwahl-Hilfe | Bearbeitet & verifiziert |

## Angewandtes kritisches Denken

### Code-Review
- Sämtlicher KI-generierter Code wurde auf Korrektheit geprüft
- Imports und Abhängigkeiten wurden verifiziert
- Grenzfälle wurden manuell getestet

### Methodologische Validierung
- Bootstrap-Konfidenzintervalle: Mathematische Korrektheit verifiziert
- Fairness-Metriken: Definitionen mit akademischen Quellen abgeglichen
- Kreuzvalidierung: Standard-Machine-Learning-Praxis bestätigt

### Output-Verifikation
- Modell-Outputs mit erwarteten Bereichen verglichen
- Metriken gegen sklearn-Implementierungen validiert
- Pipeline-Reproduzierbarkeit getestet (mehrere Durchläufe)

## Bestätigung

Ich bestätige hiermit:
1. Ich verstehe sämtlichen Code in diesem Projekt
2. KI-Werkzeuge wurden als Coding-Assistenten genutzt, nicht als Ersatz für Problemlösung
3. Alle kritischen Entscheidungen wurden von mir nach Prüfung getroffen

---

*Diese Offenlegung folgt den FHNW-Richtlinien für die Nutzung von KI-Werkzeugen in akademischen Projekten.*
