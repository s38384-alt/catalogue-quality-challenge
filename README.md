# Catalogue Quality Challenge — DSC PJATK

**Grupa:** <NR>  
**Skład:** <Imię Nazwisko>, <Imię Nazwisko>  
**Wynik Kaggle:** 0.97256 ROC-AUC (widoczny w rankingu pod nazwą zespołu `<TEAM_NAME>`)

## Opis zadania

Przewidujemy prawdopodobieństwo, że listing zawiera symulowany błąd (target=1) w syntetycznym katalogu produktów (16 000 listingów treningowych, 10 000 testowych, 9 kategorii, 5 marek). Metryka: ROC-AUC.

## Wyniki

| Konfiguracja | OOF AUC | LB |
|---|---|---|
| `ref` (finalna) | 0.9688 | **0.97256** |

## Struktura repo

- `notebooks/01_eda.ipynb` — analiza danych z wizualizacjami i wnioskami
- `notebooks/02_model.ipynb` — pełny pipeline modelowania
- `src/pipeline.py` — kod produkcyjny (ekstrakcja cech, CV, inferencja)
- `src/features.py` — ekstrakcja cech (wyciągnięta z pipeline.py)
- `results/cv_ref.json` — metryki CV (per-fold + pooled)
- `results/leaderboard_screenshot.png` — potwierdzenie wyniku
- `docs/approach.md` — opis podejścia
- `docs/challenges.md` — największe wyzwania
- `presentation/presentation.pdf` — prezentacja

## Reprodukcja

```bash
pip install -r requirements.txt
python -m src.pipeline --data-dir data/ --out-dir results/
