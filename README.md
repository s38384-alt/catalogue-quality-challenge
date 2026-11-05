# Catalogue Quality Challenge — DSC PJATK

**Grupa:** Team 2 
**Skład:** Piotr Egeman, Maciej Gruza
**Wynik Kaggle:** 0.97256 ROC-AUC, 0.96594 ROC-AUC


## Opis zadania

Przewidujemy prawdopodobieństwo, że listing zawiera symulowany błąd (target=1) w syntetycznym katalogu produktów (16 000 listingów treningowych, 10 000 testowych, 9 kategorii, 5 marek). Metryka: ROC-AUC.

## Wyniki

| Konfiguracja | OOF AUC | LB |
|---|---|---|
| finalna | 0.9688 | **0.97256** |
| pierwsza | 0.9683 | 0.96594 |

## Pliki

- `02_model.ipynb` — pipeline modelowania z opisem metod i wynikami CV

## Reprodukcja

Otwórz `02_model.ipynb` na Kaggle (dane dołączone jako input) i uruchom **Run All**.

## Wymagania

Python 3.13, numpy 2.1.3, pandas 2.3.3, scikit-learn 1.6.1, lightgbm 4.6.0.

## Licencja

MIT
