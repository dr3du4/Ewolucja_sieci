# Ewolucja sieci - symulacja UAV-ISAC

Symulacja systemu ISAC (Integrated Sensing and Communication) z wykorzystaniem dronów (UAV). Projekt porównuje scenariusze z 1 i 2 UAV pod kątem trade-off między komunikacją a sensowaniem.

## Uruchomienie

```bash
pip install -r requirements.txt
python main.py
```

Wyniki (wykresy PNG + dane CSV) zapisują się w folderze `results/`.

## Struktura projektu

- `uav_isac/channel_model.py` - model kanału air-to-ground (1/d²)
- `uav_isac/scenario.py` - generator scenariuszy z użytkownikami i celami
- `uav_isac/simulation.py` - silnik symulacji ISAC
- `uav_isac/sensing.py` - proxy sensingu radarowego (CRB)
- `uav_isac/visualization.py` - generowanie wykresów Pareto, trajektorii, porównań
- `main.py` - punkt wejścia, uruchamia oba scenariusze
