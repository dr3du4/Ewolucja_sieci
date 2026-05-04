# Ewolucja sieci - symulacja UAV-ISAC

Symulacja systemu ISAC (Integrated Sensing and Communication) z wykorzystaniem dronów (UAV). Projekt porównuje scenariusze z 1 i 2 UAV pod kątem trade-off między komunikacją a sensowaniem.

## Scenariusze

### Wspólne założenia

Oba scenariusze są uruchamiane na identycznym środowisku, żeby porównanie 1 vs 2 UAV było uczciwe:

- **Obszar**: kwadrat 500 × 500 m
- **Użytkownicy naziemne**: 4, statyczni, pozycje losowe (z=0, seed=42)
- **Cele radarowe**: 2, statyczne, pozycje losowe (z=0, seed=42)
- **Misja**: 60 s, krok czasowy `dt = 1 s` (60 kroków symulacji)
- **UAV**: wysokość 100 m, prędkość maks. 20 m/s
- **Kanał komunikacyjny** (`ChannelParams`): 2 GHz, moc Tx 30 dBm, pasmo 10 MHz, model air-to-ground Al-Hourani et al. (GLOBECOM 2014)
- **Sensing** (`SensingParams`): 8 anten, 64 impulsy, RCS = 1.0; metryką jakości jest CRB RMSE estymacji odległości
- **Polityka trajektorii**: `greedy` — w każdym kroku każdy UAV leci w stronę punktu `(1-α)·centroid_najbliższych_userów + α·najbliższy_cel`, gdzie α to udział mocy przeznaczony na sensing
- **Sweep Pareto**: α ∈ {0.1, 0.2, …, 0.9} (9 punktów); α = 0 → cała moc na komunikację, α = 1 → cała moc na sensing

### Scenariusz 1 — 1 UAV

Jeden dron startuje na środku obszaru (250, 250, 100) i musi sam obsłużyć wszystkich 4 użytkowników oraz estymować pozycje obu celów. W polityce greedy całe stado użytkowników jest przypisane do tego jednego drona (`n_per_uav = 4`), więc komunikacyjnym celem ruchu jest centroid wszystkich userów. Sensingowym celem jest najbliższy z dwóch celów radarowych. Wagą kompromisu jest α — przy małym α dron leci ku userom (wysoki rate, słabe CRB), przy dużym α „przykleja się” do najbliższego celu (niski rate, dobre CRB). Ten scenariusz to baseline pokazujący limit pojedynczego węzła ISAC.

### Scenariusz 2 — 2 UAV

Dwa drony rozstawione w narożach obszaru (~(100, 100, 100) i ~(400, 100, 100), 20% margines). Polityka greedy partycjonuje userów: każdy UAV ciągnie do centroidu **swoich 2 najbliższych** userów (`n_per_uav = 4 // 2 = 2`) oraz do swojego najbliższego celu. Skutkiem jest:

- krótszy średni dystans UAV–user → mniejsze straty propagacyjne → wyższy rate przy tym samym α,
- równoległa obsługa dwóch celów radarowych → lepsze CRB nawet przy niskim α,
- naturalny podział obszaru bez explicite kodowanej koordynacji.

Porównanie krzywych Pareto z wariantem 1 UAV pokazuje, ile faktycznie zyskuje system z drugiego drona — i przy jakich α ten zysk jest największy.

## Wyniki

Skrypt generuje wykresy w folderze `results/`:
- krzywe Pareto (trade-off rate vs CRB) dla każdego scenariusza
- porównanie Pareto 1 UAV vs 2 UAV
- serie czasowe rate i CRB
- mapy trajektorii dronów z pozycjami userów i celów
- wykresy słupkowe porównujące oba scenariusze
- dane CSV z wynikami sweepów

## Uruchomienie
``bash
pip install -r requirements.txt
python main.py``
## Struktura projektu

- `uav_isac/channel_model.py` - model kanału air-to-ground (Al-Hourani et al., GLOBECOM 2014)
- `uav_isac/scenario.py` - generator scenariuszy z użytkownikami i celami
- `uav_isac/simulation.py` - silnik symulacji ISAC
- `uav_isac/sensing.py` - proxy sensingu radarowego (CRB)
- `uav_isac/visualization.py` - generowanie wykresów
- `main.py` - punkt wejścia, uruchamia oba scenariusze