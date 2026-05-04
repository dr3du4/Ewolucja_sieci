# Cel projektu w skrócie

Celem jest zbadanie, jak liczba UAV-ów (dronów) i kooperacyjne projektowanie ich trajektorii wpływa na trade-off między komunikacją a sensingiem (efektywnością radaru) w systemach multi-UAV ISAC, w realistycznych warunkach mobilności i ograniczeń energetycznych.

Chcemy to wykonać poprzez:
- zdefiniowanie modelu systemu multi-UAV ISAC/JCAS (N dronów na wysokości H, K użytkowników naziemnych, Q celów sensingu) i zaimplementowanie go w Pythonie (NumPy/SciPy/Matplotlib);
- sformułowanie problemu optymalizacyjnego "w metrykach": np. maksymalizacja sum-rate przy ograniczeniach CRB, mocy, trajektorii i unikania kolizji (formuła ma być uniwersalna, łatwa do rozszerzenia)
- zaimplementowanie solvera (BCD/SCA) i przeprowadzenie symulacji w scenariuszu: obszar 500×500 m², 2–4 UAV-y na 100 m, 4–8 użytkowników (statycznych i mobilnych), 2–4 cele, misja 60–120 s, vmax = 20 m/s
- porównanie optymalizacji łącznej i osobnej, zbadanie wpływu liczby UAV-ów, wygenerowanie krzywych Pareto (rate vs. CRB)
- przeprowadzenie analizy wyników: agregacja danych, wykresy, statystyki, benchmarking względem baselineów

# Co nas odróżnia od literatury?

Najważniejsze wyróżniki to:
1. założenie mobilności użytkowników naziemnych - większość prac zakłada statycznych użytkowników, my testujemy zarówno statycznych, jak i mobilnych w tym samym frameworku;
2. kooperacja multi-UAV zamiast pojedynczego drona - większość literatury skupia się na single-UAV; my badamy jak liczba dronów i wspólne projektowanie trajektorii wpływa na trade-off.

# Uproszczony opis modelu systemu

## Główne elementy systemu

- **N dronów (UAV)** lecących na stałej wysokości H = 100 m nad obszarem 500 × 500 m²
- **K użytkowników naziemnych** - odbierają dane (statyczni lub mobilni)
- **Q celów sensingu** - drony muszą je wykryć i zlokalizować (mogą pokrywać się z użytkownikami)

Każdy dron pełni **dwie role naraz**: nadaje sygnał komunikacyjny do użytkowników i jednocześnie wykorzystuje ten sam sygnał jako radar (ISAC).

## Przebieg czasowy scenariusza symulacyjnego

Jedna misja trwa 60-120 s (*T* - czas misji) i jest podzielona na małe sloty czasowe $\delta t$. W każdym ze slotów:
1. Każdy UAV ma określoną pozycję $\mathbf{q}_n[t] = (x_n, y_n, H)$
2. Nadaje wiązkę (beam) z mocą $p_n[t]$ skierowaną na przypisanych użytkowników/cele
3. Odbiera echo z celów → liczy ich pozycję
4. Przesuwa się o maks. $v_{\max} \cdot \Delta t = 20 \cdot \Delta t$ metrów

## Założenia dot. kanału transmisyjnego

- **Dominuje LoS (Line-of-Sight)** - drony są wysoko, więc zakładamy bezpośrednią widoczność z użytkownikami
- **Tłumienie** zależy tylko od odległości UAV–użytkownik: 
$$
h_{n,k}[t] = \frac{\beta_0}{|\mathbf{q}_n[t] - \mathbf{u}_k|^2}
$$
gdzie $\beta_0$ to tłumienie referencyjne na 1 m, a $\mathbf{u}_k$ to pozycja użytkownika $k$.
- **Brak fading'u multipath** w pierwszej wersji (można dodać później)

## Dodatkowe, świadome uproszczenia

- **Idealna synchronizacja** między UAV-ami
- **Brak przeszkód** w obszarze (otwarta przestrzeń)
- **Doskonała znajomość pozycji** użytkowników statycznych
- **Brak jamming'u** i interferencji zewnętrznej
- **Dyskretny czas** - zamiast ciągłej trajektorii mamy ciąg punktów co $\Delta t$

Każdy UAV jednocześnie:
- nadaje dane do **users**
- "skanuje" radarem **targets**
- używa do tych celów **tej samej anteny** i tego samego sygnału (ISAC)

# Co mierzymy

| Wymiar | Metryka | Wzór (przybliżony) |
| :--: | :--: | :--: |
| Komunikacja | Rate użytkownika $k$ | $R_k[t] = \log_2(1 + \text{SINR}_k[t])$ |
| Sensing | Dokładność lokalizacji celu $q$ | min CRB (Cramér-Rao Bound, im niższe, tym lepiej) |
| Wspólny/trade-off | Energia | propulsja UAV + transmisja |

# Problem optymalizacji 

## Zmienne decyzyjne modelu

- **Trajektorie UAV-ów** - pozycje $\mathbf{q}_n[t]$ każdego drona $n$ w czasie $t$
- **Beamforming** - wektory $\mathbf{w}_n[t]$ kierujące wiązkę na użytkowników/cele
- **Alokacja mocy** - $p_n[t]$ pomiędzy komunikacją a sensingiem
- **Przypisanie użytkowników do UAV-ów** - $\alpha_{n,k}[t] \in {0,1}$

## Funkcja celu

Maksymalizujemy funkcję rate'u u użytkowników:

$$
\max \quad \sum_{t} \sum_{k} R_k[t]
$$

przy zachowaniu podanych poniżej ograniczeń:

| Ograniczenie | Forma | Znaczenie |
| :-- | :-- | :-- |
| Jakość sensingu | $\text{CRB}_q[t] \leq \gamma$ | błąd lokalizacji każdego celu poniżej progu |
| Moc nadawcza | $\|\mathbf{w}n[t]\|^2 \leq P{\max}$ | limit mocy każdego drona
Prędkość UAV | $\|\mathbf{q}_n[t+1] - \mathbf{q}n[t]\| \leq v{\max}\Delta t$ | maks. 20 m/s |
| Unikanie kolizji | $\|\mathbf{q}_n[t] - \mathbf{q}m[t]\| \geq d{\min}$ | bezpieczna odległość między dronami |
| Energia | $E_n^{\text{total}} \leq E_{\max}$ | budżet bateryjny (lot + transmisja) |
| Obszar lotu | $\mathbf{q}_n[t] \in \mathcal{A}$ | drony zostają w obszarze 500×500 m² |

## Komentarz - skąd się bierze trudność

Problem jest niewypukły i sprzężony z trzech powodów:
1. **Sprzężenie zmiennych** - trajektoria wpływa na kanał, kanał wpływa na beamforming, beamforming wpływa na alokację mocy, a wszystko razem na CRB
2. **Trade-off communication vs. sensing** - ta sama moc i ten sam beam służą dwóm celom jednocześnie
3. **Wymiar czasowy** - decyzje w chwili $t$ wpływają na to, gdzie UAV może być w $t + 1$

# Obraz strategii rozwiązania

Stosujemy Block Coordinate Descent (BCD) - rozbijamy duży problem na bloki zmiennych i optymalizujemy je naprzemiennie:

1. ustal trajektorie → zoptymalizuj beamforming + moc (problem wypukły, SCA)
2. ustal beamforming → zoptymalizuj trajektorie (SCA z aproksymacją Taylora)
3. Powtarzaj 1.-2. do zbieżności

Rezultatem jest krzywa Pareto pokazująca trade-off: ile rate'u tracimy, żeby dostać lepsze CRB (lub odwrotnie).
