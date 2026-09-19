# Corpus descriptive robustness findings (second pass)

Stress test of the first descriptive analysis. Layer labels match
[analysis-corpus-descriptive-v1.md](analysis-corpus-descriptive-v1.md).

Second-pass stress test of the first descriptive analysis on the same convenience / coverage corpus. Not a probability sample, not representative of all board games, and not a rank or best-of list.

- Corpus: `bgg_boardgames_v1` (`corpus_v1.jsonl`)
- n games in JSONL: **195** (load errors: 0)
- Corpus payload_id: `d2706e9003916f4aa3b082d4feb2567fa8498487c19b8ba5cb43d85fc026884f`
- Generated: 2026-09-19T21:22:12.354803+00:00

## 1. Play time vs weight — representation sensitivity

Spearman rank correlations with BGG weight (no p-values):

- Reported minimum minutes (raw): n=192, missing_x=2, missing_y=2, ρ=0.62
- Reported maximum minutes (raw): n=192, missing_x=2, missing_y=2, ρ=0.81
- Midpoint proxy (min+max)/2 (raw): n=192, missing_x=2, missing_y=2, ρ=0.78
- Duration span (max−min) (raw): n=192, missing_x=2, missing_y=2, ρ=0.58

Log-transformed positive values:

- Reported minimum minutes (log): n=192, missing_x=2, missing_y=2, ρ=0.62
- Reported maximum minutes (log): n=192, missing_x=2, missing_y=2, ρ=0.81
- Midpoint proxy (min+max)/2 (log): n=192, missing_x=2, missing_y=2, ρ=0.78
- Duration span (max−min) (log): n=108, missing_x=86, missing_y=2, ρ=0.69

Extreme-value leave-one-out (games flagged for extreme player or play-time intervals):

- `Cartographers` (max_players=100; player_span=99), min_play_time_minutes: ρ 0.62 → 0.61 (Δ=-0.00)
- `Cartographers` (max_players=100; player_span=99), max_play_time_minutes: ρ 0.81 → 0.81 (Δ=-0.00)
- `Cartographers` (max_players=100; player_span=99), play_time_midpoint: ρ 0.78 → 0.78 (Δ=-0.00)
- `Cartographers` (max_players=100; player_span=99), play_time_span: ρ 0.58 → 0.58 (Δ=+0.00)
- `Pictionary` (max_players=16), min_play_time_minutes: ρ 0.62 → 0.63 (Δ=+0.02)
- `Pictionary` (max_players=16), max_play_time_minutes: ρ 0.81 → 0.82 (Δ=+0.01)
- `Pictionary` (max_players=16), play_time_midpoint: ρ 0.78 → 0.79 (Δ=+0.01)
- `Pictionary` (max_players=16), play_time_span: ρ 0.58 → 0.57 (Δ=-0.00)
- `Wavelength` (max_players=12), min_play_time_minutes: ρ 0.62 → 0.61 (Δ=-0.00)
- `Wavelength` (max_players=12), max_play_time_minutes: ρ 0.81 → 0.81 (Δ=+0.00)
- `Wavelength` (max_players=12), play_time_midpoint: ρ 0.78 → 0.78 (Δ=-0.00)
- `Wavelength` (max_players=12), play_time_span: ρ 0.58 → 0.58 (Δ=+0.00)
- `Deception: Murder in Hong Kong` (max_players=12), min_play_time_minutes: ρ 0.62 → 0.61 (Δ=-0.00)
- `Deception: Murder in Hong Kong` (max_players=12), max_play_time_minutes: ρ 0.81 → 0.81 (Δ=-0.00)
- `Deception: Murder in Hong Kong` (max_players=12), play_time_midpoint: ρ 0.78 → 0.78 (Δ=-0.00)
- `Deception: Murder in Hong Kong` (max_players=12), play_time_span: ρ 0.58 → 0.57 (Δ=-0.00)
- `Civilization` (max_play_time_minutes=360), min_play_time_minutes: ρ 0.62 → 0.61 (Δ=-0.00)
- `Civilization` (max_play_time_minutes=360), max_play_time_minutes: ρ 0.81 → 0.81 (Δ=-0.00)
- `Civilization` (max_play_time_minutes=360), play_time_midpoint: ρ 0.78 → 0.78 (Δ=-0.00)
- `Civilization` (max_play_time_minutes=360), play_time_span: ρ 0.58 → 0.59 (Δ=+0.01)
- `Diplomacy` (max_play_time_minutes=360), min_play_time_minutes: ρ 0.62 → 0.62 (Δ=-0.00)
- `Diplomacy` (max_play_time_minutes=360), max_play_time_minutes: ρ 0.81 → 0.81 (Δ=+0.00)
- `Diplomacy` (max_play_time_minutes=360), play_time_midpoint: ρ 0.78 → 0.78 (Δ=+0.00)
- `Diplomacy` (max_play_time_minutes=360), play_time_span: ρ 0.58 → 0.58 (Δ=+0.01)
- `Twilight Imperium: Fourth Edition` (max_play_time_minutes=480), min_play_time_minutes: ρ 0.62 → 0.61 (Δ=-0.01)
- `Twilight Imperium: Fourth Edition` (max_play_time_minutes=480), max_play_time_minutes: ρ 0.81 → 0.81 (Δ=-0.00)
- `Twilight Imperium: Fourth Edition` (max_play_time_minutes=480), play_time_midpoint: ρ 0.78 → 0.78 (Δ=-0.00)
- `Twilight Imperium: Fourth Edition` (max_play_time_minutes=480), play_time_span: ρ 0.58 → 0.57 (Δ=-0.01)
- `The 7th Continent` (max_play_time_minutes=1000), min_play_time_minutes: ρ 0.62 → 0.62 (Δ=+0.01)
- `The 7th Continent` (max_play_time_minutes=1000), max_play_time_minutes: ρ 0.81 → 0.81 (Δ=+0.00)
- `The 7th Continent` (max_play_time_minutes=1000), play_time_midpoint: ρ 0.78 → 0.79 (Δ=+0.00)
- `The 7th Continent` (max_play_time_minutes=1000), play_time_span: ρ 0.58 → 0.58 (Δ=+0.00)
- `Paths of Glory` (max_play_time_minutes=480), min_play_time_minutes: ρ 0.62 → 0.61 (Δ=-0.00)
- `Paths of Glory` (max_play_time_minutes=480), max_play_time_minutes: ρ 0.81 → 0.81 (Δ=-0.00)
- `Paths of Glory` (max_play_time_minutes=480), play_time_midpoint: ρ 0.78 → 0.78 (Δ=-0.00)
- `Paths of Glory` (max_play_time_minutes=480), play_time_span: ρ 0.58 → 0.59 (Δ=+0.01)

## 2. Player count as interval data

Interval distributions (this file only):

- Minimum players: n=195; median 2.0 (IQR 1.0–2.0); range 1.0–5.0
- Maximum players: n=195; median 4.0 (IQR 4.0–6.0); range 2.0–100.0
- Player span (max−min): n=195; median 3.0 (IQR 2.0–4.0); range 0.0–99.0
- Midpoint proxy (min+max)/2: n=195; median 3.0 (IQR 2.5–3.5); range 1.5–50.5

Associations with complexity:

- min_players: n=193, missing_x=0, missing_y=2, ρ=-0.34
- max_players: n=193, missing_x=0, missing_y=2, ρ=-0.20
- player_span: n=193, missing_x=0, missing_y=2, ρ=-0.07
- player_midpoint_proxy: n=193, missing_x=0, missing_y=2, ρ=-0.27

Associations with play-time midpoint proxy:

- min_players vs play_time_midpoint: n=193, ρ=-0.18
- max_players vs play_time_midpoint: n=193, ρ=0.01
- player_span vs play_time_midpoint: n=193, ρ=0.07
- player_midpoint_proxy vs play_time_midpoint: n=193, ρ=-0.03

Extreme published intervals (not removed):

- `Cartographers` (bgg-263918): max_players=100; player_span=99
- `Pictionary` (bgg-2281): max_players=16
- `Wavelength` (bgg-262543): max_players=12
- `Deception: Murder in Hong Kong` (bgg-156129): max_players=12
- `Civilization` (bgg-71): max_play_time_minutes=360
- `Diplomacy` (bgg-483): max_play_time_minutes=360
- `Twilight Imperium: Fourth Edition` (bgg-233078): max_play_time_minutes=480
- `The 7th Continent` (bgg-180263): max_play_time_minutes=1000

Subset without max_players ≥ 12:
- n=193; max_players vs complexity: ρ=-0.18

Subset without player_span ≥ 20:
- n=194; max_players vs complexity: ρ=-0.19

## 3. Mechanic prevalence vs association

- Distinct labels: 163
- Labels appearing once: 19
- Labels appearing twice: 18

Labels surviving minimum-game thresholds:

- ≥3 games: 126 labels (77.3% of distinct labels)
- ≥5 games: 94 labels (57.7% of distinct labels)
- ≥10 games: 57 labels (35.0% of distinct labels)
- ≥20 games: 29 labels (17.8% of distinct labels)

Profile-lift sensitivity (lift ≥ 1.5 somewhere):

- min_games=3: 126 labels compared, 67 with lift ≥ 1.5
- min_games=5: 94 labels compared, 42 with lift ≥ 1.5
- min_games=10: 57 labels compared, 18 with lift ≥ 1.5
- min_games=20: 29 labels compared, 6 with lift ≥ 1.5

Prevalence (top labels) is not association. Most labels are rare.

## 4. Corpus composition audit

- Games: 195

Field missingness:

- `release_year`: present=193, null=2 (1.0%)
- `min_players`: present=195, null=0 (0.0%)
- `max_players`: present=195, null=0 (0.0%)
- `min_play_time_minutes`: present=193, null=2 (1.0%)
- `max_play_time_minutes`: present=193, null=2 (1.0%)
- `popularity`: present=0, null=195 (100.0%)
- `rating`: present=195, null=0 (0.0%)
- `complexity`: present=193, null=2 (1.0%)
- `mechanics`: present=194, null=0 (0.0%)

Missing play time:

- `Chess` (bgg-171)
- `Bucking Broncho` (bgg-42048)

Missing complexity:

- `Bucking Broncho` (bgg-42048)
- `Battle Royale: Flick to the Death` (bgg-359971)

Empty mechanic lists:

- `Bucking Broncho` (bgg-42048)

## 5. Analytical conclusions

### Robust observations

- Positive play-time / weight rank association persists across minimum, maximum, and midpoint representations (ρ roughly 0.62–0.81 in this file).
- Weak negative max_players vs complexity association (ρ≈-0.20) is small in magnitude.
- Hand Management–scale prevalence (Hand Management: 45% of games) reflects label frequency, not a profile association.

### Representation-sensitive observations

- Duration span (ρ≈0.58) differs from midpoint (ρ≈0.78); wide published ranges carry different signal.

### Sparse-data observations

- 19 of 163 labels (12%) appear on only one game.
- Profile lift at min_games=5 compares sparse cells; threshold changes which labels enter the comparison.
- At threshold 5: 42 labels show lift ≥ 1.5 somewhere — interpret as exploratory only.

### Data limitations

- Convenience sample: cannot infer BGG or hobby-wide distributions.
- `popularity` is unmapped; `Mechanic.category` is always null.
- Published min/max intervals are not observed play durations.
- Player midpoint is a proxy, not typical table size.
- Multi-label mechanics: prevalence shares sum above 100%.
- Play time missing for 2 games.
- Complexity missing for 2 games.

## Figures

- `figures/fig_time_weight_correlations.png`
- `figures/fig_time_vs_weight_scatter.png`
- `figures/fig_player_intervals.png`
- `figures/fig_mechanic_frequency_histogram.png`
- `figures/fig_mechanic_threshold_survival.png`
- `figures/fig_composition_audit.png`
