# Corpus descriptive findings

Generated from `report.json`. Layer labels are those in
[analysis-corpus-descriptive-v1.md](analysis-corpus-descriptive-v1.md).

This is a convenience / coverage corpus, not a probability sample, not representative of all board games, and not a rank or best-of list. Numbers describe this file only.

- Corpus: `bgg_boardgames_v1` (`corpus_v1.jsonl`)
- n games in JSONL: **195** (load errors: 0)
- Ingest list: requested=200, ok=195, skipped=2, errors=3
- Corpus payload_id: `d2706e9003916f4aa3b082d4feb2567fa8498487c19b8ba5cb43d85fc026884f`
- Generated: 2026-09-19T21:17:47.027051+00:00

## Coverage

- `min_players`: present=195, null=0 (0.0%)
- `max_players`: present=195, null=0 (0.0%)
- `min_play_time_minutes`: present=193, null=2 (1.0%)
- `max_play_time_minutes`: present=193, null=2 (1.0%)
- `complexity`: present=193, null=2 (1.0%)
- `popularity`: present=0, null=195 (100.0%)
- `mechanics`: nonempty=194, empty_list=1, null=0

`Mechanic.category` is not a corpus field; ingest leaves it null.

## Q1. Player-count profiles (descriptive observation)

Profiles are exclusive partitions of the **reported** min–max interval.
They are not typical table size.

- `two_only`: 26 (13.3%)
- `upto_four`: 79 (40.5%)
- `five_plus`: 90 (46.2%)
- `unknown`: 0 (0.0%)

- Player-range width (max − min): n=195 (missing=0); median 3 (IQR 2–4); mean 3.36 (SD 7.13); range 0–99

## Q2. Play time by profile (exploratory association)

Play time is the published BGG range. A midpoint is a proxy only.

- Point estimates (min = max): 84; ranges: 109; unknown: 2
- Reported minimum minutes: n=193 (missing=2); median 45 (IQR 30–70); mean 61.82 (SD 56.68); range 5–480
- Reported maximum minutes: n=193 (missing=2); median 90 (IQR 45–120); mean 98.61 (SD 96.94); range 10–1000
- Range span (minutes): n=193 (missing=2); median 15 (IQR 0–60); mean 36.79 (SD 80.52); range 0–995

- `two_only` (n=26): midpoint-proxy median 30 min (IQR 30–60); span median 0
- `upto_four` (n=79): midpoint-proxy median 90 min (IQR 60–105); span median 30
- `five_plus` (n=90): midpoint-proxy median 60 min (IQR 37.50–99.38); span median 15

Do not read this as “larger groups cause longer games.” Publisher
time bands, rulebooks, and this convenience sample all confound it.

## Q3. Complexity / weight (exploratory association)

BGG `averageweight` is a community poll on a 1–5 scale, present only
when voters exist. It is not a measured cognitive load.

- Complexity: n=193 (missing=2); median 2.66 (IQR 1.89–3.53); mean 2.71 (SD 0.92); range 1.03–4.44

- `two_only`: n=26 median 2.12 (IQR 1.83–3.38)
- `upto_four`: n=78 median 3.03 (IQR 2.41–3.80)
- `five_plus`: n=89 median 2.40 (IQR 1.74–3.25)

Spearman rank correlations (no p-values):

- play_time_midpoint vs complexity: n=192, ρ=0.78
- min_play_time_minutes vs complexity: n=192, ρ=0.62
- max_play_time_minutes vs complexity: n=192, ρ=0.81
- max_players vs complexity: n=193, ρ=-0.20
- player_span vs complexity: n=193, ρ=-0.07

A positive play-time / weight rank association in this file would
still not imply that longer games are heavier *because* of time,
nor that the same pattern holds outside this list.

## Q4. Mechanic labels (exploratory association)

Labels are multi-label BGG strings. Shares sum to more than 100%.
There is no mechanic-family taxonomy in this ingest.

- Distinct labels: 163; empty mechanic lists: 1
- Mechanics listed per game: n=195 (missing=0); median 9 (IQR 6–12); mean 9.21 (SD 4.44); range 0–27

Most common labels (share of games listing the name):

- Hand Management: 88 (45.1%)
- Variable Player Powers: 65 (33.3%)
- Variable Set-up: 62 (31.8%)
- Set Collection: 52 (26.7%)
- Solo / Solitaire Game: 51 (26.2%)
- End Game Bonuses: 49 (25.1%)
- Dice Rolling: 47 (24.1%)
- Open Drafting: 44 (22.6%)
- Tile Placement: 36 (18.5%)
- Modular Board: 35 (17.9%)
- Take That: 35 (17.9%)
- Area Majority / Influence: 35 (17.9%)
- Income: 30 (15.4%)
- Grid Movement: 30 (15.4%)
- Hexagon Grid: 28 (14.4%)

Profile lift for labels on at least 5 games is in
`report.json` (`tables.mechanics.profile_lifts`). Lift > 1 means the
label is more common in that profile than the corpus mix. Small
cells are noisy; do not treat lift as a discovery of a mechanic
type that “belongs” to a player count.

## Era (corpus composition only)

These bins describe how the convenience list was filled, not the
history of board games.

- `pre_1970`: 18 (9.2%)
- `1970_1994`: 10 (5.1%)
- `1995_2009`: 48 (24.6%)
- `2010_2016`: 65 (33.3%)
- `2017_plus`: 52 (26.7%)
- `unknown`: 2 (1.0%)

## What this does not show

- No causal effects of player count, time, or mechanics.
- No ranking of games or claims about “the hobby.”
- No extracted interpretations (cooperative, hidden information, …).
- Ratings were not modeled as quality.
