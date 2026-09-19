# Analysis specification v0

The first **empirical** pass over ingested Game metadata is
[analysis-corpus-descriptive-v1.md](analysis-corpus-descriptive-v1.md).
This document remains the ontology measurement catalog; it is not that
corpus analysis.

This document proposes dimensions we may eventually measure computationally.
It is a research framework, not a validated psychometrics paper and not an
analysis engine.

Labels used throughout:

- **proposed measure** — a quantity we might operationalize later
- **hypothesis** — a claim to test against human or simulated data
- **future measurement** — requires models, traces, or data we do not have

No numbers in this document are empirical results. The ten-game corpus is
used only to ask whether the *ontology* could ever support a given measure.

Related: [ontology.md](ontology.md).

## 1. Three layers (do not collapse)

| Layer | What it is | Example | Model |
| --- | --- | --- | --- |
| Observed facts | Reported by a source | 2–5 players, 2010, BGG rating 7.9 | `Game`, `SourceReference` |
| Extracted interpretations | Judgments about the rules | cooperative, constrained communication | `ExtractedInterpretation`, some `GameDefinition` fields |
| Derived measurements | Calculated from a model | branching factor, information asymmetry | `DerivedMeasurement` |

A published rating is an **observed fact** that can later be used as an
**outcome variable**. It is not a structural property of the rules, and it
is not "fun" stored on `Game`.

Interpretations such as "asymmetric information" are not measurements.
Counting how asymmetric two `InformationSpace`s are would be a measurement.

## 2. Levels of evidence

These are different kinds of claim. Do not treat them as a single score.

```text
RULE (text, components)
  ↓
Game model (ontology objects)
  ↓
Structural measurement (what the model contains)
  ↓
Simulation result (what agents do in the model)
  ↓
Human rating / playtest (what people report or do)
```

| Level | Question |
| --- | --- |
| What the game contains | Rules and encoded `GameState` / spaces |
| What our model thinks it contains | Completeness, encoding choices, `complete=False` samples |
| What we observe humans doing or feeling | Ratings, playtests, behavior |

A large decision space in our fixture may be an encoding choice (we listed
three chess moves). A high BGG rating does not mean our branching-factor
proxy caused acclaim.

## 3. Research taxonomy

### Descriptive measurements

What is structurally present?

Examples: available action count, hidden-information presence, whether
observations of one state differ.

### Predictive measurements / indicators

What might later predict an outcome? These are **hypotheses** until tested.

Examples: decision diversity, agency, tension indicators.

### Outcome variables

What we may try to predict.

Examples: source rating, rating count, ranking, review sentiment,
playtest ratings, replayability, retention if ever available.

The eventual question is of the form:

> Which structural properties of games predict human judgments of quality,
> enjoyment, tension, or replayability?

Correlation is not causation. Selection, marketing, components, and
community effects will confound any naive regression.

## 4. Information-space dimensions

All of the following are **proposed**. Item counts are not information
content.

### Information volume

How much information is available to a player at a point?

Possible operationalizations: count `InformationItem`s (naive); weight by
payload or component coverage; bits relative to a later formal encoding of
`GameState.data`.

v0 can count items. It cannot measure bits. A Chess FEN in one payload is
not "less information" than four Hanabi card dicts.

### Information visibility

How much of the *relevant* underlying state is directly observable
(`content_known`), versus hidden, inferred, or unknown-to-self?

"Relevant" is undefined in v0. A possible later operationalization is the
fraction of omniscient `GameState.data` that appears as known payloads in
an `Observation`.

### Information asymmetry

How different are players' information spaces at the same `GameState`?

The ontology supports the comparison (multiple `Observation`s, one state).
No distance function is chosen. Set differences on `content_known` items
are a crude start; they treat every item as equal.

### Information uncertainty

How uncertain is a player about relevant hidden state?

This is **not** "a hidden card exists." Poker hole cards are hidden;
uncertainty is a belief over ranges. Chess has no hidden cards and still
has strategic uncertainty about the opponent's plan — which v0 does not
encode at all.

Possible later operationalizations: entropy of a belief distribution;
use of `inferred` payloads; count of known gaps. None is implemented.

### Information ownership

Who knows what: `holder_id`, `known_to`, observer id.

This is largely descriptive bookkeeping and is close to what v0 already
stores.

### Information flow

How information moves over time: clues, communication, passing a chain.

Needs traces. `communicated_by` and `predecessor_id` are hooks, not a flow
metric.

### Information gain / loss

How an action changes knowledge.

Possible operationalizations: new `content_known` items after a
`StateTransition`; items that become unknown; later KL divergence.

Gain and loss are distinct (Telestrations can gain a drawing and lose the
original word).

### Self-knowledge asymmetry

Can a player observe others' private components that they cannot observe
about themselves?

Hanabi is the motivating case: `unknown_to_self` on own cards,
`other_private` + `content_known` on others. This is a **proposed**
descriptive flag/ratio, not a theory of fun.

## 5. Decision-space dimensions

Keep these terms distinct:

```text
legal action types     GameDefinition (players may clue, bet, move)
legal actions          instances the rules allow at a state
available actions      instances executable now (legal and possible)
strategically distinct future: clustering / outcome divergence
dominated actions      future: utility / agents
meaningful actions     future: several methods, none chosen
```

Do not define "meaningful" as "number of legal choices."

### Legal decision count / available decision count / branching factor

Available count is the only one v0 can attempt when
`DecisionSpace.complete` is true: count options with `legal` and
`available`. Chess fixtures set `complete=False`; listing three opening
moves is not a branching factor.

Legal instance count needs a move generator we do not have.

### Decision diversity

Ten near-identical pawn pushes are not ten kinds of choice.

Possible operationalizations: unique `action_type` (crude); cluster
`parameters`; cluster by simulated outcome. **Future.** No similarity
function lives in the ontology.

### Constraint

How much the state restricts the player: legal-but-unavailable options
(Pandemic charter flight), or available/legal ratio if both counts exist.

### Reversibility, consequence, agency

**Future / hypotheses.** They need a forward model or value function.
State-dict edit distance after a transition is a toy proxy, not consequence.

Agency is not option count. Many options can leave the outcome unchanged.

### Strategic dominance

**Future. Not implemented.** Options must not be labeled good or
meaningful on the ontology.

### Meaningful decisions

Open research question. Candidate identification methods (none privileged):

- counterfactual outcome differences
- simulation
- agent disagreement
- utility / value differences
- sensitivity of future state
- human empirical choice data

## 6. Temporal dimensions

The corpus is a handful of moments. Trajectories **cannot** be estimated
from them.

Proposed later series: state complexity; decision-space size/diversity;
per-player information; information convergence vs divergence.

**Tension trajectory** is a **hypothesis**, not a measure. See §9.

## 7. Interaction dimensions

Distinguish:

**Structural interaction** — the rules require mutual effect (shared
board, required clues, blocking). Partly visible on `GameDefinition`
(`interaction`, action types that name other players) and on
interpretations such as `constrained_communication`.

**Behavioral interaction** — players happen to interact in play. Needs
simulation or empirical data.

Related notions, not yet metrics: direct vs indirect interference,
communication, negotiation, dependency between decision spaces.

Cooperation vs competition is rules structure. It is not a derived
"how cooperative was this session" score.

## 8. Hidden information, uncertainty, randomness

Not synonyms.

| Concept | Roughly | Chess | Poker | Hanabi |
| --- | --- | --- | --- | --- |
| Hidden information | Exists in state, not in this observation | No (standard) | Opponent hole | Own identities |
| Uncertainty | Observer's beliefs about unknowns | Strategic, not encoded | Range over hole cards | Color/rank of own slots |
| Randomness | Stochastic rule events | No | Shuffle / dealing | Shuffle / dealing |

Do not reduce these to one "uncertainty" score.

v0 can mark hidden items and chance-kind transitions. It has no
probability calculus and no encoding of strategic (non-hidden) uncertainty.

## 9. Tension (hypothesis)

Tension is **not** an established metric. The research move is: list
candidate contributors, later test them against human tension ratings.

Candidates (not a checklist that sums to "tension"):

- uncertainty (hidden and/or strategic)
- consequence
- agency
- time pressure (not in the ontology)
- scarcity (tokens, lives, tricks)
- irreversibility
- competition
- information asymmetry
- proximity to objectives
- volatility of state

**Toy hypothesis (not adopted):** some monotone combination of uncertainty,
consequence, and agency. Writing `tension = u × c × a` would pretend
calibration we do not have. If a later notebook tries a product, it must
be labeled as a toy and tested, not shipped as truth.

## 10. Fun / acclaim (outcome variables)

Do **not** encode fun as a field on `Game`.

Possible later targets (mostly external):

- user rating, rating count, ranking (`Game.rating` / `rating_count` as facts)
- review sentiment
- retention / play frequency
- human playtest ratings (quality, tension, replayability)

These sit at evidence level "human outcome." Structural measurements may
correlate with them. That would not imply the structure caused the rating.

## 11. Measurement object

`DerivedMeasurement` records a result (or an empty value) for a named
metric:

- `name`, optional `value`, `unit`
- `scope` (`game`, `player`, `state`, `observation`, `trajectory`)
- `player_id` / `state_id` when scoped
- `method`, `version`, `model_ref`, optional `confidence`

Proposed metrics without values live in `board_game_analysis.analysis.spec`
as `MeasurementSpec`. The engine that fills `value` does not exist yet.

## 12. Corpus stress test

No invented quantities. "Representable" means the ontology has hooks.
"Not calculable" means v0 data or theory is insufficient.

| Game | Naturally in reach (structure) | Not calculable now |
| --- | --- | --- |
| Hanabi | Self-knowledge asymmetry; per-player observations; clue as gain/flow hop | Bits of information; dominance; tension; full-game trajectories |
| The Crew | Private vs other-private hands; constrained communication as interpretation | How binding a communicate action is; trick-play branching over a round |
| The Gang | Cooperative structure + poker-like hidden holes | Behavioral coordination; randomness of the remaining deck |
| Just One | Role split on the target; simultaneous actions; duplicate removal as loss | Guesser entropy; why duplicates feel punishing |
| Codenames | Spymaster vs operative key; clue then guess space | Board-word similarity; "good clue" quality |
| Wingspan | Private hand vs public mats; action-type menu | Engine value; engine-building trajectories |
| Chess | Shared public observation; empty vs non-empty decision spaces | True branching factor (`complete=False`); strategic uncertainty; agency |
| Poker | Private / public / other-private / inferred; betting transition | Calibrated uncertainty; pot-odds agency; randomness of unseen cards |
| Telestrations | Predecessor chain; partial views; medium change as transform | Bits lost in a drawing; humor/tension; full-pad trajectories |
| Pandemic | Shared public board; hidden deck; legal-but-unavailable option | Infection-deck entropy; outbreak tension; epidemic trajectories |

Shared weaknesses: sampled or incomplete decision spaces; no component
ontology so volume is ill-defined; no forward model so consequence,
reversibility, dominance, and meaningfulness are blocked; no full games
so all trajectories are blocked.

## 13. Research questions (eventual)

### Information

- Which information structures are common vs rare in a large catalog?
- Does asymmetric information correlate with acclaim?
- Which kinds of asymmetry coincide with high interaction?
- When is self-knowledge inversion (Hanabi-type) used, and with what else?

### Decisions

- Do acclaimed games have larger or smaller *available* spaces?
- Is diversity more predictive than raw count?
- When does a large space become noise rather than agency?

### Temporal

- How does decision richness change over a game?
- Do acclaimed games share trajectory shapes?

### Interaction

- Which structures co-occur with cooperation, negotiation, or blocking?
- When is structural interaction unused in actual play?

### Generation (later)

- What combinations are common or unexplored in design space?

None of these can be answered with ten fixtures.

## 14. Deliberately undefined

v0 does **not** define: a bit-level information measure; a distance on
information spaces; calibrated uncertainty; randomness as a distribution;
action similarity; dominance; meaningfulness; tension; fun; legal-move
completeness; time pressure; proximity to objectives.

Where a concept is ambiguous, the catalog records operationalizations and
`requires` instead of a formula.

## 15. What this phase is not

No analysis engine, no metrics run on the corpus, no database, no
scraper, no LLM, no simulator, no ML model, no frontend.
