# Ontology v0

This document describes the v0 representation used to compare board games
by information structure, decision structure, and state change. It is a
research artifact, not a complete game-description language.

v0 was validated against a small manually encoded corpus (Hanabi, The Crew,
The Gang, Just One, Codenames, Wingspan, Chess, Poker, Telestrations,
Pandemic). Those fixtures are representative moments, not full rulesets.

## 1. Game

`Game` is catalog metadata observed from a source: title, year, player
count, ratings, designers, and so on.

`Game` is not the rules. It does not say what players may do, what they
know, or how good a position is.

Rules-level structure belongs on `GameDefinition`. Classifications such as
"asymmetric information" belong on `ExtractedInterpretation`. Numeric
analysis belongs on `DerivedMeasurement`.

## 2. GameState

`GameState` is an omniscient snapshot of what is true at a moment: hands,
board position, token counts, phase.

`data` is intentionally an open map. v0 does not define cards, boards, or
tokens as first-class types. The important convention is:

- `GameState` is the **underlying** state.
- It may contain information no player currently knows (deck order, hole
  cards, a Hanabi player's own identities).

Player-specific views are not stored on `GameState`.

## 3. Observation

An `Observation` is the binding:

`observer × GameState → InformationSpace`

Two observations may share a `game_state_id` and differ. That is how v0
says "Alice and Bob are looking at the same position and do not see the
same things."

Observation is not inference and not a fact about the catalog. It is a
view of a state.

## 4. InformationSpace

An `InformationSpace` is the set of `InformationItem`s in one observation.

Each item has:

- `visibility` — open string. v0 vocabulary: `public`, `private`,
  `other_private`, `hidden`, `inferred`, `communicated`, `unknown_to_self`
- `content_known` — whether this observer has the actual value
- `holder_id` — who possesses the component, if anyone
- `communicated_by` — who told the observer, if this is a clue
- `predecessor_id` — prior item, when information is transformed
- `payload` — game-specific content

These fields are how v0 distinguishes, without epistemic logic:

- public board state (Chess, Pandemic)
- own known private information (Poker hole, Crew hand)
- another player's private information, seen or unseen (Hanabi others'
  cards vs Poker opponent hole)
- hidden existence (decks)
- inferred estimates (Poker range)
- communicated content (Hanabi clue, Codenames clue)
- known gaps (Hanabi own card identity)

## 5. DecisionSpace

A `DecisionSpace` is what a player can choose at a state.

`GameDefinition.legal_action_types` is the static rules list ("players may
give clues").

A `DecisionOption` is a concrete candidate:

- `legal` — the rules permit this kind of act
- `available` — it is possible in this state (clue tokens remain, a city
  has cubes, the piece can move there)

v0 does not mark options as meaningful, dominant, or strategically
distinct. That is analysis.

`complete=False` records that the listed options are a sample. Chess uses
this so a large move set is not fully enumerated.

## 6. Action

An `Action` is what a player actually chose: actor, `action_type`,
parameters, optional turn number.

Simultaneous choice (Just One clues) is several actions attached to one
`StateTransition` via `action_ids`.

## 7. StateTransition

`StateTransition` is `GameState → Action(s) → GameState`.

`kind` is an open string. v0 vocabulary: `player`, `chance`, `system`.

After a transition, later observations and decision spaces may change.
That is represented by additional `GameState`s, `Observation`s, and
`DecisionSpace`s in the same `PlaySituation`, not by mutating the prior
objects.

## 8. Facts, interpretations, and measurements

| Layer | Model | Example |
| --- | --- | --- |
| Observed fact | `Game`, `SourceReference`, `Mechanic` | released 2010, 2–5 players |
| Static rules structure | `GameDefinition` | cooperative, sequential, legal action types |
| Extracted interpretation | `ExtractedInterpretation` | constrained communication |
| Play representation | `GameState`, `Observation`, `DecisionSpace`, `Action`, `StateTransition` | Alice cannot see her Hanabi hand |
| Derived measurement | `DerivedMeasurement` | (none in v0 corpus) |

Do not store interpretations or measurements on `Game`. Cooperative vs
competitive is treated as rules structure on `GameDefinition` because it
is part of what the rules define, not a later analytic score.

## 9. Underlying state vs player observation

```
GameDefinition          what the rules permit
        |
GameState S             omniscient snapshot
        |
        +-- Observation(Alice, S) --> InformationSpace_A
        +-- Observation(Bob, S)   --> InformationSpace_B
        |
        +-- DecisionSpace(Alice, S)
        |
     Action
        |
StateTransition
        |
GameState S'
```

If a representation cannot give Alice and Bob different spaces for the
same `GameState`, it is not sufficient for this project.

## 10. Known limitations of v0

- No formal epistemic logic (possible worlds, nested knowledge, common
  knowledge).
- No first-class components (cards, boards, cubes). Those sit in `payload`
  / `data`.
- No probability calculus; inferred items carry ad hoc payload.
- No encoding of illegal-but-mentioned actions beyond `legal`/`available`.
- Chance events are a `kind` label, not a distribution.
- Simultaneous play is multiple actions on one transition, not a true
  joint action type.
- Roles (`Player.role`) are labels, not a permission system.
- Information transformation (Telestrations) is a predecessor link plus
  medium in payload, not a typed rewrite system.
- Decision-space completeness is declared, not verified.
- The corpus is a handful of moments. Coverage of genres is not a proof
  that every game will fit.

The success criterion for v0 is narrower: fundamentally different
information and decision structures can be expressed with the same
models, without a per-game special case in the type system.

Proposed measurements on top of this ontology:
[analysis.md](analysis.md).
