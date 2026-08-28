"""Map parsed BGG things onto canonical `Game` objects."""

from board_game_analysis.domain.game import Game
from board_game_analysis.domain.mechanic import Mechanic
from board_game_analysis.domain.source import SourceReference
from board_game_analysis.ingestion.bgg.parser import parse_thing_xml
from board_game_analysis.ingestion.bgg.types import BggThing, RawArtifact

SOURCE = "boardgamegeek"


def normalize_bgg_artifact(artifact: RawArtifact) -> Game:
    """Deterministic: same artifact body and provenance → same Game."""
    thing = parse_thing_xml(artifact.body)
    return normalize_bgg_thing(thing, artifact)


def normalize_bgg_thing(thing: BggThing, artifact: RawArtifact) -> Game:
    source = SourceReference(
        source=SOURCE,
        source_type="xmlapi2",
        url=artifact.request_url,
        source_identifier=thing.bgg_id,
        retrieved_at=artifact.retrieved_at,
    )
    mechanics = [
        Mechanic(id=f"bgg-{link.source_id}", name=link.value)
        for link in thing.mechanics
    ]
    return Game(
        id=f"bgg-{thing.bgg_id}",
        title=thing.primary_name,
        release_year=thing.year_published,
        min_players=thing.min_players,
        max_players=thing.max_players,
        min_play_time_minutes=thing.min_play_time_minutes,
        max_play_time_minutes=thing.max_play_time_minutes,
        popularity=None,
        rating=thing.rating,
        rating_count=thing.rating_count,
        complexity=thing.complexity,
        designers=list(thing.designers),
        publishers=list(thing.publishers),
        categories=list(thing.categories),
        mechanics=mechanics,
        sources=[source],
    )
