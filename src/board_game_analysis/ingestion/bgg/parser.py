"""Parse BGG XML API2 `/thing` documents into source-specific types."""

from __future__ import annotations

import xml.etree.ElementTree as ET

from board_game_analysis.ingestion.bgg.types import BggLink, BggThing
from board_game_analysis.ingestion.errors import BggNotFoundError, IngestionError


def parse_thing_xml(body: str) -> BggThing:
    """Parse a `/thing?stats=1` XML document containing one item."""
    try:
        root = ET.fromstring(body)
    except ET.ParseError as exc:
        msg = "BGG response was not valid XML"
        raise IngestionError(msg) from exc
    item = root.find("item")
    if item is None:
        msg = "BGG response contained no <item>"
        raise BggNotFoundError(msg)
    bgg_id = item.attrib.get("id")
    if not bgg_id:
        msg = "BGG <item> is missing id"
        raise IngestionError(msg)
    primary = _primary_name(item)
    stats = item.find("statistics/ratings")
    rating_count = _int_attr(stats, "usersrated", sentinel_zero_is_missing=False)
    min_play, max_play = _play_times(item)
    return BggThing(
        bgg_id=bgg_id,
        item_type=item.attrib.get("type", "boardgame"),
        primary_name=primary,
        year_published=_int_attr(item, "yearpublished"),
        min_players=_int_attr(item, "minplayers"),
        max_players=_int_attr(item, "maxplayers"),
        min_play_time_minutes=min_play,
        max_play_time_minutes=max_play,
        rating=_rating(stats, rating_count),
        rating_count=rating_count,
        complexity=_complexity(stats),
        designers=_link_values(item, "boardgamedesigner"),
        publishers=_link_values(item, "boardgamepublisher"),
        categories=_link_values(item, "boardgamecategory"),
        mechanics=_links(item, "boardgamemechanic"),
    )


def _child_value(parent: ET.Element | None, tag: str) -> str | None:
    if parent is None:
        return None
    child = parent.find(tag)
    if child is None:
        return None
    return child.attrib.get("value")


def _int_attr(
    parent: ET.Element | None,
    tag: str,
    *,
    sentinel_zero_is_missing: bool = True,
) -> int | None:
    raw = _child_value(parent, tag) if parent is not None else None
    if raw is None or raw.strip() == "":
        return None
    value = int(raw)
    if sentinel_zero_is_missing and value == 0:
        return None
    return value


def _play_times(item: ET.Element) -> tuple[int | None, int | None]:
    min_play = _int_attr(item, "minplaytime")
    max_play = _int_attr(item, "maxplaytime")
    playing = _int_attr(item, "playingtime")
    if min_play is None and max_play is None and playing is not None:
        return playing, playing
    return min_play, max_play


def _float_attr(parent: ET.Element | None, tag: str) -> float | None:
    raw = _child_value(parent, tag)
    if raw is None or raw.strip() == "":
        return None
    return float(raw)


def _rating(stats: ET.Element | None, rating_count: int | None) -> float | None:
    if stats is None or rating_count is None or rating_count == 0:
        return None
    return _float_attr(stats, "average")


def _complexity(stats: ET.Element | None) -> float | None:
    if stats is None:
        return None
    weights = _int_attr(stats, "numweights", sentinel_zero_is_missing=False)
    if weights is None or weights == 0:
        return None
    return _float_attr(stats, "averageweight")


def _primary_name(item: ET.Element) -> str:
    for name in item.findall("name"):
        if name.attrib.get("type") == "primary":
            value = name.attrib.get("value", "").strip()
            if value:
                return value
    fallback = item.find("name")
    if fallback is not None:
        value = fallback.attrib.get("value", "").strip()
        if value:
            return value
    msg = "BGG <item> is missing a name"
    raise IngestionError(msg)


def _links(item: ET.Element, kind: str) -> list[BggLink]:
    found: list[BggLink] = []
    for link in item.findall("link"):
        if link.attrib.get("type") != kind:
            continue
        source_id = link.attrib.get("id", "")
        value = link.attrib.get("value", "").strip()
        if not value:
            continue
        found.append(BggLink(kind=kind, source_id=source_id, value=value))
    return found


def _link_values(item: ET.Element, kind: str) -> list[str]:
    return [link.value for link in _links(item, kind)]
