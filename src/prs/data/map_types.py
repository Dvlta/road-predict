"""Project map-chunk types (design doc §9). 0 is reserved for padding."""

from __future__ import annotations

from enum import IntEnum


class MapType(IntEnum):
    PAD = 0
    LANE_UNDEFINED = 1
    LANE_FREEWAY = 2
    LANE_SURFACE_STREET = 3
    LANE_BIKE = 4
    ROAD_LINE_BROKEN = 5
    ROAD_LINE_SOLID = 6
    ROAD_LINE_UNKNOWN = 7
    ROAD_EDGE_BOUNDARY = 8
    ROAD_EDGE_MEDIAN = 9
    ROAD_EDGE_UNKNOWN = 10
    CROSSWALK = 11
    SPEED_BUMP = 12
    DRIVEWAY = 13
    STOP_SIGN = 14


NUM_MAP_TYPES = len(MapType)  # embedding-table size, including PAD

# WOMD subtype enum values (map.proto) -> project type. Paint colors collapse to broken / solid;
# a passing double yellow (one solid, one broken line) counts as solid.
_SUBTYPES = {
    "lane": {
        0: MapType.LANE_UNDEFINED,
        1: MapType.LANE_FREEWAY,
        2: MapType.LANE_SURFACE_STREET,
        3: MapType.LANE_BIKE,
    },
    "road_line": {
        0: MapType.ROAD_LINE_UNKNOWN,
        1: MapType.ROAD_LINE_BROKEN,  # broken single white
        2: MapType.ROAD_LINE_SOLID,  # solid single white
        3: MapType.ROAD_LINE_SOLID,  # solid double white
        4: MapType.ROAD_LINE_BROKEN,  # broken single yellow
        5: MapType.ROAD_LINE_BROKEN,  # broken double yellow
        6: MapType.ROAD_LINE_SOLID,  # solid single yellow
        7: MapType.ROAD_LINE_SOLID,  # solid double yellow
        8: MapType.ROAD_LINE_SOLID,  # passing double yellow
    },
    "road_edge": {
        0: MapType.ROAD_EDGE_UNKNOWN,
        1: MapType.ROAD_EDGE_BOUNDARY,
        2: MapType.ROAD_EDGE_MEDIAN,
    },
}
_WITHOUT_SUBTYPE = {
    "crosswalk": MapType.CROSSWALK,
    "speed_bump": MapType.SPEED_BUMP,
    "driveway": MapType.DRIVEWAY,
    "stop_sign": MapType.STOP_SIGN,
}


def map_type(kind: str, subtype: int) -> MapType:
    """Project type for a raw map feature; unknown kinds or subtypes fail loudly."""
    if kind in _SUBTYPES:
        if subtype not in _SUBTYPES[kind]:
            raise ValueError(f"unknown {kind} subtype {subtype}")
        return _SUBTYPES[kind][subtype]
    if kind in _WITHOUT_SUBTYPE:
        return _WITHOUT_SUBTYPE[kind]
    raise ValueError(f"unknown map feature kind {kind!r}")
