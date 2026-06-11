"""2D 적합(conforming) 삼각 메시 생성기.

모든 파트 경계선을 노딩(unary_union)해 평면 배치(arrangement)를 만들고,
각 면(face)을 재질 영역으로 분류한 뒤 Shewchuk Triangle로 영역별
크기 제약을 건 품질 메시를 생성한다. 인접 영역 경계가 정확히 공유되어
크랙 없는 적합 메시가 보장된다.

영역별 목표 요소 크기는 Maxwell 메시 연산(Band 5000 / Magnet 400 /
Coil 200 / Rotor·Stator·Region 1000)의 밀도 비율을 반영했다.
"""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import triangle as tr
from shapely.geometry import LineString, MultiLineString, Polygon, Point
from shapely.ops import unary_union, polygonize

from .geometry import MotorGeometry, _arc

# 영역 id 부여 순서(우선순위 높은 것 먼저 분류)
REGION_KINDS = ["magnet", "coil", "rotor", "stator",
                "air_gap_in", "air_gap_out", "air_slot", "air_shaft", "air_outer"]

# 목표 요소 변 길이 h [mm]
H_DEFAULT = {
    "magnet": 0.9, "coil": 1.4, "rotor": 2.2, "stator": 1.8,
    "air_gap_in": 0.30, "air_gap_out": 0.30, "air_slot": 0.9,
    "air_shaft": 3.5, "air_outer": 3.5,
}


def _h_to_area(h: float) -> float:
    return (math.sqrt(3) / 4.0) * h * h


def build_mesh(geo: MotorGeometry, h: Dict[str, float] | None = None,
               min_angle: float = 25.0) -> dict:
    h = {**H_DEFAULT, **(h or {})}

    # ---- 1) 경계 선분 수집 → 노딩 ---------------------------------
    lines: List[LineString] = []

    def add_boundary(poly):
        if poly.geom_type == "MultiPolygon":
            for g in poly.geoms:
                add_boundary(g)
            return
        lines.append(LineString(poly.exterior.coords))
        for hole in poly.interiors:
            lines.append(LineString(hole.coords))

    add_boundary(geo.rotor)
    add_boundary(geo.stator)
    for poly, _, _ in geo.magnets:
        add_boundary(poly)
    for c in geo.coils:
        add_boundary(c)
    lines.append(LineString(_arc(geo.band_radius, 0, 2 * math.pi, 256)))
    lines.append(LineString(_arc(geo.region_radius, 0, 2 * math.pi, 192)))

    import shapely
    noded = shapely.union_all(lines, grid_size=1e-6)
    faces = [f for f in polygonize(noded) if f.area > 1e-7]

    # ---- 2) 면 분류 -------------------------------------------------
    mag_polys = [(p, ang, pol) for p, ang, pol in geo.magnets]
    region_of_face: List[dict] = []
    for f in faces:
        rp = f.representative_point()
        kind, meta = None, {}
        for i, (p, ang, pol) in enumerate(mag_polys):
            if p.contains(rp):
                kind, meta = "magnet", {"index": i, "mag_angle": ang,
                                        "polarity": pol}
                break
        if kind is None:
            for j, c in enumerate(geo.coils):
                if c.contains(rp):
                    kind, meta = "coil", {"index": j}
                    break
        if kind is None and geo.rotor.contains(rp):
            kind = "rotor"
        if kind is None and geo.stator.contains(rp):
            kind = "stator"
        if kind is None:
            r = math.hypot(rp.x, rp.y)
            if r < geo.shaft_radius + 1e-6:
                kind = "air_shaft"
            elif r < geo.band_radius:
                kind = "air_gap_in"
            elif r < geo.bore_radius:
                kind = "air_gap_out"
            elif r < geo.region_radius * 0.985:
                kind = "air_slot"
            else:
                kind = "air_outer"
        region_of_face.append({"kind": kind, **meta, "face": f})

    # ---- 3) PSLG 구성 ----------------------------------------------
    pt_index: Dict[tuple, int] = {}
    vertices: List[tuple] = []
    segments: List[tuple] = []

    def vid(x: float, y: float) -> int:
        key = (round(x, 6), round(y, 6))
        if key not in pt_index:
            pt_index[key] = len(vertices)
            vertices.append(key)
        return pt_index[key]

    geoms = noded.geoms if isinstance(noded, MultiLineString) else [noded]
    seg_set = set()
    for ls in geoms:
        cs = list(ls.coords)
        for a, b in zip(cs[:-1], cs[1:]):
            ia, ib = vid(*a), vid(*b)
            if ia == ib:
                continue
            key = (min(ia, ib), max(ia, ib))
            if key not in seg_set:
                seg_set.add(key)
                segments.append((ia, ib))

    regions = []
    region_table = []
    for rid, info in enumerate(region_of_face):
        rp = info["face"].representative_point()
        regions.append([rp.x, rp.y, rid, _h_to_area(h[info["kind"]])])
        region_table.append({k: v for k, v in info.items() if k != "face"})

    pslg = {
        "vertices": np.asarray(vertices, dtype=float),
        "segments": np.asarray(segments, dtype=np.int32),
        "regions": np.asarray(regions, dtype=float),
    }
    mesh = tr.triangulate(pslg, f"pq{min_angle}Aa")
    mesh["region_table"] = region_table
    return mesh


def mesh_stats(mesh: dict) -> dict:
    attr = mesh["triangle_attributes"][:, 0].astype(int)
    table = mesh["region_table"]
    counts: Dict[str, int] = {}
    for rid, n in zip(*np.unique(attr, return_counts=True)):
        kind = table[rid]["kind"]
        counts[kind] = counts.get(kind, 0) + int(n)
    return {
        "nodes": int(len(mesh["vertices"])),
        "elements": int(len(mesh["triangles"])),
        "by_region": counts,
    }
