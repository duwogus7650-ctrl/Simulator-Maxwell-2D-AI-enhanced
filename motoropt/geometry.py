"""파라메트릭 모터 형상 생성기 (2D 단면, 단위 mm).

Maxwell 작도 히스토리를 그대로 재현한다. 같은 반경을 공유하는
인터페이스(자석↔로터 r_ri, 슈↔코일링 r_b2)는 통일 각도 그리드로
생성해 인접 영역 경계가 좌표 수준에서 정확히 일치(적합 메시 보장)한다.

자석 크라운 스타일 2종:
  'spline'        외측면 3점 스플라인 — 원호 근사 (정점 (0, D_ro/2))
  'eccentric_arc' 중심 (0, Magnet_R_Offset) 편심 원호 (theta_two*2)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import shapely
from shapely.geometry import Polygon, box
from shapely.affinity import rotate as shp_rotate
from shapely.ops import unary_union

MM = 1e3
SNAP = 1e-6  # mm 그리드 — 부동소수 노이즈 스냅


def _grid_angles(feature_angles: List[float], n_base: int = 720) -> np.ndarray:
    """균일 그리드 + 피처 각도 병합(중복 1e-9 rad 제거), 정렬."""
    base = np.linspace(-math.pi, math.pi, n_base, endpoint=False)
    allang = np.concatenate([base, np.asarray(feature_angles, float)])
    allang = np.angle(np.exp(1j * allang))  # [-pi, pi) 정규화
    allang.sort()
    keep = np.ones(len(allang), bool)
    keep[1:] = np.diff(allang) > 1e-9
    return allang[keep]


def _arc_from_grid(r: float, grid: np.ndarray, a0: float, a1: float):
    """그리드 부분집합으로 a0→a1(둘 다 그리드에 존재) 원호 점열 (CCW)."""
    a0n = math.atan2(math.sin(a0), math.cos(a0))
    a1n = math.atan2(math.sin(a1), math.cos(a1))
    i0 = int(np.argmin(np.abs(grid - a0n)))
    i1 = int(np.argmin(np.abs(grid - a1n)))
    idx = []
    i = i0
    while True:
        idx.append(i)
        if i == i1:
            break
        i = (i + 1) % len(grid)
    ang = grid[idx]
    return list(zip(r * np.cos(ang), r * np.sin(ang)))


def _arc(r, a0, a1, n=48):
    th = np.linspace(a0, a1, n)
    return list(zip(r * np.cos(th), r * np.sin(th)))


def _arc_about(cx, cy, r, a0, a1, n=48):
    th = np.linspace(a0, a1, n)
    return list(zip(cx + r * np.cos(th), cy + r * np.sin(th)))


def _circle(r: float, n: int = 192) -> Polygon:
    return Polygon(_arc(r, -math.pi, math.pi - 2 * math.pi / n, n))


def _circle_from_grid(r: float, grid: np.ndarray) -> Polygon:
    return Polygon(list(zip(r * np.cos(grid), r * np.sin(grid))))


def _circle3pt_arc(p1, p2, p3, n=40):
    ax, ay = p1; bx, by = p2; cx, cy = p3
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay)
          + (cx**2 + cy**2) * (ay - by)) / d
    uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx)
          + (cx**2 + cy**2) * (bx - ax)) / d
    r = math.hypot(ax - ux, ay - uy)
    aa = math.atan2(ay - uy, ax - ux)
    cc = math.atan2(cy - uy, cx - ux)
    # 짧은 호(중간점 경유) 선택
    if cc < aa:
        cc += 2 * math.pi
    mid = math.atan2(by - uy, bx - ux)
    if mid < aa:
        mid += 2 * math.pi
    if not (aa < mid < cc):  # 반대 방향
        aa, cc = aa + 2 * math.pi, cc
        aa, cc = cc, aa - 2 * math.pi + 2 * math.pi
    return _arc_about(ux, uy, r, aa, cc, n)


def _bezier(p0, pc, p1, n=8):
    t = np.linspace(0, 1, n)[:, None]
    p0 = np.asarray(p0); pc = np.asarray(pc); p1 = np.asarray(p1)
    pts = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * pc + t ** 2 * p1
    return [tuple(p) for p in pts]


@dataclass
class MotorGeometry:
    rotor: Polygon
    stator: Polygon
    magnets: List[tuple] = field(default_factory=list)  # (poly, 착자각, 극성)
    coils: List[Polygon] = field(default_factory=list)
    band_radius: float = 0.0
    region_radius: float = 0.0
    shaft_radius: float = 0.0
    bore_radius: float = 0.0
    params: Dict[str, float] = field(default_factory=dict)


def _build_magnet(phi: float, grid_ri: np.ndarray, v: dict,
                  style: str) -> Polygon:
    """중심각 phi(rad)에 자석 1개를 절대좌표로 생성."""
    th1 = v["theta_one"]
    r_ri = v["D_ro"] / 2 - v["T_m"]
    rot = phi - math.pi / 2  # 베이스(+Y 중심) → phi 회전량
    R = np.array([[math.cos(rot), -math.sin(rot)],
                  [math.sin(rot), math.cos(rot)]])

    def xf(pts):
        return [tuple(R @ np.asarray(p)) for p in pts]

    sx = r_ri * math.sin(th1)
    cy = r_ri * math.cos(th1)
    top_r = (sx, cy + v["T_m2"])   # 베이스 좌표계 우측벽 상단
    top_l = (-sx, cy + v["T_m2"])
    if style == "spline":
        crown = _circle3pt_arc(top_r, (0, v["D_ro"] / 2), top_l, 48)
    elif style == "eccentric_arc":
        off = v["Magnet_R_Offset"]
        crown = _arc_about(0, off, v["D_ro"] / 2 - off,
                           math.pi / 2 - v["theta_two"],
                           math.pi / 2 + v["theta_two"], 48)
        crown = list(reversed(crown))  # top_r → top_l 방향
        crown[0], crown[-1] = top_r, top_l
    else:
        raise ValueError(style)

    r_f = v.get("MagnetR", 0.0)
    # 상부 모서리 라운딩(베지어): 측벽에서 r_f 아래 ↔ 크라운에서 r_f 진행점
    bot_r = (sx, cy)
    wall_r_end = (sx, cy + v["T_m2"] - r_f) if r_f > 0 else top_r
    wall_l_start = (-sx, cy + v["T_m2"] - r_f) if r_f > 0 else top_l
    if r_f > 0:
        crown_np = np.asarray(crown)
        seg = np.linalg.norm(np.diff(crown_np, axis=0), axis=1).cumsum()
        k_r = int(np.searchsorted(seg, r_f)) + 1
        k_l = len(crown) - 1 - int(np.searchsorted(seg[::-1].cumsum()
                                                   if False else seg, 0))
        # 좌측: 끝에서 r_f 만큼 떨어진 인덱스
        seg_rev = np.linalg.norm(np.diff(crown_np[::-1], axis=0),
                                 axis=1).cumsum()
        k_l = len(crown) - 1 - (int(np.searchsorted(seg_rev, r_f)) + 1)
        fil_r = _bezier(wall_r_end, top_r, tuple(crown_np[k_r]), 8)
        fil_l = _bezier(tuple(crown_np[k_l]), top_l, wall_l_start, 8)
        crown_mid = [tuple(p) for p in crown_np[k_r:k_l + 1]]
    else:
        fil_r, fil_l, crown_mid = [], crown, []

    # 절대좌표 폴리곤: 내호(그리드, phi-th1 → phi+th1 CCW) → 좌측벽 위로
    # → 크라운(좌→우) → 우측벽 아래로
    inner = _arc_from_grid(r_ri, grid_ri, phi - th1, phi + th1)
    left_wall = xf([wall_l_start]) if r_f > 0 else []
    pts = inner + xf([wall_l_start] if r_f > 0 else []) \
        + xf(list(reversed(fil_l))) + xf(list(reversed(crown_mid))) \
        + xf(list(reversed(fil_r))) + xf([wall_r_end] if r_f > 0 else [])
    poly = Polygon(pts)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


def build_motor(v_si: Dict[str, float], magnet_style: str = "spline",
                rotor_angle_deg: float = 0.0) -> MotorGeometry:
    mm_keys = ("D_ro", "T_m", "D_shaft", "D_so", "T_Yoke", "g", "D_si",
               "W_t", "d_1", "d_2", "H_t", "W_so", "MagnetR", "T_m2",
               "Magnet_R_Offset", "L_stk")
    v = {k: (val * MM if k in mm_keys else val) for k, val in v_si.items()}

    n_slot = int(round(v["N_slot"]))
    n_pole = int(round(v["N_pole"]))
    th1 = v["theta_one"]
    r_ri = v["D_ro"] / 2 - v["T_m"]
    r_sh = v["D_shaft"] / 2
    r_yo = v["D_so"] / 2
    r_yi = r_yo - v["T_Yoke"]
    r_b0 = v["D_si"] / 2
    r_b1 = r_b0 + v["d_1"]
    r_b2 = r_b1 + v["d_2"]
    th_ss, th_ss2, th_st = v["theta_ss"], v["theta_ss2"], v["theta_st"]

    # ---- 공유 인터페이스 각도 그리드 -------------------------------
    rot0 = math.radians(rotor_angle_deg)
    mag_centers = [math.pi / 2 + math.pi / n_pole + 2 * math.pi / n_pole * k
                   + rot0 for k in range(n_pole)]
    grid_ri = _grid_angles(
        [c + s * th1 for c in mag_centers for s in (-1, 1)], 720)
    tooth_centers = [2 * math.pi / n_slot * k for k in range(n_slot)]
    grid_b2 = _grid_angles(
        [c + s * th_st / 2 for c in tooth_centers for s in (-1, 1)], 720)

    # ---- 로터 -------------------------------------------------------
    rotor = _circle_from_grid(r_ri, grid_ri).difference(_circle(r_sh))

    # ---- 스테이터 ---------------------------------------------------
    yoke = _circle(r_yo).difference(_circle(r_yi))
    teeth = []
    for c in tooth_centers:
        rot = math.degrees(c)
        tooth = box(r_b0 - 0.05, -v["W_t"] / 2,
                    r_b0 + v["H_t"] + v["d_1"] + v["d_2"], v["W_t"] / 2)
        shoe_pts = [(r_b1 * math.cos(th_ss2 / 2), r_b1 * math.sin(th_ss2 / 2))]
        shoe_pts += _arc(r_b0, th_ss / 2, -th_ss / 2, 64)
        shoe_pts.append((r_b1 * math.cos(th_ss2 / 2),
                         -r_b1 * math.sin(th_ss2 / 2)))
        # 슈 외호: 절대각 그리드 사용 (코일링과 공유)
        shoe_outer = _arc_from_grid(r_b2, grid_b2, c - th_st / 2, c + th_st / 2)
        local = unary_union([shp_rotate(Polygon(
            shoe_pts + [(r_b2 * math.cos(th_st / 2),
                         -r_b2 * math.sin(th_st / 2)),
                        (r_b2 * math.cos(th_st / 2),
                         r_b2 * math.sin(th_st / 2))]), rot, origin=(0, 0)),
            shp_rotate(tooth, rot, origin=(0, 0))])
        # 회전 노이즈 제거: 슈 외호 모서리만 그리드 좌표로 정밀 교체는
        # set_precision 스냅으로 처리
        teeth.append(local)
    stator = unary_union([yoke] + teeth)

    # ---- 자석 -------------------------------------------------------
    magnets = []
    for k, phi in enumerate(mag_centers):
        poly = _build_magnet(phi, grid_ri, v, magnet_style)
        magnets.append((poly, phi, 1 if k % 2 == 0 else -1))

    # ---- 코일 -------------------------------------------------------
    coil_ring = _circle(r_yi).difference(_circle_from_grid(r_b2, grid_b2))
    slot_air = coil_ring.difference(stator)
    slots = sorted(slot_air.geoms,
                   key=lambda p: math.atan2(p.centroid.y, p.centroid.x)
                   % (2 * math.pi))
    coils = []
    big = 3 * r_yo
    for s in slots:
        a = math.atan2(s.centroid.y, s.centroid.x)
        nx, ny = -math.sin(a), math.cos(a)
        half = Polygon([(0, 0), (big * math.cos(a), big * math.sin(a)),
                        (big * (math.cos(a) + nx), big * (math.sin(a) + ny)),
                        (big * nx, big * ny)])
        for piece in (s.difference(half), s.intersection(half)):
            if piece.is_empty:
                continue
            if piece.geom_type == "MultiPolygon":
                piece = max(piece.geoms, key=lambda g: g.area)
            coils.append(piece)

    # ---- 스냅 (부동소수 노이즈 제거) -------------------------------
    snap = lambda p: shapely.set_precision(p, SNAP)
    rotor, stator = snap(rotor), snap(stator)
    magnets = [(snap(p), a, s) for p, a, s in magnets]
    coils = [snap(c) for c in coils]

    return MotorGeometry(
        rotor=rotor, stator=stator, magnets=magnets, coils=coils,
        band_radius=v["D_ro"] / 2 + v["g"] / 2,
        region_radius=r_yo * 1.01,
        shaft_radius=r_sh, bore_radius=r_b0, params=v)
