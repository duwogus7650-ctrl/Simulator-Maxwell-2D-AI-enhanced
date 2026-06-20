"""슬라이딩 밴드 메시 (Maxwell Motion Band 등가).

로터측(축공~r_i)과 스테이터측(r_o~Region) 메시를 각 1회만 생성하고,
회전 시 로터측은 좌표 강체회전, 공극 띠(r_i~r_o)는 양쪽 경계의
균일 N분할 노드를 각도 오프셋만큼 시프트해 결정론적으로 재연결한다.
→ 위치 간 메시 위상이 동일해 코깅 같은 미세 토크의 메시 노이즈가
원천 제거된다.
"""
from __future__ import annotations

import math
import warnings
from typing import Dict

import numpy as np
import shapely
import triangle as tr
from shapely.geometry import LineString
from shapely.ops import polygonize

from .geometry import MotorGeometry, _arc
from .meshing import H_DEFAULT, _h_to_area


def _pslg_from_lines(lines, seg_tol: float = 0.0, face_tol: float = 1e-7):
    """배치(arrangement) → PSLG.

    seg_tol [mm]: 이보다 짧은 세그먼트는 제거 — 부동소수 노이즈로 생긴
    µm급 미세 변이 품질(q) 세분화를 연쇄 유발해 메시가 폭주하는 것을
    차단한다(모터 스케일에서 2µm 미만의 의도적 형상은 없음).
    face_tol [mm²]: 슬리버 면 제거 — 미세 변 제거로 경계가 열리며
    이웃 면에 자연 흡수된다.
    """
    noded = shapely.union_all(lines, grid_size=1e-6)
    faces = [f for f in polygonize(noded) if f.area > face_tol]
    pts, V, S, seen = {}, [], [], set()

    def vid(x, y):
        k = (round(x, 6), round(y, 6))
        if k not in pts:
            pts[k] = len(V)
            V.append(k)
        return pts[k]

    tol2 = seg_tol * seg_tol
    geoms = noded.geoms if hasattr(noded, "geoms") else [noded]
    for ls in geoms:
        cs = list(ls.coords)
        for a, b in zip(cs[:-1], cs[1:]):
            if tol2 > 0 and (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 < tol2:
                continue                          # 미세 세그먼트 스킵(기본 꺼짐)
            ia, ib = vid(*a), vid(*b)
            if ia == ib:
                continue
            key = (min(ia, ib), max(ia, ib))
            if key not in seen:
                seen.add(key)
                S.append((ia, ib))
    return np.asarray(V, float), np.asarray(S, np.int32), faces


def _uniform_circle_pts(r: float, n: int):
    a = np.arange(n) * (2 * math.pi / n)
    return np.column_stack([r * np.cos(a), r * np.sin(a)])


def _classify(faces, geo: MotorGeometry, inner: bool):
    table = []
    for f in faces:
        rp = f.representative_point()
        kind, meta = None, {}
        if inner:
            for i, (p, ang, pol) in enumerate(geo.magnets):
                if p.contains(rp):
                    kind, meta = "magnet", {"index": i, "polarity": pol}
                    break
            if kind is None and geo.rotor.contains(rp):
                kind = "rotor"
            if kind is None:
                r = math.hypot(rp.x, rp.y)
                kind = "air_shaft" if r < geo.shaft_radius + 1e-6 \
                    else "air_gap_in"
        else:
            for j, c in enumerate(geo.coils):
                if c.contains(rp):
                    kind, meta = "coil", {"index": j}
                    break
            if kind is None and geo.stator.contains(rp):
                kind = "stator"
            if kind is None:
                r = math.hypot(rp.x, rp.y)
                kind = "air_gap_out" if r < geo.bore_radius + 0.05 \
                    else ("air_slot" if r < geo.region_radius * 0.985
                          else "air_outer")
        table.append({"kind": kind, **meta})
    return table


def _mesh_side(lines, geo, h, inner: bool, min_angle=25.0):
    V, S, faces = _pslg_from_lines(lines)
    table = _classify(faces, geo, inner)
    regions = []
    for rid, (f, info) in enumerate(zip(faces, table)):
        rp = f.representative_point()
        regions.append([rp.x, rp.y, rid, _h_to_area(h[info["kind"]])])
    pslg = {"vertices": V, "segments": S,
            "regions": np.asarray(regions, float)}
    if not inner:                       # r_o 원 내부는 외측 도메인이 아님
        pslg["holes"] = np.asarray([[0.0, 0.0]])
    mesh = tr.triangulate(pslg, f"pq{min_angle}Aa")
    if len(mesh["triangles"]) > 600_000:
        raise ValueError(f"메시 폭주: {len(mesh['triangles'])} 요소")
    mesh["region_table"] = table
    return mesh


class SlidingBandMesh:
    """분리 메시 보유 + merge(angle)로 전체 메시 합성."""

    def __init__(self, geo: MotorGeometry, n_band: int = 2880,
                 gap_frac: tuple = (0.2, 0.8),
                 h: Dict[str, float] | None = None):
        self.geo = geo
        self.n = n_band
        self.pitch = 2 * math.pi / n_band
        h = {**H_DEFAULT, **(h or {})}
        r_mag = geo.params["D_ro"] / 2          # 자석 크라운 정점
        r_bore = geo.bore_radius
        gap = r_bore - r_mag
        self.r_i = r_mag + gap * gap_frac[0]
        self.r_o = r_mag + gap * gap_frac[1]

        # ---- 로터측 메시 (1회) -----------------------------------------
        lines_in = []
        self._add(lines_in, geo.rotor)
        for p, _, _ in geo.magnets:
            self._add(lines_in, p)
        ci = _uniform_circle_pts(self.r_i, n_band)
        lines_in.append(LineString(np.vstack([ci, ci[:1]])))
        self.inner = _mesh_side(lines_in, geo, h, inner=True)

        # ---- 스테이터측 메시 (1회) -------------------------------------
        lines_out = []
        self._add(lines_out, geo.stator)
        from .meshing import weld_boundary
        _stb = geo.stator.boundary
        for c in geo.coils:
            lines_out.append(LineString(weld_boundary(c.exterior.coords, _stb)))
            for hole in c.interiors:
                lines_out.append(LineString(weld_boundary(hole.coords, _stb)))
        co = _uniform_circle_pts(self.r_o, n_band)
        lines_out.append(LineString(np.vstack([co, co[:1]])))
        lines_out.append(LineString(_arc(geo.region_radius, 0, 2 * math.pi, 192)))
        self.outer = _mesh_side(lines_out, geo, h, inner=False)

        # ---- 밴드 경계 노드 인덱스 (각도순) -----------------------------
        self.idx_i = self._ring_nodes(self.inner, self.r_i)
        self.idx_o = self._ring_nodes(self.outer, self.r_o)
        assert len(self.idx_i) == n_band, \
            f"내측 링 노드 {len(self.idx_i)} != {n_band} (Steiner 삽입 발생)"
        assert len(self.idx_o) == n_band, \
            f"외측 링 노드 {len(self.idx_o)} != {n_band}"

    @staticmethod
    def _add(lines, poly):
        if poly.geom_type == "MultiPolygon":
            for g in poly.geoms:
                SlidingBandMesh._add(lines, g)
            return
        lines.append(LineString(poly.exterior.coords))
        for hole in poly.interiors:
            lines.append(LineString(hole.coords))

    @staticmethod
    def _ring_nodes(mesh, r, tol=1e-4):
        V = mesh["vertices"]
        rr = np.hypot(V[:, 0], V[:, 1])
        idx = np.where(np.abs(rr - r) < tol)[0]
        ang = np.arctan2(V[idx, 1], V[idx, 0]) % (2 * math.pi)
        return idx[np.argsort(ang)]

    # --------------------------------------------------------------
    def merge(self, angle_deg: float) -> dict:
        """로터 회전각 angle_deg에서 전체 메시 dict 반환."""
        th = math.radians(angle_deg)
        c, s = math.cos(th), math.sin(th)
        Vi = self.inner["vertices"]
        Vi_rot = Vi @ np.array([[c, s], [-s, c]])      # (x,y)·Rᵀ = 회전
        Vo = self.outer["vertices"]
        n_in = len(Vi)

        V = np.vstack([Vi_rot, Vo])
        Ti = np.asarray(self.inner["triangles"], np.int64)
        To = np.asarray(self.outer["triangles"], np.int64) + n_in

        # ---- 밴드 연결: 내측 j ↔ 외측 m(j) -----------------------------
        # 내측 노드 j의 현재 각도 = j·pitch + th (idx_i가 각도순이므로)
        # 알려진 한계: 로터 정점은 정확히 th로 회전하지만 밴드 연결은
        # 정수 셀 시프트 k로 양자화돼 ±pitch/2의 각도 전단 잔차가 남는다
        # (코깅 검증은 별도 과제). 잔차가 크면 진단 경고만 낸다.
        k = int(round(th / self.pitch))
        resid = abs(th / self.pitch - k)
        if resid > 0.1 and not getattr(self, "_resid_warned", False):
            # 인스턴스당 1회만 — 미세각 스윕(코깅)에서 매 스텝 스팸 방지
            self._resid_warned = True
            warnings.warn(
                f"슬라이딩 밴드 각도 잔차 {resid:.2f}셀 — "
                f"코깅 미세토크에 계통오차 가능 (이후 동일 경고 생략)")
        n = self.n
        j = np.arange(n)
        in_a = self.idx_i[j]
        in_b = self.idx_i[(j + 1) % n]
        m = (j + k) % n
        out_a = self.idx_o[m] + n_in
        out_b = self.idx_o[(m + 1) % n] + n_in
        # 쿼드(in_a→in_b→out_b→out_a) 분할 — CCW
        Tband = np.empty((2 * n, 3), np.int64)
        Tband[0::2] = np.column_stack([in_a, out_a, out_b])
        Tband[1::2] = np.column_stack([in_a, out_b, in_b])

        T = np.vstack([Ti, Tband, To])

        ai = self.inner["triangle_attributes"][:, 0].astype(int)
        ao = self.outer["triangle_attributes"][:, 0].astype(int)
        n_ri = len(self.inner["region_table"])
        band_id = n_ri + len(self.outer["region_table"])
        attr = np.concatenate([ai, np.full(2 * n, band_id),
                               ao + n_ri])[:, None].astype(float)
        table = (self.inner["region_table"]
                 + self.outer["region_table"]
                 + [{"kind": "air_gap_band"}])
        return {"vertices": V, "triangles": T,
                "triangle_attributes": attr, "region_table": table}
