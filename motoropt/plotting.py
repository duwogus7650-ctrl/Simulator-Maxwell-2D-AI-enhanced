"""형상/메시 플로터 — Maxwell 모델러 화면과 같은 색상 체계."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
for _f in ("/usr/share/fonts/truetype/nanum/NanumGothic.ttf",):
    try:
        fm.fontManager.addfont(_f)
        matplotlib.rcParams["font.family"] = "NanumGothic"
    except Exception:
        pass
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly
import numpy as np

COL = {
    "rotor": "#c8c8c8", "stator": "#c8c8c8",
    "coil": "#ff8c00", "magnet_n": "#e02020", "magnet_s": "#2040e0",
    "edge": "#404040",
}


def _draw_poly(ax, poly, fc, ec=COL["edge"], lw=0.4, alpha=1.0):
    if poly.geom_type == "MultiPolygon":
        for g in poly.geoms:
            _draw_poly(ax, g, fc, ec, lw, alpha)
        return
    ext = np.asarray(poly.exterior.coords)
    ax.add_patch(MplPoly(ext, closed=True, facecolor=fc, edgecolor=ec,
                         linewidth=lw, alpha=alpha))
    for hole in poly.interiors:
        ax.add_patch(MplPoly(np.asarray(hole.coords), closed=True,
                             facecolor="white", edgecolor=ec, linewidth=lw))


def plot_geometry(geo, path: str, title: str = ""):
    fig, ax = plt.subplots(figsize=(9, 9), dpi=130)
    _draw_poly(ax, geo.stator, COL["stator"])
    _draw_poly(ax, geo.rotor, COL["rotor"])
    for poly, ang, pol in geo.magnets:
        _draw_poly(ax, poly, COL["magnet_n"] if pol > 0 else COL["magnet_s"])
    for c in geo.coils:
        _draw_poly(ax, c, COL["coil"])
    for r, ls in ((geo.band_radius, "--"), (geo.region_radius, "-")):
        th = np.linspace(0, 2 * np.pi, 256)
        ax.plot(r * np.cos(th), r * np.sin(th), ls, color="#208020", lw=0.7)
    # 착자 방향 화살표
    for poly, ang, pol in geo.magnets:
        c = poly.centroid
        d = 2.2 * pol
        ax.annotate("", xy=(c.x + d * np.cos(ang), c.y + d * np.sin(ang)),
                    xytext=(c.x, c.y),
                    arrowprops=dict(arrowstyle="->", color="k", lw=0.9))
    lim = geo.region_radius * 1.06
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal"); ax.set_title(title)
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_mesh(mesh: dict, path: str, title: str = ""):
    """triangle 결과(dict: vertices, triangles, triangle_attributes) 플롯."""
    import matplotlib.tri as mtri
    v = mesh["vertices"]; t = mesh["triangles"]
    attr = mesh["triangle_attributes"][:, 0].astype(int)
    tri = mtri.Triangulation(v[:, 0], v[:, 1], t)
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.6), dpi=130)
    for ax, zoom in zip(axes, (False, True)):
        ax.tripcolor(tri, facecolors=attr, cmap="tab20", alpha=0.55)
        ax.triplot(tri, color="k", lw=0.08)
        ax.set_aspect("equal")
        if zoom:
            ax.set_xlim(8, 36); ax.set_ylim(8, 36)
            ax.set_title("공극부 확대")
        else:
            lim = np.abs(v).max() * 1.03
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
            ax.set_title(title)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_field(solver, res, path: str, title: str = ""):
    """|B| 컬러맵 + 자속선(Az 등고선) — Maxwell 필드 오버레이 스타일."""
    import matplotlib.tri as mtri
    V = solver.V * 1e3  # m → mm
    tri = mtri.Triangulation(V[:, 0], V[:, 1], solver.T)
    fig, ax = plt.subplots(figsize=(10, 9), dpi=130)
    tp = ax.tripcolor(tri, facecolors=res.Bmag, cmap="jet",
                      vmin=0, vmax=max(2.2, res.Bmag.max() * 0.92))
    levels = np.linspace(res.A.min(), res.A.max(), 31)
    ax.tricontour(tri, res.A, levels=levels, colors="k", linewidths=0.35)
    cb = fig.colorbar(tp, ax=ax, shrink=0.85)
    cb.set_label("|B| [T]")
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
