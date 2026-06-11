"""2D 비선형 정자기 FEM 솔버 (벡터 포텐셜 Az, 1차 삼각요소).

지배방정식: ∇×(ν ∇×A) = J_z + ∇×(ν_m B_r)
약형식(요소별):
    잔차   r_e = ν_e S_e a_e − f_e
    야코비안 J_e = ν_e S_e + (2 ν'_e / Δ_e) (S_e a_e)(S_e a_e)ᵀ
    S_e(ij) = (b_i b_j + c_i c_j) / (4Δ)
자석 소스: f_i = ν_m (Brx c_i − Bry b_i) / 2   (요소별 방사 착자)
코일 소스: f_i = J_z Δ / 3
경계: 외곽 Region 원 위 A = 0 (Dirichlet)
단위: 내부 SI(m) — 메시는 mm로 들어오므로 1e-3 변환.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .materials import NuCurve, PMLinear, NU0, MU0


@dataclass
class SolverResult:
    A: np.ndarray            # 절점 Az [Wb/m]
    Bx: np.ndarray           # 요소 Bx [T]
    By: np.ndarray
    Bmag: np.ndarray
    iterations: int
    residual: float


class Magnetostatic2D:
    def __init__(self, mesh: dict, materials: Dict[str, dict],
                 steel_name: str, magnet_name: str):
        self.V = np.asarray(mesh["vertices"], float) * 1e-3       # mm → m
        self.T = np.asarray(mesh["triangles"], np.int64)
        self.attr = mesh["triangle_attributes"][:, 0].astype(int)
        self.table: List[dict] = mesh["region_table"]
        self.kind = np.array([self.table[r]["kind"] for r in self.attr])

        # ---- 요소 기하 ------------------------------------------------
        x = self.V[self.T, 0]; y = self.V[self.T, 1]              # (ne,3)
        self.b = np.stack([y[:, 1] - y[:, 2], y[:, 2] - y[:, 0],
                           y[:, 0] - y[:, 1]], axis=1)
        self.c = np.stack([x[:, 2] - x[:, 1], x[:, 0] - x[:, 2],
                           x[:, 1] - x[:, 0]], axis=1)
        det = (x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0]) \
            - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0])
        self.area = 0.5 * det
        assert (self.area > 0).all(), "CW 요소 존재 — 메시 방향 확인 필요"
        # S_e = (b bᵀ + c cᵀ)/(4Δ)
        self.S = (self.b[:, :, None] * self.b[:, None, :]
                  + self.c[:, :, None] * self.c[:, None, :]) \
            / (4.0 * self.area[:, None, None])
        self.centroid = np.stack([x.mean(1), y.mean(1)], axis=1)

        # ---- 재질 배치 ------------------------------------------------
        self.steel = NuCurve(materials[steel_name]["bh_curve"],
                             materials[steel_name].get("stacking_factor", 1.0))
        self.pm = PMLinear(materials[magnet_name]["bh_curve"])
        self.is_steel = np.isin(self.kind, ("rotor", "stator"))
        self.is_magnet = self.kind == "magnet"
        self.is_coil = self.kind == "coil"

        self.nu_const = np.full(len(self.T), NU0)
        self.nu_const[self.is_magnet] = self.pm.nu

        # 자석 방사 착자 벡터 (요소 도심 기준 ±r̂)
        self.Br_vec = np.zeros((len(self.T), 2))
        pol = np.array([self.table[r].get("polarity", 0) for r in self.attr])
        th = np.arctan2(self.centroid[:, 1], self.centroid[:, 0])
        m = self.is_magnet
        self.Br_vec[m, 0] = pol[m] * self.pm.Br * np.cos(th[m])
        self.Br_vec[m, 1] = pol[m] * self.pm.Br * np.sin(th[m])

        # ---- Dirichlet 절점 (외곽 원) ----------------------------------
        r_nodes = np.hypot(self.V[:, 0], self.V[:, 1])
        self.r_outer = r_nodes.max()
        self.fixed = np.where(r_nodes > self.r_outer - 3e-5)[0]
        self.free = np.setdiff1d(np.arange(len(self.V)), self.fixed)

        self.n_nodes = len(self.V)
        rows = np.repeat(self.T, 3, axis=1).ravel()
        cols = np.tile(self.T, (1, 3)).ravel()
        self._ij = (rows, cols)

    # ------------------------------------------------------------------
    def set_coil_currents(self, coil_currents: Dict[int, float]):
        """코일 인덱스(0..35) → 총 암페어턴 [A·turns]. J = NI/면적."""
        self.Jz = np.zeros(len(self.T))
        if not coil_currents:
            return
        coil_idx = np.array([self.table[r].get("index", -1)
                             for r in self.attr])
        for ci, ampturns in coil_currents.items():
            sel = self.is_coil & (coil_idx == ci)
            area = self.area[sel].sum()
            if area > 0:
                self.Jz[sel] = ampturns / area

    # ------------------------------------------------------------------
    def _load_vector(self) -> np.ndarray:
        f = np.zeros(self.n_nodes)
        # 코일
        if hasattr(self, "Jz"):
            fe = (self.Jz * self.area / 3.0)[:, None].repeat(3, 1)
            np.add.at(f, self.T.ravel(), fe.ravel())
        # 자석: f_i = ν_m (Brx c_i − Bry b_i)/2
        m = self.is_magnet
        fm = self.pm.nu * (self.Br_vec[m, 0:1] * self.c[m]
                           - self.Br_vec[m, 1:2] * self.b[m]) / 2.0
        np.add.at(f, self.T[m].ravel(), fm.ravel())
        return f

    def _nu_field(self, A: np.ndarray):
        """요소별 ν, dν/dB², B 성분."""
        a_e = A[self.T]                                   # (ne,3)
        dAdx = (a_e * self.b).sum(1) / (2 * self.area)
        dAdy = (a_e * self.c).sum(1) / (2 * self.area)
        Bx, By = dAdy, -dAdx
        b2 = Bx * Bx + By * By
        nu = self.nu_const.copy()
        dnu = np.zeros_like(nu)
        if self.is_steel.any():
            nu_s, dnu_s = self.steel.nu_and_dnu(b2[self.is_steel])
            nu[self.is_steel] = nu_s
            dnu[self.is_steel] = dnu_s
        return nu, dnu, Bx, By

    # ------------------------------------------------------------------
    def solve(self, max_iter: int = 60, tol: float = 1e-4,
              verbose: bool = False) -> SolverResult:
        f = self._load_vector()
        A = np.zeros(self.n_nodes)

        # 초기해: 선형(초기 투자율) 풀이
        nu, dnu, Bx, By = self._nu_field(A)
        K = self._assemble(nu[:, None, None] * self.S)
        A = self._solve_dirichlet(K, f, A, full_solve=True)

        f_norm = np.linalg.norm(f[self.free]) + 1e-30
        res = np.inf
        it = 0
        for it in range(1, max_iter + 1):
            nu, dnu, Bx, By = self._nu_field(A)
            Sa = np.einsum("eij,ej->ei", self.S, A[self.T])
            r = np.zeros(self.n_nodes)
            np.add.at(r, self.T.ravel(), (nu[:, None] * Sa).ravel())
            r -= f
            res = np.linalg.norm(r[self.free]) / f_norm
            if verbose:
                print(f"  NR {it:2d}: |r|/|f| = {res:.3e}")
            if res < tol:
                break
            # 야코비안
            coef = 2.0 * dnu / self.area
            Je = nu[:, None, None] * self.S \
                + coef[:, None, None] * Sa[:, :, None] * Sa[:, None, :]
            J = self._assemble(Je)
            dA = self._solve_dirichlet(J, -r, np.zeros_like(A),
                                       full_solve=False)
            # 감쇠 라인서치 (잔차 증가 시 절반씩)
            alpha = 1.0
            for _ in range(8):
                A_try = A + alpha * dA
                nu_t, _, _, _ = self._nu_field(A_try)
                Sa_t = np.einsum("eij,ej->ei", self.S, A_try[self.T])
                r_t = np.zeros(self.n_nodes)
                np.add.at(r_t, self.T.ravel(), (nu_t[:, None] * Sa_t).ravel())
                r_t -= f
                if np.linalg.norm(r_t[self.free]) < (1 - 0.1 * alpha) \
                        * np.linalg.norm(r[self.free]):
                    break
                alpha *= 0.5
            A = A + alpha * dA

        nu, dnu, Bx, By = self._nu_field(A)
        return SolverResult(A=A, Bx=Bx, By=By,
                            Bmag=np.hypot(Bx, By),
                            iterations=it, residual=float(res))

    # ------------------------------------------------------------------
    def _assemble(self, Ke: np.ndarray) -> sp.csr_matrix:
        return sp.coo_matrix((Ke.ravel(), self._ij),
                             shape=(self.n_nodes, self.n_nodes)).tocsr()

    def _solve_dirichlet(self, K: sp.csr_matrix, rhs: np.ndarray,
                         A0: np.ndarray, full_solve: bool) -> np.ndarray:
        fr = self.free
        Kff = K[fr][:, fr]
        b = rhs[fr]
        x = spla.spsolve(Kff.tocsc(), b)
        out = A0.copy() if full_solve else np.zeros_like(A0)
        out[fr] = x
        out[self.fixed] = 0.0
        return out
