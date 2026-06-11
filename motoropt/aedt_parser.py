"""AEDT(.aedt) 프로젝트 파일 파서.

.aedt는 '$begin <name>' / '$end <name>' 블록 트리 + key=value /
함수형 라인으로 구성된 텍스트 포맷이다. 이 모듈은:
  1) 범용 블록 트리 파서
  2) Maxwell2D 디자인에서 변수/재질(BH 포함)/형상 파트/권선·코일/
     경계/모션/솔브/메시 연산을 추출해 내부 표준 스키마(dict)로 변환
을 제공한다.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .expressions import resolve_variables

# ---------------------------------------------------------------- 블록 트리


@dataclass
class Node:
    name: str
    props: List[str] = field(default_factory=list)   # 원시 라인
    children: List["Node"] = field(default_factory=list)

    def find_all(self, name: str) -> List["Node"]:
        return [c for c in self.children if c.name == name]

    def find(self, name: str) -> Optional["Node"]:
        for c in self.children:
            if c.name == name:
                return c
        return None

    def find_deep(self, name: str) -> Optional["Node"]:
        if self.name == name:
            return self
        for c in self.children:
            r = c.find_deep(name)
            if r is not None:
                return r
        return None

    def find_all_deep(self, name: str) -> List["Node"]:
        out = []
        if self.name == name:
            out.append(self)
        for c in self.children:
            out.extend(c.find_all_deep(name))
        return out

    def get(self, key: str) -> Optional[str]:
        """key='...' 또는 key=value 형태 속성 조회 (따옴표 제거)."""
        pat = key + "="
        for line in self.props:
            s = line.strip()
            if s.startswith(pat):
                v = s[len(pat):]
                return v.strip("'")
        return None


_BEGIN_RE = re.compile(r"^\s*\$begin '(.*)'\s*$")
_END_RE = re.compile(r"^\s*\$end '(.*)'\s*$")


def parse_tree(text: str) -> Node:
    root = Node("ROOT")
    stack = [root]
    for line in text.splitlines():
        m = _BEGIN_RE.match(line)
        if m:
            node = Node(m.group(1))
            stack[-1].children.append(node)
            stack.append(node)
            continue
        if _END_RE.match(line):
            stack.pop()
            continue
        if line.strip():
            stack[-1].props.append(line)
    return root


# ---------------------------------------------------------------- 추출기


def _extract_variables(design: Node) -> Dict[str, str]:
    """VariableProp('name','UD','','expr') 라인 수집."""
    out: Dict[str, str] = {}
    pat = re.compile(r"VariableProp\('([^']+)', '[^']*', '[^']*', '([^']*)'")
    for node in design.find_all_deep("Properties"):
        for line in node.props:
            m = pat.search(line)
            if m:
                out[m.group(1)] = m.group(2)
    return out


def _extract_bh(points_line: str) -> List[List[float]]:
    m = re.search(r"Points\[\d+:\s*(.*)\]", points_line)
    vals = [float(v) for v in m.group(1).split(",")]
    return [[vals[i], vals[i + 1]] for i in range(0, len(vals), 2)]


def _extract_materials(project: Node) -> Dict[str, dict]:
    mats: Dict[str, dict] = {}
    mat_block = project.find_deep("Materials")
    if mat_block is None:
        return mats
    for m in mat_block.children:
        info: dict = {"name": m.name}
        perm = m.find("permeability")
        if perm is not None and perm.get("property_type") == "nonlinear":
            bh_node = perm.find("BHCoordinates")
            for line in bh_node.props:
                if line.strip().startswith("Points["):
                    info["bh_curve"] = _extract_bh(line)
            info["permeability"] = "nonlinear"
        else:
            mu = m.get("permeability")
            info["permeability"] = float(mu) if mu else 1.0
        coer = m.find("magnetic_coercivity")
        if coer is not None:
            mag = coer.get("Magnitude")
            info["coercivity_A_per_m"] = float(re.sub(r"[A-Za-z_]+$", "", mag))
            info["coercivity_dir"] = [
                float(coer.get("DirComp1") or 0),
                float(coer.get("DirComp2") or 0),
                float(coer.get("DirComp3") or 0),
            ]
        for k_src, k_dst, cast in [
            ("conductivity", "conductivity_S_per_m", float),
            ("mass_density", "mass_density_kg_per_m3", float),
            ("core_loss_kh", "core_loss_kh", float),
            ("core_loss_kc", "core_loss_kc", float),
            ("core_loss_ke", "core_loss_ke", float),
            ("stacking_factor", "stacking_factor", float),
        ]:
            v = m.get(k_src)
            if v is not None:
                info[k_dst] = cast(v)
        mats[m.name] = info
    return mats


# ---- 형상 연산 ----------------------------------------------------------

def _params_block(op: Node) -> Optional[Node]:
    for c in op.children:
        if c.name.endswith("Parameters") or c.name.endswith("Parameter"):
            return c
    return None


def _extract_segments(params: Node) -> List[dict]:
    """PolylineParameters 안의 세그먼트(Line/AngularArc/Spline) + 점 목록."""
    segs: List[dict] = []
    pts_node = params.find_deep("PolylinePoints")
    points: List[List[str]] = []
    if pts_node is not None:
        for p in pts_node.find_all("PLPoint"):
            points.append([p.get("X"), p.get("Y"), p.get("Z")])
    seg_node = params.find_deep("PolylineSegments")
    if seg_node is not None:
        for s in seg_node.find_all("PLSegment"):
            seg = {
                "type": s.get("SegmentType"),
                "start_index": int(s.get("StartIndex") or 0),
                "num_points": int(s.get("NoOfPoints") or 0),
            }
            if seg["type"] == "AngularArc":
                seg["arc_angle"] = s.get("ArcAngle")
                seg["arc_center"] = [
                    s.get("ArcCenterX"), s.get("ArcCenterY"), s.get("ArcCenterZ")]
                seg["arc_plane"] = s.get("ArcPlane")
            segs.append(seg)
    return [{"points": points, "segments": segs}]


def _extract_operation(op: Node) -> dict:
    info: dict = {"type": op.get("OperationType"), "id": int(op.get("ID") or -1)}
    params = _params_block(op)
    if params is None:
        return info
    t = info["type"]
    if t == "Circle":
        info.update(center=[params.get("XCenter"), params.get("YCenter")],
                    radius=params.get("Radius"))
    elif t == "Rectangle":
        info.update(position=[params.get("XStart"), params.get("YStart")],
                    xsize=params.get("Width"), ysize=params.get("Height"),
                    axis=params.get("WhichAxis"))
        if info["position"] == [None, None]:
            info["position"] = [params.get("XPosition"), params.get("YPosition")]
            info["xsize"] = info["xsize"] or params.get("XSize")
            info["ysize"] = info["ysize"] or params.get("YSize")
    elif t == "Polyline":
        info["polyline"] = _extract_segments(params)
    elif t == "DuplicateAroundAxis":
        info.update(axis=params.get("WhichAxis"), angle=params.get("AngleStr"),
                    count=params.get("NumClones"))
    elif t == "DuplicateMirror":
        info.update(base=[params.get("DuplicateMirrorBaseX"),
                          params.get("DuplicateMirrorBaseY")],
                    normal=[params.get("DuplicateMirrorNormalX"),
                            params.get("DuplicateMirrorNormalY")])
    elif t == "Rotate":
        info.update(axis=params.get("RotateAxis"), angle=params.get("RotateAngle"))
    elif t == "SplitEdit":
        info.update(plane=params.get("SplitPlane"), side=params.get("SplitWhichSide"))
    elif t == "Fillet":
        info.update(radius=params.get("Radii") or params.get("Radius"))
    elif t in ("Subtract", "Unite", "Intersect"):
        info.update(keep_originals=params.get("KeepOriginals"))
    return info


def _extract_parts(design: Node) -> List[dict]:
    parts: List[dict] = []
    for scope_name in ("ToplevelParts", "OperandParts"):
        scope = design.find_deep(scope_name)
        if scope is None:
            continue
        for gp in scope.find_all("GeometryPart"):
            attrs = gp.find("Attributes")
            ops_node = gp.find("Operations")
            ops = [_extract_operation(o) for o in ops_node.find_all("Operation")] \
                if ops_node else []
            parts.append({
                "name": attrs.get("Name"),
                "material": (attrs.get("MaterialValue") or "").strip('"'),
                "scope": scope_name,
                "operations": ops,
            })
    return parts


# ---- 경계/권선/모션/솔브/메시 -------------------------------------------

def _extract_boundaries(design: Node) -> dict:
    bnode = design.find_deep("Boundaries")
    windings, coils, vector_potential = {}, [], None
    if bnode is None:
        return {"windings": windings, "coils": coils, "vector_potential": None}
    for b in bnode.children:
        bt = b.get("BoundType")
        if bt == "Winding Group":
            windings[b.name] = {
                "id": int(b.get("ID")),
                "type": b.get("Type"),
                "current_expr": b.get("Current"),
                "parallel_branches": b.get("ParallelBranchesNum"),
                "is_solid": b.get("IsSolid") == "true",
            }
        elif bt == "Coil":
            objs = re.search(r"Objects\((.*)\)", "\n".join(b.props))
            coils.append({
                "name": b.name,
                "winding_id": int(b.get("Winding")),
                "conductor_number": b.get("'Conductor number'") or b.get("Conductor number"),
                "polarity": b.get("PolarityType"),
                "object_ids": [int(x) for x in objs.group(1).split(",")] if objs else [],
            })
        elif bt == "Vector Potential":
            vector_potential = {"name": b.name, "value": b.get("Value")}
    return {"windings": windings, "coils": coils, "vector_potential": vector_potential}


def _extract_motion(design: Node) -> dict:
    m = design.find_deep("MotionSetup1")
    if m is None:
        return {}
    return {
        "type": m.get("MotionType"),
        "move_type": m.get("'Move Type'") or "Rotate",
        "axis": m.get("Axis"),
        "init_pos_expr": m.get("InitPos"),
        "angular_velocity_expr": m.get("'Angular Velocity'"),
    }


def _extract_solve(design: Node) -> dict:
    s = design.find_deep("SolveSetups")
    if s is None:
        return {}
    setup = s.children[0] if s.children else None
    if setup is None:
        return {}
    return {
        "name": setup.name,
        "type": setup.get("SetupType"),
        "stop_time_expr": setup.get("StopTime"),
        "time_step_expr": setup.get("TimeStep"),
        "nonlinear_residual": float(setup.get("NonlinearSolverResidual") or 1e-4),
    }


def _extract_mesh_ops(design: Node) -> List[dict]:
    mnode = design.find_deep("MeshOperations")
    out = []
    if mnode is None:
        return out
    for op in mnode.children:
        out.append({
            "name": op.name,
            "restrict_length": op.get("RestrictLength") == "true",
            "max_length": op.get("MaxLength"),
            "restrict_elem": op.get("RestrictElem") == "true",
            "max_elems": int(op.get("NumMaxElem") or 0),
        })
    return out


# ---------------------------------------------------------------- 메인 API


def parse_aedt(path: str) -> dict:
    """단일 Maxwell2D 디자인을 내부 표준 스키마로 파싱."""
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    root = parse_tree(text)
    project = root.find_deep("AnsoftProject")
    design = project.find_deep("Maxwell2DModel")
    if design is None:
        raise ValueError("Maxwell2D 디자인을 찾을 수 없습니다")

    raw_vars = _extract_variables(design)
    model = {
        "design_name": design.get("Name"),
        "solution_type": (design.find_deep("SolutionType").get("SoluteType")
                          if design.find_deep("SolutionType") else None),
        "variables_raw": raw_vars,
        "variables": resolve_variables(raw_vars),
        "materials": _extract_materials(project),
        "parts": _extract_parts(design),
        "boundaries": _extract_boundaries(design),
        "motion": _extract_motion(design),
        "solve": _extract_solve(design),
        "mesh_operations": _extract_mesh_ops(design),
    }
    return model


def detect_magnet_style(model: dict) -> str:
    """자석 파트의 폴리라인 세그먼트로 작도 스타일 판별.

    Spline 세그먼트 존재 → 'spline' (3점 스플라인 브레드로프)
    AngularArc 2개(외호가 편심 중심) → 'eccentric_arc'
    """
    for part in model["parts"]:
        if part["name"] == "Magnet":
            for op in part["operations"]:
                if op["type"] == "Polyline":
                    segs = op["polyline"][0]["segments"]
                    types = [s["type"] for s in segs]
                    if "Spline" in types:
                        return "spline"
                    if types.count("AngularArc") >= 2:
                        return "eccentric_arc"
    return "unknown"


if __name__ == "__main__":
    import sys
    m = parse_aedt(sys.argv[1])
    m_out = dict(m)
    print(json.dumps({k: v for k, v in m_out.items() if k != "materials"},
                     ensure_ascii=False, indent=1, default=str)[:4000])
