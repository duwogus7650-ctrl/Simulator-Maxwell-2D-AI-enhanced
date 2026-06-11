"""P1 데모: aedt 파싱 → 형상 재구성 → 적합 메시 → 플롯."""
import argparse, json, warnings
warnings.filterwarnings("ignore")
from motoropt.aedt_parser import parse_aedt, detect_magnet_style
from motoropt.geometry import build_motor
from motoropt.meshing import build_mesh, mesh_stats
from motoropt.plotting import plot_geometry, plot_mesh

p = argparse.ArgumentParser()
p.add_argument("aedt")
p.add_argument("-o", "--outdir", default=".")
args = p.parse_args()

model = parse_aedt(args.aedt)
style = detect_magnet_style(model)
geo = build_motor(model["variables"], style)
mesh = build_mesh(geo)
st = mesh_stats(mesh)
print(json.dumps(st, ensure_ascii=False, indent=1))
plot_geometry(geo, f"{args.outdir}/geometry.png",
              f"{model['design_name']} (magnet={style})")
plot_mesh(mesh, f"{args.outdir}/mesh.png",
          f"{st['elements']:,} elements / {st['nodes']:,} nodes")
