"""P2 데모: 무부하 비선형 정자기 해석 → |B| 분포 + Az 검증."""
import argparse, warnings
warnings.filterwarnings("ignore")
from motoropt.aedt_parser import parse_aedt, detect_magnet_style
from motoropt.geometry import build_motor
from motoropt.meshing import build_mesh
from motoropt.solver_ms import Magnetostatic2D
from motoropt.plotting import plot_field

p = argparse.ArgumentParser()
p.add_argument("aedt")
p.add_argument("--steel", default="20PNX1200F_20C")
p.add_argument("--magnet", default="Arnold_Magnetics_N45UH_80C")
p.add_argument("-o", "--outdir", default=".")
args = p.parse_args()

model = parse_aedt(args.aedt)
geo = build_motor(model["variables"], detect_magnet_style(model))
mesh = build_mesh(geo)
s = Magnetostatic2D(mesh, model["materials"], args.steel, args.magnet)
s.set_coil_currents({})
res = s.solve(verbose=True)
print(f"수렴 {res.iterations}회, 잔차 {res.residual:.2e}")
print(f"Az: [{res.A.min():.5f}, {res.A.max():.5f}] Wb/m | "
      f"|B|max: {res.Bmag.max():.3f} T")
plot_field(s, res, f"{args.outdir}/field_noload.png",
           f"No-load |B| — max {res.Bmag.max():.2f} T")
