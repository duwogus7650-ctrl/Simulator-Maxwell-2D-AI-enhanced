import sys, time, warnings; warnings.filterwarnings("ignore")
import numpy as np
sys.path.insert(0, ".")
from motoropt.aedt_parser import parse_aedt, detect_magnet_style
from motoropt.geometry import build_motor
from motoropt.sliding import SlidingBandMesh
from motoropt.solver_ms import Magnetostatic2D
from motoropt.postproc import torque_arkkio, coenergy, build_winding_map, flux_linkages

m = parse_aedt("/mnt/user-data/uploads/400W.aedt")
v = m["variables"]
geo = build_motor(v, detect_magnet_style(m))
sbm = SlidingBandMesh(geo, n_band=5760, gap_frac=(0.35,0.65),
                      h={"air_gap_in":0.18,"air_gap_out":0.18,"magnet":0.6})
L = v["L_stk"]; Zc = int(round(v["Zc"])); Ia = v["I_rms"]*np.sqrt(2); pp=8
wm = {}
def solve(theta, de, load):
    mesh = sbm.merge(theta)
    s = Magnetostatic2D(mesh, m["materials"], "20PNX1200F_20C", "Arnold_Magnetics_N45UH_80C")
    if "w" not in wm: wm["w"] = build_winding_map(s)
    if load:
        te = pp*np.radians(theta)+np.radians(de)
        iph = {"A":Ia*np.sin(te),"B":Ia*np.sin(te-2*np.pi/3),"C":Ia*np.sin(te+2*np.pi/3)}
        at = {}
        for ph,sides in wm["w"].items():
            for ci,d in sides: at[ci]=d*Zc*iph[ph]
        s.set_coil_currents(at)
    else:
        iph={"A":0,"B":0,"C":0}; s.set_coil_currents({})
    res = s.solve(tol=1e-6)
    lam = flux_linkages(s,res,wm["w"],Zc,L)
    return (torque_arkkio(s,res,sbm.r_i+0.003,sbm.r_o-0.003,L), coenergy(s,res,L),
            lam, iph, float(res.Bmag.max()))

t0=time.time()
# 1) 무부하 EMF 재검증
angs_e = np.arange(0,45+1e-9,1.25)
lamA=[]
for a in angs_e:
    _,_,lam,_,_ = solve(a,0,False); lamA.append([lam["A"],lam["B"],lam["C"]])
lamA=np.array(lamA); th=np.radians(angs_e); w=1000/60*2*np.pi
for j,nm in enumerate("ABC"):
    e=np.gradient(lamA[:,j],th)*w
    print(f"EMF {nm}: RMS {np.sqrt(np.mean(e[1:-1]**2)):.3f} V (목표 6.17)", flush=True)
# 2) 부하 스윕
rows=[]
for a in np.arange(0,45+1e-9,0.625):
    T,Wc,lam,iph,Bm = solve(a,290,True)
    rows.append([a,T,Wc,lam["A"],lam["B"],lam["C"],iph["A"],iph["B"],iph["C"],Bm])
arr=np.array(rows); np.savez("p4_final.npz", data=arr)
ang,T,Wc = arr[:,0],arr[:,1]*1e3,arr[:,2]
th=np.radians(ang)
Tvw=(np.gradient(Wc,th)-(arr[:,3]*np.gradient(arr[:,6],th)+arr[:,4]*np.gradient(arr[:,7],th)+arr[:,5]*np.gradient(arr[:,8],th)))*1e3
print(f"Arkkio 평균 {T.mean():.1f} | pk2pk {T.max()-T.min():.1f} mNm")
print(f"가상일 평균 {Tvw[2:-2].mean():.1f} | pk2pk {Tvw[2:-2].max()-Tvw[2:-2].min():.1f} mNm (목표 848.7 / ~24)")
print(f"부하 Bmax {arr[:,9].max():.3f} (목표 2.322) | {time.time()-t0:.0f}s")
