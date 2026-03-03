import numpy as np, glob, sys
sys.path.insert(0, '.')
from utils.math.sbto_utils import yaw_from_quat, yaw_to_rot_matrix

data_dir = '/home/pranish/pranav/diffusion-mp/simple_diffusion/test_datasets/pick_place/SBTO_OmniRetarget_diffusion'
files = sorted(glob.glob(data_dir + '/*/*.npz'))
ws = 10
stride = 2

rows = []
for f in files:
    d = np.load(f, allow_pickle=True)
    obj_pos = d['object_pos']
    root_rot = d['root_rot']
    for b in range(obj_pos.shape[0]):
        z   = obj_pos[b, :, 2]
        xy  = obj_pos[b, :, :2]
        T   = len(z)
        final_obj = obj_pos[b, -1, :2]
        for w in range(0, (T - ws) // stride + 1):
            ts, te = w * stride, w * stride + ws - 1
            if te >= T: break
            progress  = te / T
            cum_dz    = z[te] - z[0]
            win_dz    = z[te] - z[ts]
            quat0     = root_rot[b, ts]
            R         = yaw_to_rot_matrix(-yaw_from_quat(quat0))[:2,:2]
            dxy_world = xy[te] - xy[ts]
            win_xy    = float(np.linalg.norm(R @ dxy_world))
            rem       = float(np.linalg.norm(final_obj - xy[te]))
            rows.append((progress, rem, cum_dz, win_xy, win_dz))

arr = np.array(rows)
prog, rem, cdz, wxy, wdz = arr[:,0], arr[:,1], arr[:,2], arr[:,3], arr[:,4]

print("=== XY per-window displacement by progress ===")
edges = [(i/10, i/10+0.1) for i in range(10)]
for lo, hi in edges:
    sel = (prog >= lo) & (prog < hi + 0.001)
    if sel.sum():
        print(f"  [{lo:.1f}-{hi:.1f}): mean={wxy[sel].mean():.4f}  std={wxy[sel].std():.4f}  P50={np.percentile(wxy[sel],50):.4f}  P90={np.percentile(wxy[sel],90):.4f}")

print("\n=== Remaining distance when z is lowering (wdz < -0.02) ===")
lsel = wdz < -0.02
r = rem[lsel]
if len(r):
    print(f"  n={len(r)}  mean={r.mean():.4f}  P10={np.percentile(r,10):.4f}  P50={np.percentile(r,50):.4f}  P90={np.percentile(r,90):.4f}  max={r.max():.4f}")

print("\n=== cum_dz vs remaining distance ===")
for lo, hi in [(0,.05),(.05,.1),(.1,.2),(.2,.3),(.3,.5),(.5,1.0),(1.0,2.0)]:
    sel = (rem >= lo) & (rem < hi)
    if sel.sum():
        print(f"  rem [{lo:.2f}-{hi:.2f}m): cum_dz mean={cdz[sel].mean():.4f}  P50={np.percentile(cdz[sel],50):.4f}  n={sel.sum()}")

print("\n=== Lifting windows remaining dist ===")
lup = wdz > 0.02
if lup.sum():
    print(f"  n={lup.sum()}  rem mean={rem[lup].mean():.3f}  P10={np.percentile(rem[lup],10):.3f}  P50={np.percentile(rem[lup],50):.3f}  P90={np.percentile(rem[lup],90):.3f}")

print("\n=== Total XY displacement per trajectory ===")
tot = []
for f in files:
    d = np.load(f, allow_pickle=True)
    obj_pos = d['object_pos']
    for b in range(obj_pos.shape[0]):
        xy = obj_pos[b,:,:2]
        tot.append(np.linalg.norm(xy[-1]-xy[0]))
tot = np.array(tot)
print(f"  n={len(tot)}  mean={tot.mean():.4f}  P10={np.percentile(tot,10):.4f}  P50={np.percentile(tot,50):.4f}  P90={np.percentile(tot,90):.4f}  max={tot.max():.4f}")

# Key threshold: at what remaining dist does lowering begin?
print("\n=== At what remaining dist is z > 0.1 (box is lifted)? ===")
lifted = cdz > 0.10
if lifted.sum():
    print(f"  n={lifted.sum()}  rem mean={rem[lifted].mean():.3f}  P10={np.percentile(rem[lifted],10):.3f}  P50={np.percentile(rem[lifted],50):.3f}  P90={np.percentile(rem[lifted],90):.3f}  max={rem[lifted].max():.3f}")

print("\n=== When does lowering start relative to remaining dist ===")
# Find where z transitions from high to low: currently lifted (cdz>0.1) AND wdz < -0.01
trans = (cdz > 0.10) & (wdz < -0.01)
if trans.sum():
    print(f"  n={trans.sum()}  rem mean={rem[trans].mean():.3f}  P10={np.percentile(rem[trans],10):.3f}  P50={np.percentile(rem[trans],50):.3f}  P90={np.percentile(rem[trans],90):.3f}")
