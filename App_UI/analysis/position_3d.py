"""
position_3d.py — Double-integration position reconstruction at 6.67 kHz.

Gravity vector is estimated from the flat-on-paper baseline period:
  G = mean(acc_xyz) during the stillest window → defines the "down" direction.

After subtracting G, the residual acceleration is (approximately) the
writing motion in sensor frame. Double-integrating gives 3D position.
Projecting onto the plane perpendicular to G gives the 2D on-paper trace.

HP filters on velocity and position suppress long-timescale integration drift.
"""

import numpy as np
import pandas as pd
from scipy import signal as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import os

CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   '..', 'data_logs', 'Data_Log_26_05_04_12_13_22.csv')

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV, comment='#', skiprows=1)
df.columns = df.columns.str.strip()

t_us  = df['time[us]'].values.astype(np.float64)
acc   = df[['acc_x[mg]','acc_y[mg]','acc_z[mg]']].values.astype(np.float64)
gyro  = df[['gyro_x[mdps]','gyro_y[mdps]','gyro_z[mdps]']].values.astype(np.float64)

# Actual sample interval & rate from timestamps
dt_us = np.median(np.diff(t_us))
FS    = 1e6 / dt_us
dt    = dt_us * 1e-6
print(f"Samples: {len(acc)}   FS: {FS:.1f} Hz   dt: {dt_us:.1f} us")
print(f"Duration: {len(acc)*dt:.2f} s")

# ── Find stillest window for gravity calibration ──────────────────────────────
# Use a 200-sample (~30 ms at 6.67 kHz) rolling variance on acc magnitude
WIN_CAL = 500   # samples for stillness window
acc_mag = np.linalg.norm(acc, axis=1)
# Rolling variance (std) of magnitude
roll_std = np.array([acc_mag[i:i+WIN_CAL].std() for i in range(0, len(acc)-WIN_CAL, WIN_CAL//2)])
best_win  = int(np.argmin(roll_std))
base_start = best_win * (WIN_CAL//2)
base_end   = base_start + WIN_CAL
G_vec = acc[base_start:base_end].mean(axis=0)    # gravity vector in mg, sensor frame
G_mag = np.linalg.norm(G_vec)
g_hat = G_vec / G_mag                            # unit "down" vector

print(f"\nGravity calibration window: samples {base_start}-{base_end} ({base_start/FS:.2f}-{base_end/FS:.2f} s)")
print(f"  G_vec  = [{G_vec[0]:.2f}, {G_vec[1]:.2f}, {G_vec[2]:.2f}] mg")
print(f"  |G|    = {G_mag:.2f} mg  (expected ~1000 mg)")
print(f"  g_hat  = [{g_hat[0]:.3f}, {g_hat[1]:.3f}, {g_hat[2]:.3f}]")

# ── Subtract gravity vector ────────────────────────────────────────────────────
acc_dyn = acc - G_vec    # dynamic acceleration (writing motion), mg

# Convert to m/s²
GRAV_MS2 = 9.80665 / 1000.0   # mg → m/s²
acc_ms2  = acc_dyn * GRAV_MS2

# ── Downsample to ~500 Hz for numerical stability ────────────────────────────
DS_FACTOR = max(1, int(round(FS / 500)))
if DS_FACTOR > 1:
    # Anti-alias filter then decimate
    acc_ds  = sp.decimate(acc_ms2,  DS_FACTOR, axis=0, zero_phase=True)
    gyro_ds = sp.decimate(gyro,      DS_FACTOR, axis=0, zero_phase=True)
    t_ds    = t_us[::DS_FACTOR][:len(acc_ds)]
    FS_ds   = FS / DS_FACTOR
    dt_ds   = 1.0 / FS_ds
else:
    acc_ds, gyro_ds, t_ds, FS_ds, dt_ds = acc_ms2, gyro, t_us, FS, dt

print(f"\nDownsampled: FS_ds={FS_ds:.1f} Hz  dt_ds={dt_ds*1000:.2f} ms  samples={len(acc_ds)}")

# ── Double integration with HP drift correction ───────────────────────────────
# Build HP filter using second-order sections for numerical stability
HP_VEL = 0.8    # Hz — removes slow velocity drift while keeping strokes (>0.5 s)
HP_POS = 0.5    # Hz — removes slow position drift
sos_vel = sp.butter(2, HP_VEL / (FS_ds/2), 'high', output='sos')
sos_pos = sp.butter(2, HP_POS / (FS_ds/2), 'high', output='sos')

# Integrate velocity
vel = np.cumsum(acc_ds * dt_ds, axis=0)
# HP filter velocity to remove integration drift
for i in range(3):
    vel[:, i] = sp.sosfiltfilt(sos_vel, vel[:, i])

# Integrate position
pos3d = np.cumsum(vel * dt_ds, axis=0)
# HP filter position
for i in range(3):
    pos3d[:, i] = sp.sosfiltfilt(sos_pos, pos3d[:, i])

print(f"\n3D position range (m):")
for i, ax in enumerate('xyz'):
    print(f"  {ax}: {pos3d[:,i].min():.4f} to {pos3d[:,i].max():.4f}  "
          f"range={pos3d[:,i].max()-pos3d[:,i].min():.4f}")

# ── Project onto writing plane (perpendicular to g_hat) ──────────────────────
# Build orthonormal basis for the writing plane
# u: perpendicular to g_hat in the xz-plane
ref  = np.array([0., 1., 0.]) if abs(g_hat[1]) < 0.9 else np.array([1., 0., 0.])
u_hat = np.cross(g_hat, ref);  u_hat /= np.linalg.norm(u_hat)
v_hat = np.cross(g_hat, u_hat)                  # already unit length

pos2d_u = pos3d @ u_hat
pos2d_v = pos3d @ v_hat

# Residual "depth" (along g_hat — should be near zero for writing)
pos_depth = pos3d @ g_hat

print(f"\n2D on-paper range (m):")
print(f"  u: {pos2d_u.min():.4f} to {pos2d_u.max():.4f}  range={pos2d_u.max()-pos2d_u.min():.4f}")
print(f"  v: {pos2d_v.min():.4f} to {pos2d_v.max():.4f}  range={pos2d_v.max()-pos2d_v.min():.4f}")
print(f"  depth: {pos_depth.min():.4f} to {pos_depth.max():.4f}  (should be small)")

# Time axis (seconds from start)
t_s = (t_ds - t_ds[0]) * 1e-6

# ── acc magnitude before/after gravity subtraction ───────────────────────────
acc_raw_mag = np.linalg.norm(acc[::DS_FACTOR][:len(acc_ds)], axis=1)
acc_dyn_mag = np.linalg.norm(acc_ds / GRAV_MS2, axis=1)   # back to mg for display

# ── Detect active writing strokes (for coloring) ──────────────────────────────
WIN_S = int(0.05 * FS_ds)   # 50 ms smoothing
kern  = np.ones(max(1, WIN_S)) / max(1, WIN_S)
acc_sm = np.convolve(acc_dyn_mag, kern, mode='same')
THR   = np.percentile(acc_sm, 30)   # adaptive: bottom 30% = quiet
is_act = acc_sm > THR

# ── Plot ──────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(24, 20))
fig.patch.set_facecolor('#111')

# ── Panel 1: raw acc magnitude + dynamic magnitude ──────────────────────────
ax1 = fig.add_subplot(4, 2, 1)
ax1.set_facecolor('#1a1a1a')
ax1.plot(t_s[::10], acc_raw_mag[::10], color='#ff6666', lw=0.4, label='raw |acc| (mg)')
ax1.axhline(G_mag, color='#ffaa33', lw=1.0, ls='--', label=f'|G|={G_mag:.0f} mg')
ax1.set_title('Raw acc magnitude vs time', color='#ddd', fontsize=9, loc='left')
ax1.legend(fontsize=7, facecolor='#222', labelcolor='#ccc')
ax1.tick_params(colors='#aaa', labelsize=7)
for sp_ in ax1.spines.values(): sp_.set_edgecolor('#444')

ax2 = fig.add_subplot(4, 2, 2)
ax2.set_facecolor('#1a1a1a')
ax2.plot(t_s[::10], acc_dyn_mag[::10], color='#44aaff', lw=0.4, label='dynamic |acc-G| (mg)')
ax2.axhline(THR, color='#ff5555', lw=0.8, ls='--', label=f'activity thr={THR:.1f}')
ax2.set_title('Gravity-subtracted acc magnitude', color='#ddd', fontsize=9, loc='left')
ax2.legend(fontsize=7, facecolor='#222', labelcolor='#ccc')
ax2.tick_params(colors='#aaa', labelsize=7)
for sp_ in ax2.spines.values(): sp_.set_edgecolor('#444')

# ── Panel 2: velocity xyz ─────────────────────────────────────────────────────
ax3 = fig.add_subplot(4, 2, 3)
ax3.set_facecolor('#1a1a1a')
for i, col in enumerate(['#ff6666','#66ff66','#6699ff']):
    ax3.plot(t_s[::5], vel[::5, i]*100, color=col, lw=0.5, label=f'v{"xyz"[i]} (cm/s)')
ax3.set_title('Integrated velocity (HP filtered)', color='#ddd', fontsize=9, loc='left')
ax3.legend(fontsize=7, facecolor='#222', labelcolor='#ccc')
ax3.tick_params(colors='#aaa', labelsize=7)
for sp_ in ax3.spines.values(): sp_.set_edgecolor('#444')

# ── Panel 3: position xyz ─────────────────────────────────────────────────────
ax4 = fig.add_subplot(4, 2, 4)
ax4.set_facecolor('#1a1a1a')
for i, col in enumerate(['#ff6666','#66ff66','#6699ff']):
    ax4.plot(t_s[::5], pos3d[::5, i]*100, color=col, lw=0.5, label=f'p{"xyz"[i]} (cm)')
ax4.set_title('3D position (HP filtered)', color='#ddd', fontsize=9, loc='left')
ax4.legend(fontsize=7, facecolor='#222', labelcolor='#ccc')
ax4.tick_params(colors='#aaa', labelsize=7)
for sp_ in ax4.spines.values(): sp_.set_edgecolor('#444')

# ── Panel 4: 3D trajectory ────────────────────────────────────────────────────
ax5 = fig.add_subplot(4, 2, 5, projection='3d')
ax5.set_facecolor('#1a1a1a')
fig.patch.set_facecolor('#111')
step = max(1, len(pos3d)//5000)
cmap = plt.cm.plasma
colors_t = cmap(np.linspace(0, 1, len(pos3d[::step])))
ax5.scatter(pos3d[::step, 0]*100, pos3d[::step, 1]*100, pos3d[::step, 2]*100,
            c=colors_t, s=0.3, alpha=0.6)
# Draw gravity vector from origin
scale = 0.05  # 5 cm scale
ax5.quiver(0, 0, 0, g_hat[0]*scale*100, g_hat[1]*scale*100, g_hat[2]*scale*100,
           color='#ffaa33', linewidth=2, label='gravity (down)')
ax5.set_xlabel('X (cm)', color='#888', fontsize=7)
ax5.set_ylabel('Y (cm)', color='#888', fontsize=7)
ax5.set_zlabel('Z (cm)', color='#888', fontsize=7)
ax5.set_title('3D trajectory (time: purple→yellow)', color='#ddd', fontsize=9)
ax5.tick_params(colors='#777', labelsize=6)

# ── Panel 5: 2D on-paper projection — full trace ──────────────────────────────
ax6 = fig.add_subplot(4, 2, 6)
ax6.set_facecolor('#1a1a1a')
ax6.plot(pos2d_u*100, pos2d_v*100, color='#222', lw=0.3, alpha=0.4)
# Color active strokes
step2 = max(1, len(pos2d_u)//8000)
ax6.scatter(pos2d_u[::step2]*100, pos2d_v[::step2]*100,
            c=is_act[::step2].astype(float),
            cmap='RdYlGn', s=0.5, alpha=0.7, vmin=0, vmax=1)
ax6.set_aspect('equal', adjustable='datalim')
ax6.set_xlabel('u (cm) — right', color='#888', fontsize=8)
ax6.set_ylabel('v (cm) — up along paper', color='#888', fontsize=8)
ax6.set_title('2D on-paper projection  (green=active, red=quiet)', color='#ddd', fontsize=9, loc='left')
ax6.tick_params(colors='#aaa', labelsize=7)
for sp_ in ax6.spines.values(): sp_.set_edgecolor('#444')

# ── Panel 6: 2D projection — active strokes only, colored by time ─────────────
ax7 = fig.add_subplot(4, 2, 7)
ax7.set_facecolor('#1a1a1a')
act_idx = np.where(is_act)[0]
if len(act_idx):
    sc = ax7.scatter(pos2d_u[act_idx[::max(1,len(act_idx)//6000)]]*100,
                     pos2d_v[act_idx[::max(1,len(act_idx)//6000)]]*100,
                     c=act_idx[::max(1,len(act_idx)//6000)] / len(t_s),
                     cmap='plasma', s=0.8, alpha=0.8)
    plt.colorbar(sc, ax=ax7, label='time (fraction)', shrink=0.7)
ax7.set_aspect('equal', adjustable='datalim')
ax7.set_xlabel('u (cm)', color='#888', fontsize=8)
ax7.set_ylabel('v (cm)', color='#888', fontsize=8)
ax7.set_title('2D active strokes only — colored by time (purple=early, yellow=late)',
              color='#ddd', fontsize=9, loc='left')
ax7.tick_params(colors='#aaa', labelsize=7)
for sp_ in ax7.spines.values(): sp_.set_edgecolor('#444')

# ── Panel 7: depth component (should be ~0 for flat writing) ──────────────────
ax8 = fig.add_subplot(4, 2, 8)
ax8.set_facecolor('#1a1a1a')
ax8.plot(t_s[::5], pos_depth[::5]*100, color='#ffaa33', lw=0.5)
ax8.axhline(0, color='#555', lw=0.5)
ax8.set_title('Depth component along gravity (cm) — should be near zero for flat writing',
              color='#ddd', fontsize=9, loc='left')
ax8.set_xlabel('Time (s)', color='#888', fontsize=8)
ax8.set_ylabel('depth (cm)', color='#888', fontsize=7)
ax8.tick_params(colors='#aaa', labelsize=7)
for sp_ in ax8.spines.values(): sp_.set_edgecolor('#444')

plt.suptitle(
    f'3D Position Reconstruction — {os.path.basename(CSV)}\n'
    f'FS={FS:.0f} Hz → downsampled to {FS_ds:.0f} Hz  |  '
    f'gravity vector: [{G_vec[0]:.0f}, {G_vec[1]:.0f}, {G_vec[2]:.0f}] mg  |  '
    f'HP vel={HP_VEL} Hz  pos={HP_POS} Hz',
    color='white', fontsize=10
)
plt.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output', 'position_3d.png')
plt.savefig(out, dpi=130, bbox_inches='tight', facecolor='#111')
print(f"\nSaved {out}")
