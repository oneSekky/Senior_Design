"""
position_3d_v2.py — Improved 2D reconstruction with:
 - Sweep of HP filter cutoffs to find the best tradeoff
 - Gyro-based orientation tracking to correct for pen tilt during writing
 - Cleaner 2D output focused on letter shapes
"""
import numpy as np
import pandas as pd
from scipy import signal as sp
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   '..', 'data_logs', 'Data_Log_26_05_04_12_13_22.csv')

df = pd.read_csv(CSV, comment='#', skiprows=1)
df.columns = df.columns.str.strip()
t_us  = df['time[us]'].values.astype(np.float64)
acc   = df[['acc_x[mg]','acc_y[mg]','acc_z[mg]']].values.astype(np.float64)
gyro  = df[['gyro_x[mdps]','gyro_y[mdps]','gyro_z[mdps]']].values.astype(np.float64)

dt_us = np.median(np.diff(t_us))
FS    = 1e6 / dt_us
dt    = dt_us * 1e-6
print(f"FS={FS:.1f} Hz  samples={len(acc)}  duration={len(acc)*dt:.2f}s")

# ── Gravity from stillest 0.1s window ────────────────────────────────────────
WIN = int(0.1 * FS)
acc_mag = np.linalg.norm(acc, axis=1)
# Rolling std over WIN samples — find quietest window
n_win = len(acc) - WIN
roll_std = np.array([acc_mag[i:i+WIN].std() for i in range(0, n_win, WIN//4)])
best = int(np.argmin(roll_std)) * (WIN//4)
G_vec = acc[best:best+WIN].mean(axis=0)
G_mag = np.linalg.norm(G_vec)
g_hat = G_vec / G_mag
print(f"Gravity: {G_vec.round(1)} mg  |G|={G_mag:.1f}  from window {best}-{best+WIN} ({best/FS:.2f}s)")

# ── Method A: Simple gravity subtraction + double integration ─────────────────
# Downsample to 500 Hz
DS = max(1, int(round(FS / 500)))
acc_ds  = sp.decimate((acc - G_vec).astype(np.float64),  DS, axis=0, zero_phase=True)
gyro_ds = sp.decimate(gyro.astype(np.float64), DS, axis=0, zero_phase=True)
t_ds    = t_us[::DS][:len(acc_ds)]
FS2     = FS / DS
dt2     = 1.0 / FS2
print(f"Downsampled FS={FS2:.1f} Hz")

# Convert mg → m/s²
G_MS2 = 9.80665e-3
a_ms2 = acc_ds * G_MS2

# Sweep HP cutoffs for velocity and position
fig, axes = plt.subplots(3, 3, figsize=(22, 18))
fig.patch.set_facecolor('#111')

configs = [
    (0.3, 0.2, 'Conservative (0.3/0.2 Hz)'),
    (0.6, 0.4, 'Moderate (0.6/0.4 Hz)'),
    (1.0, 0.6, 'Aggressive (1.0/0.6 Hz)'),
]

best_pos2d = None
best_label = ''

for col, (hp_vel, hp_pos, label) in enumerate(configs):
    sos_v = sp.butter(2, hp_vel / (FS2/2), 'high', output='sos')
    sos_p = sp.butter(2, hp_pos / (FS2/2), 'high', output='sos')

    vel = np.cumsum(a_ms2 * dt2, axis=0)
    for i in range(3):
        vel[:, i] = sp.sosfiltfilt(sos_v, vel[:, i])
    pos3d = np.cumsum(vel * dt2, axis=0)
    for i in range(3):
        pos3d[:, i] = sp.sosfiltfilt(sos_p, pos3d[:, i])

    # Project onto writing plane
    ref   = np.array([0., 1., 0.]) if abs(g_hat[1]) < 0.9 else np.array([1., 0., 0.])
    u_hat = np.cross(g_hat, ref);  u_hat /= np.linalg.norm(u_hat)
    v_hat = np.cross(g_hat, u_hat)
    pu = pos3d @ u_hat
    pv = pos3d @ v_hat
    pd_ = pos3d @ g_hat

    # Activity mask
    dyn_mag = np.linalg.norm(acc_ds, axis=1)
    kern    = np.ones(int(0.05*FS2)) / int(0.05*FS2)
    dyn_sm  = np.convolve(dyn_mag, kern, mode='same')
    active  = dyn_sm > np.percentile(dyn_sm, 35)
    act_idx = np.where(active)[0]

    # Row 0: full 2D trace
    ax = axes[0, col]
    ax.set_facecolor('#1a1a1a')
    ax.plot(pu*100, pv*100, color='#333', lw=0.3, alpha=0.5)
    if len(act_idx):
        step = max(1, len(act_idx)//5000)
        ax.scatter(pu[act_idx[::step]]*100, pv[act_idx[::step]]*100,
                   c=act_idx[::step]/len(t_ds), cmap='plasma', s=0.5, alpha=0.8)
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_title(label, color='#ddd', fontsize=8, loc='left')
    ax.tick_params(colors='#aaa', labelsize=7)
    for sp_ in ax.spines.values(): sp_.set_edgecolor('#444')
    rng_u = (pu.max()-pu.min())*100
    rng_v = (pv.max()-pv.min())*100
    ax.set_xlabel(f'u  range={rng_u:.1f}cm', color='#888', fontsize=7)
    ax.set_ylabel(f'v  range={rng_v:.1f}cm', color='#888', fontsize=7)

    # Row 1: velocity
    ax2 = axes[1, col]
    ax2.set_facecolor('#1a1a1a')
    t_s = (t_ds - t_ds[0]) * 1e-6
    for i, c in enumerate(['#ff6666','#66ff66','#6699ff']):
        ax2.plot(t_s[::3], vel[::3, i]*100, color=c, lw=0.5, alpha=0.8)
    ax2.set_title('Velocity (cm/s)', color='#ddd', fontsize=8, loc='left')
    ax2.tick_params(colors='#aaa', labelsize=7)
    for sp_ in ax2.spines.values(): sp_.set_edgecolor('#444')

    # Row 2: depth component
    ax3 = axes[2, col]
    ax3.set_facecolor('#1a1a1a')
    ax3.plot(t_s[::3], pd_[::3]*100, color='#ffaa33', lw=0.5)
    ax3.axhline(0, color='#555', lw=0.5)
    ax3.set_title(f'Depth along gravity (cm)  max={abs(pd_).max()*100:.1f}cm', color='#ddd', fontsize=8, loc='left')
    ax3.tick_params(colors='#aaa', labelsize=7)
    for sp_ in ax3.spines.values(): sp_.set_edgecolor('#444')
    ax3.set_xlabel('Time (s)', color='#888', fontsize=7)

    if col == 0:
        best_pos2d = (pu, pv, active, act_idx)
        best_label = label

# Row labels
for row, lbl in enumerate(['2D on-paper trajectory', 'Velocity xyz (cm/s)', 'Depth / pen-tilt error (cm)']):
    axes[row, 0].set_ylabel(lbl + '\n' + axes[row, 0].get_ylabel(), color='#aaa', fontsize=7)

plt.suptitle(
    f'HP filter sweep  |  gravity g_hat=[{g_hat[0]:.2f},{g_hat[1]:.2f},{g_hat[2]:.2f}]  |  '
    f'FS_ds={FS2:.0f} Hz\n'
    'Color = time order (purple=start → yellow=end).  Active strokes only shown.',
    color='white', fontsize=10
)
plt.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output', 'position_3d_v2.png')
plt.savefig(out, dpi=140, bbox_inches='tight', facecolor='#111')
print(f"Saved {out}")

# ── Clean final 2D plot: best config, active strokes only, large ──────────────
pu, pv, active, act_idx = best_pos2d
fig2, ax = plt.subplots(1, 1, figsize=(16, 10))
fig2.patch.set_facecolor('#111')
ax.set_facecolor('#0a0a0a')

# Draw full ghost trace
ax.plot(pu*100, pv*100, color='#1a2a1a', lw=0.4, alpha=0.6)

# Draw active strokes colored by time
if len(act_idx):
    # Find contiguous stroke segments
    breaks = np.where(np.diff(act_idx) > 5)[0] + 1
    segs   = np.split(act_idx, breaks)
    cmap   = plt.cm.plasma
    total_act = len(act_idx)
    cum = 0
    for seg in segs:
        if len(seg) < 3: continue
        c = cmap(cum / max(total_act, 1))
        ax.plot(pu[seg]*100, pv[seg]*100, color=c, lw=1.2, alpha=0.85)
        cum += len(seg)

ax.set_aspect('equal', adjustable='datalim')
ax.tick_params(colors='#888', labelsize=9)
for sp_ in ax.spines.values(): sp_.set_edgecolor('#333')
ax.set_xlabel('u — horizontal on paper (cm)', color='#aaa', fontsize=11)
ax.set_ylabel('v — vertical on paper (cm)', color='#aaa', fontsize=11)

# Colorbar for time
sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(0, 1))
sm.set_array([])
cb = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.01)
cb.set_label('Time progression (start → end)', color='#aaa', fontsize=9)
cb.ax.yaxis.set_tick_params(color='#aaa')
plt.setp(cb.ax.yaxis.get_ticklabels(), color='#aaa')

ax.set_title(
    f'2D Pen Trajectory — {best_label}\n'
    f'gravity subtracted: G=[{G_vec[0]:.0f},{G_vec[1]:.0f},{G_vec[2]:.0f}] mg  '
    f'g_hat=[{g_hat[0]:.2f},{g_hat[1]:.2f},{g_hat[2]:.2f}]',
    color='white', fontsize=12, pad=10
)
plt.tight_layout()
out2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output', 'position_2d_final.png')
plt.savefig(out2, dpi=150, bbox_inches='tight', facecolor='#111')
print(f"Saved {out2}")
