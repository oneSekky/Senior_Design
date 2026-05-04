"""
position_zupt.py — Dead-reckoning pen position with zero-velocity updates (ZUPT).

The HP filter approach was wrong: it removes the slow left-to-right drift which
IS the inter-letter spacing. Instead we use ZUPT:
  - During quiet moments (pen paused between letters/words), force v = 0.
  - This bounds drift to at most one stroke's worth of integration time,
    while preserving the overall left-to-right motion across the page.

Gravity is estimated from the stillest window in the first 2 seconds only
(the flat-on-paper baseline), not from anywhere in the file.
"""
import numpy as np
import pandas as pd
from scipy import signal as sp
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
N     = len(acc)
print(f"FS={FS:.1f} Hz  samples={N}  duration={N*dt:.2f}s")

# ── Gravity: stillest 0.3s window within the FIRST 2 seconds only ─────────────
WIN_CAL  = int(0.3 * FS)
MAX_CAL  = int(2.0 * FS)
search   = min(MAX_CAL, N - WIN_CAL)
acc_mag  = np.linalg.norm(acc, axis=1)
# Rolling std over WIN_CAL
step = max(1, WIN_CAL // 4)
stds = [acc_mag[i:i+WIN_CAL].std() for i in range(0, search, step)]
best = int(np.argmin(stds)) * step
G_vec = acc[best:best+WIN_CAL].mean(axis=0)
G_mag = np.linalg.norm(G_vec)
g_hat = G_vec / G_mag
print(f"Gravity from t={best/FS:.3f}-{(best+WIN_CAL)/FS:.3f}s")
print(f"  G = [{G_vec[0]:.1f}, {G_vec[1]:.1f}, {G_vec[2]:.1f}] mg  |G|={G_mag:.1f} mg")
print(f"  g_hat = [{g_hat[0]:.3f}, {g_hat[1]:.3f}, {g_hat[2]:.3f}]")

# ── Build writing-plane basis ─────────────────────────────────────────────────
# u_hat: right across the page; v_hat: up the page
ref   = np.array([0., 1., 0.]) if abs(g_hat[1]) < 0.9 else np.array([1., 0., 0.])
u_hat = np.cross(g_hat, ref);  u_hat /= np.linalg.norm(u_hat)
v_hat = np.cross(g_hat, u_hat)
print(f"  Writing plane: u={u_hat.round(3)}, v={v_hat.round(3)}")

# ── Downsample to ~500 Hz for numerical stability ─────────────────────────────
DS     = max(1, int(round(FS / 500)))
acc_ds = sp.decimate((acc - G_vec), DS, axis=0, zero_phase=True)
t_ds   = t_us[::DS][:len(acc_ds)]
FS2    = FS / DS
dt2    = 1.0 / FS2
Nds    = len(acc_ds)
print(f"Downsampled: FS={FS2:.1f} Hz  samples={Nds}")

G_MS2   = 9.80665e-3           # mg → m/s²
a_ms2   = acc_ds * G_MS2       # dynamic acceleration (m/s²)

# ── ZUPT: detect quiet (pen still) moments ────────────────────────────────────
# Use dynamic acceleration magnitude (should be ~0 when pen is still after
# gravity subtraction). Smooth over 80ms window.
dyn_mag = np.linalg.norm(acc_ds, axis=1)
WIN_SM  = max(1, int(0.08 * FS2))
kern    = np.ones(WIN_SM) / WIN_SM
dyn_sm  = np.convolve(dyn_mag, kern, mode='same')

# Threshold: look at the distribution. Quiet = bottom portion.
# Use Otsu-like split: find the valley between noise and writing peaks.
hist, edges = np.histogram(dyn_sm, bins=200)
# Simple threshold: 5× noise floor estimate (p10 of the distribution)
noise_floor = np.percentile(dyn_sm, 10)
ZUPT_THR    = max(noise_floor * 4.0, 15.0)   # at least 15 mg
is_quiet    = dyn_sm < ZUPT_THR
print(f"ZUPT threshold: {ZUPT_THR:.1f} mg  ({is_quiet.mean()*100:.1f}% samples quiet)")

# ── Dead-reckoning with ZUPT ──────────────────────────────────────────────────
vel   = np.zeros((Nds, 3))
pos3d = np.zeros((Nds, 3))

for i in range(1, Nds):
    vel[i] = vel[i-1] + a_ms2[i] * dt2
    if is_quiet[i]:
        vel[i] = 0.0            # ZUPT: pen is still → velocity = 0
    pos3d[i] = pos3d[i-1] + vel[i] * dt2

# ── Project to 2D writing plane ───────────────────────────────────────────────
pu = pos3d @ u_hat    # horizontal across page (m)
pv = pos3d @ v_hat    # vertical on page (m)
pd_ = pos3d @ g_hat   # depth (should be small)

t_s = (t_ds - t_ds[0]) * 1e-6

print(f"\n2D range:  u={pu.min()*100:.1f} to {pu.max()*100:.1f} cm "
      f"(range {(pu.max()-pu.min())*100:.1f} cm)")
print(f"           v={pv.min()*100:.1f} to {pv.max()*100:.1f} cm "
      f"(range {(pv.max()-pv.min())*100:.1f} cm)")
print(f"     depth={pd_.min()*100:.1f} to {pd_.max()*100:.1f} cm")

# ── Detect active strokes for coloring ───────────────────────────────────────
is_active = ~is_quiet

# Find contiguous active segments
act_changes = np.diff(is_active.astype(int))
seg_starts  = np.where(act_changes ==  1)[0] + 1
seg_ends    = np.where(act_changes == -1)[0] + 1
if is_active[0]:  seg_starts = np.insert(seg_starts, 0, 0)
if is_active[-1]: seg_ends   = np.append(seg_ends, Nds)

n_strokes = len(seg_starts)
print(f"\nActive segments (strokes): {n_strokes}")
for i, (s, e) in enumerate(zip(seg_starts, seg_ends)):
    dur_ms = (e - s) * 1000 / FS2
    if dur_ms > 50:
        print(f"  {i+1:2d}: t={t_s[s]:.2f}-{t_s[min(e,Nds-1)]:.2f}s  dur={dur_ms:.0f}ms  "
              f"pos=({pu[s]*100:.1f},{pv[s]*100:.1f})cm")

# ── Figure ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 16))
fig.patch.set_facecolor('#0a0a0a')

# ── Top row: diagnostics ──────────────────────────────────────────────────────
ax1 = fig.add_subplot(3, 3, 1)
ax1.set_facecolor('#111')
ax1.plot(t_s[::3], dyn_sm[::3], color='#44aaff', lw=0.5, label='dyn |acc| smoothed')
ax1.axhline(ZUPT_THR, color='#ff5555', lw=1.0, ls='--', label=f'ZUPT thr={ZUPT_THR:.0f}mg')
ax1.fill_between(t_s[::3], 0, dyn_sm[::3] * is_quiet[::3],
                 color='#22ff44', alpha=0.3, label='quiet (v=0)')
ax1.legend(fontsize=7, facecolor='#1a1a1a', labelcolor='#ccc')
ax1.set_title('Dynamic acc magnitude + ZUPT regions', color='#ddd', fontsize=8, loc='left')
ax1.tick_params(colors='#888', labelsize=7)
for s in ax1.spines.values(): s.set_edgecolor('#333')
ax1.set_xlabel('Time (s)', color='#777', fontsize=7)
ax1.set_ylabel('mg', color='#777', fontsize=7)

ax2 = fig.add_subplot(3, 3, 2)
ax2.set_facecolor('#111')
for i, col in enumerate(['#ff6666','#66ff66','#6699ff']):
    ax2.plot(t_s[::3], vel[::3, i]*100, color=col, lw=0.5, alpha=0.8)
ax2.set_title('Velocity after ZUPT (cm/s)', color='#ddd', fontsize=8, loc='left')
ax2.tick_params(colors='#888', labelsize=7)
for s in ax2.spines.values(): s.set_edgecolor('#333')
ax2.set_xlabel('Time (s)', color='#777', fontsize=7)

ax3 = fig.add_subplot(3, 3, 3)
ax3.set_facecolor('#111')
ax3.plot(t_s[::3], pu[::3]*100, color='#ff9933', lw=0.7, label='u (horiz)')
ax3.plot(t_s[::3], pv[::3]*100, color='#33ffaa', lw=0.7, label='v (vert)')
ax3.plot(t_s[::3], pd_[::3]*100, color='#aa55ff', lw=0.5, alpha=0.6, label='depth')
ax3.legend(fontsize=7, facecolor='#1a1a1a', labelcolor='#ccc')
ax3.set_title('Position components over time (cm)', color='#ddd', fontsize=8, loc='left')
ax3.tick_params(colors='#888', labelsize=7)
for s in ax3.spines.values(): s.set_edgecolor('#333')
ax3.set_xlabel('Time (s)', color='#777', fontsize=7)

# ── Middle row: 2D full trace ─────────────────────────────────────────────────
ax4 = fig.add_subplot(3, 3, 4)
ax4.set_facecolor('#111')
ax4.plot(pu*100, pv*100, color='#1a2a1a', lw=0.3, alpha=0.5)
# Color active by time
step = max(1, Nds // 8000)
act_mask_plot = is_active[::step]
sc = ax4.scatter(pu[::step][act_mask_plot]*100, pv[::step][act_mask_plot]*100,
                 c=np.where(act_mask_plot)[0] / max(act_mask_plot.sum(), 1),
                 cmap='plasma', s=0.8, alpha=0.8)
ax4.set_aspect('equal', adjustable='datalim')
ax4.set_title('2D: full trace with ZUPT (active colored purple→yellow)',
              color='#ddd', fontsize=8, loc='left')
ax4.set_xlabel('u — right (cm)', color='#888', fontsize=8)
ax4.set_ylabel('v — up (cm)',    color='#888', fontsize=8)
ax4.tick_params(colors='#888', labelsize=7)
for s in ax4.spines.values(): s.set_edgecolor('#333')

# ── Middle: active strokes only, individual colored segments ──────────────────
ax5 = fig.add_subplot(3, 3, 5)
ax5.set_facecolor('#0a0a0a')
cmap = plt.cm.plasma
valid = [(s, e) for s, e in zip(seg_starts, seg_ends) if (e - s) * 1000 / FS2 > 30]
for k, (s, e) in enumerate(valid):
    c = cmap(k / max(len(valid) - 1, 1))
    ax5.plot(pu[s:e]*100, pv[s:e]*100, color=c, lw=1.5, alpha=0.9)
    # Mark start
    ax5.plot(pu[s]*100, pv[s]*100, 'o', color=c, ms=3)
ax5.set_aspect('equal', adjustable='datalim')
ax5.set_title(f'Active strokes only — {len(valid)} strokes  (purple=1st, yellow=last)',
              color='#ddd', fontsize=8, loc='left')
ax5.set_xlabel('u — right (cm)', color='#888', fontsize=8)
ax5.set_ylabel('v — up (cm)',    color='#888', fontsize=8)
ax5.tick_params(colors='#888', labelsize=7)
for s_ in ax5.spines.values(): s_.set_edgecolor('#333')

# ── Middle right: depth component ─────────────────────────────────────────────
ax6 = fig.add_subplot(3, 3, 6)
ax6.set_facecolor('#111')
ax6.plot(t_s[::3], pd_[::3]*100, color='#ffaa33', lw=0.6)
ax6.axhline(0, color='#444', lw=0.5)
ax6.fill_between(t_s, 0, pd_*100, where=is_active, color='#ffaa33', alpha=0.15)
ax6.set_title('Depth component (cm) — pen tilt error, should be small',
              color='#ddd', fontsize=8, loc='left')
ax6.set_xlabel('Time (s)', color='#777', fontsize=7)
ax6.set_ylabel('cm', color='#777', fontsize=7)
ax6.tick_params(colors='#888', labelsize=7)
for s in ax6.spines.values(): s.set_edgecolor('#333')

# ── Bottom: big clean 2D ──────────────────────────────────────────────────────
ax7 = fig.add_subplot(3, 1, 3)
ax7.set_facecolor('#050505')
# Ghost
ax7.plot(pu*100, pv*100, color='#111', lw=0.3)
# Each active segment as a colored line
for k, (s, e) in enumerate(valid):
    c = cmap(k / max(len(valid) - 1, 1))
    ax7.plot(pu[s:e]*100, pv[s:e]*100, color=c, lw=2.0, alpha=0.9, solid_capstyle='round')
    ax7.plot(pu[s]*100, pv[s]*100, 'o', color=c, ms=4, zorder=5)

# Mark word gap (longest quiet gap between active segments)
if len(valid) > 1:
    gap_lens = [(valid[i+1][0] - valid[i][1], i) for i in range(len(valid)-1)]
    gap_lens.sort(reverse=True)
    # Top 1 gap = word gap (between 3-letter and 4-letter word)
    for gap_len, gi in gap_lens[:1]:
        mid_t = (valid[gi][1] + valid[gi+1][0]) // 2
        ax7.axvline(pu[mid_t]*100, color='#ffffff', lw=0.8, ls=':', alpha=0.4,
                    label=f'word gap ({gap_len/FS2*1000:.0f}ms)')

ax7.set_aspect('equal', adjustable='datalim')
ax7.set_xlabel('u — horizontal across page (cm)', color='#aaa', fontsize=11)
ax7.set_ylabel('v — vertical on page (cm)', color='#aaa', fontsize=11)
ax7.set_title(
    f'Dead-reckoning with ZUPT  |  gravity g=[{G_vec[0]:.0f},{G_vec[1]:.0f},{G_vec[2]:.0f}]mg '
    f'g_hat=[{g_hat[0]:.2f},{g_hat[1]:.2f},{g_hat[2]:.2f}]  |  '
    f'{len(valid)} strokes detected',
    color='white', fontsize=10, pad=8
)
ax7.legend(fontsize=8, facecolor='#1a1a1a', labelcolor='#ccc')
ax7.tick_params(colors='#888', labelsize=9)
for s in ax7.spines.values(): s.set_edgecolor('#333')

sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(0, len(valid)-1))
sm.set_array([])
cb = plt.colorbar(sm, ax=ax7, shrink=0.5, pad=0.01)
cb.set_label('Stroke order (0=first)', color='#aaa', fontsize=8)
cb.ax.yaxis.set_tick_params(color='#aaa', labelsize=7)
plt.setp(cb.ax.yaxis.get_ticklabels(), color='#aaa')

plt.suptitle(
    'Pen Position Reconstruction — Dead-reckoning + ZUPT\n'
    'No HP filter: inter-letter spacing is preserved in the trajectory',
    color='white', fontsize=11, y=1.002
)
plt.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output', 'position_zupt.png')
plt.savefig(out, dpi=140, bbox_inches='tight', facecolor='#0a0a0a')
print(f"\nSaved {out}")
