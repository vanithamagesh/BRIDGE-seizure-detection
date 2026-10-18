"""Redraw the result figures of the paper (Figs. 5-9 of the full run and the one-column versions) from a results folder
of bridge_chbmit.py (no retraining), with the labels used in the paper.
    python make_paper_figs.py <results folder> <features_local.npz> <edf cache folder> <output folder>
<results folder>: the output folder of the full run, or results/loso_fixed_split of this repository.
features_local.npz: the same windows and features as the run (feat_local.py), used for the correlation maps; the EEG of
the example recording is downloaded once from the PhysioNet AWS mirror into the edf cache folder."""
import csv, json, math, os, shutil, sys, urllib.parse, urllib.request
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt

RES, FEAT, EDF, OUT = sys.argv[1:5]; os.makedirs(OUT, exist_ok=True); os.makedirs(EDF, exist_ok=True)
plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8, 'legend.fontsize': 6.5,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.spines.top': False, 'axes.spines.right': False})
COL = {'BRIDGE': '#1f5a96', 'BRIDGE-GT': '#d0661f', 'Transformer': '#8c8c8c', 'MLP': '#b5b5b5', 'LogReg': '#2e7d32',
       'RandomForest': '#8bc34a', 'RawCNN': '#6d4c41'}
LBL = {'BRIDGE': 'BRIDGE (proposed)', 'BRIDGE-GT': 'BRIDGE-GT (proposed)', 'Transformer': 'Transformer (no graph)', 'MLP': 'MLP',
       'LogReg': 'Logistic regression', 'RandomForest': 'Random forest', 'RawCNN': 'Raw 1-D CNN'}
MODELS = list(LBL)
CH18 = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
        'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2', 'FZ-CZ', 'CZ-PZ']
POS = {'FP1': (-0.31, 0.95), 'FP2': (0.31, 0.95), 'F7': (-0.81, 0.59), 'F3': (-0.41, 0.51), 'FZ': (0, 0.47),
       'F4': (0.41, 0.51), 'F8': (0.81, 0.59), 'T7': (-1, 0), 'C3': (-0.5, 0), 'CZ': (0, 0), 'C4': (0.5, 0), 'T8': (1, 0),
       'P7': (-0.81, -0.59), 'P3': (-0.41, -0.51), 'PZ': (0, -0.47), 'P4': (0.41, -0.51), 'P8': (0.81, -0.59),
       'O1': (-0.31, -0.95), 'O2': (0.31, -0.95)}
N_CH = 18; IU = np.triu_indices(N_CH, 1); FS = 256
mids = np.array([np.mean([POS[e] for e in ch.split('-')], 0) for ch in CH18])
rows = list(csv.DictReader(open(os.path.join(RES, 'per_subject.csv'))))
for r in rows:
    for k, v in r.items():
        if k != 'model':
            try: r[k] = float(v)
            except ValueError: r[k] = np.nan
SUMMARY = json.load(open(os.path.join(RES, 'summary.json'))); EXTRA = json.load(open(os.path.join(RES, 'extras.json')))
GW = json.load(open(os.path.join(RES, 'learned_graph_weights.json')))


def savefig(fig, name):
    fig.savefig(os.path.join(OUT, name + '.png'), dpi=600, bbox_inches='tight'); fig.savefig(os.path.join(OUT, name + '.pdf'), bbox_inches='tight')
    plt.close(fig); print('figure', name)


def fig5():
    gs = ['BRIDGE (full)', 'BRIDGE w/o baseline reference', 'BRIDGE anatomical graph only', 'BRIDGE functional graph only',
          'BRIDGE without graph', 'BRIDGE + 16-s temporal context']
    names = ['full\nmodel', 'without\nbaseline\nreference', 'anatomical\ngraph\nonly', 'functional\ngraph\nonly', 'without\ngraph', '+16-s\ntemporal\ncontext']
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.5), gridspec_kw=dict(width_ratios=[1.8, 1]))
    for i, g in enumerate(gs):
        mu, lo, hi = EXTRA['ablation'][g]['auroc']; axs[0].bar(i, mu, color=COL['BRIDGE'] if i == 0 else '#9db7d5', width=0.6)
        axs[0].errorbar(i, mu, yerr=[[mu - lo], [hi - mu]], color='k', capsize=2, lw=0.8); axs[0].text(i, hi + 0.004, f'{mu:.3f}', ha='center', fontsize=6)
    axs[0].set_xticks(range(len(gs))); axs[0].set_xticklabels(names, fontsize=6.5); axs[0].set_ylim(0.78, 1.0)
    axs[0].set_ylabel('Mean AUROC (held-out subjects)'); axs[0].set_title('(a) BRIDGE: one component removed or context added', loc='left')
    L = EXTRA['leakage']; v = [L['random_split_auroc'], L['subject_wise_pooled_auroc']]
    axs[1].bar([0, 1], v, color=['#c0504d', COL['BRIDGE']], width=0.55)
    for i, x in enumerate(v): axs[1].text(i, x + 0.004, f'{x:.3f}', ha='center', va='bottom', fontsize=7)
    axs[1].set_xticks([0, 1]); axs[1].set_xticklabels(['random window\nsplit (leaky)', 'leave-one-\nsubject-out']); axs[1].set_ylabel('Pooled AUROC')
    axs[1].set_ylim(0.85, 1.0); axs[1].set_title('(b) evaluation leakage', loc='left'); fig.tight_layout(); savefig(fig, 'fig5_ablation_leakage')


def fig6():
    E = {r['model']: r for r in csv.DictReader(open(os.path.join(RES, 'efficiency.csv'))) if r.get('params')}
    off = {'BRIDGE': (6, 4), 'BRIDGE-GT': (8, 2), 'Transformer': (8, -8), 'MLP': (6, -3), 'LogReg': (6, -3), 'RandomForest': (-22, 7), 'RawCNN': (-25, 6)}
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    for m in MODELS:
        if m not in E: continue
        x, y = float(E[m]['params']), SUMMARY[m]['auroc'][0]; macs = float(E[m]['macs']) if E[m].get('macs') else 1e4
        ax.scatter(x, y, s=12 + 50 * np.log10(1 + macs) / 7, color=COL[m], zorder=3, edgecolor='k' if m.startswith('BRIDGE') else 'none', lw=0.5)
        ax.annotate(LBL[m].replace(' (proposed)', ''), (x, y), xytext=off[m], textcoords='offset points', fontsize=6)
    ax.set_xscale('log'); ax.set_xlim(150, 4e5); ax.set_ylim(0.8, 0.96)
    ax.set_xlabel('Parameters (random forest: tree nodes)'); ax.set_ylabel('Mean AUROC (held-out subjects)')
    savefig(fig, 'fig6_efficiency')


def read_edf(path):
    raw = open(path, 'rb').read(); ns = int(raw[252:256].decode().strip()); h = raw[256:256 + 256 * ns]
    fld = lambda off, w, typ=str: [typ(h[off * ns + i * w: off * ns + (i + 1) * w].decode('latin-1').strip()) for i in range(ns)]
    labels, pmin, pmax = fld(0, 16), fld(104, 8, float), fld(112, 8, float)
    dmin, dmax, nsamp = fld(120, 8, float), fld(128, 8, float), fld(216, 8, int)
    rec = sum(nsamp); data = np.frombuffer(raw, dtype='<i2', offset=256 + 256 * ns)
    n_rec = len(data) // rec; data = data[:n_rec * rec].reshape(n_rec, rec); cum = np.cumsum([0] + nsamp); out = {}
    for i, lab in enumerate(labels):
        key = lab.upper().replace(' ', '')
        if key not in CH18 or key in out: continue
        g = (pmax[i] - pmin[i]) / (dmax[i] - dmin[i])
        out[key] = ((data[:, cum[i]:cum[i + 1]].reshape(-1).astype(np.float32) - dmin[i]) * g + pmin[i]).astype(np.float32)
    return out


def fig7(FILES, SZB):
    q = np.load(os.path.join(RES, 'pred_BRIDGE.npz')); g = np.load(os.path.join(RES, 'pred_BRIDGE-GT.npz'))
    au = {int(r['subject']): r['auroc'] for r in rows if r['model'] == 'BRIDGE'}
    det = {int(r['subject']): r['sz_sens_fa'] for r in rows if r['model'] == 'BRIDGE'}
    cand = sorted([s for s in au if det[s] >= 0.5], key=au.get); s = cand[len(cand) // 2]
    fids = np.unique(q['fid'][q['subject'] == s]); best = None
    for f in fids:
        for z in SZB.get(int(f), []):
            k = (q['fid'] == f) & (q['t0'] + 4 > z['onset']) & (q['t0'] < z['offset'])
            alarm = (q['p'][k] >= q['t_fa'][k]).sum()
            if best is None or alarm > best[0]: best = (alarm, int(f), z)
    _, f, z = best; name = FILES[f]['file']; case = FILES[f]['case']; sel = np.where(q['fid'] == f)[0]; sel = sel[np.argsort(q['t0'][sel])]
    sel = sel[(q['t0'][sel] >= z['onset'] - 600) & (q['t0'][sel] <= z['offset'] + 600)]
    dest = os.path.join(EDF, name)
    if not os.path.exists(dest):
        req = urllib.request.Request(f'https://physionet-open.s3.amazonaws.com/chbmit/1.0.0/chb{case:02d}/' + urllib.parse.quote(name), headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=300) as r, open(dest, 'wb') as fo: shutil.copyfileobj(r, fo)
    ch = read_edf(dest); sos = butter(4, [0.5, 45], btype='bandpass', fs=FS, output='sos'); chans = [0, 2, 6, 16]
    a, b = int(q['t0'][sel[0]] * FS), int((q['t0'][sel[-1]] + 4) * FS)
    X = sosfiltfilt(sos, np.stack([ch[CH18[c]] for c in chans]), axis=-1)[:, a:b][:, ::2]; tt = a / FS + np.arange(X.shape[1]) / (FS / 2)
    fig, axs = plt.subplots(3, 1, figsize=(7.0, 3.5), sharex=True, gridspec_kw=dict(height_ratios=[1.6, 1, 0.45]))
    for j, c in enumerate(chans):
        y = X[j] / (np.percentile(np.abs(X[j]), 99) + 1e-6); axs[0].plot(tt, y - 2.2 * j, lw=0.25, color='0.15')
    axs[0].set_yticks([-2.2 * j for j in range(len(chans))]); axs[0].set_yticklabels([CH18[c] for c in chans])
    tw = q['t0'][sel] + 2
    for m, d in (('BRIDGE', q), ('BRIDGE-GT', g)):
        o = np.argsort(d['idx']); ii = o[np.searchsorted(d['idx'][o], q['idx'][sel])]
        axs[1].plot(tw, d['p'][ii], color=COL[m], lw=0.9, label=LBL[m]); axs[1].axhline(d['t_fa'][ii].mean(), color=COL[m], ls='--', lw=0.6)
    p, t, tau = q['p'][sel], q['t'][sel].mean(), q['tau_0.95'][sel].mean(); t_fa = q['t_fa'][sel].mean()
    cf = np.where(p >= t, (p - t) / max(1 - t, 1e-6), (t - p) / max(t, 1e-6)); acc = cf >= tau
    alarm = np.convolve(((p >= t_fa) & acc).astype(int), np.ones(3, int))[:len(p)] >= 2
    axs[2].fill_between(tw, 0, alarm.astype(float), step='mid', color='#c0504d', lw=0, label='alarm (BRIDGE)')
    axs[2].fill_between(tw, 0, (~acc).astype(float) * 0.5, step='mid', color='#f2c14e', lw=0, label='referred (coverage 0.95)')
    for ax in axs: ax.axvspan(z['onset'], z['offset'], color='#c0504d', alpha=0.12, lw=0)
    axs[1].set_ylabel('Seizure\nprobability'); axs[1].legend(frameon=True, framealpha=0.85, edgecolor='none', loc='upper left', fontsize=6)
    axs[2].set_yticks([]); axs[2].legend(frameon=True, framealpha=0.85, edgecolor='none', loc='upper left', ncol=2, fontsize=6)
    axs[2].set_xlabel(f'Time in recording {name} (s)')
    axs[0].set_title(f'Held-out subject {s}; shaded: annotated seizure; dashed: threshold for at most 1 false alarm/h', loc='left')
    savefig(fig, 'fig7_example_timeline'); return s, name, z


def fig8(Ca, Y, AUG, EXT):
    A_STATIC = np.zeros((N_CH, N_CH))
    for i in range(N_CH):
        for j in range(N_CH):
            if i == j: continue
            d = np.linalg.norm(mids[i] - mids[j])
            A_STATIC[i, j] = 1.0 if set(CH18[i].split('-')) & set(CH18[j].split('-')) else (math.exp(-(d / 0.35) ** 2) if d < 0.6 else 0.0)
    na = ~AUG & ~EXT; rng = np.random.default_rng(0); ii = np.where(na & (Y == 1))[0]; jj = rng.choice(np.where(na & (Y == 0))[0], 50000, replace=False)
    full = lambda v: (lambda M: M + M.T)(np.bincount(IU[0] * N_CH + IU[1], weights=v, minlength=N_CH * N_CH).reshape(N_CH, N_CH))
    A1 = full(np.abs(Ca[ii].astype(np.float32)).mean(0)); A0 = full(np.abs(Ca[jj].astype(np.float32)).mean(0)); D = A1 - A0
    fig = plt.figure(figsize=(7.0, 2.5)); gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1.05, 0.95], wspace=0.35)
    for k, (M, ttl) in enumerate(((A0, '(a) interictal |corr|'), (A1, '(b) ictal |corr|'))):
        ax = fig.add_subplot(gs[k]); ax.imshow(M, vmin=0, vmax=1, cmap='viridis'); ax.set_title(ttl, loc='left')
        ax.set_xticks(range(N_CH)); ax.set_yticks(range(N_CH)); ax.set_xticklabels(CH18, rotation=90, fontsize=4); ax.set_yticklabels(CH18, fontsize=4)
    ax = fig.add_subplot(gs[2]); th = np.linspace(0, 2 * np.pi, 200); ax.plot(1.08 * np.cos(th), 1.08 * np.sin(th), color='0.6', lw=0.6)
    for qq in np.argsort(-np.abs(D[IU]))[:25]:
        i, j = IU[0][qq], IU[1][qq]; w = D[i, j]
        ax.plot(mids[[i, j], 0], mids[[i, j], 1], color='#c0504d' if w > 0 else '#1f5a96', lw=0.5 + 6 * abs(w), alpha=0.8)
    ax.scatter(mids[:, 0], mids[:, 1], s=8, color='k', zorder=3); ax.set_aspect('equal'); ax.axis('off')
    ax.set_title('(c) ictal - interictal', loc='left')
    ax = fig.add_subplot(gs[3]); mx = np.array(list(GW['MIX'].values())); b = np.array(list(GW['BETA'].values())).mean(axis=(1, 3))
    ax.bar([0, 1], mx.mean(0), yerr=mx.std(0), color=[COL['BRIDGE'], '#9db7d5'], width=0.6, capsize=2)
    ax.bar([2.4, 3.4], b.mean(0), yerr=b.std(0), color=[COL['BRIDGE-GT'], '#f0b27a'], width=0.6, capsize=2)
    ax.set_xticks([0, 1, 2.4, 3.4]); ax.set_xticklabels(['anat.', 'func.', 'anat.', 'func.'], fontsize=6, rotation=45)
    ax.text(0.5, -0.25, 'BRIDGE\n(mix weight)', ha='center', va='top', fontsize=6, transform=ax.get_xaxis_transform())
    ax.text(2.9, -0.25, 'BRIDGE-GT\n(mean ' + r'$\beta$' + ')', ha='center', va='top', fontsize=6, transform=ax.get_xaxis_transform())
    ax.set_title('(d) learned graph weights', loc='left'); savefig(fig, 'fig8_graphs'); return mx.mean(0), b.mean(0)


def fig9():
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.4))
    for m in ('BRIDGE', 'BRIDGE-GT', 'Transformer', 'RandomForest'):
        v = np.array([r['latency_fa'] for r in rows if r['model'] == m and np.isfinite(r['latency_fa'])])
        axs[0].hist(v, bins=np.arange(0, 64, 4), histtype='step', color=COL[m], lw=1.1, label=LBL[m])
        R = [r for r in rows if r['model'] == m]; axs[1].scatter([r['fa_h_fa'] for r in R], [r['sz_sens_fa'] for r in R], s=10, color=COL[m], alpha=0.8, lw=0, label=LBL[m])
    axs[0].set_xlabel('Median detection latency per subject (s)'); axs[0].set_ylabel('Subjects'); axs[0].legend(frameon=False)
    axs[1].set_xscale('symlog', linthresh=0.1); axs[1].set_xlabel('False alarms per hour'); axs[1].set_ylabel('Seizures detected per subject')
    axs[0].set_title('(a) detection latency', loc='left'); axs[1].set_title('(b) one point per held-out subject', loc='left')
    fig.text(0.5, -0.04, 'Threshold: at most 1 false alarm per hour on the inner validation subjects (training subjects only).', ha='center', fontsize=6.5)
    fig.tight_layout(); savefig(fig, 'fig9_latency_falsealarms')


def fig2_1col():
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    for i, m in enumerate(MODELS):
        v = np.array([r['auroc'] for r in rows if r['model'] == m and np.isfinite(r['auroc'])])
        ax.scatter(i + np.random.default_rng(i).uniform(-0.18, 0.18, len(v)), v, s=6, color=COL[m], alpha=0.8, lw=0)
        mu, lo, hi = SUMMARY[m]['auroc']; ax.errorbar(i, mu, yerr=[[mu - lo], [hi - mu]], fmt='_', color='k', ms=11, capsize=2, lw=0.8)
    ax.set_xticks(range(len(MODELS))); ax.set_xticklabels([LBL[m].replace(' (proposed)', '').replace(' (no graph)', '') for m in MODELS], rotation=35, ha='right', fontsize=6.5)
    ax.set_ylabel('AUROC per held-out subject'); ax.set_ylim(0.45, 1.01); savefig(fig, 'fig2b_per_subject_auroc_1col')


# ------------------------------------------------------------------ one-column versions for the 6-page paper
def fig5_1col():
    gs = ['BRIDGE (full)', 'BRIDGE w/o baseline reference', 'BRIDGE anatomical graph only', 'BRIDGE functional graph only',
                                  'BRIDGE without graph', 'BRIDGE + 16-s temporal context']
    names = ['full\nmodel', 'no\nbaseline\nref.', 'anat.\ngraph\nonly', 'func.\ngraph\nonly', 'no\ngraph', '+16-s\ncontext']
    fig, axs = plt.subplots(2, 1, figsize=(3.5, 3.4), gridspec_kw=dict(height_ratios=[1.9, 1]))
    for i, g in enumerate(gs):
        mu, lo, hi = EXTRA['ablation'][g]['auroc']; axs[0].bar(i, mu, color=COL['BRIDGE'] if i == 0 else '#9db7d5', width=0.62)
        axs[0].errorbar(i, mu, yerr=[[mu - lo], [hi - mu]], color='k', capsize=1.5, lw=0.7); axs[0].text(i, hi + 0.004, f'{mu:.3f}', ha='center', fontsize=5.5)
    axs[0].set_xticks(range(len(gs))); axs[0].set_xticklabels(names, fontsize=5.8); axs[0].set_ylim(0.78, 1.0)
    axs[0].set_ylabel('Mean AUROC'); axs[0].set_title('(a) BRIDGE: component removed or added', loc='left', fontsize=7)
    L = EXTRA['leakage']; v = [L['random_split_auroc'], L['subject_wise_pooled_auroc']]
    axs[1].barh([1, 0], v, color=['#c0504d', COL['BRIDGE']], height=0.55)
    for y, x in zip([1, 0], v): axs[1].text(x + 0.002, y, f'{x:.3f}', va='center', fontsize=6)
    axs[1].set_yticks([1, 0]); axs[1].set_yticklabels(['random window split\n(leaky)', 'leave-one-subject-out'], fontsize=6)
    axs[1].set_xlim(0.85, 1.0); axs[1].set_xlabel('Pooled AUROC'); axs[1].set_title('(b) evaluation leakage', loc='left', fontsize=7)
    fig.tight_layout(h_pad=0.6); savefig(fig, 'fig5_ablation_leakage_1col')


def fig7_1col(FILES, SZB, s=19, f=128):
    q = np.load(os.path.join(RES, 'pred_BRIDGE.npz')); g = np.load(os.path.join(RES, 'pred_BRIDGE-GT.npz'))
    z = max(SZB[f], key=lambda z: z['offset'] - z['onset']); name = FILES[f]['file']
    sel = np.where(q['fid'] == f)[0]; sel = sel[np.argsort(q['t0'][sel])]; sel = sel[(q['t0'][sel] >= z['onset'] - 300) & (q['t0'][sel] <= z['offset'] + 120)]
    ch = read_edf(os.path.join(EDF, name)); sos = butter(4, [0.5, 45], btype='bandpass', fs=FS, output='sos'); chans = [0, 2, 6, 16]
    a, b = int(q['t0'][sel[0]] * FS), int((q['t0'][sel[-1]] + 4) * FS)
    X = sosfiltfilt(sos, np.stack([ch[CH18[c]] for c in chans]), axis=-1)[:, a:b][:, ::2]; tt = a / FS + np.arange(X.shape[1]) / (FS / 2)
    fig, axs = plt.subplots(3, 1, figsize=(3.5, 3.0), sharex=True, gridspec_kw=dict(height_ratios=[1.5, 1, 0.45]))
    for j, c in enumerate(chans):
        y = X[j] / (np.percentile(np.abs(X[j]), 99) + 1e-6); axs[0].plot(tt, y - 2.2 * j, lw=0.2, color='0.15')
    axs[0].set_yticks([-2.2 * j for j in range(len(chans))]); axs[0].set_yticklabels([CH18[c] for c in chans], fontsize=6)
    tw = q['t0'][sel] + 2
    for m, d in (('BRIDGE', q), ('BRIDGE-GT', g)):
        o = np.argsort(d['idx']); ii = o[np.searchsorted(d['idx'][o], q['idx'][sel])]
        axs[1].plot(tw, d['p'][ii], color=COL[m], lw=0.8, label=m); axs[1].axhline(d['t_fa'][ii].mean(), color=COL[m], ls='--', lw=0.5)
    p, t, tau = q['p'][sel], q['t'][sel].mean(), q['tau_0.95'][sel].mean(); t_fa = q['t_fa'][sel].mean()
    cf = np.where(p >= t, (p - t) / max(1 - t, 1e-6), (t - p) / max(t, 1e-6)); acc = cf >= tau
    alarm = np.convolve(((p >= t_fa) & acc).astype(int), np.ones(3, int))[:len(p)] >= 2
    axs[2].fill_between(tw, 0, alarm.astype(float), step='mid', color='#c0504d', lw=0, label='alarm')
    axs[2].fill_between(tw, 0, (~acc).astype(float) * 0.5, step='mid', color='#f2c14e', lw=0, label='referred')
    for ax in axs: ax.axvspan(z['onset'], z['offset'], color='#c0504d', alpha=0.12, lw=0)
    axs[1].set_ylabel('Probability', fontsize=6.5); axs[1].legend(frameon=True, framealpha=0.85, edgecolor='none', loc='upper left', fontsize=5.5)
    axs[2].set_yticks([]); axs[2].legend(frameon=True, framealpha=0.85, edgecolor='none', loc='upper left', ncol=2, fontsize=5.5)
    axs[2].set_xlabel(f'Time in {name} (s)', fontsize=6.5); savefig(fig, 'fig7_example_timeline_1col')


def fig8_1col(Ca, Y, AUG, EXT):
    na = ~AUG & ~EXT; rng = np.random.default_rng(0); ii = np.where(na & (Y == 1))[0]; jj = rng.choice(np.where(na & (Y == 0))[0], 50000, replace=False)
    full = lambda v: (lambda M: M + M.T)(np.bincount(IU[0] * N_CH + IU[1], weights=v, minlength=N_CH * N_CH).reshape(N_CH, N_CH))
    A1 = full(np.abs(Ca[ii].astype(np.float32)).mean(0)); A0 = full(np.abs(Ca[jj].astype(np.float32)).mean(0)); D = A1 - A0
    fig = plt.figure(figsize=(3.5, 3.5)); gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35)
    for k, (M, ttl) in enumerate(((A0, '(a) interictal |corr|'), (A1, '(b) ictal |corr|'))):
        ax = fig.add_subplot(gs[0, k]); ax.imshow(M, vmin=0, vmax=1, cmap='viridis'); ax.set_title(ttl, loc='left', fontsize=6.5)
        ax.set_xticks(range(N_CH)); ax.set_yticks(range(N_CH)); ax.set_xticklabels(CH18, rotation=90, fontsize=3.2); ax.set_yticklabels(CH18, fontsize=3.2)
        ax.tick_params(length=1, pad=1)
    ax = fig.add_subplot(gs[1, 0]); th = np.linspace(0, 2 * np.pi, 200); ax.plot(1.08 * np.cos(th), 1.08 * np.sin(th), color='0.6', lw=0.5)
    for qq in np.argsort(-np.abs(D[IU]))[:25]:
        i, j = IU[0][qq], IU[1][qq]; w = D[i, j]
        ax.plot(mids[[i, j], 0], mids[[i, j], 1], color='#c0504d' if w > 0 else '#1f5a96', lw=0.4 + 5 * abs(w), alpha=0.8)
    ax.scatter(mids[:, 0], mids[:, 1], s=5, color='k', zorder=3); ax.set_aspect('equal'); ax.axis('off'); ax.set_title('(c) ictal - interictal', loc='left', fontsize=6.5)
    ax = fig.add_subplot(gs[1, 1]); mx = np.array(list(GW['MIX'].values())); b = np.array(list(GW['BETA'].values())).mean(axis=(1, 3))
    ax.bar([0, 1], mx.mean(0), yerr=mx.std(0), color=[COL['BRIDGE'], '#9db7d5'], width=0.6, capsize=1.5)
    ax.bar([2.4, 3.4], b.mean(0), yerr=b.std(0), color=[COL['BRIDGE-GT'], '#f0b27a'], width=0.6, capsize=1.5)
    ax.set_xticks([0, 1, 2.4, 3.4]); ax.set_xticklabels(['anat.', 'func.', 'anat.', 'func.'], fontsize=5.5)
    ax.text(0.5, -0.17, 'BRIDGE\nmix weight', ha='center', va='top', fontsize=5.5, transform=ax.get_xaxis_transform())
    ax.text(2.9, -0.17, 'BRIDGE-GT\nmean ' + r'$\beta$', ha='center', va='top', fontsize=5.5, transform=ax.get_xaxis_transform())
    ax.tick_params(labelsize=5.5); ax.set_title('(d) learned weights', loc='left', fontsize=6.5); savefig(fig, 'fig8_graphs_1col')


def fig11_1col(F, M):
    import pandas as pd
    na = (M[:, 5] == 0) & (M[:, 7] == 0); fid = M[:, 2].astype(int); t0 = M[:, 3]; subj = M[:, 0].astype(int); y = M[:, 4].astype(int)
    absv = np.full(len(M), np.nan); refv = np.full(len(M), np.nan)
    for f in np.unique(fid[na]):
        ii = np.where(na & (fid == f))[0]; ii = ii[np.argsort(t0[ii])]; x = F[ii][:, :, 7]
        med = pd.DataFrame(x).rolling(150, min_periods=1).median().to_numpy(); absv[ii] = x.mean(1); refv[ii] = (x - med).mean(1)
    subjects = sorted(np.unique(subj[na]))
    fig, axs = plt.subplots(2, 1, figsize=(3.5, 2.9), sharex=True)
    for ax, v, ttl in ((axs[0], absv, '(a) absolute log line length'), (axs[1], refv, '(b) referenced to the patient baseline')):
        for i, s_ in enumerate(subjects):
            for c, col, dx in ((0, '#9e9e9e', -0.2), (1, '#c0504d', 0.2)):
                w = v[na & (subj == s_) & (y == c)]
                if len(w) > 2000: w = np.random.default_rng(i).choice(w, 2000, replace=False)
                if len(w): ax.boxplot(w, positions=[i + dx], widths=0.34, showfliers=False, patch_artist=True,
                                      boxprops=dict(facecolor=col, lw=0.3), medianprops=dict(color='k', lw=0.5), whiskerprops=dict(lw=0.3), capprops=dict(lw=0.3))
        ax.set_title(ttl, loc='left', fontsize=6.5); ax.set_ylabel('log10', fontsize=6.5); ax.tick_params(labelsize=5.5)
    axs[1].set_xticks(range(len(subjects))); axs[1].set_xticklabels([str(s_) for s_ in subjects], fontsize=5)
    axs[1].set_xlabel('Subject (grey: interictal, red: ictal)', fontsize=6.5); fig.tight_layout(h_pad=0.4); savefig(fig, 'fig11_baseline_reference_1col')


d = np.load(FEAT, allow_pickle=False); M = d['meta']; FILES_ALL = json.loads(str(d['files'])); SZ = json.loads(str(d['seizures']))
main = [i for i, f in enumerate(FILES_ALL) if not f['extra']]; loc2v = {li: vi for vi, li in enumerate(main)}
FILES = [FILES_ALL[i] for i in main]; SZB = {}
for z in SZ:
    if z['fid'] in loc2v: SZB.setdefault(loc2v[z['fid']], []).append(z)
fig2_1col(); fig5(); fig6(); fig9()
print('example', fig7(FILES, SZB))
print('graph weights', fig8(d['C'], M[:, 4].astype(int), M[:, 5].astype(bool), M[:, 7].astype(bool)))
fig5_1col(); fig7_1col(FILES, SZB); fig8_1col(d['C'], M[:, 4].astype(int), M[:, 5].astype(bool), M[:, 7].astype(bool)); fig11_1col(d['F'], M)
