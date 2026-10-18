"""BRIDGE (Baseline-Referenced Interpretable Dual-Graph Encoder): cross-patient seizure detection on the CHB-MIT Scalp EEG
Database with a causal patient-baseline reference and dual-graph learning for edge devices. Code of the conference paper
"BRIDGE: Patient-Baseline-Referenced Dual-Graph Learning for Cross-Patient Seizure Detection on Edge Devices".

Run on Google Colab (Runtime > Change runtime type > GPU) or on any machine with Python 3.10+, PyTorch and a GPU:
    python bridge_chbmit.py --smoke     # about 5 min: 3 cases, 2 seizure files each, 1 epoch (checks the pipeline)
    python bridge_chbmit.py             # full run: about 7.5 GB download and 1.5-2 h on an A100 GPU
    python bridge_chbmit.py --isolate   # isolating ablations only (reuses the feature cache), about 30 min
Options: --data <download folder> (default /content/chbmit), --out <output folder> (default /content/chbmit_out),
--extra <seizure-free files per case> (default 1), --skip_extras, --skip_fixed. If the run stops, start it again with the
same command: finished subjects, ablations and seeds are loaded from checkpoints. Outputs: log, per-subject metrics,
summary, predictions, ablation, leakage, quantization, fixed split, efficiency, results tables, figures/ (PNG and PDF) and
the zip bridge_results.zip (bridge_results_isolate.zip for --isolate), also copied to MyDrive if Drive is mounted.

Data: CHB-MIT Scalp EEG Database v1.0.0 (PhysioNet, ODC-BY 1.0; Shoeb 2009, Goldberger et al. 2000), read from the
AWS open-data mirror (s3://physionet-open) with physionet.org as fallback. Leave-one-subject-out set: all files with
seizures plus --extra seizure-free files per case; chb21 is a later recording of the subject of chb01, so the two cases form
one subject (23 subjects). Fixed split: train chb01-19, validation chb20-21, test every recording of chb22-23. The 18
bipolar channels common to the montage are used; files without them are skipped.

BRIDGE (Baseline-Referenced Interpretable Dual-Graph Encoder): 4-s windows, 0.5-45 Hz band-pass; node features per
channel: 7 log band powers, log line length, log variance, kurtosis, zero-crossing rate. (1) Patient-baseline reference:
the log features minus their running median over the previous 10 min of the same recording (causal, label-free), given
next to the absolute values. (2) Dual graph: static electrode-distance graph and per-window correlation graph (top-4
neighbours), mixed with learned weights; 2 message-passing layers (BRIDGE) or attention biased by both graphs
(BRIDGE-GT). Variant tested in the ablation: a GRU over the embeddings of the current and the 3 preceding windows (16 s).
Baselines on the same referenced features: channel Transformer without graph, MLP, logistic regression, random forest;
raw 1-D CNN on the 128-Hz signals. Thresholds (maximum F1, at most 1 false alarm per hour) and abstention thresholds
(coverage 0.98, 0.95, 0.9) come from 3 inner subject-wise folds of the training subjects only. Metrics: window AUROC/AUPRC,
seizure sensitivity, false alarms per hour (alarm = 2 of the last 3 windows), latency, referrals per hour; fixed split:
balanced accuracy (threshold from the validation cases), AUPRC, AUROC over 3 seeds.
"""
import argparse, copy, csv, io, json, math, os, pickle, re, shutil, sys, time, urllib.parse, urllib.request, zipfile
import concurrent.futures as cf
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import butter, sosfilt, sosfiltfilt
from scipy.stats import kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument('--smoke', action='store_true', help='3 cases, 2 seizure files each, 1 epoch')
ap.add_argument('--data', default='/content/chbmit', help='download folder')
ap.add_argument('--out', default='/content/chbmit_out', help='output folder')
ap.add_argument('--extra', type=int, default=1, help='seizure-free files added per case')
ap.add_argument('--drive', default='/content/drive/MyDrive', help='results zip is copied here if it exists')
ap.add_argument('--skip_extras', action='store_true', help='skip ablation, leakage and quantization')
ap.add_argument('--skip_fixed', action='store_true', help='skip the fixed-split comparison (no extra download)')
ap.add_argument('--isolate', action='store_true', help='only the isolating ablations (views, graph mixing, attention biases)')
args, _ = ap.parse_known_args()
SMOKE = args.smoke
if args.isolate: args.skip_fixed = True   # the isolating ablations use the leave-one-subject-out set only
T_START = time.time()
os.makedirs(args.out, exist_ok=True); os.makedirs(args.data, exist_ok=True)
LOG = open(os.path.join(args.out, 'log.txt'), 'a')


def log(*a):
    s = f'[{(time.time() - T_START) / 60:6.1f} min] ' + ' '.join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + '\n'); LOG.flush()


dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FS, WIN_S = 256, 4
WIN, HOP, AUG_HOP = FS * WIN_S, FS * WIN_S, FS
BANDS = [(0.5, 2), (2, 4), (4, 8), (8, 13), (13, 20), (20, 30), (30, 45)]
CH18 = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
        'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2', 'FZ-CZ', 'CZ-PZ']
N_CH, N_F = len(CH18), len(BANDS) + 4
N_LOG = len(BANDS) + 2            # log band powers, log line length, log variance: referenced to the patient baseline
N_FR = N_F + N_LOG                # node features: 11 absolute + 9 referenced
N_REL = N_F                       # relative view only: 9 referenced + kurtosis and zero-crossing rate
W_BASE, K_CTX = 150, 4            # baseline: previous 150 windows (10 min); context variant: current + 3 preceding windows (16 s)
FA_BUDGET = 1.0                   # false alarms per hour allowed on the inner validation subjects
CASES = [1, 2, 3] if SMOKE else list(range(1, 25))
SUBJECT = lambda case: 1 if case == 21 else case
FIX_TRAIN, FIX_VAL, FIX_TEST = ([1], [2], [3]) if SMOKE else (list(range(1, 20)), [20, 21], [22, 23])
FIX_SEEDS = (0,) if SMOKE else (0, 1, 2)
MIRRORS = ['https://physionet-open.s3.amazonaws.com/chbmit/1.0.0/', 'https://physionet.org/files/chbmit/1.0.0/']
EPOCHS, EPOCHS_RAW = (1, 1) if SMOKE else (15, 10)
INNER, RATIO, BS, SEED = (2 if SMOKE else 3), 4, 1024, 0
KAPPAS = (0.98, 0.95, 0.9)   # coverages: referring 2-10% of windows keeps the clinical review load small
MODELS = ['BRIDGE', 'BRIDGE-GT', 'Transformer', 'MLP', 'LogReg', 'RandomForest', 'RawCNN']
ABLATIONS = ['BRIDGE w/o baseline reference', 'BRIDGE anatomical graph only', 'BRIDGE functional graph only', 'BRIDGE without graph',
             'BRIDGE + 16-s temporal context']
FIXED_MODELS = MODELS + ['BRIDGE w/o baseline reference']
warnings.filterwarnings('ignore', message='.*quantiz.*')
safe = lambda name: re.sub(r'[^A-Za-z0-9_.-]+', '_', name)   # model name -> file name
log(f'start; device {dev}; smoke={SMOKE}; torch {torch.__version__}')



# ------------------------------------------------------------------ download
def fetch(rel, tries=3):
    dest = os.path.join(args.data, rel)
    if os.path.exists(dest) and os.path.getsize(dest) > 0: return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True); err = None
    for base in MIRRORS:
        for k in range(tries):
            try:
                req = urllib.request.Request(base + urllib.parse.quote(rel), headers={'User-Agent': 'Mozilla/5.0'})   # '+' in names
                with urllib.request.urlopen(req, timeout=180) as r, open(dest + '.part', 'wb') as f:
                    n = int(r.headers.get('Content-Length', -1)); shutil.copyfileobj(r, f, 1 << 20)
                if n >= 0 and os.path.getsize(dest + '.part') != n: raise IOError('incomplete download')   # retried
                os.replace(dest + '.part', dest); return dest
            except Exception as e:
                err = e
                if getattr(e, 'code', None) in (403, 404): break
                time.sleep(2 + 3 * k)
    raise RuntimeError(f'{rel}: {err}')


def parse_summary(txt):
    files, cur = {}, None
    for line in txt.splitlines():
        line = line.strip()
        m = re.match(r'File Name:\s*(\S+\.edf)', line)
        if m: cur = m.group(1); files[cur] = []; continue
        m = re.match(r'Seizure\s*\d*\s*Start Time:\s*(\d+)', line)
        if m and cur: files[cur].append([int(m.group(1)), None]); continue
        m = re.match(r'Seizure\s*\d*\s*End Time:\s*(\d+)', line)
        if m and cur and files[cur] and files[cur][-1][1] is None: files[cur][-1][1] = int(m.group(1))
    return {f: [(a, b) for a, b in s if b is not None and b > a] for f, s in files.items()}


# ------------------------------------------------------------------ EDF reader and features
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
    return out, nsamp[0] / float(raw[244:252].decode().strip())


SOS = butter(4, [0.5, 45], btype='bandpass', fs=FS, output='sos')
FREQ = np.fft.rfftfreq(WIN, 1 / FS); BMASK = [(FREQ >= lo) & (FREQ < hi) for lo, hi in BANDS]
HANN = np.hanning(WIN).astype(np.float32); IU = np.triu_indices(N_CH, 1)


def window_features(seg):
    """seg: (n, 18, WIN) band-passed EEG in microvolts -> node features (n, 18, 11) and correlations (n, 153)."""
    P = np.abs(np.fft.rfft(seg * HANN, axis=-1)) ** 2
    bp = np.stack([P[..., m].sum(-1) for m in BMASK], -1)
    ll = np.abs(np.diff(seg, axis=-1)).mean(-1); var = seg.var(-1); ku = kurtosis(seg, axis=-1)
    sb = np.signbit(seg); zc = (sb[..., 1:] != sb[..., :-1]).mean(-1)
    f = np.concatenate([np.log10(bp + 1e-6), np.log10(ll + 1e-6)[..., None], np.log10(var + 1e-6)[..., None],
                        (np.clip(ku, -3, 30) / 10)[..., None], (zc * 10)[..., None]], -1)
    z = (seg - seg.mean(-1, keepdims=True)) / (seg.std(-1, keepdims=True) + 1e-6)
    C = np.einsum('nct,ndt->ncd', z, z, optimize=True) / seg.shape[-1]
    return np.nan_to_num(f).astype(np.float32), np.nan_to_num(C[:, IU[0], IU[1]]).astype(np.float16)


POS = {'FP1': (-0.31, 0.95), 'FP2': (0.31, 0.95), 'F7': (-0.81, 0.59), 'F3': (-0.41, 0.51), 'FZ': (0, 0.47),
       'F4': (0.41, 0.51), 'F8': (0.81, 0.59), 'T7': (-1, 0), 'C3': (-0.5, 0), 'CZ': (0, 0), 'C4': (0.5, 0), 'T8': (1, 0),
       'P7': (-0.81, -0.59), 'P3': (-0.41, -0.51), 'PZ': (0, -0.47), 'P4': (0.41, -0.51), 'P8': (0.81, -0.59),
       'O1': (-0.31, -0.95), 'O2': (0.31, -0.95)}


def static_graph():
    """Channels sharing an electrode: weight 1; others: Gaussian of the distance between bipolar midpoints (< 0.6)."""
    mid = np.array([np.mean([POS[e] for e in ch.split('-')], 0) for ch in CH18]); A = np.zeros((N_CH, N_CH), np.float32)
    for i in range(N_CH):
        for j in range(N_CH):
            if i == j: continue
            d = np.linalg.norm(mid[i] - mid[j])
            A[i, j] = 1.0 if set(CH18[i].split('-')) & set(CH18[j].split('-')) else (math.exp(-(d / 0.35) ** 2) if d < 0.6 else 0.0)
    return A / A.sum(1, keepdims=True)


A_STATIC = static_graph()


# ------------------------------------------------------------------ dataset
def build(plan, cache, raw_path, fid0, sz0, extra):
    """Download the planned EDF files and compute window features, the 128-Hz raw windows (memmap) and the metadata.
    meta columns: subject, case, file id, window start (s), ictal, augmented, seizure id, extra test file."""
    ok = []
    with cf.ThreadPoolExecutor(8) as ex:
        futs = {ex.submit(fetch, f'chb{c:02d}/{f}'): (c, f, s) for c, f, s in plan}
        for k, fu in enumerate(cf.as_completed(futs)):
            try: fu.result(); ok.append(futs[fu])
            except Exception as e: log('download failed', futs[fu][1], e)
            if (k + 1) % 20 == 0: log(f'  downloaded {k + 1}/{len(plan)}')
    ok.sort(key=lambda r: (r[0], r[1]))
    cap = sum(os.path.getsize(os.path.join(args.data, f'chb{c:02d}', f)) // (2 * FS * N_CH * WIN_S) + 2 +
              sum(int(e - a) + 2 * WIN_S for a, e in s) for c, f, s in ok)
    mm = np.memmap(raw_path, dtype=np.float16, mode='w+', shape=(max(cap, 1), N_CH, WIN // 2))
    Fl, Cl, meta, files, sz_tab, skipped, k = [], [], [], [], [], [], 0
    for fi, (c, f, seiz) in enumerate(ok):
        try: ch, fs = read_edf(os.path.join(args.data, f'chb{c:02d}', f))
        except Exception as e: skipped.append([f, str(e)]); continue
        if abs(fs - FS) > 1e-6 or any(x not in ch for x in CH18): skipped.append([f, 'missing channels or rate']); continue
        X = sosfiltfilt(SOS, np.stack([ch[x] for x in CH18]), axis=-1).astype(np.float32); T = X.shape[1]; fid = fid0 + len(files); sids = []
        for a, e in seiz:
            sz_tab.append(dict(case=c, subject=SUBJECT(c), file=f, fid=fid, onset=a, offset=e)); sids.append(sz0 + len(sz_tab) - 1)
        starts = [(st, 0) for st in range(0, T - WIN + 1, HOP)]
        for a, e in seiz:
            starts += [(st, 1) for st in range(int(a * FS) - WIN // 2, int(e * FS) - WIN // 2, AUG_HOP) if 0 <= st <= T - WIN and st % HOP]
        for b in range(0, len(starts), 256):
            chunk = starts[b:b + 256]; seg = np.stack([X[:, st:st + WIN] for st, _ in chunk])
            fe, co = window_features(seg); Fl.append(fe); Cl.append(co)
            mm[k:k + len(chunk)] = np.clip(seg[..., ::2], -6e4, 6e4).astype(np.float16)
            for st, aug in chunk:
                ctr = (st + WIN / 2) / FS; sid = next((sids[j] for j, (a, e) in enumerate(seiz) if a <= ctr < e), -1)
                meta.append((SUBJECT(c), c, fid, st / FS, int(sid >= 0), aug, sid, extra))
            k += len(chunk)
        files.append(dict(case=c, file=f, hours=T / FS / 3600, seizures=seiz, extra=extra))
        if (fi + 1) % 10 == 0 or fi + 1 == len(ok): log(f'  features {fi + 1}/{len(ok)} files, {k} windows')
    mm.flush(); del mm
    out = dict(F=np.concatenate(Fl) if Fl else np.zeros((0, N_CH, N_F), np.float32), C=np.concatenate(Cl) if Cl else np.zeros((0, len(IU[0])), np.float16),
               meta=np.array(meta, dtype=np.float64).reshape(-1, 8), cap=max(cap, 1), nraw=k, files=json.dumps(files), seizures=json.dumps(sz_tab), skipped=json.dumps(skipped))
    np.savez(cache, **out); return out


def load_or_build(tag, make_plan, fid0, sz0, extra):
    cache = os.path.join(args.out if SMOKE else args.data, f'features_{tag}' + ('_smoke' if SMOKE else '') + '.npz')
    raw_path = os.path.join(args.data, f'raw128_{tag}' + ('_smoke' if SMOKE else '') + '.f16')
    if os.path.exists(cache) and os.path.exists(raw_path):
        d = dict(np.load(cache, allow_pickle=False)); log(f'loaded cached features ({tag}): {len(d["F"])} windows')
        if d['meta'].shape[1] == 7: d['meta'] = np.column_stack([d['meta'], np.zeros(len(d['meta']))])   # v1.0/1.1 cache
    else:
        plan = make_plan(); log(f'{len(plan)} EDF files planned ({tag}); downloading'); d = build(plan, cache, raw_path, fid0, sz0, extra)
        log(f'features saved ({tag}): {len(d["F"])} windows; skipped files: {len(json.loads(str(d["skipped"])))}')
    RAWMM = np.memmap(raw_path, dtype=np.float16, mode='r', shape=(int(d['cap']), N_CH, WIN // 2))
    return d, RAWMM


SUMMARIES = {}
def summary(case):
    if case not in SUMMARIES:
        rel = f'chb{case:02d}/chb{case:02d}-summary.txt'
        try: SUMMARIES[case] = parse_summary(open(fetch(rel), errors='ignore').read())
        except Exception as e: log('summary missing', rel, e); SUMMARIES[case] = {}
    return SUMMARIES[case]


def main_plan():
    plan = []
    for case in CASES:
        files = summary(case); sz = [f for f, s in files.items() if s]; free = [f for f, s in files.items() if not s]
        if SMOKE: sz, free = sz[:2], []
        plan += [(case, f, files[f]) for f in sz + free[:args.extra]]
    return plan


dM, RAW_M = load_or_build('v1_0', main_plan, 0, 0, 0)          # same files and cache as v1.0 and v1.1
FILES, SZ_TAB, SKIPPED = json.loads(str(dM['files'])), json.loads(str(dM['seizures'])), json.loads(str(dM['skipped']))
main_names = {f['file'] for f in FILES} | {s[0] for s in SKIPPED}


def extra_plan():
    """Every remaining recording of the fixed-split test cases (chb22, chb23), for the BIOT/CBraMod protocol."""
    plan = []
    for case in FIX_TEST:
        files = summary(case); rest = [f for f in files if f not in main_names]
        plan += [(case, f, files[f]) for f in (rest[:1] if SMOKE else rest)]
    return plan


if args.skip_fixed: dE, RAW_E = dict(F=dM['F'][:0], C=dM['C'][:0], meta=dM['meta'][:0], nraw=0, files='[]', seizures='[]', skipped='[]'), None
else: dE, RAW_E = load_or_build('v1_2_extra', extra_plan, len(FILES), len(SZ_TAB), 1)
FILES += json.loads(str(dE['files'])); SZ_TAB += json.loads(str(dE['seizures'])); SKIPPED += json.loads(str(dE['skipped']))
Fa, Ca, META = np.concatenate([dM['F'], dE['F']]), np.concatenate([dM['C'], dE['C']]), np.concatenate([dM['meta'], dE['meta']])
NRAW_M, NRAW_E = int(dM['nraw']), int(dE['nraw'])

SUBJ, CASE, FID = META[:, 0].astype(int), META[:, 1].astype(int), META[:, 2].astype(int)
T0, Y, AUG, SID = META[:, 3], META[:, 4].astype(int), META[:, 5].astype(bool), META[:, 6].astype(int)
EXTRA_F = META[:, 7].astype(bool)
ALL = np.arange(len(Y)); MAIN = ALL[~EXTRA_F]; SZ_BY_FILE = {}
for z in SZ_TAB: SZ_BY_FILE.setdefault(z['fid'], []).append(z)
subjects = sorted(np.unique(SUBJ[MAIN]).tolist())
mf = [f for f in FILES if not f.get('extra')]
DATASET = dict(windows=int((~AUG & ~EXTRA_F).sum()), ictal_windows=int(((Y == 1) & ~AUG & ~EXTRA_F).sum()), augmented_ictal=int((AUG & ~EXTRA_F).sum()),
               subjects=subjects, files=len(mf), hours=float(sum(f['hours'] for f in mf)), seizures=sum(1 for z in SZ_TAB if not FILES[z['fid']].get('extra')),
               skipped=SKIPPED, fixed_split_extra_test_files=len(FILES) - len(mf), fixed_split_extra_test_hours=float(sum(f['hours'] for f in FILES if f.get('extra'))))
json.dump(DATASET, open(os.path.join(args.out, 'dataset.json'), 'w'), indent=1)
log(f"dataset: {DATASET['files']} files, {DATASET['hours']:.1f} h, {DATASET['seizures']} seizures, {len(subjects)} subjects, "
    f"{DATASET['windows']} windows ({DATASET['ictal_windows']} ictal); fixed-split extra test: {DATASET['fixed_split_extra_test_files']} files, "
    f"{DATASET['fixed_split_extra_test_hours']:.1f} h")


# ------------------------------------------------------------------ patient-baseline reference and temporal context
def baseline_and_context():
    """Causal per-recording baseline: running median of the previous W_BASE non-augmented windows (current included) of each
    node feature; the referenced features are the log features minus this baseline. Context: indices of the K_CTX - 1
    preceding windows of the same recording (the window itself where the recording has not started yet)."""
    B = np.empty_like(Fa); PREV = np.empty((len(Y), max(K_CTX - 1, 1)), np.int64)
    for f in np.unique(FID):
        ii = np.where(FID == f)[0]; na = ii[~AUG[ii]]; na = na[np.argsort(T0[na])]; t_na = T0[na]
        med = pd.DataFrame(Fa[na].reshape(len(na), -1)).rolling(W_BASE, min_periods=1).median().to_numpy(np.float32).reshape(-1, N_CH, N_F)
        B[ii] = med[np.clip(np.searchsorted(t_na, T0[ii] + 1e-6, side='right') - 1, 0, len(na) - 1)]
        for j in range(1, max(K_CTX, 2)):
            pj = np.searchsorted(t_na, T0[ii] - WIN_S * j + 1e-6, side='right') - 1
            PREV[ii, j - 1] = np.where(pj >= 0, na[np.clip(pj, 0, None)], ii)
    return B, PREV


t1 = time.time(); BASE, PREV = baseline_and_context()
FR = np.concatenate([Fa, Fa[..., :N_LOG] - BASE[..., :N_LOG]], -1)   # node features: 11 absolute + 9 baseline-referenced
del BASE; log(f'patient-baseline reference and context indices in {time.time() - t1:.0f} s')

Ft = torch.tensor(FR, device=dev); Yt = torch.tensor(Y, device=dev); PREVt = torch.tensor(PREV, device=dev)
IU_t = torch.tensor(np.stack(IU), device=dev)
Ct = torch.tensor(Ca, device=dev)
raw_bytes = (NRAW_M + NRAW_E) * N_CH * (WIN // 2) * 2
Rt = torch.empty((NRAW_M + NRAW_E, N_CH, WIN // 2), dtype=torch.float16)   # filled in chunks to keep the peak memory low
for src, off, n in ((RAW_M, 0, NRAW_M), (RAW_E, NRAW_M, NRAW_E)):
    for b in range(0, n, 20000): Rt[off + b:off + min(n, b + 20000)] = torch.from_numpy(np.asarray(src[b:min(n, b + 20000)]))
if dev.type == 'cuda' and torch.cuda.get_device_properties(0).total_memory > 4 * raw_bytes: Rt = Rt.to(dev)
log(f'raw signals: {raw_bytes / 1e9:.2f} GB on {Rt.device}')


# ------------------------------------------------------------------ models
def full_corr(c):
    M = torch.zeros(c.shape[0], N_CH, N_CH, device=c.device, dtype=torch.float32)
    M[:, IU_t[0], IU_t[1]] = c.float(); return M + M.transpose(1, 2)


def topk_norm(M, k=4):
    M = M.abs(); v, i = M.topk(k, dim=-1); A = torch.zeros_like(M).scatter_(-1, i, v); A = 0.5 * (A + A.transpose(1, 2))
    return A / (A.sum(-1, keepdim=True) + 1e-6)


class GLayer(nn.Module):
    def __init__(s, d):
        super().__init__(); s.a = nn.Linear(d, d); s.b = nn.Linear(d, d, bias=False); s.n = nn.LayerNorm(d); s.dp = nn.Dropout(0.1)

    def forward(s, h, A): return s.n(h + s.dp(F.gelu(s.a(h) + torch.matmul(A, s.b(h)))))


class DualGraphEncoder(nn.Module):
    """One 4-s window: node features -> 2 message-passing layers over a learned mix of the anatomical graph and the
    window's functional graph -> mean and max pooling. graph: 'both' (proposed), 'static', 'dynamic' or 'none'."""
    def __init__(s, f, d=32, graph='both'):
        super().__init__(); s.graph = graph; s.inp = nn.Linear(f, d); s.L = nn.ModuleList([GLayer(d), GLayer(d)])
        s.mix = nn.Parameter(torch.zeros(2)); s.register_buffer('As', torch.tensor(A_STATIC))

    def forward(s, x, c):
        n = x.shape[0]
        if s.graph == 'none': A = torch.zeros(n, N_CH, N_CH, device=x.device)
        elif s.graph == 'static': A = s.As.expand(n, -1, -1)
        else:
            Ad = topk_norm(full_corr(c))
            if s.graph == 'fixed': A = 0.5 * s.As + 0.5 * Ad   # equal, not learned
            else: A = Ad if s.graph == 'dynamic' else (lambda w: w[0] * s.As + w[1] * Ad)(torch.softmax(s.mix, 0))
        h = F.gelu(s.inp(x))
        for layer in s.L: h = layer(h, A)
        return torch.cat([h.mean(1), h.amax(1)], -1)


class GTLayer(nn.Module):
    """Pre-norm self-attention whose scores are biased per head by the anatomical graph (log weights) and the
    functional graph of the window (absolute correlation), followed by a feed-forward block."""
    def __init__(s, d=32, h=4, use=(1.0, 1.0)):
        super().__init__(); s.h = h; s.register_buffer('use', torch.tensor(use)); s.qkv = nn.Linear(d, 3 * d); s.o = nn.Linear(d, d); s.n1 = nn.LayerNorm(d); s.n2 = nn.LayerNorm(d)
        s.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(0.1), nn.Linear(2 * d, d)); s.dp = nn.Dropout(0.1)
        s.beta = nn.Parameter(torch.full((2, h), 0.5))

    def forward(s, x, Bs, Bd):
        B, N, d = x.shape; q, k, v = s.qkv(s.n1(x)).view(B, N, 3, s.h, d // s.h).permute(2, 0, 3, 1, 4)
        bias = s.use[0] * s.beta[0].view(1, -1, 1, 1) * Bs + s.use[1] * s.beta[1].view(1, -1, 1, 1) * Bd.unsqueeze(1)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=0.1 if s.training else 0.0)
        x = x + s.dp(s.o(a.transpose(1, 2).reshape(B, N, d))); return x + s.dp(s.ff(s.n2(x)))


class DualGraphAttentionEncoder(nn.Module):
    """One 4-s window: global attention over channel tokens that prefers anatomically close and functionally coupled
    channels by learned per-head amounts (beta); beta = 0 gives a plain Transformer."""
    def __init__(s, f, d=32, h=4, bias='both'):
        super().__init__(); s.inp = nn.Linear(f, d); s.pos = nn.Parameter(torch.zeros(1, N_CH, d))
        use = {'both': (1.0, 1.0), 'anat': (1.0, 0.0), 'func': (0.0, 1.0), 'none': (0.0, 0.0)}[bias]
        s.L = nn.ModuleList([GTLayer(d, h, use), GTLayer(d, h, use)]); s.n = nn.LayerNorm(d)
        s.register_buffer('Bs', torch.log(torch.tensor(A_STATIC) + 1e-3))

    def forward(s, x, c):
        Bd = full_corr(c).abs(); h = s.inp(x) + s.pos
        for layer in s.L: h = layer(h, s.Bs, Bd)
        h = s.n(h); return torch.cat([h.mean(1), h.amax(1)], -1)


class BRIDGE(nn.Module):
    """BRIDGE (Baseline-Referenced Interpretable Dual-Graph Encoder), proposed. Input: node features referenced to the
    patient's own causal baseline (absolute and referenced views). A dual-graph encoder embeds the window and a small
    classifier reads the pooled embedding. enc 'gnn': message passing (BRIDGE); enc 'gt': dual-graph-biased attention
    (BRIDGE-GT). With k > 1 (ablation variant) a GRU reads the embeddings of the current and k-1 preceding windows."""
    def __init__(s, f=None, d=32, graph='both', enc='gnn', k=None, bias='both'):
        super().__init__(); f = N_FR if f is None else f; s.k = 1 if k is None else k
        s.enc = DualGraphEncoder(f, d, graph) if enc == 'gnn' else DualGraphAttentionEncoder(f, d, bias=bias)
        s.gru = nn.GRU(2 * d, d, batch_first=True) if s.k > 1 else None
        s.head = nn.Sequential(nn.Linear(3 * d if s.k > 1 else 2 * d, d), nn.GELU(), nn.Dropout(0.1), nn.Linear(d, 2))

    def forward(s, x, c):   # x: (B, K, 18, f), c: (B, K, 153)
        B, K = x.shape[:2]; e = s.enc(x.reshape(B * K, N_CH, -1), c.reshape(B * K, -1)).view(B, K, -1)
        if s.gru is None: return s.head(e[:, 0])
        o, _ = s.gru(e.flip(1)); return s.head(torch.cat([o[:, -1], e[:, 0]], -1))


class ChanTransformer(nn.Module):
    """Channels as tokens with learned channel embeddings; full self-attention instead of a graph prior."""
    def __init__(s, f=None, d=32):
        super().__init__(); f = N_FR if f is None else f; s.inp = nn.Linear(f, d); s.pos = nn.Parameter(torch.zeros(1, N_CH, d))
        s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 64, 0.1, batch_first=True, norm_first=True), 2,
                                      enable_nested_tensor=False)
        s.head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(0.1), nn.Linear(d, 2))

    def forward(s, x, c): h = s.enc(s.inp(x[:, 0]) + s.pos); return s.head(torch.cat([h.mean(1), h.amax(1)], -1))


class MLP(nn.Module):
    def __init__(s, f=None):
        super().__init__(); f = N_FR if f is None else f
        s.net = nn.Sequential(nn.Flatten(), nn.Linear(N_CH * f, 64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 2))

    def forward(s, x, c): return s.net(x[:, 0])


class RawCNN(nn.Module):
    def __init__(s):
        super().__init__(); L, ch = [], [N_CH, 32, 64, 64, 64]
        for a, b in zip(ch[:-1], ch[1:]): L += [nn.Conv1d(a, b, 7, 2, 3), nn.BatchNorm1d(b), nn.ReLU()]
        s.net = nn.Sequential(*L, nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(64, 2))

    def forward(s, r, c=None):
        r = r.float(); return s.net((r - r.mean(-1, keepdim=True)) / (r.std(-1, keepdim=True) + 1e-3))


# name: (constructor, input kind, windows of context); 'ref' = absolute + baseline-referenced features, 'abs' = absolute only
MAKERS = {'BRIDGE': (lambda: BRIDGE(), 'ref', 1), 'BRIDGE-GT': (lambda: BRIDGE(enc='gt'), 'ref', 1),
          'Transformer': (lambda: ChanTransformer(), 'ref', 1), 'MLP': (lambda: MLP(), 'ref', 1), 'RawCNN': (RawCNN, 'raw', 1),
          # ablations of BRIDGE: one component removed (w/o baseline reference = the v1.0 Tiny GNN) or temporal context added
          'BRIDGE w/o baseline reference': (lambda: BRIDGE(f=N_F), 'abs', 1),
          'BRIDGE anatomical graph only': (lambda: BRIDGE(graph='static'), 'ref', 1),
          'BRIDGE functional graph only': (lambda: BRIDGE(graph='dynamic'), 'ref', 1),
          'BRIDGE without graph': (lambda: BRIDGE(graph='none'), 'ref', 1),
          'BRIDGE + 16-s temporal context': (lambda: BRIDGE(k=K_CTX), 'ref', K_CTX),
          # isolating ablations (v1.3)
          'BRIDGE relative view only': (lambda: BRIDGE(f=N_REL), 'rel', 1),
          'BRIDGE fixed equal graph mix': (lambda: BRIDGE(graph='fixed'), 'ref', 1),
          'BRIDGE-GT anatomical bias only': (lambda: BRIDGE(enc='gt', bias='anat'), 'ref', 1),
          'BRIDGE-GT functional bias only': (lambda: BRIDGE(enc='gt', bias='func'), 'ref', 1),
          'BRIDGE-GT without graph bias': (lambda: BRIDGE(enc='gt', bias='none'), 'ref', 1)}


def norm_stats(idx):
    sel = idx[~AUG[idx]]
    if len(sel) > 200000: sel = np.random.default_rng(0).choice(sel, 200000, replace=False)
    x = Ft[torch.as_tensor(sel, device=dev)]; return x.mean(0), x.std(0) + 1e-4


def get_batch(bi, kind, st, k=1):
    bt = torch.as_tensor(bi, device=dev)
    if kind == 'raw': return (Rt[bt] if Rt.device == dev else Rt[torch.as_tensor(bi)].to(dev, non_blocking=True)), None
    ix = torch.cat([bt[:, None], PREVt[bt]], 1)[:, :k]; x = (Ft[ix] - st[0]) / st[1]
    if kind == 'rel': return torch.cat([x[..., N_F:], x[..., N_LOG:N_F]], -1), Ct[ix]
    return (x[..., :N_F] if kind == 'abs' else x), Ct[ix]


def balanced(tr, rng):
    pos, neg = tr[Y[tr] == 1], tr[Y[tr] == 0]
    return np.concatenate([pos, rng.choice(neg, min(len(neg), RATIO * max(len(pos), 1)), replace=False)])


def train_torch(name, tr, seed=SEED):
    make_m, kind, k = MAKERS[name]; torch.manual_seed(seed); rng = np.random.default_rng(seed); m = make_m().to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4); st = norm_stats(tr) if kind != 'raw' else None
    for ep in range(EPOCHS_RAW if kind == 'raw' else EPOCHS):
        idx = balanced(tr, rng); rng.shuffle(idx); m.train()
        for b in range(0, len(idx), BS):
            bi = idx[b:b + BS]; x, c = get_batch(bi, kind, st, k)
            loss = F.cross_entropy(m(x, c), Yt[torch.as_tensor(bi, device=dev)])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    m.eval(); return m, kind, st, k


@torch.no_grad()
def predict_torch(model, idx):
    m, kind, st, k = model; out = []
    for b in range(0, len(idx), 8192):
        x, c = get_batch(idx[b:b + 8192], kind, st, k); out.append(torch.softmax(m(x, c).float(), -1)[:, 1].cpu())
    return torch.cat(out).numpy() if out else np.zeros(0)


def train_sk(name, tr, seed=SEED):
    idx = balanced(tr, np.random.default_rng(seed)); X = FR[idx].reshape(len(idx), -1); mu, sd = X.mean(0), X.std(0) + 1e-6
    clf = LogisticRegression(C=0.1, max_iter=500) if name == 'LogReg' else RandomForestClassifier(
        150, max_depth=14, min_samples_leaf=3, max_samples=0.5, n_jobs=-1, random_state=seed)
    clf.fit((X - mu) / sd, Y[idx]); return clf, mu, sd


def predict_sk(model, idx):
    clf, mu, sd = model
    out = [clf.predict_proba((FR[idx[b:b + 50000]].reshape(-1, N_CH * N_FR) - mu) / sd)[:, 1] for b in range(0, len(idx), 50000)]
    return np.concatenate(out) if out else np.zeros(0)


def fit(name, tr, seed=SEED): return ('sk', train_sk(name, tr, seed)) if name in ('LogReg', 'RandomForest') else ('torch', train_torch(name, tr, seed))
def pred(model, idx): return predict_sk(model[1], idx) if model[0] == 'sk' else predict_torch(model[1], idx)


# ------------------------------------------------------------------ calibration and metrics
def conf(p, t): return np.where(p >= t, (p - t) / max(1 - t, 1e-6), (t - p) / max(t, 1e-6))


def best_threshold(p, y):
    best, bt = -1.0, 0.5
    for t in np.unique(np.quantile(p, np.linspace(0.5, 0.9995, 300))):
        pr = p >= t; f1 = 2 * (pr & (y == 1)).sum() / (pr.sum() + (y == 1).sum() + 1e-9)
        if f1 > best: best, bt = f1, float(t)
    return bt


def runs(b):
    d = np.diff(np.r_[0, b.astype(int), 0]); return list(zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1))


def events(idx, p, t, tau=None):
    """Alarm = 2 of the last 3 accepted positive windows. False alarm: alarm run not overlapping a seizure (+30 s)."""
    acc = np.ones(len(p), bool) if tau is None else conf(p, t) >= tau; alarm = (p >= t) & acc
    r = dict(n=0, det=0, cov=0, lat=[], fa=0, ref=0, inter_h=0.0, hours=0.0)
    for fid in np.unique(FID[idx]):
        sel = np.where(FID[idx] == fid)[0]; o = sel[np.argsort(T0[idx][sel])]; t0 = T0[idx][o]
        s = np.convolve(alarm[o].astype(int), np.ones(3, int))[:len(o)] >= 2; seiz = SZ_BY_FILE.get(fid, [])
        r['inter_h'] += (Y[idx][o] == 0).sum() * WIN_S / 3600; r['hours'] += len(o) * WIN_S / 3600
        for i0, i1 in runs(s):
            if not any(t0[i0] < z['offset'] + 30 and t0[i1] + WIN_S > z['onset'] for z in seiz): r['fa'] += 1
        ref = ~acc[o]; r['ref'] += len(runs(ref))
        for z in seiz:
            inside = (t0 + WIN_S > z['onset']) & (t0 < z['offset']); hit = np.where(s & inside)[0]; r['n'] += 1
            if len(hit): r['det'] += 1; r['lat'].append(max(0.0, t0[hit[0]] + WIN_S - z['onset']))
            if len(hit) or (ref & inside).any(): r['cov'] += 1
    return r


def fa_threshold(idx, p, budget=FA_BUDGET):
    """Lowest threshold whose alarms on the inner validation subjects stay within the false-alarm budget (bisection
    over quantiles of the inner predictions)."""
    cand = np.unique(np.quantile(p, np.linspace(0.8, 0.99995, 256))); fa = lambda t: (lambda e: e['fa'] / max(e['inter_h'], 1e-9))(events(idx, p, t))
    lo, hi = 0, len(cand) - 1
    if fa(cand[hi]) > budget: return float(cand[hi])
    while lo < hi:
        mid = (lo + hi) // 2
        if fa(cand[mid]) <= budget: hi = mid
        else: lo = mid + 1
    return float(cand[lo])


def calibrate(name, tr):
    """Decision threshold (max F1), false-alarm-budget threshold and abstention thresholds from inner subject-wise
    folds of the training subjects only."""
    groups = np.array_split(np.random.default_rng(1).permutation(np.unique(SUBJ[tr])), INNER); P, Yv, V = [], [], []
    for g in groups:
        inner, va = tr[~np.isin(SUBJ[tr], g)], tr[np.isin(SUBJ[tr], g) & ~AUG[tr]]
        if (Y[inner] == 1).sum() == 0 or len(va) == 0: continue
        P.append(pred(fit(name, inner), va)); Yv.append(Y[va]); V.append(va)
    p, y, v = np.concatenate(P), np.concatenate(Yv), np.concatenate(V); t = best_threshold(p, y); c = conf(p, t)
    return t, {k: float(np.quantile(c, 1 - k)) for k in KAPPAS}, fa_threshold(v, p)


def subject_metrics(idx, p, t, tau, t_fa):
    y = Y[idx]; pr = p >= t; e = events(idx, p, t); e3 = events(idx, p, t_fa)
    r = dict(n_windows=len(idx), n_seizures=e['n'], hours=e['hours'],
             auroc=roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan,
             auprc=average_precision_score(y, p) if y.sum() else np.nan,
             win_sens=(pr & (y == 1)).sum() / max(1, (y == 1).sum()), win_spec=(~pr & (y == 0)).sum() / max(1, (y == 0).sum()),
             sz_sens=e['det'] / max(1, e['n']), fa_h=e['fa'] / max(e['inter_h'], 1e-9),
             latency=float(np.median(e['lat'])) if e['lat'] else np.nan, threshold=t,
             sz_sens_fa=e3['det'] / max(1, e3['n']), fa_h_fa=e3['fa'] / max(e3['inter_h'], 1e-9),
             latency_fa=float(np.median(e3['lat'])) if e3['lat'] else np.nan, threshold_fa=t_fa)
    for k in KAPPAS:
        acc = conf(p, t) >= tau[k]; e2 = events(idx, p, t, tau[k]); ya, pa = y[acc], pr[acc]
        r.update({f'cov_{k}': acc.mean(), f'sel_sens_{k}': (pa & (ya == 1)).sum() / (ya == 1).sum() if (ya == 1).sum() else np.nan,
                  f'sel_spec_{k}': (~pa & (ya == 0)).sum() / max(1, (ya == 0).sum()), f'sz_sens_{k}': e2['det'] / max(1, e2['n']),
                  f'sz_detected_or_referred_{k}': e2['cov'] / max(1, e2['n']), f'fa_h_{k}': e2['fa'] / max(e2['inter_h'], 1e-9),
                  f'referrals_h_{k}': e2['ref'] / max(e2['hours'], 1e-9)})
    return r


def boot(v, B=1000):
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    if not len(v): return [np.nan, np.nan, np.nan]
    m = np.random.default_rng(0).choice(v, (B, len(v))).mean(1)
    return [float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def write_csv(rows, name):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(args.out, name), 'w', newline='') as f:
        w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)


def save_ckpt(path, **kw):
    pickle.dump(dict(kw, smoke=SMOKE, fingerprint=FINGERPRINT), open(path + '.tmp', 'wb')); os.replace(path + '.tmp', path)


def load_ckpt(path, **must):
    if not os.path.exists(path): return None
    ck = pickle.load(open(path, 'rb'))
    if ck.get('smoke') == SMOKE and ck.get('fingerprint') == FINGERPRINT and all(ck.get(k) == v for k, v in must.items()): return ck
    log(f'{os.path.basename(path)} made with other settings or data; starting afresh'); return None


# ------------------------------------------------------------------ isolating ablations only (--isolate)
if args.isolate:
    from scipy.stats import wilcoxon
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    GROUPS = [('feature views', 'BRIDGE', ['BRIDGE w/o baseline reference', 'BRIDGE relative view only']),
              ('graph mixing', 'BRIDGE', ['BRIDGE fixed equal graph mix', 'BRIDGE without graph']),
              ('attention biases', 'BRIDGE-GT', ['BRIDGE-GT anatomical bias only', 'BRIDGE-GT functional bias only', 'BRIDGE-GT without graph bias'])]
    VARS = list(dict.fromkeys([g[1] for g in GROUPS] + [v for g in GROUPS for v in g[2]]))
    FINGERPRINT = [len(Y), DATASET['files'], DATASET['seizures'], int(Y.sum()), W_BASE, EPOCHS]
    CK = os.path.join(args.out, 'checkpoint_v1_3_isolate.pkl'); ck = load_ckpt(CK, variants=VARS); res = ck['res'] if ck else []
    log(f'isolating ablations: {len(VARS)} variants x {len(subjects)} held-out subjects (no inner calibration; threshold-free metrics)')
    for v in VARS:
        t1 = time.time()
        for s in subjects:
            if any(r['variant'] == v and r['subject'] == s for r in res): continue
            tr, te = MAIN[SUBJ[MAIN] != s], MAIN[(SUBJ[MAIN] == s) & ~AUG[MAIN]]; p = pred(fit(v, tr), te); y = Y[te]
            res.append(dict(variant=v, subject=s, auroc=roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan,
                            auprc=average_precision_score(y, p) if y.sum() else np.nan))
            save_ckpt(CK, res=res, variants=VARS)
        log(f"  {v}: AUROC {boot([r['auroc'] for r in res if r['variant'] == v])[0]:.3f}, "
            f"AUPRC {boot([r['auprc'] for r in res if r['variant'] == v])[0]:.3f} ({(time.time() - t1) / 60:.1f} min)")
    write_csv(res, 'isolate_per_subject.csv')
    get = lambda v, k: np.array([next(r[k] for r in res if r['variant'] == v and r['subject'] == s) for s in subjects], float)
    SUMI = {}
    for name, ref, vs in GROUPS:
        for v in [ref] + vs:
            row = {k: boot(get(v, k)) for k in ('auroc', 'auprc')}
            if v != ref:
                for k in ('auroc', 'auprc'):
                    d = get(ref, k) - get(v, k); ok = np.isfinite(d)
                    row[f'delta_{k}_vs_{ref}'] = float(np.mean(d[ok])); row[f'wins_{k}'] = int((d[ok] > 0).sum())
                    row[f'p_{k}'] = float(wilcoxon(d[ok]).pvalue) if np.any(d[ok] != 0) else 1.0
            SUMI[v] = row
    json.dump(dict(groups=[(g[0], g[1], g[2]) for g in GROUPS], summary=SUMI), open(os.path.join(args.out, 'isolate_summary.json'), 'w'), indent=1)
    for v, r in SUMI.items():
        log(f"  {v:34s} AUROC {r['auroc'][0]:.3f} AUPRC {r['auprc'][0]:.3f}" + (f" | reference minus variant: AUROC {r[[k for k in r if k.startswith('delta_auroc')][0]]:+.3f} "
            f"(reference better on {r['wins_auroc']}/{len(subjects)}, p={r['p_auroc']:.3g}), AUPRC {r[[k for k in r if k.startswith('delta_auprc')][0]]:+.3f} (p={r['p_auprc']:.3g})" if 'p_auroc' in r else ''))
    plt.rcParams.update({'font.family': 'serif', 'font.size': 7, 'axes.spines.top': False, 'axes.spines.right': False})
    FIG = os.path.join(args.out, 'figures'); os.makedirs(FIG, exist_ok=True)
    short = {'BRIDGE': 'both views,\nlearned mix', 'BRIDGE w/o baseline reference': 'absolute\nview only', 'BRIDGE relative view only': 'relative\nview only',
             'BRIDGE fixed equal graph mix': 'fixed equal\nmix', 'BRIDGE without graph': 'no graph', 'BRIDGE-GT': 'both\nbiases',
             'BRIDGE-GT anatomical bias only': 'anatomical\nbias only', 'BRIDGE-GT functional bias only': 'functional\nbias only', 'BRIDGE-GT without graph bias': 'no graph\nbias'}
    for key, lab in (('auroc', 'Mean AUROC'), ('auprc', 'Mean AUPRC')):
        fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.3), gridspec_kw=dict(width_ratios=[3, 3, 4]))
        for ax, (name, ref, vs) in zip(axs, GROUPS):
            for i, v in enumerate([ref] + vs):
                mu, lo, hi = SUMI[v][key]; ax.bar(i, mu, color='#1f5a96' if i == 0 else '#9db7d5', width=0.62)
                ax.errorbar(i, mu, yerr=[[mu - lo], [hi - mu]], color='k', capsize=1.5, lw=0.7); ax.text(i, hi + 0.005, f'{mu:.3f}', ha='center', fontsize=5.5)
            ax.set_xticks(range(len(vs) + 1)); ax.set_xticklabels([short[v] for v in [ref] + vs], fontsize=5.8); ax.set_title(f'({"abc"[GROUPS.index((name, ref, vs))]}) {name}', loc='left')
            ax.set_ylim(0.75 if key == 'auroc' else 0.3, 1.0 if key == 'auroc' else 0.8)
        axs[0].set_ylabel(lab + ' (held-out subjects)'); fig.tight_layout()
        for ext in ('png', 'pdf'): fig.savefig(os.path.join(FIG, f'fig12_isolating_ablations_{key}.{ext}'), dpi=600, bbox_inches='tight')
        plt.close(fig); log('  figure', f'fig12_isolating_ablations_{key}')
    zpath = os.path.join(args.out, f"bridge_results_isolate{'_smoke' if SMOKE else ''}.zip")
    with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as z:
        for fn in ('isolate_per_subject.csv', 'isolate_summary.json', 'log.txt', 'dataset.json'): z.write(os.path.join(args.out, fn), fn)
        for fn in sorted(os.listdir(FIG)):
            if fn.startswith('fig12'): z.write(os.path.join(FIG, fn), 'figures/' + fn)
    if os.path.isdir(args.drive): shutil.copy(zpath, args.drive)
    log(f'done in {(time.time() - T_START) / 60:.1f} min; results: {zpath}')
    log(f"to download in Colab: from google.colab import files; files.download('{zpath}')"); sys.exit(0)


# ------------------------------------------------------------------ leave-one-subject-out comparison
FINGERPRINT = [len(Y), DATASET['files'], DATASET['seizures'], int(Y.sum()), K_CTX, W_BASE, EPOCHS]
rows, PRED, BR_FOLDS = [], {m: dict(idx=[], p=[], t=[], t_fa=[], **{f'tau_{k}': [] for k in KAPPAS}) for m in MODELS}, {}
MIX, BETA, DONE = {}, {}, []
CKPT = os.path.join(args.out, 'checkpoint_v1_2.pkl'); ck = load_ckpt(CKPT, models=MODELS)
if ck: rows, PRED, BR_FOLDS, MIX, BETA, DONE = ck['rows'], ck['PRED'], ck['BR'], ck['MIX'], ck['BETA'], ck['done']; log(f'resumed: {len(DONE)} subjects done')
t_loop, n_new = time.time(), 0
for si, s in enumerate(subjects):
    if s in DONE: continue
    tr, te = MAIN[SUBJ[MAIN] != s], MAIN[(SUBJ[MAIN] == s) & ~AUG[MAIN]]
    for name in MODELS:
        t1 = time.time(); t, tau, t_fa = calibrate(name, tr); model = fit(name, tr); p = pred(model, te)
        P = PRED[name]; P['idx'].append(te); P['p'].append(p); P['t'].append(np.full(len(te), t)); P['t_fa'].append(np.full(len(te), t_fa))
        for k in KAPPAS: P[f'tau_{k}'].append(np.full(len(te), tau[k]))
        r = subject_metrics(te, p, t, tau, t_fa); r.update(model=name, subject=s, seconds=round(time.time() - t1, 1)); rows.append(r)
        if name == 'BRIDGE':
            m, kind, st, k = model[1]; MIX[s] = torch.softmax(m.enc.mix.detach(), 0).cpu().numpy().tolist()
            BR_FOLDS[s] = ({kk: v.detach().cpu() for kk, v in m.state_dict().items()}, (st[0].cpu(), st[1].cpu()), te)
        if name == 'BRIDGE-GT':
            BETA[s] = torch.stack([layer.beta.detach() for layer in model[1][0].enc.L]).cpu().numpy().tolist()
    DONE.append(s); n_new += 1
    save_ckpt(CKPT, rows=rows, PRED=PRED, BR=BR_FOLDS, MIX=MIX, BETA=BETA, done=DONE, models=MODELS)
    eta = (time.time() - t_loop) / n_new * (len(subjects) - len(DONE)) / 60
    log(f'subject {s} ({len(DONE)}/{len(subjects)}): ' + ', '.join(f"{r['model']} {r['auroc']:.3f}" for r in rows[-len(MODELS):])
        + f' | AUROC; ETA main loop {eta:.0f} min')
    write_csv(rows, 'per_subject.csv')

SUMMARY = {}
for name in MODELS:
    R = [r for r in rows if r['model'] == name]; P = PRED[name]; idx, p = np.concatenate(P['idx']), np.concatenate(P['p'])
    np.savez_compressed(os.path.join(args.out, f'pred_{safe(name)}.npz'), idx=idx, p=p, t=np.concatenate(P['t']), t_fa=np.concatenate(P['t_fa']),
                        y=Y[idx], subject=SUBJ[idx], fid=FID[idx], t0=T0[idx], **{f'tau_{k}': np.concatenate(P[f'tau_{k}']) for k in KAPPAS})
    keys = [k for k in R[0] if k not in ('model', 'subject', 'seconds', 'n_windows', 'hours', 'threshold', 'threshold_fa')]
    SUMMARY[name] = {k: boot([r[k] for r in R]) for k in keys}
    SUMMARY[name]['pooled_auroc'] = float(roc_auc_score(Y[idx], p)); SUMMARY[name]['pooled_auprc'] = float(average_precision_score(Y[idx], p))
    SUMMARY[name]['seconds_per_fold'] = float(np.mean([r['seconds'] for r in R]))
json.dump(SUMMARY, open(os.path.join(args.out, 'summary.json'), 'w'), indent=1)
log('main comparison, leave-one-subject-out (mean over subjects [95% CI]):')
for name in MODELS:
    S = SUMMARY[name]
    log(f"  {name:12s} AUROC {S['auroc'][0]:.3f} [{S['auroc'][1]:.3f}-{S['auroc'][2]:.3f}] pooled {S['pooled_auroc']:.3f} | "
        f"AUPRC {S['auprc'][0]:.3f} | at <= {FA_BUDGET:g} FA/h (inner): seizures detected {S['sz_sens_fa'][0]:.2f}, FA/h {S['fa_h_fa'][0]:.2f}, "
        f"latency {S['latency_fa'][0]:.1f} s | max-F1: detected {S['sz_sens'][0]:.2f}, FA/h {S['fa_h'][0]:.2f} | "
        f"cov 0.95: FA/h {S['fa_h_0.95'][0]:.2f}, detected or referred {S['sz_detected_or_referred_0.95'][0]:.2f}, referrals/h {S['referrals_h_0.95'][0]:.2f}")

# ------------------------------------------------------------------ extras
EXTRA = {}
if not args.skip_extras:
    log('component ablation of BRIDGE (one component removed or temporal context added; no inner calibration; threshold-free metrics)')
    CK2 = os.path.join(args.out, 'checkpoint_v1_2_ablation.pkl'); ck = load_ckpt(CK2, ablations=ABLATIONS)
    abl = ck['abl'] if ck else []
    for g in ABLATIONS:
        for s in subjects:
            if any(a['variant'] == g and a['subject'] == s for a in abl): continue
            tr, te = MAIN[SUBJ[MAIN] != s], MAIN[(SUBJ[MAIN] == s) & ~AUG[MAIN]]; p = pred(fit(g, tr), te); y = Y[te]
            abl.append(dict(variant=g, subject=s, auroc=roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan,
                            auprc=average_precision_score(y, p) if y.sum() else np.nan))
        save_ckpt(CK2, abl=abl, ablations=ABLATIONS)
        log(f"  {g}: AUROC {boot([a['auroc'] for a in abl if a['variant'] == g])[0]:.3f}")
    abl += [dict(variant='BRIDGE (full)', subject=r['subject'], auroc=r['auroc'], auprc=r['auprc']) for r in rows if r['model'] == 'BRIDGE']
    write_csv(abl, 'ablation_per_subject.csv')
    EXTRA['ablation'] = {g: {k: boot([a[k] for a in abl if a['variant'] == g]) for k in ('auroc', 'auprc')} for g in ['BRIDGE (full)'] + ABLATIONS}

    log('leakage check: BRIDGE on a random window-level 5-fold split (overlapping seizure windows may enter training)')
    na = MAIN[~AUG[MAIN]]; perm = np.random.default_rng(0).permutation(na); lk = []
    for f in np.array_split(perm, 5):
        test = np.sort(f); tr = np.setdiff1d(MAIN, test); p = pred(fit('BRIDGE', tr), test); lk.append(float(roc_auc_score(Y[test], p)))
    EXTRA['leakage'] = dict(random_split_auroc_folds=lk, random_split_auroc=float(np.mean(lk)),
                            subject_wise_pooled_auroc=SUMMARY['BRIDGE']['pooled_auroc'])
    log(f"  random split AUROC {np.mean(lk):.3f} vs subject-wise pooled {SUMMARY['BRIDGE']['pooled_auroc']:.3f}")

    log('int8 dynamic quantization of BRIDGE (CPU; linear and GRU layers)')
    q = []
    for s, (sd, st, te) in BR_FOLDS.items():
        m = BRIDGE(); m.load_state_dict(sd); m.eval(); k = m.k
        qm = torch.ao.quantization.quantize_dynamic(copy.deepcopy(m), {nn.Linear, nn.GRU}, dtype=torch.qint8)
        tt = torch.as_tensor(te, device=dev); ix = torch.cat([tt[:, None], PREVt[tt]], 1)[:, :k]
        x = ((Ft[ix] - st[0].to(dev)) / st[1].to(dev)).cpu(); c = Ct[ix].cpu(); y = Y[te]
        with torch.no_grad():
            pf = torch.cat([torch.softmax(m(x[b:b + 8192], c[b:b + 8192]), -1)[:, 1] for b in range(0, len(te), 8192)]).numpy()
            pq = torch.cat([torch.softmax(qm(x[b:b + 8192], c[b:b + 8192]), -1)[:, 1] for b in range(0, len(te), 8192)]).numpy()
        if 0 < y.sum() < len(y): q.append(dict(subject=s, auroc_fp32=roc_auc_score(y, pf), auroc_int8=roc_auc_score(y, pq),
                                               max_abs_prob_diff=float(np.abs(pf - pq).max())))
    write_csv(q, 'quantization_per_subject.csv')
    EXTRA['quantization'] = {k: boot([r[k] for r in q]) for k in ('auroc_fp32', 'auroc_int8', 'max_abs_prob_diff')}
    log(f"  AUROC fp32 {EXTRA['quantization']['auroc_fp32'][0]:.3f} -> int8 {EXTRA['quantization']['auroc_int8'][0]:.3f}")

# ------------------------------------------------------------------ fixed split of BIOT / CBraMod / CSBrain
# train chb01-chb19, validation chb20-chb21 (threshold for balanced accuracy), test every recording of chb22-chb23
LIT = [dict(model='SPaRCNet', source='Jing et al. 2023, reported in BIOT (NeurIPS 2023)', bacc=0.5876, auprc=0.1247, auroc=0.8143),
       dict(model='ContraWR', source='Yang et al., reported in BIOT', bacc=0.6344, auprc=0.2264, auroc=0.8097),
       dict(model='CNN-Transformer', source='Peh et al. 2022, reported in BIOT', bacc=0.6389, auprc=0.2479, auroc=0.8662),
       dict(model='FFCL', source='Li et al. 2022, reported in BIOT', bacc=0.6262, auprc=0.2049, auroc=0.8271),
       dict(model='ST-Transformer', source='Song et al. 2021, reported in BIOT', bacc=0.5915, auprc=0.1422, auroc=0.8237),
       dict(model='BIOT', source='Yang, Westover, Sun, NeurIPS 2023', bacc=0.7068, auprc=0.3277, auroc=0.8761),
       dict(model='CBraMod', source='Wang et al., ICLR 2025', bacc=np.nan, auprc=0.3689, auroc=0.8892),
       dict(model='CSBrain', source='Zhou et al., NeurIPS 2025', bacc=np.nan, auprc=0.5164, auroc=0.8915)]
FIXED = []
if not args.skip_fixed:
    log('fixed split (train chb01-19, validation chb20-21, test all recordings of chb22-23)')
    tr = ALL[np.isin(CASE, FIX_TRAIN) & ~EXTRA_F]; va = ALL[np.isin(CASE, FIX_VAL) & ~EXTRA_F & ~AUG]; te = ALL[np.isin(CASE, FIX_TEST) & ~AUG]
    log(f'  test: {len(te)} windows, {te.size * WIN_S / 3600:.1f} h, ictal fraction {Y[te].mean():.4f}')
    CK3 = os.path.join(args.out, 'checkpoint_v1_2_fixed.pkl'); ck = load_ckpt(CK3, fixed_models=FIXED_MODELS); FIXED = ck['fixed'] if ck else []
    for name in FIXED_MODELS:
        for seed in FIX_SEEDS:
            if any(r['model'] == name and r['seed'] == seed for r in FIXED): continue
            model = fit(name, tr, seed); pv, pt = pred(model, va), pred(model, te)
            ths = np.unique(np.quantile(pv, np.linspace(0.5, 0.9995, 400)))
            thr = ths[int(np.argmax([balanced_accuracy_score(Y[va], pv >= q) for q in ths]))]
            FIXED.append(dict(model=name, seed=seed, bacc=balanced_accuracy_score(Y[te], pt >= thr), auprc=average_precision_score(Y[te], pt),
                              auroc=roc_auc_score(Y[te], pt), threshold=float(thr)))
            if seed == FIX_SEEDS[0]: np.savez_compressed(os.path.join(args.out, f'pred_fixed_{safe(name)}.npz'), idx=te, p=pt, y=Y[te], fid=FID[te], t0=T0[te])
        save_ckpt(CK3, fixed=FIXED, fixed_models=FIXED_MODELS)
        R = [r for r in FIXED if r['model'] == name]
        log(f"  {name}: balanced acc {np.mean([r['bacc'] for r in R]):.4f}, AUPRC {np.mean([r['auprc'] for r in R]):.4f}, "
            f"AUROC {np.mean([r['auroc'] for r in R]):.4f} (seeds {len(R)})")
    write_csv(FIXED, 'fixed_split_per_seed.csv')
    comp = [dict(model=l['model'], source=l['source'], params='', bacc=l['bacc'], auprc=l['auprc'], auroc=l['auroc'], ours=0) for l in LIT]
    for name in FIXED_MODELS:
        R = [r for r in FIXED if r['model'] == name]
        comp.append(dict(model=name, source='this study', ours=1, **{k: float(np.mean([r[k] for r in R])) for k in ('bacc', 'auprc', 'auroc')},
                         **{f'{k}_sd': float(np.std([r[k] for r in R])) for k in ('bacc', 'auprc', 'auroc')}))
    write_csv(comp, 'comparison_fixed_split.csv'); EXTRA['fixed_split'] = comp

# ------------------------------------------------------------------ efficiency
def count_macs(m, kind, k):
    tot = [0]
    def hook(mod, inp, out):
        if isinstance(mod, nn.Linear): tot[0] += mod.in_features * mod.out_features * (out.numel() // out.shape[-1])
        elif isinstance(mod, nn.Conv1d): tot[0] += mod.in_channels // mod.groups * mod.kernel_size[0] * out.numel()
        elif isinstance(mod, nn.GRU): tot[0] += 3 * (mod.input_size + mod.hidden_size) * mod.hidden_size * inp[0].shape[0] * inp[0].shape[1]
    hs = [mod.register_forward_hook(hook) for mod in m.modules() if isinstance(mod, (nn.Linear, nn.Conv1d, nn.GRU))]
    m.train()
    with torch.no_grad(): m(*dummy(kind, k))
    m.eval(); [h.remove() for h in hs]; extra = 0
    enc = getattr(m, 'enc', None)
    if isinstance(enc, DualGraphEncoder): extra = k * len(enc.L) * N_CH * N_CH * enc.inp.out_features
    if isinstance(enc, DualGraphAttentionEncoder): extra = k * len(enc.L) * 2 * N_CH * N_CH * enc.inp.out_features   # QK^T and AV
    if isinstance(m, ChanTransformer):
        for mod in m.modules():
            if isinstance(mod, nn.MultiheadAttention): extra += N_CH * 4 * mod.embed_dim ** 2 + 2 * N_CH * N_CH * mod.embed_dim
    return tot[0] + extra


def dummy(kind, k=1):
    if kind == 'raw': return torch.randn(1, N_CH, WIN // 2), None
    return torch.randn(1, k, N_CH, N_F if kind == 'abs' else N_FR), torch.randn(1, k, len(IU[0]))


def cpu_latency_ms(fn, n=300):
    for _ in range(20): fn()
    ts = []
    for _ in range(n): t1 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t1) * 1e3)
    return float(np.median(ts))


nt = torch.get_num_threads(); torch.set_num_threads(1); EFF = []
for name in ('BRIDGE', 'BRIDGE-GT', 'BRIDGE + 16-s temporal context', 'Transformer', 'MLP', 'RawCNN'):
    make_m, kind, k = MAKERS[name]; m = make_m().cpu().eval(); n_par = sum(p.numel() for p in m.parameters()); macs = count_macs(m, kind, k)
    x, c = dummy(kind, k)
    with torch.no_grad(): lat = cpu_latency_ms(lambda: m(x, c))
    buf = io.BytesIO(); torch.save(m.state_dict(), buf); row = dict(model=name, params=n_par, macs=macs, size_fp32_kB=buf.tell() / 1024, cpu_latency_ms=lat)
    if name not in ('RawCNN', 'Transformer'):   # the fused nn.TransformerEncoder path does not take dynamically quantized layers
        qm = torch.ao.quantization.quantize_dynamic(copy.deepcopy(m), {nn.Linear, nn.GRU}, dtype=torch.qint8)
        buf = io.BytesIO(); torch.save(qm.state_dict(), buf); row['size_int8_kB'] = buf.tell() / 1024
        with torch.no_grad(): row['cpu_latency_int8_ms'] = cpu_latency_ms(lambda: qm(x, c))
    EFF.append(row)
seg1 = np.random.randn(1, N_CH, WIN).astype(np.float32) * 30; zi = np.zeros((SOS.shape[0], N_CH, 2))
ring = np.random.randn(W_BASE, N_CH * N_LOG).astype(np.float16)
EFF.append(dict(model='feature extraction (one 4-s window, 18 channels)', cpu_latency_ms=cpu_latency_ms(lambda: window_features(seg1), 100),
                filter_ms=cpu_latency_ms(lambda: sosfilt(SOS, seg1[0], axis=-1, zi=zi), 100)))
EFF.append(dict(model=f'patient-baseline reference (median of a {W_BASE}-window ring buffer)', memory_kB=ring.nbytes / 1024,
                cpu_latency_ms=cpu_latency_ms(lambda: np.median(ring, axis=0), 100)))
torch.set_num_threads(nt)
for name in ('LogReg', 'RandomForest'):
    model = fit(name, MAIN[SUBJ[MAIN] != subjects[0]])[1][0]
    EFF.append(dict(model=name, params=int(model.coef_.size + model.intercept_.size) if name == 'LogReg'
                    else int(sum(e.tree_.node_count for e in model.estimators_))))
write_csv(EFF, 'efficiency.csv'); EXTRA['efficiency'] = EFF
for r in EFF: log('  efficiency', r)
json.dump(EXTRA, open(os.path.join(args.out, 'extras.json'), 'w'), indent=1, default=float)

# ------------------------------------------------------------------ figures (600 dpi PNG and PDF; data in fig_data.npz)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import precision_recall_curve, roc_curve
plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8, 'legend.fontsize': 6.5,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.spines.top': False, 'axes.spines.right': False})
FIG = os.path.join(args.out, 'figures'); os.makedirs(FIG, exist_ok=True)
COL = {'BRIDGE': '#1f5a96', 'BRIDGE-GT': '#d0661f', 'Transformer': '#8c8c8c', 'MLP': '#b5b5b5', 'LogReg': '#2e7d32',
       'RandomForest': '#8bc34a', 'RawCNN': '#6d4c41'}
LBL = {'BRIDGE': 'BRIDGE (proposed)', 'BRIDGE-GT': 'BRIDGE-GT (proposed)', 'Transformer': 'Transformer (no graph)', 'MLP': 'MLP',
       'LogReg': 'Logistic regression', 'RandomForest': 'Random forest', 'RawCNN': 'Raw 1-D CNN'}
OURS = ('BRIDGE', 'BRIDGE-GT')
FD = {}
PALL = {m: np.full(len(Y), np.nan) for m in MODELS}; TALL = {m: np.full(len(Y), np.nan) for m in MODELS}; TFA = {m: np.full(len(Y), np.nan) for m in MODELS}
TAU = {m: {k: np.full(len(Y), np.nan) for k in KAPPAS} for m in MODELS}
for m in MODELS:
    idx = np.concatenate(PRED[m]['idx']); PALL[m][idx] = np.concatenate(PRED[m]['p']); TALL[m][idx] = np.concatenate(PRED[m]['t'])
    TFA[m][idx] = np.concatenate(PRED[m]['t_fa'])
    for k in KAPPAS: TAU[m][k][idx] = np.concatenate(PRED[m][f'tau_{k}'])
mids = np.array([np.mean([POS[e] for e in ch.split('-')], 0) for ch in CH18])


def savefig(fig, name):
    fig.savefig(os.path.join(FIG, name + '.png'), dpi=600, bbox_inches='tight'); fig.savefig(os.path.join(FIG, name + '.pdf'), bbox_inches='tight')
    plt.close(fig); log('  figure', name)


def make(name, fn):
    try: fn()
    except Exception as e: log('  figure failed', name, repr(e)); plt.close('all')


def head_graph(ax, A, color, thr=0.0, lw=4.0):
    th = np.linspace(0, 2 * np.pi, 120); ax.plot(1.1 * np.cos(th), 1.1 * np.sin(th), color='0.55', lw=0.5)
    for i, j in zip(*np.triu_indices(N_CH, 1)):
        if A[i, j] > thr: ax.plot(mids[[i, j], 0], mids[[i, j], 1], color=color, lw=0.3 + lw * A[i, j], alpha=0.7)
    ax.scatter(mids[:, 0], mids[:, 1], s=4, color='k', zorder=3); ax.set_aspect('equal'); ax.axis('off')


def fig_architecture():
    fig = plt.figure(figsize=(7.0, 2.8)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 42); ax.axis('off')
    def box(x, y, w, h, txt, fc, fs=6.2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.25,rounding_size=0.8', fc=fc, ec='0.3', lw=0.6))
        ax.text(x + w / 2, y + h / 2, txt, ha='center', va='center', fontsize=fs)
    def arrow(x0, y0, x1, y1): ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='-|>', lw=0.7, color='0.25', mutation_scale=7))
    rng = np.random.default_rng(3); tt = np.linspace(0, 1, 300)
    for j in range(5): ax.plot(1 + 11 * tt, 33 - 2.3 * j + 0.8 * np.sin(40 * tt + j) * rng.uniform(0.3, 1, 300) ** 0.2, lw=0.35, color='0.2')
    ax.text(6.5, 40, 'scalp EEG, 18 bipolar\nchannels, 4-s windows', ha='center', va='center', fontsize=6.2)
    box(15, 27, 13, 9, '11 node features\nper channel\n(7 band powers, line\nlength, variance, ...)', '#eef2f7'); arrow(12.5, 31.5, 15, 31.5)
    box(30.5, 27, 19, 9, 'Patient-baseline reference\n(causal, label-free): running\nmedian of the last 10 min\nnode = [x, x - baseline(x)]', '#fde9d9'); arrow(28, 31.5, 30.5, 31.5)
    a1 = fig.add_axes([0.515, 0.60, 0.10, 0.28]); head_graph(a1, A_STATIC / A_STATIC.max(), '#1f5a96', 0.05, 1.5); a1.set_title('anatomical $A_s$', fontsize=5.5, pad=1)
    Afun = np.abs(np.corrcoef(rng.standard_normal((N_CH, 40)) + np.repeat(rng.standard_normal((3, 40)), 6, 0)))
    np.fill_diagonal(Afun, 0); a2 = fig.add_axes([0.625, 0.60, 0.10, 0.28]); head_graph(a2, Afun * (Afun > np.sort(Afun, 1)[:, [-4]]), '#c0504d', 0.0, 1.5)
    a2.set_title('functional $A_f(t)$', fontsize=5.5, pad=1)
    ax.text(61.5, 25.5, r'$A = w_s A_s + w_f A_f(t)$,  $(w_s, w_f)$ = softmax(learned)', ha='center', fontsize=6.2)
    arrow(49.5, 31.5, 51.2, 31.5)
    box(75, 27, 23, 9, 'Dual-graph encoder\nBRIDGE: 2 message-passing layers\nBRIDGE-GT: attention biased by\n' + r'$\beta_s \log A_s + \beta_f |C(t)|$ per head', '#e3edf9'); arrow(73.5, 31.5, 75, 31.5)
    box(36, 9.5, 22, 7, 'mean + max pooling over channels\n-> classifier (2 layers)', '#e3edf9')
    ax.plot([86.5, 86.5], [27, 13], lw=0.7, color='0.25'); arrow(86.5, 13, 58, 13)
    box(62, 1.2, 24, 6, 'ablation variant: GRU over the embeddings\nof the last 4 windows (16 s)', '#f6f6f6', 5.6)
    ax.annotate('', xy=(50, 9.5), xytext=(62, 4.2), arrowprops=dict(arrowstyle='-|>', lw=0.6, color='0.5', ls='--', mutation_scale=6))
    box(15, 9.5, 17, 7, 'threshold and abstention\nfrom training subjects only\n(inner subject-wise folds)', '#e8f3e6'); arrow(36, 13, 32, 13)
    box(1, 9.5, 11, 7, 'alarm (2 of 3)\nor refer to\na clinician', '#f6f6f6'); arrow(15, 13, 12, 13)
    ax.text(98, 40, f'about {sum(p.numel() for p in BRIDGE().parameters()) / 1e3:.0f}k parameters; int8 on CPU; baseline buffer '
            f'{W_BASE} windows x {N_CH * N_LOG} values', ha='right', fontsize=5.8, color='0.25')
    savefig(fig, 'fig1_architecture')


def fig_auroc():
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.4))
    for j, key in enumerate(('auroc', 'auprc')):
        ax = axs[j]
        for i, m in enumerate(MODELS):
            v = np.array([r[key] for r in rows if r['model'] == m and np.isfinite(r[key])]); FD[f'{key}_{m}'] = v
            ax.scatter(i + np.random.default_rng(i).uniform(-0.18, 0.18, len(v)), v, s=7, color=COL[m], alpha=0.75, lw=0)
            mu, lo, hi = SUMMARY[m][key]; ax.errorbar(i, mu, yerr=[[mu - lo], [hi - mu]], fmt='_', color='k', ms=12, capsize=2, lw=0.8)
        ax.set_xticks(range(len(MODELS))); ax.set_xticklabels([LBL[m] for m in MODELS], rotation=35, ha='right')
        ax.set_ylabel(('AUROC' if key == 'auroc' else 'AUPRC') + ' per held-out subject'); ax.set_title(f'({"ab"[j]})', loc='left')
    savefig(fig, 'fig2_per_subject_auroc_auprc')


def fig_roc_pr():
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for m in MODELS:
        ok = ~np.isnan(PALL[m]); y, p = Y[ok], PALL[m][ok]
        fpr, tpr, _ = roc_curve(y, p); pr, rc, _ = precision_recall_curve(y, p); lw = 1.4 if m in OURS else 0.8
        axs[0].plot(fpr, tpr, color=COL[m], lw=lw, label=f"{LBL[m]} ({SUMMARY[m]['pooled_auroc']:.3f})")
        axs[1].plot(rc, pr, color=COL[m], lw=lw, label=f"{LBL[m]} ({SUMMARY[m]['pooled_auprc']:.3f})")
    axs[0].plot([0, 1], [0, 1], ':', color='0.6', lw=0.6); axs[0].set_xlabel('False positive rate'); axs[0].set_ylabel('True positive rate')
    axs[1].set_xlabel('Recall (ictal windows)'); axs[1].set_ylabel('Precision'); axs[0].set_title('(a) pooled ROC', loc='left')
    axs[1].set_title('(b) pooled precision-recall', loc='left'); axs[0].legend(frameon=False, loc='lower right'); axs[1].legend(frameon=False)
    savefig(fig, 'fig3_roc_pr')


def fig_abstention():
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.5))
    for m in MODELS:
        S = SUMMARY[m]; ref = [0] + [S[f'referrals_h_{k}'][0] for k in KAPPAS]; fa = [S['fa_h'][0]] + [S[f'fa_h_{k}'][0] for k in KAPPAS]
        cov = [S['sz_sens'][0]] + [S[f'sz_detected_or_referred_{k}'][0] for k in KAPPAS]; FD[f'abst_{m}'] = np.array([ref, fa, cov])
        lw = 1.4 if m in OURS else 0.8
        axs[0].plot(ref, fa, '-o', color=COL[m], ms=3, lw=lw, label=LBL[m]); axs[1].plot(ref, cov, '-o', color=COL[m], ms=3, lw=lw)
    axs[0].set_xlabel('Referrals to a clinician per hour'); axs[0].set_ylabel('False alarms per hour (accepted)')
    axs[1].set_xlabel('Referrals to a clinician per hour'); axs[1].set_ylabel('Seizures detected or referred')
    axs[0].set_title('(a) points: coverage 1.0, ' + ', '.join(str(k) for k in KAPPAS), loc='left'); axs[1].set_title('(b)', loc='left')
    h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.16))
    savefig(fig, 'fig4_abstention_tradeoff')


def fig_ablation_leakage():
    if 'ablation' not in EXTRA: return
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.5), gridspec_kw=dict(width_ratios=[1.7, 1]))
    gs = ['BRIDGE (full)'] + ABLATIONS; FD['ablation'] = np.array([EXTRA['ablation'][g]['auroc'] for g in gs])
    for i, g in enumerate(gs):
        mu, lo, hi = EXTRA['ablation'][g]['auroc']; axs[0].bar(i, mu, color=COL['BRIDGE'] if i == 0 else '#9db7d5', width=0.6)
        axs[0].errorbar(i, mu, yerr=[[mu - lo], [hi - mu]], color='k', capsize=2, lw=0.8); axs[0].text(i, hi + 0.004, f'{mu:.3f}', ha='center', fontsize=5.5)
    axs[0].set_xticks(range(len(gs))); axs[0].set_xticklabels([g.replace('BRIDGE ', '').replace(' (', '\n(') for g in gs], rotation=30, ha='right', fontsize=6)
    lo_ = min(EXTRA['ablation'][g]['auroc'][1] for g in gs); axs[0].set_ylim(max(0, lo_ - 0.04), 1.0)
    axs[0].set_ylabel('Mean AUROC (held-out subjects)'); axs[0].set_title('(a) BRIDGE: one component removed or context added', loc='left')
    L = EXTRA['leakage']; v = [L['random_split_auroc'], L['subject_wise_pooled_auroc']]; FD['leakage'] = np.array(v)
    axs[1].bar([0, 1], v, color=['#c0504d', COL['BRIDGE']], width=0.55)
    for i, x in enumerate(v): axs[1].text(i, x + 0.005, f'{x:.3f}', ha='center', va='bottom', fontsize=7)
    axs[1].set_xticks([0, 1]); axs[1].set_xticklabels(['random window split\n(leaky)', 'leave-one-subject-out']); axs[1].set_ylabel('Pooled AUROC')
    axs[1].set_ylim(min(v) - 0.1, 1.0); axs[1].set_title('(b) evaluation leakage', loc='left'); savefig(fig, 'fig5_ablation_leakage')


def fig_efficiency():
    fig, ax = plt.subplots(figsize=(3.5, 2.5)); E = {r['model']: r for r in EFF if 'params' in r}
    ms = sorted([m for m in MODELS if m in E], key=lambda m: SUMMARY[m]['auroc'][0])
    for k, m in enumerate(ms):
        x, y = E[m]['params'], SUMMARY[m]['auroc'][0]; sz = 10 + 60 * np.log10(1 + E[m].get('macs', 1e4)) / 7
        ax.scatter(x, y, s=sz, color=COL[m], zorder=3)
        ax.annotate(LBL[m], (x, y), xytext=(5, 6 if k % 2 else -9), textcoords='offset points', fontsize=6)
    ax.set_xscale('log'); ax.set_xlabel('Parameters (random forest: tree nodes)'); ax.set_ylabel('Mean AUROC (leave-one-subject-out)')
    ax.set_title('Accuracy against model size', loc='left'); savefig(fig, 'fig6_efficiency')


def fig_timeline():
    au = {r['subject']: r['auroc'] for r in rows if r['model'] == 'BRIDGE' and np.isfinite(r['auroc'])}
    order = sorted(au, key=au.get); s = order[len(order) // 2]; z = next(z for z in SZ_TAB if z['subject'] == s)
    sel = np.where((FID == z['fid']) & ~AUG)[0]; sel = sel[np.argsort(T0[sel])]
    sel = sel[(T0[sel] >= z['onset'] - 600) & (T0[sel] <= z['offset'] + 600)]
    chans = [0, 2, 6, 16]; tr = np.concatenate([Rt[i][chans].float().cpu().numpy() for i in sel], -1); tt = T0[sel[0]] + np.arange(tr.shape[1]) / (FS / 2)
    fig, axs = plt.subplots(3, 1, figsize=(7.0, 3.6), sharex=True, gridspec_kw=dict(height_ratios=[1.6, 1, 0.5]))
    for j, c in enumerate(chans):
        y = tr[j] / (np.percentile(np.abs(tr[j]), 99) + 1e-6); axs[0].plot(tt, y - 2.2 * j, lw=0.25, color='0.15')
    axs[0].set_yticks([-2.2 * j for j in range(len(chans))]); axs[0].set_yticklabels([CH18[c] for c in chans])
    tw = T0[sel] + WIN_S / 2; FD['timeline'] = dict(t=tw, subject=s, onset=z['onset'], offset=z['offset'])
    for m in OURS:
        axs[1].plot(tw, PALL[m][sel], color=COL[m], lw=0.9, label=LBL[m]); axs[1].axhline(np.nanmean(TFA[m][sel]), color=COL[m], ls='--', lw=0.6)
        FD[f'timeline_p_{m}'] = PALL[m][sel]
    m = 'BRIDGE'; p, t, tau = PALL[m][sel], np.nanmean(TALL[m][sel]), np.nanmean(TAU[m][0.95][sel]); acc = conf(p, t) >= tau
    alarm = np.convolve(((p >= np.nanmean(TFA[m][sel])) & acc).astype(int), np.ones(3, int))[:len(p)] >= 2
    axs[2].fill_between(tw, 0, alarm.astype(float), step='mid', color='#c0504d', lw=0, label='alarm')
    axs[2].fill_between(tw, 0, (~acc).astype(float) * 0.5, step='mid', color='#f2c14e', lw=0, label='referred (coverage 0.95)')
    for ax in axs: ax.axvspan(z['onset'], z['offset'], color='#c0504d', alpha=0.12, lw=0)
    axs[1].set_ylabel('Seizure probability'); axs[1].legend(frameon=True, framealpha=0.85, edgecolor='none', loc='upper right', fontsize=6)
    axs[2].set_yticks([]); axs[2].legend(frameon=True, framealpha=0.85, edgecolor='none', loc='upper right', ncol=2, fontsize=6)
    axs[2].set_xlabel(f'Time in recording {FILES[z["fid"]]["file"]} (s); shaded: annotated seizure; dashed: false-alarm-budget threshold')
    axs[0].set_title(f'Held-out subject {s} (median AUROC of BRIDGE)', loc='left'); savefig(fig, 'fig7_example_timeline')


def fig_graphs():
    na = MAIN[~AUG[MAIN]]; rng = np.random.default_rng(0); ii = na[Y[na] == 1]; jj = na[Y[na] == 0]
    jj = rng.choice(jj, min(len(jj), 50000), replace=False)
    full = lambda v: (lambda M: M + M.T)(np.bincount(IU[0] * N_CH + IU[1], weights=v, minlength=N_CH * N_CH).reshape(N_CH, N_CH))
    A1 = full(np.abs(Ca[ii].astype(np.float32)).mean(0)); A0 = full(np.abs(Ca[jj].astype(np.float32)).mean(0)); D = A1 - A0
    FD.update(adj_ictal=A1, adj_interictal=A0, adj_static=A_STATIC, mix=np.array(list(MIX.values())), beta=np.array(list(BETA.values())))
    fig = plt.figure(figsize=(7.0, 2.5)); gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1.1, 0.9])
    for k, (M, ttl) in enumerate(((A0, '(a) interictal |corr|'), (A1, '(b) ictal |corr|'))):
        ax = fig.add_subplot(gs[k]); ax.imshow(M, vmin=0, vmax=1, cmap='viridis'); ax.set_title(ttl, loc='left')
        ax.set_xticks(range(N_CH)); ax.set_yticks(range(N_CH)); ax.set_xticklabels(CH18, rotation=90, fontsize=4); ax.set_yticklabels(CH18, fontsize=4)
    ax = fig.add_subplot(gs[2]); th = np.linspace(0, 2 * np.pi, 200); ax.plot(1.08 * np.cos(th), 1.08 * np.sin(th), color='0.6', lw=0.6)
    for q in np.argsort(-np.abs(D[IU]))[:25]:
        i, j = IU[0][q], IU[1][q]; w = D[i, j]
        ax.plot(mids[[i, j], 0], mids[[i, j], 1], color='#c0504d' if w > 0 else '#1f5a96', lw=0.5 + 6 * abs(w), alpha=0.8)
    ax.scatter(mids[:, 0], mids[:, 1], s=8, color='k', zorder=3); ax.set_aspect('equal'); ax.axis('off'); ax.set_title('(c) ictal - interictal', loc='left')
    ax = fig.add_subplot(gs[3])
    if len(MIX): mx = np.array(list(MIX.values())); ax.bar([0, 1], mx.mean(0), yerr=mx.std(0), color=[COL['BRIDGE'], '#9db7d5'], width=0.5, capsize=2)
    if len(BETA): b = np.array(list(BETA.values())).mean(axis=(1, 3)); ax.bar([2.2, 3.2], b.mean(0), yerr=b.std(0), color=[COL['BRIDGE-GT'], '#f0b27a'], width=0.5, capsize=2)
    ax.set_xticks([0, 1, 2.2, 3.2]); ax.set_xticklabels(['BRIDGE:\nanat.', 'BRIDGE:\nfunc.', 'GT:\nanat.', 'GT:\nfunc.'], fontsize=6)
    ax.set_title('(d) learned graph weights', loc='left'); savefig(fig, 'fig8_graphs')


def fig_latency_fa():
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.3))
    for m in ('BRIDGE', 'BRIDGE-GT', 'Transformer', 'RandomForest'):
        v = np.array([r['latency_fa'] for r in rows if r['model'] == m and np.isfinite(r['latency_fa'])]); FD[f'latency_{m}'] = v
        if len(v): axs[0].hist(v, bins=np.arange(0, 64, 4), histtype='step', color=COL[m], lw=1.1, label=LBL[m])
        R = [r for r in rows if r['model'] == m]; axs[1].scatter([r['fa_h_fa'] for r in R], [r['sz_sens_fa'] for r in R], s=9, color=COL[m], alpha=0.8, lw=0, label=LBL[m])
    axs[0].set_xlabel('Median detection latency per subject (s)'); axs[0].set_ylabel('Subjects'); axs[0].legend(frameon=False)
    axs[1].set_xscale('symlog', linthresh=0.1); axs[1].set_xlabel('False alarms per hour'); axs[1].set_ylabel('Seizures detected (per subject)')
    axs[0].set_title(f'(a) threshold: <= {FA_BUDGET:g} false alarm/h on training subjects', loc='left'); axs[1].set_title('(b) each point: one held-out subject', loc='left')
    savefig(fig, 'fig9_latency_falsealarms')


def fig_fixed():
    if not FIXED: return
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9))
    ours = [(LBL.get(n, n), np.mean([r['auroc'] for r in FIXED if r['model'] == n]), np.std([r['auroc'] for r in FIXED if r['model'] == n]),
             np.mean([r['auprc'] for r in FIXED if r['model'] == n]), np.std([r['auprc'] for r in FIXED if r['model'] == n]), n in OURS) for n in FIXED_MODELS]
    lit = [(l['model'] + ' [rep.]', l['auroc'], 0, l['auprc'], 0, None) for l in LIT]; allr = ours + lit; FD['fixed'] = np.array([[a[1], a[3]] for a in allr])
    for j, (key, ttl) in enumerate(((1, '(a) AUROC'), (3, '(b) AUPRC'))):
        order = sorted(range(len(allr)), key=lambda i: allr[i][key]); ax = axs[j]
        for y, i in enumerate(order):
            a = allr[i]; c = '#bdbdbd' if a[5] is None else (COL['BRIDGE'] if a[5] else '#6f8fb5')
            ax.barh(y, a[key], color=c, height=0.7, xerr=a[key + 1] if a[5] is not None else None, error_kw=dict(lw=0.6, capsize=1.5))
            ax.text(a[key] + 0.005, y, f'{a[key]:.3f}', va='center', fontsize=5.5)
        ax.set_yticks(range(len(allr))); ax.set_yticklabels([allr[i][0] for i in order], fontsize=6)
        ax.set_xlim(0.7 if key == 1 else 0, 1.02); ax.set_title(ttl + ', test chb22-23', loc='left')
    fig.text(0.5, -0.02, 'Same split as BIOT, CBraMod and CSBrain (train chb01-19, validation chb20-21, test chb22-23); grey: values reported '
             'in those papers; ours: mean and s.d. over seeds, all recordings of the test cases', ha='center', fontsize=6)
    fig.tight_layout(); savefig(fig, 'fig10_fixed_split_vs_published')


def fig_baseline_reference():
    na = MAIN[~AUG[MAIN]]; f_abs = FR[na][:, :, 7].mean(1); f_ref = FR[na][:, :, N_F + 7].mean(1); s_ = SUBJ[na]; y = Y[na]
    fig, axs = plt.subplots(2, 1, figsize=(7.0, 3.2), sharex=True); FD['bref'] = np.array([[np.median(f_abs[(s_ == s) & (y == c)]) if ((s_ == s) & (y == c)).any() else np.nan
                                                                                            for s in subjects for c in (0, 1)]])
    for ax, v, ttl in ((axs[0], f_abs, '(a) absolute log line length (channel mean)'), (axs[1], f_ref, '(b) referenced to the patient baseline')):
        for i, s in enumerate(subjects):
            for c, col, dx in ((0, '#9e9e9e', -0.18), (1, '#c0504d', 0.18)):
                w = v[(s_ == s) & (y == c)]
                if len(w) > 2000: w = np.random.default_rng(i).choice(w, 2000, replace=False)
                if len(w): ax.boxplot(w, positions=[i + dx], widths=0.3, showfliers=False, patch_artist=True,
                                      boxprops=dict(facecolor=col, lw=0.4), medianprops=dict(color='k', lw=0.6), whiskerprops=dict(lw=0.4), capprops=dict(lw=0.4))
        ax.set_title(ttl, loc='left'); ax.set_ylabel('log10 value')
    axs[1].set_xticks(range(len(subjects))); axs[1].set_xticklabels([str(s) for s in subjects]); axs[1].set_xlabel('Subject (grey: interictal, red: ictal windows)')
    savefig(fig, 'fig11_baseline_reference')


log('figures')
for name, fn in (('architecture', fig_architecture), ('per-subject AUROC/AUPRC', fig_auroc), ('ROC/PR', fig_roc_pr), ('abstention', fig_abstention),
                 ('ablation/leakage', fig_ablation_leakage), ('efficiency', fig_efficiency), ('timeline', fig_timeline),
                 ('graphs', fig_graphs), ('latency/false alarms', fig_latency_fa), ('fixed split', fig_fixed), ('baseline reference', fig_baseline_reference)):
    make(name, fn)
try: np.savez_compressed(os.path.join(FIG, 'fig_data.npz'), **{k: np.asarray(v, dtype=object) if isinstance(v, dict) else np.asarray(v) for k, v in FD.items()}, allow_pickle=True)
except Exception as e: log('fig data not saved', repr(e))
json.dump(dict(MIX={str(k): v for k, v in MIX.items()}, BETA={str(k): v for k, v in BETA.items()}), open(os.path.join(args.out, 'learned_graph_weights.json'), 'w'))

# ------------------------------------------------------------------ results tables (CSV): this study and published results
T1 = []
for name in MODELS:
    S = SUMMARY[name]; T1.append(dict(model=LBL[name], auroc=S['auroc'][0], auroc_ci=f"{S['auroc'][1]:.3f}-{S['auroc'][2]:.3f}", auprc=S['auprc'][0],
                                       pooled_auroc=S['pooled_auroc'], sz_detected_at_fa_budget=S['sz_sens_fa'][0], fa_h_at_fa_budget=S['fa_h_fa'][0],
                                       latency_s=S['latency_fa'][0], detected_or_referred_cov095=S['sz_detected_or_referred_0.95'][0],
                                       referrals_h_cov095=S['referrals_h_0.95'][0]))
write_csv(T1, 'table_loso_this_study.csv')

# ------------------------------------------------------------------ package
zpath = os.path.join(args.out, f"bridge_results{'_smoke' if SMOKE else ''}.zip")
with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as z:
    for fn in sorted(os.listdir(args.out)):
        if fn.endswith(('.csv', '.json', '.txt')) or fn.startswith('pred_'): z.write(os.path.join(args.out, fn), fn)
    for fn in sorted(os.listdir(FIG)): z.write(os.path.join(FIG, fn), 'figures/' + fn)
if os.path.isdir(args.drive): shutil.copy(zpath, args.drive); log('results zip copied to', args.drive)
log(f'done in {(time.time() - T_START) / 60:.1f} min; results: {zpath}')
log(f"to download in Colab: from google.colab import files; files.download('{zpath}')")
