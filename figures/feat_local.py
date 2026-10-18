"""CHB-MIT feature builder for the figure scripts (same windows and features as bridge_chbmit.py); streams each EDF:
download -> features -> delete. Also adds every file of chb22 and chb23 (flag extra=1) for the fixed-split test.
    python feat_local.py <edf cache folder> <output folder>      -> <output folder>/features_local.npz"""
import sys, os, re, json, time, shutil, urllib.parse, urllib.request, math
import concurrent.futures as cf
import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.stats import kurtosis
EDF, OUT = sys.argv[1], sys.argv[2]; os.makedirs(EDF, exist_ok=True); os.makedirs(OUT, exist_ok=True)
FS, WIN_S = 256, 4; WIN, HOP, AUG_HOP = FS * WIN_S, FS * WIN_S, FS
BANDS = [(0.5, 2), (2, 4), (4, 8), (8, 13), (13, 20), (20, 30), (30, 45)]
CH18 = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
        'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2', 'FZ-CZ', 'CZ-PZ']
N_CH = 18; SUBJECT = lambda c: 1 if c == 21 else c
BASE = 'https://physionet-open.s3.amazonaws.com/chbmit/1.0.0/'
T0 = time.time()
def log(*a): print(f'[{(time.time()-T0)/60:6.1f} min]', *a, flush=True)
def fetch(rel):
    dest = os.path.join(EDF, rel.replace('/', '__'))
    for k in range(6):
        try:
            req = urllib.request.Request(BASE + urllib.parse.quote(rel), headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=300) as r, open(dest + '.part', 'wb') as f:
                n = int(r.headers.get('Content-Length', -1)); shutil.copyfileobj(r, f, 1 << 20)
            if n >= 0 and os.path.getsize(dest + '.part') != n: raise IOError(f'truncated {os.path.getsize(dest + ".part")} of {n}')
            os.replace(dest + '.part', dest); return dest
        except Exception as e:
            err = e; time.sleep(3 + 5 * k)
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
    P = np.abs(np.fft.rfft(seg * HANN, axis=-1)) ** 2
    bp = np.stack([P[..., m].sum(-1) for m in BMASK], -1)
    ll = np.abs(np.diff(seg, axis=-1)).mean(-1); var = seg.var(-1); ku = kurtosis(seg, axis=-1)
    sb = np.signbit(seg); zc = (sb[..., 1:] != sb[..., :-1]).mean(-1)
    f = np.concatenate([np.log10(bp + 1e-6), np.log10(ll + 1e-6)[..., None], np.log10(var + 1e-6)[..., None],
                        (np.clip(ku, -3, 30) / 10)[..., None], (zc * 10)[..., None]], -1)
    z = (seg - seg.mean(-1, keepdims=True)) / (seg.std(-1, keepdims=True) + 1e-6)
    C = np.einsum('nct,ndt->ncd', z, z, optimize=True) / seg.shape[-1]
    return np.nan_to_num(f).astype(np.float32), np.nan_to_num(C[:, IU[0], IU[1]]).astype(np.float16)
def work(item):
    c, f, seiz, extra = item
    try: path = fetch(f'chb{c:02d}/{f}')
    except Exception as e: return item, None, f'download: {e}'
    try:
        ch, fs = read_edf(path)
    except Exception as e:
        os.remove(path); return item, None, f'read: {e}'
    os.remove(path)
    if abs(fs - FS) > 1e-6 or any(x not in ch for x in CH18): return item, None, 'missing channels or rate'
    X = sosfiltfilt(SOS, np.stack([ch[x] for x in CH18]), axis=-1).astype(np.float32); T = X.shape[1]; del ch
    starts = [(st, 0) for st in range(0, T - WIN + 1, HOP)]
    for a, e in seiz:
        starts += [(st, 1) for st in range(int(a * FS) - WIN // 2, int(e * FS) - WIN // 2, AUG_HOP) if 0 <= st <= T - WIN and st % HOP]
    Fl, Cl, meta = [], [], []
    for b in range(0, len(starts), 256):
        chunk = starts[b:b + 256]; seg = np.stack([X[:, st:st + WIN] for st, _ in chunk])
        fe, co = window_features(seg); Fl.append(fe); Cl.append(co)
        for st, aug in chunk:
            ctr = (st + WIN / 2) / FS; sid = next((j for j, (a, e) in enumerate(seiz) if a <= ctr < e), -1)
            meta.append((st / FS, int(sid >= 0), aug, sid))
    return item, (np.concatenate(Fl), np.concatenate(Cl), np.array(meta, np.float64), T / FS / 3600), None
if __name__ == '__main__':
    plan = []
    for case in range(1, 25):
        rel = f'chb{case:02d}/chb{case:02d}-summary.txt'; p = fetch(rel); files = parse_summary(open(p, errors='ignore').read()); os.remove(p)
        sz = [f for f, s in files.items() if s]; free = [f for f, s in files.items() if not s]
        main = sz + free[:1]; plan += [(case, f, files[f], 0) for f in main]
        if case in (22, 23): plan += [(case, f, files[f], 1) for f in files if f not in main]
    log(f'{len(plan)} files planned ({sum(p[3] for p in plan)} extra test files of chb22/chb23)')
    res = {}
    with cf.ProcessPoolExecutor(4) as ex:
        for k, (item, r, err) in enumerate(ex.map(work, plan)):
            res[(item[0], item[1])] = (item, r, err)
            if (k + 1) % 10 == 0: log(f'  {k + 1}/{len(plan)} files')
    Fl, Cl, M, FILES, SZ, SKIP = [], [], [], [], [], []
    for key in sorted(res):
        (c, f, seiz, extra), r, err = res[key]
        if r is None: SKIP.append([f, err]); continue
        fid = len(FILES); FILES.append(dict(case=c, file=f, hours=r[3], seizures=seiz, extra=extra))
        sid0 = len(SZ)
        for a, e in seiz: SZ.append(dict(case=c, subject=SUBJECT(c), file=f, fid=fid, onset=a, offset=e))
        m = r[2]; n = len(m)
        M.append(np.column_stack([np.full(n, SUBJECT(c)), np.full(n, c), np.full(n, fid), m[:, 0], m[:, 1], m[:, 2],
                                  np.where(m[:, 3] >= 0, m[:, 3] + sid0, -1), np.full(n, extra)]))
        Fl.append(r[0]); Cl.append(r[1])
    np.savez(os.path.join(OUT, 'features_local.npz'), F=np.concatenate(Fl), C=np.concatenate(Cl), meta=np.concatenate(M),
             files=json.dumps(FILES), seizures=json.dumps(SZ), skipped=json.dumps(SKIP))
    log(f'saved {sum(len(x) for x in Fl)} windows from {len(FILES)} files; skipped {SKIP}')
