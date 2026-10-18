"""One-column figure of the isolating ablations (bridge_chbmit.py --isolate) used in the paper.
    python make_fig_isolating_1col.py <isolate_summary.json> <output folder>
(results/isolating_ablations/isolate_summary.json of this repository reproduces the paper figure)."""
import json, os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SRC, OUT = sys.argv[1:3]; S = json.load(open(SRC))['summary']
plt.rcParams.update({'font.family': 'serif', 'font.size': 7, 'axes.spines.top': False, 'axes.spines.right': False})
ROWS = [('(a) feature views', [('BRIDGE', 'both views (BRIDGE)'), ('BRIDGE relative view only', 'relative view only'),
                               ('BRIDGE w/o baseline reference', 'absolute view only')]),
        ('(b) graph mixing', [('BRIDGE', 'learned mix (BRIDGE)'), ('BRIDGE fixed equal graph mix', 'fixed equal mix'), ('BRIDGE without graph', 'no graph')]),
        ('(c) attention biases', [('BRIDGE-GT', 'both biases (BRIDGE-GT)'), ('BRIDGE-GT anatomical bias only', 'anatomical only'),
                                  ('BRIDGE-GT functional bias only', 'functional only'), ('BRIDGE-GT without graph bias', 'no graph bias')])]
fig, axs = plt.subplots(3, 1, figsize=(3.5, 3.4), sharex=True, gridspec_kw=dict(height_ratios=[3, 3, 4]))
for ax, (title, rows) in zip(axs, ROWS):
    for i, (key, lab) in enumerate(rows):
        mu, lo, hi = S[key]['auroc']; y = len(rows) - 1 - i
        ax.barh(y, mu - 0.8, left=0.8, color='#1f5a96' if i == 0 else '#9db7d5', height=0.62)
        ax.errorbar(mu, y, xerr=[[mu - lo], [hi - mu]], color='k', capsize=1.5, lw=0.6)
        ax.text(1.02, y, f'{mu:.3f} / {S[key]["auprc"][0]:.3f}', va='center', ha='left', fontsize=5.8, transform=ax.get_yaxis_transform(), clip_on=False)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([lab for _, lab in rows][::-1], fontsize=6.2)
    ax.set_title(title, loc='left', fontsize=6.8, pad=2); ax.set_xlim(0.8, 0.98)
axs[0].text(1.02, 2.75, 'AUROC / AUPRC', ha='left', va='bottom', fontsize=5.8, color='0.3', transform=axs[0].get_yaxis_transform(), clip_on=False)
axs[-1].set_xlabel('Mean AUROC over 23 held-out subjects (95% CI)')
fig.tight_layout(h_pad=0.3); fig.subplots_adjust(right=0.76)
for ext in ('png', 'pdf'): fig.savefig(os.path.join(OUT, f'fig12_isolating_ablations_1col.{ext}'), dpi=600, bbox_inches='tight')
print('figure fig12_isolating_ablations_1col')
