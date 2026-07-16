"""Shared evaluation for the four tracking drivers.

Owns the per-frame metric loop, the clip/chunk aggregation and the summary.txt
writing.  The metric FORMULAS stay per object family (utils.metrics_wire /
metrics_cloth / metrics_fabric); each driver passes three adapters:

    edge_fn(kp, edges, reference_lengths) -> edge metric dict
    pos_fn(kp, ref_pc)                    -> position metric dict
    cd_fn(kp, ref_pc)                     -> chamfer metric dict (builds its own
                                             predicted cloud: edges or faces)

Evaluation is always on the RAW keypoints, never the smoothed ones.
"""
import numpy as np

METRIC_KEYS = [
    'edge_pct_mean', 'edge_pct_std', 'edge_pct_max', 'edge_rmse_mm',
    'edge_under_2pct', 'edge_under_5pct', 'edge_under_10pct',
    'pos_rmse_mm', 'pos_under_2mm', 'pos_under_5mm', 'pos_under_10mm',
    'cd', 'cd_pred2ref', 'cd_ref2pred',
    'precision_2mm', 'precision_5mm', 'precision_10mm',
    'recall_2mm', 'recall_5mm', 'recall_10mm',
    'f_2mm', 'f_5mm', 'f_10mm',
]


def zero_row(local_idx, global_idx):
    row = {'frame': local_idx, 'global_frame': global_idx, 'success': False}
    row.update({k: 0.0 for k in METRIC_KEYS})
    return row


def evaluate_frames(kp_seq, success_seq, global_indices, edges, reference_lengths,
                    ref_clouds, edge_fn, pos_fn, cd_fn):
    """Per-frame metric rows for one clip (RAW keypoints)."""
    rows = []
    for i, kp in enumerate(kp_seq):
        ok = (success_seq[i] and reference_lengths is not None
              and ref_clouds[i] is not None and len(ref_clouds[i]) > 0)
        if not ok:
            rows.append(zero_row(i, global_indices[i]))
            continue
        edge_m = edge_fn(kp, edges, reference_lengths)
        pos_m = pos_fn(kp, ref_clouds[i])
        cd_m = cd_fn(kp, ref_clouds[i])
        rows.append({
            'frame': i, 'global_frame': global_indices[i], 'success': True,
            'edge_pct_mean': edge_m['pct_mean'], 'edge_pct_std': edge_m['pct_std'],
            'edge_pct_max': edge_m['pct_max'], 'edge_rmse_mm': edge_m['rmse_mm'],
            'edge_under_2pct': edge_m['under_2pct'], 'edge_under_5pct': edge_m['under_5pct'],
            'edge_under_10pct': edge_m['under_10pct'],
            'pos_rmse_mm': pos_m['rmse_mm'], 'pos_under_2mm': pos_m['under_2mm'],
            'pos_under_5mm': pos_m['under_5mm'], 'pos_under_10mm': pos_m['under_10mm'],
            'cd': cd_m['cd'], 'cd_pred2ref': cd_m['pred2ref_avg'],
            'cd_ref2pred': cd_m['ref2pred_avg'],
            'precision_2mm': cd_m['precision_2mm'], 'precision_5mm': cd_m['precision_5mm'],
            'precision_10mm': cd_m['precision_10mm'],
            'recall_2mm': cd_m['recall_2mm'], 'recall_5mm': cd_m['recall_5mm'],
            'recall_10mm': cd_m['recall_10mm'],
            'f_2mm': cd_m['f_2mm'], 'f_5mm': cd_m['f_5mm'], 'f_10mm': cd_m['f_10mm'],
        })
    return rows


def summarize(rows, method='Full', skip_first=True):
    """Aggregate per-frame rows into one summary row (clip semantics: the init
    frame is skipped; >0 filters for the error means, success filter for rates)."""
    rows = rows[1:] if (skip_first and len(rows) > 1) else rows
    if not rows:
        return None

    def _mean(vals):
        return float(np.mean(vals)) if vals else 0.0

    def _std(vals):
        return float(np.std(vals)) if vals else 0.0

    edge_pct = [m['edge_pct_mean'] for m in rows if m['edge_pct_mean'] > 0]
    edge_rmse = [m['edge_rmse_mm'] for m in rows if m['edge_rmse_mm'] > 0]
    pos_rmse = [m['pos_rmse_mm'] for m in rows if m['pos_rmse_mm'] > 0]
    ok = [m for m in rows if m['success']]
    row = {
        'method': method,
        'edge_pct_mean_avg': _mean(edge_pct), 'edge_pct_mean_std': _std(edge_pct),
        'edge_rmse_avg': _mean(edge_rmse), 'edge_rmse_std': _std(edge_rmse),
        'edge_under_2pct': _mean([m['edge_under_2pct'] for m in ok]),
        'edge_under_5pct': _mean([m['edge_under_5pct'] for m in ok]),
        'edge_under_10pct': _mean([m['edge_under_10pct'] for m in ok]),
        'pos_rmse_avg': _mean(pos_rmse), 'pos_rmse_std': _std(pos_rmse),
        'pos_under_2mm': _mean([m['pos_under_2mm'] for m in ok]),
        'pos_under_5mm': _mean([m['pos_under_5mm'] for m in ok]),
        'pos_under_10mm': _mean([m['pos_under_10mm'] for m in ok]),
        'cd_avg': _mean([m['cd'] for m in ok]), 'cd_std': _std([m['cd'] for m in ok]),
        'cd_pred2ref_avg': _mean([m['cd_pred2ref'] for m in ok]),
        'cd_ref2pred_avg': _mean([m['cd_ref2pred'] for m in ok]),
    }
    for k in ['precision_2mm', 'precision_5mm', 'precision_10mm',
              'recall_2mm', 'recall_5mm', 'recall_10mm', 'f_2mm', 'f_5mm', 'f_10mm']:
        row[k] = _mean([m[k] for m in ok])
    return row


def clip_weighted_summary(per_clip_rows, method='Full'):
    """Chunk 'clip-weighted' row: unweighted mean of the per-clip summary rows."""
    rows = [r for r in per_clip_rows if r is not None]
    if not rows:
        return None
    out = {'method': method}
    for k in rows[0]:
        if k == 'method':
            continue
        out[k] = float(np.mean([r[k] for r in rows]))
    return out


# ============================================================================
# summary.txt writers (three tables + F-scores)
# ============================================================================

def write_summary_tables(f, summary_rows, title_prefix=""):
    f.write(f"{title_prefix}Edge Length Metrics\n")
    f.write("-" * 100 + "\n")
    f.write(f"{'Method':<12} | {'Edge % Mean':<18} | {'Edge RMSE (mm)':<15} | {'<2%':<8} | {'<5%':<8} | {'<10%':<8}\n")
    f.write("-" * 100 + "\n")
    for s in summary_rows:
        f.write(f"{s['method']:<12} | {s['edge_pct_mean_avg']:>5.2f}% ±{s['edge_pct_mean_std']:>5.2f}% | "
                f"{s['edge_rmse_avg']:>5.2f} ±{s['edge_rmse_std']:>4.2f} mm | "
                f"{s['edge_under_2pct']:>5.1f}% | {s['edge_under_5pct']:>5.1f}% | {s['edge_under_10pct']:>5.1f}%\n")

    f.write("\n")
    f.write(f"{title_prefix}Position RMSE Metrics\n")
    f.write("-" * 80 + "\n")
    f.write(f"{'Method':<12} | {'Pos RMSE (mm)':<18} | {'<2mm':<8} | {'<5mm':<8} | {'<10mm':<8}\n")
    f.write("-" * 80 + "\n")
    for s in summary_rows:
        f.write(f"{s['method']:<12} | {s['pos_rmse_avg']:>5.2f} ±{s['pos_rmse_std']:>5.2f} mm   | "
                f"{s['pos_under_2mm']:>5.1f}% | {s['pos_under_5mm']:>5.1f}% | {s['pos_under_10mm']:>5.1f}%\n")

    f.write("\n")
    f.write(f"{title_prefix}Chamfer Distance Metrics \n")
    f.write("-" * 130 + "\n")
    f.write(f"{'Method':<12} | {'CD (mm)':<15} | {'Pred→Ref':<10} | {'Ref→Pred':<10} | {'Prec@2mm':<8} | {'Prec@5mm':<8} | {'Prec@10mm':<8} | {'Rec@2mm':<8} | {'Rec@5mm':<8} | {'Rec@10mm':<8}\n")
    f.write("-" * 130 + "\n")
    for s in summary_rows:
        f.write(f"{s['method']:<12} | {s['cd_avg']:>5.2f} ±{s['cd_std']:>4.2f} mm | "
                f"{s['cd_pred2ref_avg']:>7.2f} mm | {s['cd_ref2pred_avg']:>7.2f} mm | "
                f"{s['precision_2mm']:>5.1f}% | {s['precision_5mm']:>5.1f}% | {s['precision_10mm']:>5.1f}% | "
                f"{s['recall_2mm']:>5.1f}% | {s['recall_5mm']:>5.1f}% | {s['recall_10mm']:>5.1f}%\n")

    f.write("\n")
    f.write(f"{title_prefix}F-Scores\n")
    f.write("-" * 60 + "\n")
    f.write(f"{'Method':<12} | {'F@2mm':<12} | {'F@5mm':<12} | {'F@10mm':<12}\n")
    f.write("-" * 60 + "\n")
    for s in summary_rows:
        f.write(f"{s['method']:<12} | {s['f_2mm']:>8.2f}% | {s['f_5mm']:>8.2f}% | {s['f_10mm']:>8.2f}%\n")


def write_clip_summary(path, title, summary_row):
    """Per-clip summary.txt: header + the three tables + F-scores."""
    rows = [summary_row] if summary_row is not None else []
    with open(path, 'w') as f:
        f.write(title + "\n")
        f.write("=" * 100 + "\n\n")
        write_summary_tables(f, rows)


def write_chunk_aggregate(path, title, pooled_rows, per_clip_summary_rows, method='Full'):
    """chunk_aggregate_summary.txt: frame-weighted (all frames pooled across
    clips) + clip-weighted (mean of the per-clip summaries) tables."""
    frame_weighted = summarize(pooled_rows, method=method, skip_first=False)
    clip_weighted = clip_weighted_summary(per_clip_summary_rows, method=method)
    with open(path, 'w') as f:
        f.write(title + "\n")
        f.write("=" * 100 + "\n\n")
        f.write("FRAME-WEIGHTED (all frames pooled across clips)\n")
        f.write("=" * 100 + "\n")
        write_summary_tables(f, [frame_weighted] if frame_weighted else [])
        f.write("\n\n")
        f.write("CLIP-WEIGHTED (average of per-clip summaries)\n")
        f.write("=" * 100 + "\n")
        write_summary_tables(f, [clip_weighted] if clip_weighted else [])


def print_summary_tables(summary_rows):
    import io
    buf = io.StringIO()
    write_summary_tables(buf, summary_rows)
    print(buf.getvalue())
