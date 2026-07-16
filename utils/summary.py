"""Chunk-summary table writers shared by the DLO/BDLO drivers."""


def write_summary_tables(f, summary_rows, title_prefix=""):
    # Table 1: Edge Length Metrics
    f.write(f"{title_prefix}Edge Length Metrics\n")
    f.write("-" * 100 + "\n")
    f.write(f"{'Method':<12} | {'Edge % Mean':<18} | {'Edge RMSE (mm)':<15} | {'<2%':<8} | {'<5%':<8} | {'<10%':<8}\n")
    f.write("-" * 100 + "\n")
    for s in summary_rows:
        f.write(f"{s['method']:<12} | {s['edge_pct_mean_avg']:>5.2f}% ±{s['edge_pct_mean_std']:>5.2f}% | "
                f"{s['edge_rmse_avg']:>5.2f} ±{s['edge_rmse_std']:>4.2f} mm | "
                f"{s['edge_under_2pct']:>5.1f}% | {s['edge_under_5pct']:>5.1f}% | {s['edge_under_10pct']:>5.1f}%\n")

    f.write("\n")

    # Table 2: Position RMSE Metrics
    f.write(f"{title_prefix}Position RMSE Metrics\n")
    f.write("-" * 80 + "\n")
    f.write(f"{'Method':<12} | {'Pos RMSE (mm)':<18} | {'<2mm':<8} | {'<5mm':<8} | {'<10mm':<8}\n")
    f.write("-" * 80 + "\n")
    for s in summary_rows:
        f.write(f"{s['method']:<12} | {s['pos_rmse_avg']:>5.2f} ±{s['pos_rmse_std']:>5.2f} mm   | "
                f"{s['pos_under_2mm']:>5.1f}% | {s['pos_under_5mm']:>5.1f}% | {s['pos_under_10mm']:>5.1f}%\n")

    f.write("\n")

    # Table 3: Chamfer Distance Metrics
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

def print_summary_tables(summary_rows):
    print("\nEdge Length Metrics")
    print("-" * 100)
    print(f"{'Method':<12} | {'Edge % Mean':<18} | {'Edge RMSE (mm)':<15} | {'<2%':<8} | {'<5%':<8} | {'<10%':<8}")
    print("-" * 100)
    for s in summary_rows:
        print(f"{s['method']:<12} | {s['edge_pct_mean_avg']:>5.2f}% ±{s['edge_pct_mean_std']:>5.2f}% | "
              f"{s['edge_rmse_avg']:>5.2f} ±{s['edge_rmse_std']:>4.2f} mm | "
              f"{s['edge_under_2pct']:>5.1f}% | {s['edge_under_5pct']:>5.1f}% | {s['edge_under_10pct']:>5.1f}%")

    print("\nPosition RMSE Metrics")
    print("-" * 80)
    print(f"{'Method':<12} | {'Pos RMSE (mm)':<18} | {'<2mm':<8} | {'<5mm':<8} | {'<10mm':<8}")
    print("-" * 80)
    for s in summary_rows:
        print(f"{s['method']:<12} | {s['pos_rmse_avg']:>5.2f} ±{s['pos_rmse_std']:>5.2f} mm   | "
              f"{s['pos_under_2mm']:>5.1f}% | {s['pos_under_5mm']:>5.1f}% | {s['pos_under_10mm']:>5.1f}%")

    print("\nChamfer Distance Metrics ")
    print("-" * 130)
    print(f"{'Method':<12} | {'CD (mm)':<15} | {'Pred→Ref':<10} | {'Ref→Pred':<10} | {'Prec@2mm':<8} | {'Prec@5mm':<8} | {'Prec@10mm':<8} | {'Rec@2mm':<8} | {'Rec@5mm':<8} | {'Rec@10mm':<8}")
    print("-" * 130)
    for s in summary_rows:
        print(f"{s['method']:<12} | {s['cd_avg']:>5.2f} ±{s['cd_std']:>4.2f} mm | "
              f"{s['cd_pred2ref_avg']:>7.2f} mm | {s['cd_ref2pred_avg']:>7.2f} mm | "
              f"{s['precision_2mm']:>5.1f}% | {s['precision_5mm']:>5.1f}% | {s['precision_10mm']:>5.1f}% | "
              f"{s['recall_2mm']:>5.1f}% | {s['recall_5mm']:>5.1f}% | {s['recall_10mm']:>5.1f}%")

    print("\nF-Scores")
    print("-" * 60)
    print(f"{'Method':<12} | {'F@2mm':<12} | {'F@5mm':<12} | {'F@10mm':<12}")
    print("-" * 60)
    for s in summary_rows:
        print(f"{s['method']:<12} | {s['f_2mm']:>8.2f}% | {s['f_5mm']:>8.2f}% | {s['f_10mm']:>8.2f}%")
