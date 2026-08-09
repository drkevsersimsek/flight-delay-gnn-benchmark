"""
Adım 13: İstatistiksel anlamlılık testleri

5 seed'den elde edilen test MAE değerleri üzerinde:
  - Paired t-test (MLP vs GCN, MLP vs GAT, GCN vs GAT)
  - Wilcoxon signed-rank test (non-parametrik, n=5 için daha uygun)
  - Cohen's d (etki büyüklüğü / effect size)

Mevcut sonuçlar (12_multi_seed_v2.py çıktısından):
  MLP  : [9.116, 8.956, 8.954, 9.034, 8.736]
  GCN  : [9.886, 10.249, 9.982, 9.904, 9.990]
  GAT  : [9.161, 9.392, 10.494, 9.186, 9.369]

Bu değerleri 12_multi_seed_v2.py çıktısından güncelleyiniz.
"""

import numpy as np
from scipy import stats

# ========= BURAYA KENDİ SONUÇLARINIZI GİRİN =========
# Sıra: seed 42, 1, 7, 123, 2024
mlp_scores = np.array([9.116, 8.956, 8.954, 9.034, 8.736])
gcn_scores = np.array([9.886, 10.249, 9.982, 9.904, 9.990])
gat_scores = np.array([9.161, 9.392, 10.494, 9.186, 9.369])
# =====================================================

models = {"MLP": mlp_scores, "GCN": gcn_scores, "GAT": gat_scores}

print("=" * 65)
print("Temel istatistikler (Test MAE, dakika)")
print("=" * 65)
for name, scores in models.items():
    print(f"  {name}: mean={scores.mean():.3f}  std={scores.std(ddof=1):.3f}  "
          f"min={scores.min():.3f}  max={scores.max():.3f}")

def cohen_d_paired(a, b):
    """Paired Cohen's d = mean(diff) / std(diff)"""
    diff = a - b
    return diff.mean() / diff.std(ddof=1)

print()
print("=" * 65)
print("Çiftli karşılaştırmalar: MLP vs diğerleri")
print("(Düşük MAE daha iyi; negatif d -> MLP daha iyi)")
print("=" * 65)

pairs = [
    ("MLP", "GCN", mlp_scores, gcn_scores),
    ("MLP", "GAT", mlp_scores, gat_scores),
    ("GCN", "GAT", gcn_scores, gat_scores),
]

for name_a, name_b, a, b in pairs:
    diff = a - b
    print(f"\n--- {name_a} vs {name_b} ---")
    print(f"  Farklar (A-B, her seed): {np.round(diff, 3).tolist()}")
    print(f"  Ortalama fark: {diff.mean():.3f} dk  ({name_a} {'daha iyi' if diff.mean() < 0 else 'daha kötü'})")

    # Paired t-test
    t_stat, p_val = stats.ttest_rel(a, b)
    print(f"  Paired t-test: t({len(a)-1})={t_stat:.3f}, p={p_val:.4f}", end="")
    if p_val < 0.001:
        print(" ***")
    elif p_val < 0.01:
        print(" **")
    elif p_val < 0.05:
        print(" *")
    else:
        print(f" (n.s., α=0.05)")

    # Wilcoxon signed-rank test
    try:
        w_stat, w_p = stats.wilcoxon(a, b, alternative='two-sided')
        print(f"  Wilcoxon signed-rank: W={w_stat:.1f}, p={w_p:.4f}", end="")
        if w_p < 0.001:
            print(" ***")
        elif w_p < 0.01:
            print(" **")
        elif w_p < 0.05:
            print(" *")
        else:
            print(f" (n.s., α=0.05)")
    except ValueError as e:
        print(f"  Wilcoxon: {e} (muhtemelen tüm farklar aynı işaret)")

    # Cohen's d (paired)
    d = cohen_d_paired(a, b)
    magnitude = ("küçük" if abs(d) < 0.5 else
                 "orta" if abs(d) < 0.8 else "büyük")
    print(f"  Cohen's d (paired): {d:.3f} ({magnitude} etki)")

    # 95% CI for mean difference
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    t_crit = stats.t.ppf(0.975, df=len(diff)-1)
    ci_low = diff.mean() - t_crit * se
    ci_high = diff.mean() + t_crit * se
    print(f"  95% CI (fark): [{ci_low:.3f}, {ci_high:.3f}] dk")

print()
print("=" * 65)
print("NOT: n=5 seed ile istatistiksel güç (power) düşük olabilir.")
print("Wilcoxon signed-rank n=5 ile minimum p=0.0625 üretebilir")
print("(iki-yönlü, tüm işaretler aynı yönde olsa bile).")
print("Bu nedenle hem p değerleri hem Cohen's d birlikte raporlanmalıdır.")
print("=" * 65)

print()
print("Makale için kopyalanabilir özet:")
print("-" * 45)
for name_a, name_b, a, b in pairs[:2]:  # Sadece MLP vs diğerleri
    diff = a - b
    t_stat, p_val = stats.ttest_rel(a, b)
    try:
        w_stat, w_p = stats.wilcoxon(a, b, alternative='two-sided')
        w_str = f"W={w_stat:.0f}, p={w_p:.4f}"
    except:
        w_str = "N/A"
    d = cohen_d_paired(a, b)
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    t_crit = stats.t.ppf(0.975, df=len(diff)-1)
    ci = (diff.mean() - t_crit*se, diff.mean() + t_crit*se)
    print(f"{name_a} vs {name_b}: mean diff={diff.mean():.3f} dk "
          f"[95% CI: {ci[0]:.3f}, {ci[1]:.3f}], "
          f"t(4)={t_stat:.3f}, p={p_val:.4f}, "
          f"{w_str}, d={d:.3f}")
