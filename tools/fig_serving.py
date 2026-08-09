#!/usr/bin/env python3
"""Figure 1 for the paper: WER vs each serving parameter, one panel per mechanism.

Every number is transcribed from a measured entry in BENCH.md (113-utt stratified subset for
the flush/prefix curves and the gain table; the sinc/test numbers are the 676-utt runs).
No smoothing, no interpolation — these are the actual measured points, and the sparse x-axes
are honest about how few configurations were run.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))

# --- flush depth (tail frames past the 0.5 s delay), subset ------------------------------
ax = axes[0]
x, y = [8, 16, 24], [11.45, 10.78, 11.51]
ax.plot(x, y, "o-", color="#1f6fb2")
ax.axhline(11.45, ls=":", c="gray", lw=0.8)
ax.annotate("reference scripts\nflush ≈ 7", xy=(8, 11.45), xytext=(10, 11.85), fontsize=8,
            arrowprops=dict(arrowstyle="->", lw=0.7))
ax.set_xlabel("flush tail (frames)")
ax.set_ylabel("WER %")
ax.set_title("flush is a window, not a dial", fontsize=9)
ax.set_xticks(x)

# --- leading-silence prefix, subset -------------------------------------------------------
ax = axes[1]
x, y = [0, 6, 12], [11.45, 10.71, 12.25]
ax.plot(x, y, "s-", color="#2a9d5c")
ax.set_xlabel("leading silence (frames)")
ax.set_title("prefix: codec settle vs LM silence prior", fontsize=9)
ax.set_xticks(x)

# --- input gain, subset --------------------------------------------------------------------
ax = axes[2]
labels = ["raw\n−46 dBFS", "raw\n+AGC", "peak-norm", "norm\n+AGC"]
y = [22.36, 11.41, 11.45, 11.08]
bars = ax.bar(range(4), y, color=["#c44", "#1f6fb2", "#888", "#1f6fb2"])
ax.set_xticks(range(4), labels, fontsize=8)
ax.set_title("input gain: the codec sees raw samples", fontsize=9)
ax.bar_label(bars, fmt="%.1f", fontsize=8)
ax.set_ylim(0, 25)

for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("Serving protocol as a quality hyperparameter (kyutai stt-1b, FLEURS-fr subset, n=113)",
             fontsize=10)
fig.tight_layout()
fig.savefig("eval-data/fig_serving.png", dpi=180)
fig.savefig("eval-data/fig_serving.pdf")
print("written eval-data/fig_serving.{png,pdf}")
