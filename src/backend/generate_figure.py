import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea
from PIL import Image


# ── DATA ──────────────────────────────────────────────────────────────────────
IMAGE_PATH = "NIH_dataset/images-224/images-224/00013689_000.png"
OUTPUT_FIG = "crc_vs_uncalibrated_cnn_figure.png"

ALPHA = 0.1
LAMBDA_HAT = 0.292214

CLASSES_CLEAN = [
    "Cardiomegaly",
    "Edema",
    "Consolidation",
    "Atelectasis",
    "Pleural Effusion",
]

CLASSES_LABEL = [
    "Cardio-\nmegaly",
    "Edema",
    "Consoli-\ndation",
    "Atelectasis",
    "Pleural\nEffusion",
]

TRUE_LABELS = [
    "Consolidation",
    "Atelectasis",
    "Pleural Effusion",
]

UNCALIBRATED_TOP_PRED = ["Consolidation"]

CRC_PREDS = [
    "Consolidation",
    "Atelectasis",
    "Pleural Effusion",
]

SIGMOIDS = {
    "Cardiomegaly": 0.1105,
    "Edema": 0.2432,
    "Consolidation": 0.8178,
    "Atelectasis": 0.4964,
    "Pleural Effusion": 0.3258,
}


# ── COLOURS ───────────────────────────────────────────────────────────────────
SOFT_YELLOW = "#E9C46A"
SOFT_GREEN = "#8AB17D"
SOFT_BLUE = "#6FA8DC"
SOFT_TEAL = "#76B7B2"
SOFT_INDIGO = "#8DA0CB"
GREY = "#D6DADF"
DGREY = "#5F6872"
BLACK = "#24313F"
WHITE = "#FFFFFF"
GREEN = "#8AB17D"
RED = "#D98982"
GRID = "#E2E6EA"
SOFT_BORDER = "#EEF1F4"

CLASS_COLORS = {
    "Atelectasis": SOFT_GREEN,
    "Cardiomegaly": SOFT_YELLOW,
    "Consolidation": SOFT_YELLOW,
    "Edema": SOFT_TEAL,
    "Pleural Effusion": SOFT_BLUE,
}


# ── AXIS STYLING HELPERS ──────────────────────────────────────────────────────
def style_bar_axis(ax):
    ax.set_ylim(0, 1.0)
    ax.tick_params(labelsize=10.5, colors=DGREY, length=0)
    ax.yaxis.grid(True, linestyle="--", alpha=0.22, color=GRID, zorder=0)
    ax.set_axisbelow(True)
    ax.set_facecolor(WHITE)

    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.0)
        ax.spines[side].set_color(SOFT_BORDER)


# ── PREDICTION LABEL HELPERS ─────────────────────────────────────────────────
def draw_token_line(ax, tokens, y, fontsize=12):
    """
    Draw a centered colored token line at a given axes y coordinate.
    tokens = [(text, color), ...]
    """

    fig = ax.get_figure()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax_bbox = ax.get_window_extent(renderer=renderer)
    ax_width = ax_bbox.width

    def token_width_axes(text, fs):
        t = ax.text(0, -10, text, fontsize=fs, transform=ax.transAxes, alpha=0)
        fig.canvas.draw()
        w = t.get_window_extent(renderer=renderer).width / ax_width
        t.remove()
        return w

    def gap_after(curr_txt, next_txt):
        if next_txt is None:
            return 0.0

        curr = curr_txt.strip()
        nxt = next_txt.strip()

        if curr == "{":
            return 0.012
        if nxt == ",":
            return 0.008
        if curr == ",":
            return 0.012
        if nxt == "}":
            return 0.010
        return 0.010

    widths = [token_width_axes(txt, fontsize) for txt, _ in tokens]
    gaps = []
    for i in range(len(tokens) - 1):
        gaps.append(gap_after(tokens[i][0], tokens[i + 1][0]))

    total_w = sum(widths) + sum(gaps)
    x_start = 0.5 - total_w / 2

    x_cursor = x_start
    for i, ((txt, color), w) in enumerate(zip(tokens, widths)):
        is_symbol = txt.strip() in ["{", "}", ","]
        weight = "bold" if (color != BLACK and not is_symbol) else "normal"

        ax.text(
            x_cursor,
            y,
            txt,
            transform=ax.transAxes,
            fontsize=fontsize,
            color=color,
            fontweight=weight,
            va="bottom",
            ha="left",
            clip_on=False,
        )

        x_cursor += w
        if i < len(tokens) - 1:
            x_cursor += gaps[i]


def draw_plain_prediction_lines(ax, lines, y_start=1.025, fontsize=12.5, line_gap=0.055):
    for idx, line_labels in enumerate(lines):
        tokens = []
        for i, label in enumerate(line_labels):
            tokens.append((label, CLASS_COLORS[label]))
            if i < len(line_labels) - 1:
                tokens.append((",", BLACK))

        draw_token_line(
            ax=ax,
            tokens=tokens,
            y=y_start - idx * line_gap,
            fontsize=fontsize,
        )


def draw_set_prediction_lines(ax, lines, y_start=1.068, fontsize=10.8, line_gap=0.043):
    total_lines = len(lines)

    for line_idx, line_labels in enumerate(lines):
        pieces = []

        if line_idx == 0:
            pieces.append(
                TextArea(
                    "{ ",
                    textprops=dict(color=BLACK, fontsize=fontsize, fontweight="bold"),
                )
            )

        for item_idx, label in enumerate(line_labels):
            is_last_item_in_line = item_idx == len(line_labels) - 1
            is_last_line = line_idx == total_lines - 1
            is_last_overall = is_last_line and is_last_item_in_line
            suffix = "" if is_last_overall else " , "

            pieces.append(
                TextArea(
                    f"{label}{suffix}",
                    textprops=dict(
                        color=CLASS_COLORS[label],
                        fontsize=fontsize,
                        fontweight="bold",
                    ),
                )
            )

        if line_idx == total_lines - 1:
            pieces.append(
                TextArea(
                    " }",
                    textprops=dict(color=BLACK, fontsize=fontsize, fontweight="bold"),
                )
            )

        packed_line = HPacker(children=pieces, align="baseline", pad=0, sep=0)
        annotation = AnnotationBbox(
            packed_line,
            (0.5, y_start - line_idx * line_gap),
            xycoords=ax.transAxes,
            box_alignment=(0.5, 0.0),
            frameon=False,
        )
        annotation.set_clip_on(False)
        ax.add_artist(annotation)


# ── BAR ANNOTATION HELPERS ───────────────────────────────────────────────────
def add_value_labels(ax, bars, values, decimals=4, y_offset=0.02):
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + y_offset,
            f"{val:.{decimals}f}",
            ha="center",
            va="bottom",
            fontsize=10.5,
            color=BLACK,
        )


# ── FIGURE LAYOUT ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(24, 8.2), facecolor=WHITE)

fig.subplots_adjust(
    top=0.80,
    bottom=0.24,
    left=0.04,
    right=0.96,
    wspace=0.28,
)

gs = gridspec.GridSpec(
    1,
    3,
    figure=fig,
    top=0.80,
    bottom=0.24,
    left=0.04,
    right=0.96,
    wspace=0.30,
    width_ratios=[1.0, 1.35, 1.35],
)

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax3 = fig.add_subplot(gs[2])


# ── FIGURE-LEVEL HEADER ───────────────────────────────────────────────────────
ground_truth_text = "Ground Truth: { " + ", ".join(TRUE_LABELS) + " }"
fig.text(
    0.5,
    0.965,
    ground_truth_text,
    ha="center",
    va="center",
    fontsize=20,
    fontweight="bold",
    color=BLACK,
)


# Common x positions with extra spacing
BAR_X = np.array([0.0, 1.45, 3.05, 4.50, 5.95])


# ─────────────────────────────────────────────────────────────────────────────
# Panel 1: Chest X-ray
# ─────────────────────────────────────────────────────────────────────────────
img = Image.open(IMAGE_PATH).convert("L")
ax1.imshow(img, cmap="gray", aspect="auto")
ax1.set_xticks([])
ax1.set_yticks([])

for side in ["top", "right", "left", "bottom"]:
    ax1.spines[side].set_visible(False)

ax1.text(
    0.5,
    1.045,
    "Chest X-ray",
    transform=ax1.transAxes,
    ha="center",
    va="bottom",
    fontsize=19,
    fontweight="bold",
    color=BLACK,
)


# ─────────────────────────────────────────────────────────────────────────────
# Panel 2: Uncalibrated model top prediction
# ─────────────────────────────────────────────────────────────────────────────
probs2 = [SIGMOIDS[c] for c in CLASSES_CLEAN]
colors2 = [CLASS_COLORS[c] if c in UNCALIBRATED_TOP_PRED else GREY for c in CLASSES_CLEAN]

bars2 = ax2.bar(
    BAR_X,
    probs2,
    color=colors2,
    edgecolor=WHITE,
    width=0.78,
    zorder=3,
)

add_value_labels(ax2, bars2, probs2, decimals=4, y_offset=0.018)

ax2.set_xticks(BAR_X)
ax2.set_xticklabels(CLASSES_LABEL, fontsize=10, ha="center")
ax2.set_xlim(-0.8, 6.8)
ax2.set_ylabel(r"$\hat{f}(x)_k$ (sigmoid)", fontsize=14, color=DGREY)

style_bar_axis(ax2)

ax2.text(
    0.5,
    1.085,
    "Uncalibrated Model Top Prediction",
    transform=ax2.transAxes,
    ha="center",
    va="bottom",
    fontsize=18,
    fontweight="bold",
    color=BLACK,
)

draw_plain_prediction_lines(
    ax=ax2,
    lines=[UNCALIBRATED_TOP_PRED],
    y_start=1.025,
    fontsize=13,
    line_gap=0.055,
)


# ─────────────────────────────────────────────────────────────────────────────
# Panel 3: CRC calibrated prediction set
# ─────────────────────────────────────────────────────────────────────────────
probs4 = [SIGMOIDS[c] for c in CLASSES_CLEAN]
colors4 = [CLASS_COLORS[c] if c in CRC_PREDS else GREY for c in CLASSES_CLEAN]

bars4 = ax3.bar(
    BAR_X,
    probs4,
    color=colors4,
    edgecolor=WHITE,
    width=0.78,
    zorder=3,
)

add_value_labels(ax3, bars4, probs4, decimals=4, y_offset=0.018)
for label, bar, value in zip(CLASSES_CLEAN, bars4, probs4):
    if label == "Edema":
        bar_center = bar.get_x() + bar.get_width() / 2
        for text in ax3.texts:
            x, y = text.get_position()
            if abs(x - bar_center) < 0.001 and abs(y - (value + 0.018)) < 0.001:
                text.set_y(LAMBDA_HAT - 0.012)
                text.set_va("top")
                break

ax3.axhline(LAMBDA_HAT, color=BLACK, linestyle="--", linewidth=1.8, zorder=4)

gap_x = (BAR_X[3] + BAR_X[4]) / 2.0
ax3.annotate(
    f"λ̂ = {LAMBDA_HAT}",
    xy=(gap_x, LAMBDA_HAT),
    xytext=(BAR_X[3] + 0.55, 0.43),
    fontsize=11,
    color=BLACK,
    fontweight="bold",
    ha="left",
    arrowprops=dict(arrowstyle="->", color=BLACK, lw=1.1),
)

ax3.set_xticks(BAR_X)
ax3.set_xticklabels(CLASSES_LABEL, fontsize=10, ha="center")
ax3.set_xlim(-0.8, 6.8)
ax3.set_ylabel(r"$\hat{f}(x)_k$ (sigmoid)", fontsize=14, color=DGREY)

style_bar_axis(ax3)

ax3.text(
    0.5,
    1.125,
    "CRC Calibrated Model Prediction Set",
    transform=ax3.transAxes,
    ha="center",
    va="bottom",
    fontsize=18,
    fontweight="bold",
    color=BLACK,
)

draw_set_prediction_lines(
    ax=ax3,
    lines=[
        ["Consolidation", "Atelectasis", "Pleural Effusion"],
    ],
    y_start=1.070,
    fontsize=13,
    line_gap=0.055,
)


# ── SAVE ──────────────────────────────────────────────────────────────────────
plt.savefig(
    OUTPUT_FIG,
    dpi=300,
    bbox_inches="tight",
    facecolor=WHITE,
    edgecolor="none",
)

print(f"Figure saved: {OUTPUT_FIG}")
