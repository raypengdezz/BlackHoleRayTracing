import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.widgets import Button, RadioButtons, Slider

from geodesics.massless_geodesics import massless_geodesics


# color
FIGURE_BG = "#0B1020"
PANEL_BG = "#121A2B"
AXIS_BG = "#09111E"
TEXT_COLOR = "#E8EEF9"
MUTED_TEXT = "#9FB3C8"
GRID_COLOR = "#22324D"
SLIDER_TRACK = "#1A2740"
SLIDER_ACTIVE = "#F4A261"
BUTTON_BG = "#1D2A44"
BUTTON_HOVER = "#2B3D63"
HORIZON_FILL = "#05070B"
LINE_COLORS = [
    "#7FDBCA",
    "#F4A261",
    "#F28482",
    "#7B9ACC",
    "#E9C46A",
    "#C77DFF",
    "#A1E3FF",
]

# initial condition
INITIAL_M = 1.0
PLOT_LIMIT = 30

trajectories = [
    {"b": 1.0},
    {"b": 3.0},
    {"b": 5.0},
]
selected_index = 0
selector = None
selector_ax = None
syncing_sliders = False


# 調整整張圖的大小與主圖/控制區比例
fig, ax = plt.subplots(figsize=(13, 8))
plt.subplots_adjust(left=0.22, bottom=0.12, right=0.98, top=0.92)
fig.patch.set_facecolor(FIGURE_BG)

ax.set_aspect("equal")
# 調整座標軸顯示範圍
ax.set_xlim(-PLOT_LIMIT, PLOT_LIMIT)
ax.set_ylim(-PLOT_LIMIT, PLOT_LIMIT)
ax.set_facecolor(AXIS_BG)
ax.set_title("Massless Geodesics", color=TEXT_COLOR, fontsize=16, pad=14)
ax.set_xlabel("x", color=MUTED_TEXT)
ax.set_ylabel("y", color=MUTED_TEXT)
ax.tick_params(colors=MUTED_TEXT, labelsize=10)
ax.grid(color=GRID_COLOR, alpha=0.35, linewidth=0.8)
for spine in ax.spines.values():
    spine.set_color(GRID_COLOR)
    spine.set_linewidth(1.2)

horizon = Circle((0, 0), 2 * INITIAL_M, facecolor=HORIZON_FILL, edgecolor="#2A3448", linewidth=1.5)
ax.add_patch(horizon)

# 儲存每條 trajectory 對應的線段物件
lines = []


def get_color(index):
    return LINE_COLORS[index % len(LINE_COLORS)]


def rebuild_selector():
    global selector, selector_ax

    if selector_ax is not None:
        selector_ax.remove()

    # 左側 trajectory 選擇面板位置與大小
    selector_ax = plt.axes((0.04, 0.56, 0.13, 0.24), facecolor=PANEL_BG)
    labels = [f"Trajectory {index + 1}" for index in range(len(trajectories))]
    selector = RadioButtons(selector_ax, labels, active=selected_index)
    selector_ax.set_title("Trajectories", color=TEXT_COLOR, fontsize=11, pad=10)
    selector_ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    for spine in selector_ax.spines.values():
        spine.set_color("#24324A")
        spine.set_linewidth(1.0)
    for label in selector.labels:
        label.set_color(TEXT_COLOR)
        label.set_fontsize(10)
    if hasattr(selector, "activecolor"):
        selector.activecolor = SLIDER_ACTIVE
    selector.on_clicked(select_trajectory)


def sync_sliders_with_selected():
    global syncing_sliders

    syncing_sliders = True
    b_slider.set_val(trajectories[selected_index]["b"])
    syncing_sliders = False


def redraw(_value=None):
    # 依照目前 M 更新 event horizon 半徑
    current_m = m_slider.val
    horizon.set_radius(2 * current_m)

    while len(lines) < len(trajectories):
        color = get_color(len(lines))
        line, = ax.plot([], [], color=color, linewidth=1.3, linestyle="-", solid_capstyle="round")
        lines.append(line)

    for index, line in enumerate(lines):
        if index < len(trajectories):
            current_b = trajectories[index]["b"]
            traj = massless_geodesics(current_b, current_m)
            line.set_data(traj.x_positions, traj.y_positions)
            line.set_color(get_color(index))
            line.set_linewidth(2.0 if index == selected_index else 1.2)
            line.set_alpha(0.96 if index == selected_index else 0.52)
            line.set_label(f"#{index + 1}: b={current_b:.1f}")
        else:
            line.set_data([], [])

    legend = ax.legend(
        loc="upper right",
        facecolor=PANEL_BG,
        edgecolor="#24324A",
        framealpha=0.95,
        fontsize=9,
    )
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)
    fig.canvas.draw_idle()


def select_trajectory(label):
    global selected_index

    selected_index = int(label.split()[-1]) - 1
    sync_sliders_with_selected()
    redraw()


def update_selected(_value):
    if syncing_sliders:
        return

    # 只更新目前選中的那條 trajectory 的 b
    trajectories[selected_index]["b"] = b_slider.val
    redraw()


def add_trajectory(_event):
    global selected_index

    # 新增一條 trajectory，初始值直接採用目前 slider 上的數字
    trajectories.append({"b": b_slider.val})
    selected_index = len(trajectories) - 1
    rebuild_selector()
    sync_sliders_with_selected()
    redraw()


rebuild_selector()

# 左側控制元件位置：可以在這裡調整 slider / button 版面
b_axis = plt.axes((0.04, 0.22, 0.13, 0.012), facecolor=SLIDER_TRACK)
m_axis = plt.axes((0.04, 0.16, 0.13, 0.012), facecolor=SLIDER_TRACK)
add_axis = plt.axes((0.04, 0.31, 0.13, 0.045), facecolor=PANEL_BG)

b_slider = Slider(b_axis, "b", 0.1, 10.0, valinit=trajectories[0]["b"], valstep=0.1)
m_slider = Slider(m_axis, "M", 0.1, 5.0, valinit=INITIAL_M, valstep=0.1)
add_button = Button(add_axis, "Add Trajectory")

for slider in (b_slider, m_slider):
    slider.label.set_color(TEXT_COLOR)
    slider.valtext.set_color(SLIDER_ACTIVE)
    slider.poly.set_facecolor(SLIDER_ACTIVE)
    slider.label.set_fontsize(9)
    slider.valtext.set_fontsize(8)
    if hasattr(slider, "track"):
        slider.track.set_color(SLIDER_TRACK)
    if hasattr(slider, "vline"):
        slider.vline.set_color(TEXT_COLOR)

add_button.label.set_color(TEXT_COLOR)
add_button.label.set_fontsize(9)
add_button.color = BUTTON_BG
add_button.hovercolor = BUTTON_HOVER

b_slider.on_changed(update_selected)
m_slider.on_changed(redraw)
add_button.on_clicked(add_trajectory)


redraw()
plt.show()
