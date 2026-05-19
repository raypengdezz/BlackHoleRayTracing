import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, Slider

from geodesics.massive_geodesics import massive_geodesics
from geodesics.massless_geodesics import massless_geodesics


# color
FIGURE_BG = "#0D1223"
PANEL_BG = "#121A2B"
AXIS_BG = "#0D1625"
TEXT_COLOR = "#E8EEF9"
MUTED_TEXT = "#9FB3C8"
GRID_COLOR = "#444B58"
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
GEODESIC_STEP = 0.1
GEODESIC_MAX_STEP = 500000

# alpha, beta 單位是 degree。
# b, v, M 決定原本平面上的 geodesic。
# alpha 是入射方向在 x-y 平面的方位角。
# beta 是入射方向相對 x-y 平面的仰角。
trajectories = [
    {"b": 5.8, "v": 0.9, "alpha": 0.0, "beta": 0.0},
    {"b": 5.0, "v": 0.8, "alpha": 75.0, "beta": 0.0},
]

selected_index = 0
current_mode = "Massive"
selector = None
selector_ax = None
mode_selector = None
mode_selector_ax = None
orientation_ax = None
syncing_sliders = False


# 讓 3D 圖的三個座標軸使用相同比例。
def set_axes_equal(ax):
    # 讓 x, y, z 三個方向的比例相同，黑洞球體才不會被拉長。
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    max_range = max(x_range, y_range, z_range) / 2
    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)
    
    # 座標軸大小 
    ax.set_xlim3d(x_middle - max_range, x_middle + max_range)
    ax.set_ylim3d(y_middle - max_range, y_middle + max_range)
    ax.set_zlim3d(z_middle - max_range, z_middle + max_range)


# 根據 alpha / beta 建立入射方向與側向偏移方向。
def incoming_basis(alpha_deg, beta_deg):
    # incoming_direction 是粒子/光線從遠方射向黑洞時的大方向。
    # side_direction 是 impact parameter 的方向；b < 0 會自動跑到反側。
    alpha = np.deg2rad(alpha_deg)
    beta = np.deg2rad(beta_deg)

    incoming_direction = np.array([
        np.cos(beta) * np.cos(alpha),
        np.cos(beta) * np.sin(alpha),
        np.sin(beta),
    ])
    side_direction = np.array([
        -np.sin(alpha),
        np.cos(alpha),
        0,
    ])
    return incoming_direction, side_direction


# 把 2D geodesic 軌跡投影到指定方向的 3D 平面。
def lift_trajectory_to_3d(x_positions, y_positions, alpha, beta):
    # 原本 coordinate.py 算的是 x-y 平面的軌道，也就是 z = 0。
    # x_positions 是沿著入射方向的座標，y_positions 是 impact parameter 方向。
    incoming_direction, side_direction = incoming_basis(alpha, beta)
    points_3d = (
        np.outer(x_positions, incoming_direction)
        + np.outer(y_positions, side_direction)
    )
    return points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]


# 畫出目前選中軌道的入射方向箭頭。
def draw_incoming_direction(alpha, beta):
    # 畫一小段入射方向箭頭，幫助看 alpha / beta 的意義。
    incoming_direction, _side_direction = incoming_basis(alpha, beta)
    start = -0.82 * PLOT_LIMIT * incoming_direction
    delta = 0.22 * PLOT_LIMIT * incoming_direction
    ax.quiver(
        start[0],
        start[1],
        start[2],
        delta[0],
        delta[1],
        delta[2],
        color="#E8EEF9",
        linewidth=1.2,
        arrow_length_ratio=0.18,
        alpha=0.78,
    )


# 軌道編號循環取得顏色。
def get_color(index):
    return LINE_COLORS[index % len(LINE_COLORS)]


# 畫出 Schwarzschild 黑洞事件視界球面。
def draw_black_hole():
    # Schwarzschild event horizon 半徑 r = 2M。
    radius = 2 * m_slider.val
    theta = np.linspace(0, np.pi, 72)
    phi = np.linspace(0, 2 * np.pi, 72)
    theta, phi = np.meshgrid(theta, phi)

    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)

    # 外層透明球面讓黑洞看起來有一點陰影/光暈。
    glow_radius = radius * 1.35
    glow_x = glow_radius * np.sin(theta) * np.cos(phi)
    glow_y = glow_radius * np.sin(theta) * np.sin(phi)
    glow_z = glow_radius * np.cos(theta)
    ax.plot_surface(glow_x, glow_y, glow_z, color="#1D2740", edgecolor="none", alpha=0.16)
    return ax.plot_surface(x, y, z, color=HORIZON_FILL, edgecolor="#1F293D", linewidth=0.15, alpha=0.99)

# 設定 3D 座標軸外觀。
def style_3d_axes():
    ax.set_axis_on()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.tick_params(colors=MUTED_TEXT, labelsize=8)
    ax.grid(color=GRID_COLOR, alpha=0.28, linewidth=0.0)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor(AXIS_BG)
        axis.pane.set_alpha(0.18)
        axis.pane.set_edgecolor((0, 0, 0, 0))   # 隱藏面板邊框
        axis.line.set_color((0, 0, 0, 0))       # 隱藏面板邊框
        axis._axinfo["grid"]["linewidth"] = 0


# 重新建立左側軌道選擇器。
def rebuild_selector():
    global selector, selector_ax

    if selector_ax is not None:
        selector_ax.remove()

    selector_ax = plt.axes((0.04, 0.60, 0.13, 0.23), facecolor=PANEL_BG)
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


# 建立 massive / massless 模式選擇器。
def build_mode_selector():
    global mode_selector, mode_selector_ax

    mode_selector_ax = plt.axes((0.04, 0.84, 0.13, 0.10), facecolor=PANEL_BG)
    mode_selector = RadioButtons(mode_selector_ax, ["Massive", "Massless"], active=0)
    mode_selector_ax.set_title("Mode", color=TEXT_COLOR, fontsize=11, pad=8)
    mode_selector_ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    for spine in mode_selector_ax.spines.values():
        spine.set_color("#24324A")
        spine.set_linewidth(1.0)
    for label in mode_selector.labels:
        label.set_color(TEXT_COLOR)
        label.set_fontsize(10)
    if hasattr(mode_selector, "activecolor"):
        mode_selector.activecolor = SLIDER_ACTIVE
    mode_selector.on_clicked(select_mode)


# 將 slider 數值同步成目前選中軌道的參數。
def sync_sliders_with_selected():
    global syncing_sliders

    syncing_sliders = True
    current = trajectories[selected_index]
    b_slider.set_val(current["b"])
    v_slider.set_val(current["v"])
    alpha_slider.set_val(current["alpha"])
    beta_slider.set_val(current["beta"])
    syncing_sliders = False


# 依照目前模式計算 massive 或 massless geodesic。
def make_trajectory(current, current_m):
    if current_mode == "Massless":
        return massless_geodesics(
            current["b"],
            current_m,
            step=GEODESIC_STEP,
            max_step=GEODESIC_MAX_STEP,
        )

    return massive_geodesics(
        current["b"],
        current["v"],
        current_m,
        step=GEODESIC_STEP,
        max_step=GEODESIC_MAX_STEP,
    )


# 產生 legend 裡顯示的軌道標籤。
def trajectory_label(index, current):
    if current_mode == "Massless":
        return (
            f"#{index + 1}: b={current['b']:.1f}, "
            f"az={current['alpha']:.1f}, el={current['beta']:.1f}"
        )

    return (
        f"#{index + 1}: b={current['b']:.1f}, v={current['v']:.2f}, "
        f"az={current['alpha']:.1f}, el={current['beta']:.1f}"
    )


# 清空並重畫整張 3D geodesic 圖。
def redraw(_value=None):
    ax.clear()
    ax.set_facecolor(AXIS_BG)

    draw_black_hole()
    selected = trajectories[selected_index]
    draw_incoming_direction(selected["alpha"], selected["beta"])

    for index, current in enumerate(trajectories):
        traj = make_trajectory(current, m_slider.val)
        x, y, z = lift_trajectory_to_3d(
            traj.x_positions,
            traj.y_positions,
            current["alpha"],
            current["beta"],
        )

        is_selected = index == selected_index
        line_color = get_color(index)

        # 未選中的軌道先畫一層透明底，選中的軌道更亮更粗。
        ax.plot(
            x,
            y,
            z,
            color=line_color,
            linewidth=3.2 if is_selected else 1.5,
            alpha=0.20 if is_selected else 0.10,
            solid_capstyle="round",
        )
        ax.plot(
            x,
            y,
            z,
            color=line_color,
            linewidth=2.0 if is_selected else 1.0,
            alpha=0.98 if is_selected else 0.48,
            solid_capstyle="round",
            label=trajectory_label(index, current),
        )

    ax.set_title(
        (
            f"3D {current_mode} Geodesics | "
        ),
        color=TEXT_COLOR,
        fontsize=14,
        pad=14,
    )

    ax.set_xlim(-PLOT_LIMIT, PLOT_LIMIT)
    ax.set_ylim(-PLOT_LIMIT, PLOT_LIMIT)
    ax.set_zlim(-PLOT_LIMIT, PLOT_LIMIT)
    set_axes_equal(ax)
    ax.view_init(elev=24, azim=38) # 初始角度
    ax.set_box_aspect((1, 1, 1))
    style_3d_axes()
    # draw_xyz_reference_axes()
    if orientation_ax is not None:
        draw_orientation_axes()

    legend = ax.legend(
        loc="upper right",
        facecolor=PANEL_BG,
        edgecolor="#24324A",
        framealpha=0.95,
        fontsize=8,
    )
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)

    fig.canvas.draw_idle()


# 切換目前選中的軌道。
def select_trajectory(label):
    global selected_index

    selected_index = int(label.split()[-1]) - 1
    sync_sliders_with_selected()
    redraw()


# 切換 massive / massless 模式。
def select_mode(label):
    global current_mode

    current_mode = label
    redraw()


# slider 改變時更新目前選中軌道的參數。
def update_selected(_value):
    if syncing_sliders:
        return

    trajectories[selected_index]["b"] = b_slider.val
    trajectories[selected_index]["v"] = v_slider.val
    trajectories[selected_index]["alpha"] = alpha_slider.val
    trajectories[selected_index]["beta"] = beta_slider.val
    redraw()


# 新增一條使用目前 slider 參數的軌道。
def add_trajectory(_event):
    global selected_index

    trajectories.append({
        "b": b_slider.val,
        "v": v_slider.val,
        "alpha": alpha_slider.val,
        "beta": beta_slider.val,
    })
    selected_index = len(trajectories) - 1
    rebuild_selector()
    sync_sliders_with_selected()
    redraw()


# 刪除目前選中的軌道。
def delete_trajectory(_event):
    global selected_index

    if len(trajectories) <= 1:
        return

    trajectories.pop(selected_index)
    selected_index = min(selected_index, len(trajectories) - 1)
    rebuild_selector()
    sync_sliders_with_selected()
    redraw()


# 畫右下角跟著主視角旋轉的 XYZ 方向指示器。
def draw_orientation_axes():
    orientation_ax.clear()
    orientation_ax.set_facecolor(FIGURE_BG)
    orientation_ax.set_axis_off()
    orientation_ax.set_xlim(-1.0, 1.0)
    orientation_ax.set_ylim(-1.0, 1.0)
    orientation_ax.set_zlim(-1.0, 1.0)
    orientation_ax.set_box_aspect((1, 1, 1))
    orientation_ax.view_init(elev=ax.elev, azim=ax.azim)

    axes = [
        ("X", "#8A4443", (0.78, 0, 0), (0.95, 0, 0)),
        ("Y", "#365550", (0, 0.78, 0), (0, 0.95, 0)),
        ("Z", "#355562", (0, 0, 0.78), (0, 0, 0.95)),
    ]

    for label, color, direction, label_position in axes:
        orientation_ax.quiver(
            0,
            0,
            0,
            direction[0],
            direction[1],
            direction[2],
            color=color,
            linewidth=1.8,
            arrow_length_ratio=0.18,
        )
        orientation_ax.text(
            label_position[0],
            label_position[1],
            label_position[2],
            label,
            color=color,
            fontsize=10,
            fontweight="bold",
            ha="center",
            va="center",
        )


# 拖曳主圖旋轉時，同步右下角方向指示器。
def sync_orientation_axes(_event):
    if orientation_ax is None:
        return
    if _event.inaxes is not ax:
        return

    draw_orientation_axes()
    fig.canvas.draw_idle()


# 建立 3D 畫布與左側控制區。
fig = plt.figure(figsize=(13, 8))
fig.patch.set_facecolor(FIGURE_BG)
ax = fig.add_subplot(111, projection="3d")
orientation_ax = fig.add_axes((0.84, 0.08, 0.11, 0.11), projection="3d")
plt.subplots_adjust(left=0.22, bottom=0.10, right=0.98, top=0.92)

build_mode_selector()
rebuild_selector()

# 左側控制元件。
b_axis = plt.axes((0.04, 0.34, 0.13, 0.012), facecolor=SLIDER_TRACK)
v_axis = plt.axes((0.04, 0.28, 0.13, 0.012), facecolor=SLIDER_TRACK)
m_axis = plt.axes((0.04, 0.22, 0.13, 0.012), facecolor=SLIDER_TRACK)
alpha_axis = plt.axes((0.04, 0.16, 0.13, 0.012), facecolor=SLIDER_TRACK)
beta_axis = plt.axes((0.04, 0.10, 0.13, 0.012), facecolor=SLIDER_TRACK)
add_axis = plt.axes((0.04, 0.43, 0.13, 0.045), facecolor=PANEL_BG)
delete_axis = plt.axes((0.04, 0.37, 0.13, 0.045), facecolor=PANEL_BG)

b_slider = Slider(b_axis, "b", -10.0, 10.0, valinit=trajectories[0]["b"], valstep=0.1)
v_slider = Slider(v_axis, "v", 0.1, 0.99, valinit=trajectories[0]["v"])
m_slider = Slider(m_axis, "M", 0.1, 5.0, valinit=INITIAL_M, valstep=0.1)
alpha_slider = Slider(alpha_axis, "alpha", -180.0, 180.0, valinit=trajectories[0]["alpha"], valstep=0.1)
beta_slider = Slider(beta_axis, "beta", -45.0, 45.0, valinit=trajectories[0]["beta"], valstep=0.1)
add_button = Button(add_axis, "Add Trajectory")
delete_button = Button(delete_axis, "Delete Trajectory")

for slider in (b_slider, v_slider, m_slider, alpha_slider, beta_slider):
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

delete_button.label.set_color(TEXT_COLOR)
delete_button.label.set_fontsize(9)
delete_button.color = BUTTON_BG
delete_button.hovercolor = BUTTON_HOVER

b_slider.on_changed(update_selected)
v_slider.on_changed(update_selected)
m_slider.on_changed(redraw)
alpha_slider.on_changed(update_selected)
beta_slider.on_changed(update_selected)
add_button.on_clicked(add_trajectory)
delete_button.on_clicked(delete_trajectory)
fig.canvas.mpl_connect("motion_notify_event", sync_orientation_axes)
fig.canvas.mpl_connect("button_release_event", sync_orientation_axes)

redraw()
draw_orientation_axes()
plt.show()
