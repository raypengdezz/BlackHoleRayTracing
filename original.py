import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from geodesics.massive_geodesics import massive_geodesics

M = 1 #blackhole

fig, ax = plt.subplots(figsize = (6, 6))

r = 2 * M # Schwarzschild radius
circle = Circle((0, 0), r, color = "#181818")

ax.add_patch(circle)

all_parameters = [
    (1, 0.9),
    (5, 0.8),
    (7, 0.6)
]

for parameters in all_parameters:
    b, v = parameters

    traj = massive_geodesics(b, v, M)

    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

    ax.plot(
        traj.x_positions, 
        traj.y_positions, 
        color = "#8B8888",
        linewidth = 2,
        linestyle = "-"
    )

plt.savefig("test.png")
