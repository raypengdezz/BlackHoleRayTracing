import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

class massive_geodesics:
    def __init__(self, b, v, M, step, max_step):
        """
        b : impact parameter
        v : velocity (when c=1)
        M : blackhole Mass
        E : energy
        L : angular momentum
        r_crtical : critical distance

        """
        if v >= 1:
            raise ValueError("The speed of a particle should not exceed the speed of light")

        self.gamma = 1 / np.sqrt(1 - v ** 2)
        self.E = self.gamma
        self.L = -self.gamma * b * v

        self.central_mass = M

        self.direction = -1

        self.r = 50
        self.phi = np.pi - np.arcsin(b / self.r)

        self.r_critical = self._critical_radius(M)

        self.x_positions = [] 
        self.y_positions = []

        self.step = step
        self.max_step = max_step
        
        self._run_trajectory()

    def get_cartesian(self):
        x = self.r * np.cos(self.phi)
        y = self.r * np.sin(self.phi)

        return x, y
    
    # 三次多項式係數
    
    def _critical_radius(self, M):
        a = 2 * M * self.L ** 2
        b = - self.L ** 2
        c = 2 * M
        d = self.E ** 2 - 1
    
        coeffs = [d, c, b, a]

        roots = np.roots(coeffs)
        real_roots = roots[np.isreal(roots)]

        return np.max(np.real(real_roots)).item() # turning point

    def in_critical(self,r):
        return r < self.r_critical
    
    def _run_trajectory(self):
        total_step = 0
        step = self.step
        max_step = self.max_step

        while (self.r <= 50) and (total_step < max_step) and (self.r >= 2 * self.central_mass):
            dr = np.sqrt(np.abs(self.E ** 2 - (1 - 2 * self.central_mass / self.r) * (1 + (self.L ** 2) / (self.r ** 2) ))) * step

            dphi = self.L / self.r ** 2 * step

            if self.in_critical(self.r - dr):
                self.direction = +1

            self.r += self.direction * dr

            self.phi += dphi

            self.phi = (self.phi + np.pi) % (2 * np.pi) - np.pi 

            x, y = self.get_cartesian()

            self.x_positions.append(x)
            self.y_positions.append(y)

            total_step += 1

fig, ax = plt.subplots(figsize = (6, 6))

ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)

# all parameters
M = 1
step = 0.01
max_step = int(1e4)
r = 2 * M


circle = Circle((0, 0), r, color = "black")

test = massive_geodesics(7, 0.9, M, step = step, max_step = max_step)

ax.add_patch(circle)
ax.plot(test.x_positions, test.y_positions, color = "#1f77b4")

plt.savefig("test_massive.png")
