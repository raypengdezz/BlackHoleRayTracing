import numpy as np

class massive_geodesics:
    def __init__(self, b, v, M):
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

        self.ingoing = True

        self.r = 50
        self.phi = np.pi - np.arcsin(b / self.r)

        self.r_critical = self._critical_radius(M)

        self.x_positions = [] 
        self.y_positions = []

        self._run_trajectory()

    def get_cartesian(self):
        x = self.r * np.cos(self.phi)
        y = self.r * np.sin(self.phi)

        return x, y
    
    def _critical_radius(self, M):
        a = 2 * M * self.L ** 2
        b = - self.L ** 2
        c = 2 * M
        d = self.E ** 2 - 1
    
        coeffs = [d, c, b, a]

        roots = np.roots(coeffs)
        real_roots = roots[np.isreal(roots)]

        return np.max(np.real(real_roots)).item()

    def arrive_critical(self):
        return abs(self.r - self.r_critical) > 1e-4
    
    def _run_trajectory(self, step = 0.01, max_step = 100000):
        total_step = 0

        while (self.r <= 50) and (total_step < max_step) and (self.r >= 2 * self.central_mass):
            dr = np.sqrt(self.E ** 2 - (1 - 2 * self.central_mass / self.r) * (1 + self.L ** 2 / self.r ** 2)) * step

            dphi = self.L / self.r ** 2 * step

            if self.ingoing: 
                self.r -= dr
                self.ingoing = self.arrive_critical()

            else:
                self.r += dr

            self.phi += dphi

            self.phi = (self.phi + np.pi) % (2 * np.pi) - np.pi 

            x, y = self.get_cartesian()

            self.x_positions.append(x)
            self.y_positions.append(y)

            total_step += 1
