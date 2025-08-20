import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from math import sin, cos, pi


# Helper functions
def dot(a, b):
    return np.dot(a, b)


def norm(v):
    return np.linalg.norm(v)


# Simulation constants
dt = 0.002
simTime = 10.0
steps = int(simTime / dt)
g = np.array([0.0, -9.81])  # Gravity vector in x-z plane
mu = 0.30  # Kinetic friction
coefRest = 1.0  # Coefficient of restitution
R = 0.20  # Radius
mass = 1.0
Icoeff = 2 / 5  # Solid sphere
I = Icoeff * mass * R ** 2
proportional = 0.1  # Proportional gain

# Slope specification
slopeAng = 15 * pi / 180  # Initial slope angle
mid_x = 5.0
mid_z = 3.0
reference = np.array([mid_x, mid_z])
slopeLen = 10.0
mid = np.array([mid_x, mid_z])


# Function to update slope geometry based on angle
def update_slope_geometry(angle):
    tHat = np.array([cos(angle), -sin(angle)])  # Unit tangent
    tHat = tHat / norm(tHat)
    nSlope = np.array([sin(angle), cos(angle)])  # Unit normal
    nSlope = nSlope / norm(nSlope)
    dSlope = -dot(nSlope, mid)  # Line equation constant
    halfL = 0.5 * slopeLen
    P1 = mid - halfL * tHat
    P2 = mid + halfL * tHat
    return tHat, nSlope, dSlope, P1, P2


# Initial slope geometry
tHat, nSlope, dSlope, P1, P2 = update_slope_geometry(slopeAng)

# Body dictionary
Phigh = P1 if P1[1] > P2[1] else P2
body = {
    'pos': Phigh + R * nSlope,
    'vel': np.array([0.0, 0.0]),
    'angVel': 0.0,
    'R': R,
    'mass': mass,
    'I': I
}


def collide_plane(obj, n, d, mid, tHat, seg_len, mu, e):
    s = dot(obj['pos'] - mid, tHat)
    if abs(s) <= 0.5 * seg_len:
        dist = dot(n, obj['pos']) + d - obj['R']
        if dist < 0:
            obj['pos'] = obj['pos'] - dist * n
            r = -obj['R'] * n
            vcp = obj['vel'] + np.array([-obj['angVel'] * r[1], obj['angVel'] * r[0]])
            vn = dot(vcp, n)
            vt_vec = vcp - vn * n
            vt_mag = norm(vt_vec)
            if vt_mag > 1e-6:
                vt_hat = vt_vec / vt_mag
                k_t = 1 / obj['mass'] + (dot(r, r) - dot(r, n) ** 2) / obj['I']
                Jn = -(1 + e) * vn * obj['mass']
                Jt = min(mu * abs(Jn), vt_mag / k_t)
                impulse = Jn * n - Jt * vt_hat
            else:
                Jn = -(1 + e) * vn * obj['mass']
                impulse = Jn * n
            obj['vel'] += impulse / obj['mass']
            torque = r[0] * impulse[1] - r[1] * impulse[0]
            obj['angVel'] += torque / obj['I']
    return obj


# Simulate and collect data
positions = []
angles = []
angle = 0.0
display_every = 10  # Reduced for smoother animation
slope_angles = []  # Track slope angle over time

for step in range(steps):
    # Compute error: projection of position error onto slope tangent
    pos_error = reference - body['pos']
    error_along_slope = dot(pos_error, tHat)

    # Proportional controller: adjust slope angle
    slopeAng += proportional * error_along_slope * dt  # Scale by dt for stability
    slopeAng = np.clip(slopeAng, -45 * pi / 180, 45 * pi / 180)  # Limit angle range

    # Update slope geometry
    tHat, nSlope, dSlope, P1, P2 = update_slope_geometry(slopeAng)

    # Update physics
    body['vel'] += dt * g
    body['pos'] += dt * body['vel']
    body = collide_plane(body, nSlope, dSlope, mid, tHat, slopeLen, mu, coefRest)
    angle += body['angVel'] * dt

    if step % display_every == 0:
        positions.append(body['pos'].copy())
        angles.append(angle)
        slope_angles.append(slopeAng)

# Plot setup
fig, ax = plt.subplots()
xmin = reference[0] - 5
xmax = reference[0] + 5
zmin = reference[1] - 5
zmax = reference[1] + 5
ax.set_xlim(xmin, xmax)
ax.set_ylim(zmin, zmax)
ax.set_aspect('equal')
ax.set_xlabel('X')
ax.set_ylabel('Z (down)')

# Plot initial slope segment
slope_line, = ax.plot([P1[0], P2[0]], [P1[1], P2[1]], 'k-', linewidth=2)
# Plot reference point
ax.plot(reference[0], reference[1], 'ro', label='Reference Point')
# Ball
circle = plt.Circle((positions[0][0], positions[0][1]), R, color='blue', alpha=0.8)
ax.add_patch(circle)
# Rotation line
line, = ax.plot([], [], 'r-', linewidth=2)
# Legend
ax.legend()


def init():
    circle.center = (positions[0][0], positions[0][1])
    line.set_data([], [])
    slope_line.set_data([P1[0], P2[0]], [P1[1], P2[1]])
    return circle, line, slope_line


def animate(i):
    pos = positions[i]
    circle.center = (pos[0], pos[1])
    theta = angles[i]
    dx = R * np.cos(theta)
    dz = R * np.sin(theta)
    line.set_data([pos[0], pos[0] + dx], [pos[1], pos[1] + dz])
    # Update slope
    tHat, nSlope, dSlope, P1, P2 = update_slope_geometry(slope_angles[i])
    slope_line.set_data([P1[0], P2[0]], [P1[1], P2[1]])
    return circle, line, slope_line


anim = FuncAnimation(fig, animate, init_func=init, frames=len(positions), interval=20, blit=True)
plt.show()