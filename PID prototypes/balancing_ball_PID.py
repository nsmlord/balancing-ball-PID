import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from math import sin, cos, pi


# Helper functions
def dot(a, b):
    return np.dot(a, b)


def norm(v):
    return np.linalg.norm(v)


# Simulation constants
dt = 0.001
simTime = 100.0
steps = int(simTime / dt)
g = np.array([0.0, -9.81])  # Gravity vector in x-z plane
mu = 0.30  # Kinetic friction
coefRest = 0.0  # Coefficient of restitution
R = 0.20  # Radius
mass = 1.0
Icoeff = 2 / 5  # Solid sphere
I = Icoeff * mass * R ** 2
proportional = 0.011  # Proportional gain
differential = 0.36  # differential gain
integral = 0.08  # integral gain

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
display_every = 150  # Reduced for smoother animation
slope_angles = []  # Track slope angle over time
error = []
p_output = []  # Proportional component
i_output = []  # Integral component
d_output = []  # Derivative component
times = []

# Initialize PID controller variables
error_integral = 0.0
previous_error = 0.0

for step in range(steps):
    # Compute error: projection of position error onto slope tangent
    pos_error = reference - body['pos']
    error_along_slope = dot(pos_error, tHat)
    error.append(error_along_slope)

    # Compute error derivative
    if step == 0:
        error_dot = 0.0
    else:
        error_dot = (error_along_slope - previous_error) / dt

    # Compute error integral using trapezoidal rule
    if step == 0:
        error_integral = 0.0
    else:
        error_integral += 0.5 * (error_along_slope + previous_error) * dt

    # PID controller components
    p_term = proportional * error_along_slope
    i_term = integral * error_integral
    d_term = differential * error_dot

    # Store individual components
    p_output.append(p_term)
    i_output.append(i_term)
    d_output.append(d_term)

    # Total control signal
    control_signal = p_term + i_term + d_term

    # Apply control signal to slope angle
    slopeAng += control_signal * dt  # Integrate control signal to get angle change

    # Optional: Add limits to prevent extreme slope angles
    slopeAng = np.clip(slopeAng, -pi / 3, pi / 3)  # Limit to ±60 degrees

    # Store time
    times.append(step * dt)

    # Update slope geometry with new angle
    tHat, nSlope, dSlope, P1, P2 = update_slope_geometry(slopeAng)

    # Update physics
    body['vel'] += dt * g
    body['pos'] += dt * body['vel']
    body = collide_plane(body, nSlope, dSlope, mid, tHat, slopeLen, mu, coefRest)
    angle += body['angVel'] * dt

    # Store previous error for next iteration
    previous_error = error_along_slope

    if step % display_every == 0:
        positions.append(body['pos'].copy())
        angles.append(angle)
        slope_angles.append(slopeAng)

# Create gridspec layout
fig = plt.figure(figsize=(15, 10))
gs = gridspec.GridSpec(3, 3, figure=fig)

# Main simulation subplot [0:2, 0:2]
ax_sim = fig.add_subplot(gs[0:2, 0:2])
xmin = reference[0] - 5
xmax = reference[0] + 5
zmin = reference[1] - 5
zmax = reference[1] + 5
ax_sim.set_xlim(xmin, xmax)
ax_sim.set_ylim(zmin, zmax)
ax_sim.set_aspect('equal')
ax_sim.set_xlabel('X')
ax_sim.set_ylabel('Z (down)')
ax_sim.set_title('Ball Simulation with PID Controller')

# Plot initial slope segment
slope_line, = ax_sim.plot([P1[0], P2[0]], [P1[1], P2[1]], 'k-', linewidth=2)
# Plot reference point
ax_sim.plot(reference[0], reference[1], 'ro', label='Reference Point')
# Ball
circle = plt.Circle((positions[0][0], positions[0][1]), R, color='blue', alpha=0.8)
ax_sim.add_patch(circle)
# Rotation line
line, = ax_sim.plot([], [], 'r-', linewidth=2)
# Legend
ax_sim.legend()

# Top right: Proportional controller output
ax_p = fig.add_subplot(gs[0, 2])
p_line, = ax_p.plot([], [], 'b-')
ax_p.set_xlabel('Time (s)')
ax_p.set_ylabel('P Output')
ax_p.set_title('Proportional Controller Output')
ax_p.grid(True)
ax_p.set_xlim(0, simTime)
if len(p_output) > 0:
    ax_p.set_ylim(min(p_output) * 1.1, max(p_output) * 1.1)

# Middle right: Integral controller output
ax_i = fig.add_subplot(gs[1, 2])
i_line, = ax_i.plot([], [], 'g-')
ax_i.set_xlabel('Time (s)')
ax_i.set_ylabel('I Output')
ax_i.set_title('Integral Controller Output')
ax_i.grid(True)
ax_i.set_xlim(0, simTime)
if len(i_output) > 0:
    ax_i.set_ylim(min(i_output) * 1.1, max(i_output) * 1.1)

# Bottom right: Derivative controller output
ax_d = fig.add_subplot(gs[2, 2])
d_line, = ax_d.plot([], [], 'r-')
ax_d.set_xlabel('Time (s)')
ax_d.set_ylabel('D Output')
ax_d.set_title('Derivative Controller Output')
ax_d.grid(True)
ax_d.set_xlim(0, simTime)
if len(d_output) > 0:
    ax_d.set_ylim(min(d_output) * 1.1, max(d_output) * 1.1)

# Bottom left: Error vs time
ax_error = fig.add_subplot(gs[2, 0])
error_line, = ax_error.plot([], [], 'purple')
ax_error.set_xlabel('Time (s)')
ax_error.set_ylabel('Error')
ax_error.set_title('Error vs Time')
ax_error.grid(True)
ax_error.set_xlim(0, simTime)
if len(error) > 0:
    ax_error.set_ylim(min(error) * 1.1, max(error) * 1.1)

# Bottom middle: Slope angle vs time
ax_slope = fig.add_subplot(gs[2, 1])
slope_line_plot, = ax_slope.plot([], [], 'm-')
ax_slope.set_xlabel('Time (s)')
ax_slope.set_ylabel('Slope Angle (rad)')
ax_slope.set_title('Slope Angle vs Time')
ax_slope.grid(True)
ax_slope.set_xlim(0, simTime)
if len(slope_angles) > 0:
    ax_slope.set_ylim(min(slope_angles) * 1.1, max(slope_angles) * 1.1)

# Adjust layout to prevent overlap
plt.tight_layout()


def init():
    circle.center = (positions[0][0], positions[0][1])
    line.set_data([], [])
    slope_line.set_data([P1[0], P2[0]], [P1[1], P2[1]])
    p_line.set_data([], [])
    i_line.set_data([], [])
    d_line.set_data([], [])
    error_line.set_data([], [])
    slope_line_plot.set_data([], [])
    return circle, line, slope_line, p_line, i_line, d_line, error_line, slope_line_plot


def animate(i):
    # Update main simulation
    pos = positions[i]
    circle.center = (pos[0], pos[1])
    theta = angles[i]
    dx = R * np.cos(theta)
    dz = R * np.sin(theta)
    line.set_data([pos[0], pos[0] + dx], [pos[1], pos[1] + dz])
    # Update slope
    tHat, nSlope, dSlope, P1, P2 = update_slope_geometry(slope_angles[i])
    slope_line.set_data([P1[0], P2[0]], [P1[1], P2[1]])

    # Update live plots - show data up to current animation frame
    current_step = i * display_every
    current_time = current_step * dt

    # Current data up to animation frame
    current_times = times[:current_step]
    current_p = p_output[:current_step]
    current_i = i_output[:current_step]
    current_d = d_output[:current_step]
    current_error = error[:current_step]

    # PID component plots
    p_line.set_data(current_times, current_p)
    i_line.set_data(current_times, current_i)
    d_line.set_data(current_times, current_d)

    # Error plot
    error_line.set_data(current_times, current_error)

    # Slope angle plot
    current_display_times = [j * display_every * dt for j in range(i + 1)]
    current_slope_angles = slope_angles[:i + 1]
    slope_line_plot.set_data(current_display_times, current_slope_angles)

    return circle, line, slope_line, p_line, i_line, d_line, error_line, slope_line_plot


anim = FuncAnimation(fig, animate, init_func=init, frames=len(positions), interval=20, blit=True)
plt.show()