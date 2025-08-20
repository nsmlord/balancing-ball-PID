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
dt = 0.0005
simTime = 100
steps = int(simTime / dt)
g = np.array([0.0, -9.81])  # Gravity vector in x-z plane
mu = 0.30  # Kinetic friction
coefRest = 1.0  # Coefficient of restitution
R = 0.20  # Radius
mass = 1.0
Icoeff = 2/5  # Solid sphere
I = Icoeff * mass * R**2

# general line equation: n . x + d = 0
# where n is the unit normal vector
# x = (x,z), belonging to every point on the line
# d is a constant chosen to pass through a specific point

# equation with our variables: nSlope . x + dSlope = 0
# we know slope line passes through point (0, planeH)
# replace x for that point and we can then solve for dSlope

slopeAng = 15 * pi / 180
planeH = 4.0
prismLen = 10.0

# slope direction vector: < cos(slopeAng), -sin(slopeAng) >
nSlope = np.array([sin(slopeAng), cos(slopeAng)]) # slope normal vector (check using dot product)
nSlope = nSlope / norm(nSlope) # to guarantee unit vector (lucked out)
dSlope = -planeH * nSlope[1] # point's coordinate is zero, thus using only [1]


# Body dictionary
body = {
    'pos': np.array([0.0, planeH + R]),
    'vel': np.array([0.0, 0.0]),
    'angVel': 0.0,
    'R': R,
    'mass': mass,
    'I': I
}

# Collision function
def collide_plane(obj, n, d, len_, mu, e):
    # Check if within slope bounds (in 2D, only x)
    if obj['pos'][0] >= 0 and obj['pos'][0] <= len_:
        dist = dot(n, obj['pos']) + d - obj['R'] # calculate obj/plane intersection rule

        if dist < 0: # obj is intersecting
            obj['pos'] = obj['pos'] - dist * n # push the obj out orthogonally to the slope
            r = -obj['R'] * n # vector from center of sphere to contact point
            vcp = obj['vel'] + np.array([-obj['angVel'] * r[1], obj['angVel'] * r[0]]) # contact point velocity
            vn = dot(vcp, n) # normal velocity (velocity into or out of the plane)
            vt_vec = vcp - vn * n # tangential velocity
            vt_mag = norm(vt_vec)

            if vt_mag > 1e-6: # if tangential velocity is noticeable, treat it as sliding
                vt_hat = vt_vec / vt_mag # unit vector along contact's tangential motion

                k_t = 1 / obj['mass'] + (dot(r, r) - dot(r, n)**2) / obj['I']

                Jn = -(1 + e) * vn * obj['mass'] # normal impulse for bounce
                Jt = min(mu * abs(Jn), vt_mag / k_t) # friction impulse capped by coloumbs law

                impulse = Jn * n - Jt * vt_hat # total impulse vector to be delivered to obj
            else: # no-slipping case
                Jn = -(1 + e) * vn * obj['mass'] # normal impulse for bounce
                impulse = Jn * n

            obj['vel'] += impulse / obj['mass'] # linear velocity change Δv = J / m
            torque = r[0] * impulse[1] - r[1] * impulse[0] # torque = r × impulse (cross product for 2d)
            obj['angVel'] += torque / obj['I'] # angular velocity changes by Δω = torque / I.
    return obj

# Simulate and collect data for animation
positions = []
angles = []
angle = 0.0
display_every = 40  # Update animation every 40 simulation steps for performance

for step in range(steps):
    body['vel'] += dt * g
    body['pos'] += dt * body['vel']
    body = collide_plane(body, nSlope, dSlope, prismLen, mu, coefRest)
    angle += body['angVel'] * dt
    if step % display_every == 0:
        positions.append(body['pos'].copy())
        angles.append(angle)

# Set up the figure and animation
fig, ax = plt.subplots() # create a figure and one axes
ax.set_xlim(-1, prismLen + 1) # bounds for x-axis
slope_end_z = planeH - prismLen * np.tan(slopeAng) # compute slope end height
ax.set_ylim(slope_end_z - 1, planeH + 1) # bounds for y-axis
ax.set_aspect('equal') # aspect ratio is 1:1 to maintain a circular ball
ax.set_xlabel('X') # x-axis label
ax.set_ylabel('Z (down)') # y-axis label

# Plot slope
x_slope = [0, prismLen] # slope x-dir vector
z_slope = [planeH, slope_end_z] # slope z-dir vector
ax.plot(x_slope, z_slope, 'k-', linewidth=2) # combine for line segment

# Ball
circle = plt.Circle((0, 0), R, color='blue', alpha=0.8)
ax.add_patch(circle) # add to axes

# Line to show rotation
line, = ax.plot([], [], 'r-', linewidth=2) # will be anchored to ball center in animation phase

def init():
    circle.center = (positions[0][0], positions[0][1])
    return circle, line

def animate(i):
    pos = positions[i] # ball center
    circle.center = (pos[0], pos[1]) # update circle center

    # create the line with desired R length and direction
    theta = angles[i]
    dx = R * np.cos(theta)
    dz = R * np.sin(theta)
    # (dx, dz) is a point on the circumference of the ball in the current spin direction.

    # set line endpoints
    line.set_data([pos[0], pos[0] + dx], [pos[1], pos[1] + dz])
    return circle, line

anim = FuncAnimation(fig, animate, init_func=init, frames=len(positions), interval=20, blit=True)

plt.show()