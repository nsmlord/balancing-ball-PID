# Ball Balancing Platform Simulation with PID Control

This project is a computer simulation of a ball balancing platform controlled by a PID controller. The simulation models the dynamics of a ball rolling on a platform and uses different controllers (P, PD, PID) to stabilize the ball at a desired position.

The simulation includes progressively more advanced controllers, as well as testing environments for both physics and tuning.

---

## Repository Structure

### 📂 PID prototypes
- **balancing_ball_proportional.py**  
  Simulation with only the proportional controller.

- **balancing_ball_PD.py**  
  Simulation with the proportional and derivative controller.

- **balancing_ball_PID.py**  
  Simulation with proportional, integral, and derivative controllers, though not necessarily tuned properly.

---

### 📂 physics engine
- **balancing_ball_proto1_slope.py**  
  Initial physics engine test for the simulation, focusing on basic slope behavior.

---

### 📂 source_
- **balancing_ball_PID_tuning.py**  
  Simulation with PID controllers tuned to the best extent possible.

- **BB_PID_playground.py**  
  Experimental simulation with controllers tuned in a wayward manner, with no specific rationale.

---

## Physics Engine Reference

Details of the physics engine underlying this simulation can be found in the appendix of the PDF included in the **references** folder. This PDF was written for a different project but contains all relevant mathematical and physical modeling details used here.

---

## How to Run

Each file can be executed independently to test the specific control strategy it demonstrates.

Example:
```bash
python balancing_ball_PID_tuning.py
