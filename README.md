<div align="center">
  <!-- Header Banner -->
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0b0f14&height=220&section=header&text=6-DOF%20Crosswind%20Landing%20Sim&fontSize=40&fontColor=22d3ee&animation=twinkle" width="100%" />

  <!-- Animated Typing Subtitle -->
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=20&pause=1000&color=F472B6&center=true&vcenter=true&width=700&lines=%E2%9C%88%FE%8F+Rigid-Body+6-DOF+Flight+Dynamics+%26+Quaternion+Kinematics;%E2%9A%A1+Non-Linear+Trim+Equilibrium+Solver+(SciPy);%F0%9F%8E%AF+Multi-Phase+Closed-Loop+Guidance+(Crab+%E2%86%92+De-Crab+%E2%86%92+Flare)" alt="Typing Subtitle" />

  <br/><br/>

  <!-- Badges -->
  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" />
    <img src="https://img.shields.io/badge/SciPy-1.10%2B-8993C5?style=for-the-badge&logo=scipy&logoColor=white" />
    <img src="https://img.shields.io/badge/NumPy-1.24%2B-013243?style=for-the-badge&logo=numpy&logoColor=white" />
    <img src="https://img.shields.io/badge/Matplotlib-3.7%2B-11557c?style=for-the-badge" />
    <img src="https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge" />
  </p>
</div>

---

## 📽️ Interactive Simulation Preview

<div align="center">
  <img src="./assets/crosswind_landing.gif" alt="6-DOF Crosswind Landing Animation" width="90%" style="border-radius: 8px; border: 1px solid #1e2a36;" />
  <p><i>Real-time dual-view visualization showing runway planform alignment, approach profile, and live telemetry HUD.</i></p>
</div>

---

## 📌 Executive Summary

The **6-DOF Crosswind Landing Simulation Engine** is a high-fidelity flight dynamics framework developed in Python to simulate light aircraft final approach, de-crab alignment, and touchdown flare under variable crosswind conditions.

The framework models full rigid-body dynamic coupling, aerodynamic load derivatives, rotational quaternion kinematics, and a multi-phase closed-loop flight control system designed to satisfy strict civil aviation touchdown safety envelopes.

---

## 📐 Mathematical & Physics Framework

### 1. 6-DOF Rigid-Body Equations of Motion (Body Frame $\mathcal{F}_b$)
Translational acceleration and rotational rates are integrated using a fixed-step 4th-Order Runge-Kutta (**RK4**) algorithm:

$$\begin{bmatrix} \dot{u} \\  \dot{v} \\  \dot{w} \end{bmatrix} = \frac{1}{m} \mathbf{F}_b - \begin{bmatrix} 0 & -r & q \\  r & 0 & -p \\  -q & p & 0 \end{bmatrix} \begin{bmatrix} u \\  v \\  w \end{bmatrix}$$

$$\begin{bmatrix} \dot{p} \\  \dot{q} \\  \dot{r} \end{bmatrix} = \mathbf{I}^{-1} \left( \mathbf{M}_b - \boldsymbol{\omega}_b \times (\mathbf{I} \boldsymbol{\omega}_b) \right)$$

### 2. Quaternion Kinematics (Gimbal Lock Elimination)
Attitude propagation utilizes unit quaternions $\mathbf{q} = [q_0, q_1, q_2, q_3]^T$ to prevent numerical singularities during large Euler angle transitions:

$$\dot{\mathbf{q}} = \frac{1}{2} \mathbf{q} \otimes \begin{bmatrix} 0 \\  p \\  q \\  r \end{bmatrix}$$

### 3. Non-Linear Trim State Equilibrium Solver
Initial approach trim conditions $(\alpha_0, \delta_{e,0}, \delta_{t,0})$ are computed using `scipy.optimize.least_squares` by enforcing state derivative residual minimization:

$$\mathbf{f}(\mathbf{x}, \mathbf{u}) = \begin{bmatrix} \frac{\sum F_x}{W} \\ \frac{\sum F_z}{W} \\ \frac{\sum M_y}{W \cdot \bar{c}} \end{bmatrix}^T \to \mathbf{0}$$

---

## 🕹️ Multi-Phase Guidance State Machine

The closed-loop guidance architecture transitions automatically across four altitude-dependent operational phases:

```text
[ CRAB PHASE ] ──────> [ DE-CRAB / WING-LOW ] ──────> [ FLARE PHASE ] ──────> [ TOUCHDOWN ]
Altitude > 15m           15m >= Altitude > 8m          8m >= Altitude > 0m         Altitude = 0m
Crab angle heading       Rudder aligns heading         Elevator pitch flare        Ground friction &
compensates crosswind.   with runway centerline.       minimizes sink rate.        rollout deceleration.