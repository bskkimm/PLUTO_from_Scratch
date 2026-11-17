"""Educational batched kinematic bicycle and finite-horizon LQR tracking.

States are [x,y,yaw,speed], controls are [longitudinal acceleration, steering
angle], in metres, radians and seconds. This simulator is independent of the
nuPlan/PDM implementation and is not validated as an official replacement.
"""

import torch


def _wrap(angle):
    return torch.atan2(angle.sin(), angle.cos())


def bicycle_step(state, control, dt=0.1, wheelbase=2.7):
    if dt <= 0 or wheelbase <= 0:
        raise ValueError("dt and wheelbase must be positive")
    x, y, yaw, speed = state.unbind(-1)
    accel, steer = control.unbind(-1)
    return torch.stack(
        (
            x + speed * yaw.cos() * dt,
            y + speed * yaw.sin() * dt,
            _wrap(yaw + speed * steer.tan() * dt / wheelbase),
            (speed + accel * dt).clamp(min=0.0),
        ),
        dim=-1,
    )


def lqr_rollout(initial, trajectory, dt=0.1, wheelbase=2.7):
    """Track [B,K,T,6] plans from [B,4], returning [B,K,T+1,4] states.

    A finite-horizon LQR is computed around the trajectory's kinematic
    linearization using fixed diagonal state/control costs. Feedforward controls
    are inferred from reference velocity and yaw, then bounded to a plausible
    range. This diagnostic implementation does not model actuator lag or
    replicate the official PDM tracker.
    """
    if dt <= 0 or wheelbase <= 0 or trajectory.ndim != 4 or trajectory.shape[-1] < 6:
        raise ValueError("invalid trajectory, dt or wheelbase")
    b, k, horizon, _ = trajectory.shape
    if initial.shape != (b, 4) or horizon < 1:
        raise ValueError("initial must be [B,4] and horizon must be nonzero")
    yaw = torch.atan2(trajectory[..., 3], trajectory[..., 2])
    speed = torch.linalg.vector_norm(trajectory[..., 4:6], dim=-1)
    future = torch.cat(
        (trajectory[..., :2], yaw.unsqueeze(-1), speed.unsqueeze(-1)), dim=-1
    )
    reference = torch.cat((initial[:, None, None, :].expand(b, k, 1, 4), future), dim=2)
    current, nxt = reference[:, :, :-1], reference[:, :, 1:]
    acceleration = (nxt[..., 3] - current[..., 3]) / dt
    yaw_delta = _wrap(nxt[..., 2] - current[..., 2])
    steering = torch.atan2(yaw_delta * wheelbase, current[..., 3].clamp(min=1e-3) * dt)
    reference_control = torch.stack(
        (acceleration.clamp(-6.0, 4.0), steering.clamp(-0.6, 0.6)), dim=-1
    )
    eye4 = torch.eye(4, dtype=trajectory.dtype, device=trajectory.device)
    q = torch.diag(trajectory.new_tensor([1.0, 4.0, 2.0, 1.0]))
    r = torch.diag(trajectory.new_tensor([0.5, 1.0]))
    p = q.expand(b, k, 4, 4).clone()
    gains = [None] * horizon
    for t in range(horizon - 1, -1, -1):
        ref = current[:, :, t]
        steer = reference_control[:, :, t, 1]
        v, theta = ref[..., 3], ref[..., 2]
        a = eye4.expand(b, k, 4, 4).clone()
        a[..., 0, 2] = -v * theta.sin() * dt
        a[..., 0, 3] = theta.cos() * dt
        a[..., 1, 2] = v * theta.cos() * dt
        a[..., 1, 3] = theta.sin() * dt
        a[..., 2, 3] = steer.tan() * dt / wheelbase
        control_matrix = trajectory.new_zeros((b, k, 4, 2))
        control_matrix[..., 3, 0] = dt
        control_matrix[..., 2, 1] = (
            v * dt / (wheelbase * steer.cos().square().clamp(min=1e-4))
        )
        bt = control_matrix.transpose(-1, -2)
        gain = torch.linalg.solve(r + bt @ p @ control_matrix, bt @ p @ a)
        gains[t] = gain
        p = q + a.transpose(-1, -2) @ p @ (a - control_matrix @ gain)
    state = reference[:, :, 0]
    states = [state]
    for t in range(horizon):
        error = state - reference[:, :, t]
        error = torch.cat((error[..., :2], _wrap(error[..., 2:3]), error[..., 3:]), -1)
        control = reference_control[:, :, t] - (gains[t] @ error.unsqueeze(-1)).squeeze(
            -1
        )
        control = torch.stack(
            (control[..., 0].clamp(-6.0, 4.0), control[..., 1].clamp(-0.6, 0.6)), -1
        )
        state = bicycle_step(state, control, dt, wheelbase)
        states.append(state)
    return torch.stack(states, dim=2)


def make_rollout_evaluator(initial, scorer, dt=0.1, wheelbase=2.7):
    """Adapt a rollout scorer to `planner.select_trajectory`'s evaluator API.

    Scorer receives rollout states [B,K,T+1,4] and a valid-candidate mask
    [B,K] and must return finite [B,K] scores for valid candidates.
    """

    def evaluate(proposals, valid):
        rollouts = lqr_rollout(initial, proposals, dt=dt, wheelbase=wheelbase)
        return scorer(rollouts, valid)

    return evaluate
