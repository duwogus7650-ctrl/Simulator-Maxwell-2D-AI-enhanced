"""P6 RL: 설계탐색 환경 + 컴팩트 SAC(torch, CPU).

환경 DesignEnv:
  상태  = [u(5), 현재 D]  (정규화 설계점 + 만족도)
  행동  = Δu ∈ [−0.08, 0.08]^5
  보상  = D(u') − D(u)  (개선 보상) + 종단에서 D 보너스
  에피소드 = 24스텝, 시작점 무작위
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DesignEnv:
    def __init__(self, obj, horizon: int = 24, step_scale: float = 0.08,
                 seed: int = 0):
        self.obj = obj
        self.h = horizon
        self.s = step_scale
        self.rng = np.random.default_rng(seed)
        self.dim = len(obj.keys)

    def reset(self):
        self.u = self.rng.random(self.dim)
        self.t = 0
        self.D = float(self.obj.D(self.u)[0])
        return self._obs()

    def _obs(self):
        return np.concatenate([self.u, [self.D]]).astype(np.float32)

    def step(self, a):
        self.u = np.clip(self.u + self.s * np.asarray(a), 0, 1)
        D2 = float(self.obj.D(self.u)[0])
        r = (D2 - self.D) * 10.0
        self.D = D2
        self.t += 1
        done = self.t >= self.h
        if done:
            r += D2 * 2.0
        return self._obs(), r, done


class MLP(nn.Module):
    def __init__(self, i, o, h=128, out_act=None):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(i, h), nn.ReLU(),
                                 nn.Linear(h, h), nn.ReLU(),
                                 nn.Linear(h, o))
        self.out_act = out_act

    def forward(self, x):
        y = self.net(x)
        return self.out_act(y) if self.out_act else y


class SAC:
    def __init__(self, sdim, adim, seed=0, lr=3e-4, gamma=0.97, tau=0.005):
        torch.manual_seed(seed)
        self.adim = adim
        self.actor = MLP(sdim, adim * 2)
        self.q1 = MLP(sdim + adim, 1)
        self.q2 = MLP(sdim + adim, 1)
        self.q1t = MLP(sdim + adim, 1)
        self.q2t = MLP(sdim + adim, 1)
        self.q1t.load_state_dict(self.q1.state_dict())
        self.q2t.load_state_dict(self.q2.state_dict())
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.opt_q = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr)
        self.log_alpha = torch.tensor(np.log(0.2), requires_grad=True)
        self.opt_al = torch.optim.Adam([self.log_alpha], lr=lr)
        self.target_ent = -adim
        self.gamma, self.tau = gamma, tau

    def _dist(self, s):
        out = self.actor(s)
        mu, log_std = out[..., :self.adim], out[..., self.adim:]
        log_std = torch.clamp(log_std, -5, 2)
        return mu, log_std.exp()

    def act(self, s, deterministic=False):
        with torch.no_grad():
            s = torch.as_tensor(s, dtype=torch.float32)
            mu, std = self._dist(s)
            z = mu if deterministic else mu + std * torch.randn_like(std)
            return torch.tanh(z).numpy()

    def _sample(self, s):
        mu, std = self._dist(s)
        z = mu + std * torch.randn_like(std)
        a = torch.tanh(z)
        logp = (-0.5 * ((z - mu) / std) ** 2 - std.log()
                - 0.5 * np.log(2 * np.pi)).sum(-1)
        logp -= torch.log(1 - a ** 2 + 1e-6).sum(-1)
        return a, logp

    def update(self, batch):
        s, a, r, s2, d = [torch.as_tensor(x, dtype=torch.float32)
                          for x in batch]
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            a2, logp2 = self._sample(s2)
            q2in = torch.cat([s2, a2], -1)
            qt = torch.min(self.q1t(q2in), self.q2t(q2in)).squeeze(-1)
            y = r + self.gamma * (1 - d) * (qt - alpha * logp2)
        qin = torch.cat([s, a], -1)
        lq = F.mse_loss(self.q1(qin).squeeze(-1), y) \
            + F.mse_loss(self.q2(qin).squeeze(-1), y)
        self.opt_q.zero_grad(); lq.backward(); self.opt_q.step()

        an, logp = self._sample(s)
        qa = torch.min(self.q1(torch.cat([s, an], -1)),
                       self.q2(torch.cat([s, an], -1))).squeeze(-1)
        la = (self.log_alpha.exp() * logp - qa).mean()
        self.opt_a.zero_grad(); la.backward(); self.opt_a.step()

        lal = -(self.log_alpha * (logp + self.target_ent).detach()).mean()
        self.opt_al.zero_grad(); lal.backward(); self.opt_al.step()

        for q, qt_ in ((self.q1, self.q1t), (self.q2, self.q2t)):
            for p, pt in zip(q.parameters(), qt_.parameters()):
                pt.data.mul_(1 - self.tau).add_(self.tau * p.data)
        return float(lq), float(la)


def train_sac(obj, steps=30000, seed=0, log_every=3000):
    env = DesignEnv(obj, seed=seed)
    agent = SAC(env.dim + 1, env.dim, seed=seed)
    buf_s, buf_a, buf_r, buf_s2, buf_d = [], [], [], [], []
    s = env.reset()
    best_u, best_D = env.u.copy(), env.D
    ep_returns, ep_ret = [], 0.0
    rng = np.random.default_rng(seed)
    for t in range(steps):
        a = (rng.uniform(-1, 1, env.dim) if t < 1500
             else agent.act(s))
        s2, r, done = env.step(a)
        buf_s.append(s); buf_a.append(a); buf_r.append(r)
        buf_s2.append(s2); buf_d.append(float(done))
        ep_ret += r
        if env.D > best_D:
            best_D, best_u = env.D, env.u.copy()
        s = env.reset() if done else s2
        if done:
            ep_returns.append(ep_ret); ep_ret = 0.0
        if len(buf_s) > 100000:
            for b in (buf_s, buf_a, buf_r, buf_s2, buf_d):
                del b[:50000]
        if t >= 1500 and t % 2 == 0:
            idx = rng.integers(0, len(buf_s), 256)
            batch = (np.asarray(buf_s)[idx], np.asarray(buf_a)[idx],
                     np.asarray(buf_r)[idx], np.asarray(buf_s2)[idx],
                     np.asarray(buf_d)[idx])
            agent.update(batch)
        if (t + 1) % log_every == 0:
            recent = np.mean(ep_returns[-20:]) if ep_returns else 0
            print(f"  step {t+1}: best D={best_D:.4f} | "
                  f"최근 에피소드 보상 {recent:.2f}", flush=True)
    # 학습된 정책으로 다중 시작 롤아웃 → 최종 후보
    cands = []
    for k in range(40):
        s = env.reset()
        for _ in range(env.h):
            s, _, done = env.step(agent.act(s, deterministic=True))
        cands.append((env.D, env.u.copy()))
        if env.D > best_D:
            best_D, best_u = env.D, env.u.copy()
    cands.sort(key=lambda c: -c[0])
    return agent, best_u, best_D, cands
