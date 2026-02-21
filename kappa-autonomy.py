"""
Agent-Based Simulation of Scientific Knowledge Production Under Weibull Reliability
==================================================================================

Preliminary experiment for: "When do AI agents become capable of autonomous
knowledge generation, and what measurable parameters gate the transition
from 'fast tool' to 'autonomous scientist'?"

MODEL
-----
The simulation connects the agent reliability literature and the science-of-science
literature.

1. Agent reliability (Weibull model).
   The probability that an AI agent successfully completes a task of length t
   (in human-equivalent hours) follows a Weibull survival function:

       P(success) = exp(-(t / λ)^κ)

   where λ is a scale parameter derived from T₅₀ (the task length at which
   the agent succeeds 50% of the time), and κ is the shape parameter. When
   κ < 1, the hazard rate *declines* over time: agents that survive early
   steps become progressively less likely to fail per step. This is consistent
   with agents that work within their competence zone but cannot adapt when
   the task evolves. Empirically, SOTA models sit at κ ≈ 0.6–0.9, and κ does
   not improve with model scale (METR 2025, Hamilton 2026).

2. Ideas production function (Bloom et al. 2020).
   At the heart of many economic growth models is an equation for how new
   ideas (knowledge, productivity improvements) are produced:

       Ȧ = α · S^λ · A^(1 - β)

   where:
   - A is the stock of existing knowledge
   - Ȧ is the flow of new knowledge (discoveries per unit time)
   - S is research effort (number of effective researchers)
   - λ < 1 captures diminishing returns to effort (two researchers are
     not twice as productive as one)
   - β > 0 is the "fishing out" exponent: as A grows, each successive
     discovery becomes harder. Higher β means ideas get harder to find
     faster.

   Equivalently, research *productivity* (Ȧ/A per researcher) is:

       Research productivity = α · S^(λ-1) · A^(-β)

   Bloom et al.'s key empirical finding: across semiconductors (Moore's Law),
   agriculture (crop yields), medicine (life expectancy), and firm-level data,
   research productivity is declining at roughly 5–10% per year. Steady growth
   only occurs because S (research spending) has been rising fast enough to
   offset falling productivity. For Moore's Law specifically, it takes 18x
   more researchers today to sustain the doubling rate than it did in the
   early 1970s.

3. How this simulation connects them.
   We populate a research landscape with N AI agents per timestep. Each agent
   attempts a task drawn from one of three regimes (recombination at 120h,
   empirical extension at 800h, new theory at 5000h). Whether the agent
   succeeds depends on the Weibull model: P(success) given the task length,
   the current T₅₀, and κ.

   Successful completions feed into the ideas production function. The number
   of successes gives us S (effective researchers). Each success contributes
   a base amount of new knowledge, weighted by regime (harder tasks produce
   more valuable discoveries). The fishing-out penalty A^(-β) reduces the
   value of each discovery as the knowledge stock grows.

   T₅₀ grows exogenously on the METR trajectory (doubling every ~89 days).
   κ is fixed within each run but swept across runs.

   This lets us ask: given a specific κ, how does cumulative knowledge evolve
   over time? Is there a critical κ below which knowledge production
   accelerates? How does κ interact with β? And where does the economic
   break-even sit — at what task length does deploying an agent become more
   expensive than hiring a human?

KEY QUESTIONS (genuinely uncertain outcomes):
1. Is there a critical κ threshold where knowledge production transitions?
2. How does the κ × β interaction determine sustained acceleration vs level boost?
3. Where exactly does the economic firebreak sit, and how does it shift with κ?

PARAMETER GROUNDING:
- κ: 0.3–1.0 (METR 2025, Hamilton 2026; SOTA ≈ 0.6–0.9, human ≈ 0.37)
- β: 0–4 (Bloom et al. 2020; central estimate ≈ 2)
- λ (returns to effort): 0.5–0.8 (standard in growth literature)
- T₅₀: ~8h for frontier models as of Feb 2026 (METR TH1.1; ~5h for Opus 4.5,
  est. 8–9h for Opus 4.6 / GPT-5.3-Codex), doubling every ~89 days
- Cost: ~$0.10–0.20/step at current API pricing; human researcher ~$100–200/hr

Author: Margot Stakenborg
Date: 21 February 2026
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import json
import os


# ──────────────────────────────────────────────
# Core model
# ──────────────────────────────────────────────

@dataclass
class SimConfig:
    """All tuneable parameters in one place."""

    # ── Agent reliability ──
    kappa: float = 0.70              # Weibull shape (the variable to watch)
    t50_initial: float = 8.0         # Current frontier T₅₀ in hours (METR Feb 2026)
    t50_doubling_days: float = 89.0  # METR TH1.1 doubling time

    # ── Knowledge frontier (Bloom et al. 2020) ──
    beta: float = 2.0                # Fishing-out exponent
    lambda_ideas: float = 0.6        # Returns to research effort
    knowledge_initial: float = 1.0   # Normalised starting knowledge stock

    # ── Research task structure (human-equivalent hours) ──
    regime_a_hours: float = 120      # Recombination: synthesise known results
    regime_b_hours: float = 800      # Empirical extension: new experiments
    regime_c_hours: float = 5000     # New theory: novel conceptual frameworks

    # ── Economics ──
    human_rate: float = 150          # $/hour for a human researcher
    cost_per_step: float = 0.15      # $/step for agent compute
    steps_per_hour: float = 80       # Agent steps per hour of equivalent task

    # ── Simulation ──
    n_agents: int = 1000             # Agents attempting research per timestep
    n_months: int = 72               # 6 years (2026–2032)
    n_seeds: int = 20                # Monte Carlo runs for confidence bands


# ── Weibull mechanics ──

def weibull_lambda(t50: float, kappa: float) -> float:
    """Convert T₅₀ (median task length at 50% success) to Weibull scale λ."""
    return t50 / (np.log(2) ** (1.0 / kappa))


def p_success(task_hours: float, t50: float, kappa: float) -> float:
    """Weibull survival probability for a task of given length."""
    lam = weibull_lambda(t50, kappa)
    return np.exp(-((task_hours / lam) ** kappa))


def expected_cost(task_hours: float, t50: float, kappa: float,
                  cost_per_step: float, steps_per_hour: float) -> float:
    """Expected agent cost for a task, accounting for retries (1/P(success))."""
    ps = p_success(task_hours, t50, kappa)
    if ps < 1e-15:
        return float('inf')
    return steps_per_hour * task_hours * cost_per_step / ps


def human_cost(task_hours: float, human_rate: float) -> float:
    return human_rate * task_hours


def cost_ratio(task_hours: float, t50: float, kappa: float,
               cost_per_step: float, steps_per_hour: float,
               human_rate: float) -> float:
    """Agent cost / human cost. < 1 means agent is cheaper."""
    hc = human_cost(task_hours, human_rate)
    ac = expected_cost(task_hours, t50, kappa, cost_per_step, steps_per_hour)
    if ac == float('inf'):
        return float('inf')
    return ac / hc


def find_breakeven(t50: float, kappa: float, cfg: SimConfig,
                   lo: float = 0.1, hi: float = 50000.0) -> float:
    """Binary search for task length where agent cost = human cost."""
    # Check if agents are always cheaper or always more expensive
    if cost_ratio(lo, t50, kappa, cfg.cost_per_step, cfg.steps_per_hour, cfg.human_rate) >= 1.0:
        return 0.0  # Never cheaper
    if cost_ratio(hi, t50, kappa, cfg.cost_per_step, cfg.steps_per_hour, cfg.human_rate) < 1.0:
        return hi  # Always cheaper within range
    for _ in range(100):
        mid = (lo + hi) / 2
        r = cost_ratio(mid, t50, kappa, cfg.cost_per_step,
                       cfg.steps_per_hour, cfg.human_rate)
        if r < 1.0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ──────────────────────────────────────────────
# Simulation engine
# ──────────────────────────────────────────────

@dataclass
class TimeStep:
    month: int
    t50: float
    knowledge: float
    regime_a_psuccess: float
    regime_b_psuccess: float
    regime_c_psuccess: float
    discoveries: float
    breakeven_hours: float
    economically_viable_a: bool
    economically_viable_b: bool
    economically_viable_c: bool


def run_simulation(cfg: SimConfig, seed: int = 42) -> List[TimeStep]:
    """Run one trajectory of the knowledge production simulation."""
    rng = np.random.default_rng(seed)
    results = []
    knowledge = cfg.knowledge_initial

    for m in range(cfg.n_months):
        # T₅₀ grows on METR trajectory
        t50 = cfg.t50_initial * (2.0 ** (m * 30.44 / cfg.t50_doubling_days))

        # Fishing out (Bloom et al. 2020):
        # As the knowledge stock A grows, each new discovery is worth less.
        # This implements the A^(-β) term from the ideas production function:
        #   Ȧ = α · S^λ · A^(1-β)
        # At β = 2 (Bloom et al. central estimate), doubling the knowledge
        # stock makes each new discovery ~4x harder to achieve.
        productivity_penalty = (knowledge / cfg.knowledge_initial) ** (-cfg.beta)

        # Success probabilities from the Weibull model.
        # Task lengths are fixed by regime (in human-equivalent hours).
        # What changes over time is T₅₀ (the agent gets better at longer tasks)
        # and knowledge (each success is worth less).
        ps_a = p_success(cfg.regime_a_hours, t50, cfg.kappa)
        ps_b = p_success(cfg.regime_b_hours, t50, cfg.kappa)
        ps_c = p_success(cfg.regime_c_hours, t50, cfg.kappa)

        # ── Ideas production function ──
        # S^λ term: effective research effort with diminishing returns.
        # We draw the number of successful agents stochastically, then apply
        # the diminishing returns exponent λ.
        regime_weights = np.array([0.6, 0.3, 0.1])  # Share of agents attempting each regime
        regime_ps = np.array([ps_a, ps_b, ps_c])
        # Base knowledge contribution per success, by regime.
        # Harder tasks (longer horizon) produce more valuable knowledge.
        regime_base_contribution = np.array([0.01, 0.05, 0.2])

        n_per_regime = (cfg.n_agents * regime_weights).astype(int)
        successes = np.array([
            rng.binomial(n_per_regime[i], np.clip(regime_ps[i], 0, 1))
            for i in range(3)
        ])

        # S^λ: diminishing returns to total research effort.
        # If 100 agents succeed, they don't produce 100x the output of 1 agent.
        effective_researchers = np.sum(successes)
        effort_factor = (effective_researchers ** cfg.lambda_ideas) if effective_researchers > 0 else 0

        # Total raw knowledge output (before fishing-out penalty)
        raw_output = np.sum(successes * regime_base_contribution)

        # Final discoveries = S^λ × raw_output × A^(-β)
        # This is the discrete-time version of Ȧ = α · S^λ · A^(1-β)
        discoveries = effort_factor * raw_output * max(productivity_penalty, 1e-15)
        knowledge += discoveries

        # Economics
        be = find_breakeven(t50, cfg.kappa, cfg)

        results.append(TimeStep(
            month=m, t50=t50, knowledge=knowledge,
            regime_a_psuccess=ps_a, regime_b_psuccess=ps_b,
            regime_c_psuccess=ps_c, discoveries=discoveries,
            breakeven_hours=be,
            economically_viable_a=expected_cost(cfg.regime_a_hours, t50, cfg.kappa, cfg.cost_per_step, cfg.steps_per_hour) < human_cost(cfg.regime_a_hours, cfg.human_rate),
            economically_viable_b=expected_cost(cfg.regime_b_hours, t50, cfg.kappa, cfg.cost_per_step, cfg.steps_per_hour) < human_cost(cfg.regime_b_hours, cfg.human_rate),
            economically_viable_c=expected_cost(cfg.regime_c_hours, t50, cfg.kappa, cfg.cost_per_step, cfg.steps_per_hour) < human_cost(cfg.regime_c_hours, cfg.human_rate),
        ))

    return results


def run_ensemble(cfg: SimConfig) -> List[List[TimeStep]]:
    return [run_simulation(cfg, seed=s) for s in range(cfg.n_seeds)]


# ──────────────────────────────────────────────
# Experiments
# ──────────────────────────────────────────────

def experiment_kappa_sweep(base_cfg: SimConfig):
    """Sweep κ from 0.30 to 1.00."""
    kappas = np.array([0.30, 0.33, 0.35, 0.37, 0.40, 0.43, 0.45,
                       0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80,
                       0.85, 0.90, 0.95, 1.00])
    results = {}

    for k in kappas:
        k = round(k, 2)
        cfg = SimConfig(**{**base_cfg.__dict__, 'kappa': k})
        ensemble = run_ensemble(cfg)

        knowledge_arr = np.array([[r.knowledge for r in run] for run in ensemble])

        results[k] = {
            'knowledge_mean': knowledge_arr.mean(axis=0),
            'knowledge_p10': np.percentile(knowledge_arr, 10, axis=0),
            'knowledge_p90': np.percentile(knowledge_arr, 90, axis=0),
            'final_knowledge_mean': knowledge_arr[:, -1].mean(),
            'final_knowledge_std': knowledge_arr[:, -1].std(),
        }

    return kappas, results


def experiment_kappa_beta_interaction(base_cfg: SimConfig):
    """Sweep κ × β."""
    kappas = np.arange(0.35, 0.95, 0.05)
    betas = np.arange(0.0, 4.5, 0.5)

    final_knowledge = np.zeros((len(betas), len(kappas)))
    knowledge_growth_rate = np.zeros((len(betas), len(kappas)))

    for i, b in enumerate(betas):
        for j, k in enumerate(kappas):
            cfg = SimConfig(**{**base_cfg.__dict__,
                              'kappa': round(k, 2), 'beta': round(b, 1), 'n_seeds': 5})
            ensemble = run_ensemble(cfg)
            knowledge_arr = np.array([[r.knowledge for r in run] for run in ensemble])

            final_knowledge[i, j] = knowledge_arr[:, -1].mean()
            early = knowledge_arr[:, :12].mean()
            late = knowledge_arr[:, -12:].mean()
            knowledge_growth_rate[i, j] = (late - early) / max(early, 1e-10)

    return kappas, betas, final_knowledge, knowledge_growth_rate


def experiment_regime_accessibility(base_cfg: SimConfig):
    """Track P(success) and viability for each regime at select κ."""
    kappas_select = [0.37, 0.50, 0.60, 0.70, 0.85]
    results = {}

    for k in kappas_select:
        cfg = SimConfig(**{**base_cfg.__dict__, 'kappa': k})
        run = run_simulation(cfg, seed=42)

        results[k] = {
            'months': [r.month for r in run],
            'years': [2026 + r.month / 12 for r in run],
            't50': [r.t50 for r in run],
            'regime_a_p': [r.regime_a_psuccess for r in run],
            'regime_b_p': [r.regime_b_psuccess for r in run],
            'regime_c_p': [r.regime_c_psuccess for r in run],
            'breakeven': [r.breakeven_hours for r in run],
            'knowledge': [r.knowledge for r in run],
            'viable_a': [r.economically_viable_a for r in run],
            'viable_b': [r.economically_viable_b for r in run],
        }

    return results


# ──────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────

PALETTE = {
    'kappa_low': '#1b6ca8',
    'kappa_mid': '#3a9e5c',
    'kappa_high': '#c44d2b',
    'regime_a': '#2d7a4f',
    'regime_b': '#3568a8',
    'regime_c': '#7c5cbf',
    'human': '#666666',
}

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
})


def plot_fig1_kappa_sweep(kappas, results, base_cfg, output_dir):
    """Figure 1: Knowledge production, phase transition, and breakeven vs κ."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel A: Knowledge trajectories
    ax = axes[0]
    colors = plt.cm.coolwarm(np.linspace(0, 1, 6))
    for idx, k in enumerate([0.37, 0.45, 0.55, 0.65, 0.75, 0.90]):
        k = round(k, 2)
        if k in results:
            months = np.arange(len(results[k]['knowledge_mean']))
            years = 2026 + months / 12
            lw = 2.5 if k == 0.37 else 1.5
            ax.plot(years, results[k]['knowledge_mean'], label=f'κ={k}',
                    linewidth=lw, color=colors[idx])
            ax.fill_between(years, results[k]['knowledge_p10'],
                           results[k]['knowledge_p90'], alpha=0.07, color=colors[idx])
    ax.set_xlabel('Year')
    ax.set_ylabel('Knowledge stock (normalised)')
    ax.set_title('A. Knowledge production over time')
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.15)

    # Panel B: Final knowledge vs κ
    ax = axes[1]
    ks = sorted(results.keys())
    finals = [results[k]['final_knowledge_mean'] for k in ks]
    stds = [results[k]['final_knowledge_std'] for k in ks]
    ax.errorbar(ks, finals, yerr=stds, fmt='o-', color='#c44d2b',
                markersize=4, capsize=3, linewidth=1.5)
    ax.axvline(x=0.37, color=PALETTE['human'], linestyle='--', alpha=0.6, label='Human κ ≈ 0.37')
    ax.axvspan(0.6, 0.9, alpha=0.06, color=PALETTE['kappa_high'], label='Current SOTA range')
    ax.set_xlabel('Weibull κ')
    ax.set_ylabel('Final knowledge stock (6 years)')
    ax.set_title('B. Phase transition in κ?')
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.15)

    # Panel C: Breakeven task length vs κ (at initial T₅₀)
    ax = axes[2]
    bes = [find_breakeven(base_cfg.t50_initial, k, base_cfg) for k in ks]
    ax.plot(ks, bes, 'o-', color='#3568a8', markersize=4, linewidth=1.5)
    ax.axvline(x=0.37, color=PALETTE['human'], linestyle='--', alpha=0.6, label='Human κ')
    ax.axvspan(0.6, 0.9, alpha=0.06, color=PALETTE['kappa_high'], label='Current SOTA')
    ax.axhline(y=120, color=PALETTE['regime_a'], linestyle=':', alpha=0.5, label='Regime A (120h)')
    ax.axhline(y=800, color=PALETTE['regime_b'], linestyle=':', alpha=0.5, label='Regime B (800h)')
    ax.set_xlabel('Weibull κ')
    ax.set_ylabel('Break-even task length (hours)')
    ax.set_title(f'C. Economic firebreak at T₅₀ = {base_cfg.t50_initial}h')
    ax.set_yscale('log')
    ax.legend(fontsize=7, framealpha=0.9)
    ax.grid(alpha=0.15)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig1_kappa_sweep.png'), dpi=180, bbox_inches='tight')
    plt.close()


def plot_fig2_kappa_beta(kappas, betas, final_knowledge, growth_rate, output_dir):
    """Figure 2: κ × β interaction surface."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    im = ax.imshow(np.log10(final_knowledge + 1), aspect='auto', origin='lower',
                   extent=[kappas[0], kappas[-1], betas[0], betas[-1]], cmap='RdYlGn')
    ax.set_xlabel('Weibull κ')
    ax.set_ylabel('Fishing-out β')
    ax.set_title('A. Log₁₀(final knowledge stock)')
    plt.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[1]
    im = ax.imshow(growth_rate, aspect='auto', origin='lower',
                   extent=[kappas[0], kappas[-1], betas[0], betas[-1]],
                   cmap='RdBu', vmin=-1, vmax=5)
    ax.set_xlabel('Weibull κ')
    ax.set_ylabel('Fishing-out β')
    ax.set_title('B. Knowledge growth: late / early ratio')
    plt.colorbar(im, ax=ax, shrink=0.8, label='Growth ratio (>1 = accelerating)')
    try:
        ax.contour(kappas, betas, growth_rate, levels=[1.0], colors='black',
                   linewidths=2, linestyles='--')
    except Exception:
        pass

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig2_kappa_beta.png'), dpi=180, bbox_inches='tight')
    plt.close()


def plot_fig3_economic_firebreak(base_cfg, output_dir):
    """Figure 3: The economic firebreak — cost ratio curves."""
    task_hours = np.logspace(np.log10(0.5), np.log10(2000), 400)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Panel A: Cost ratio vs task length at different κ (fixed T₅₀)
    ax = axes[0]
    kappas_plot = [0.37, 0.50, 0.60, 0.70, 0.85, 1.0]
    styles = ['-', '-', '-', '-', '--', ':']
    cmap = plt.cm.coolwarm(np.linspace(0, 1, len(kappas_plot)))

    for idx, (k, ls) in enumerate(zip(kappas_plot, styles)):
        ratios = np.array([
            min(cost_ratio(h, base_cfg.t50_initial, k, base_cfg.cost_per_step,
                          base_cfg.steps_per_hour, base_cfg.human_rate), 1e5)
            for h in task_hours
        ])
        lw = 2.5 if k in [0.37, 0.70] else 1.3
        ax.plot(task_hours, ratios, ls, label=f'κ = {k}', linewidth=lw, color=cmap[idx])

    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    ax.text(0.6, 0.52, 'Break-even', fontsize=9, color='gray',
            transform=ax.get_yaxis_transform())

    # Mark regime boundaries
    for rh, label, col in [(120, 'A', PALETTE['regime_a']),
                           (800, 'B', PALETTE['regime_b']),
                           (5000, 'C', PALETTE['regime_c'])]:
        ax.axvline(x=rh, color=col, linestyle=':', alpha=0.3)
        ax.text(rh, 1e4, f' {label}', fontsize=8, color=col, va='top')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Task length (hours)')
    ax.set_ylabel('Agent cost / Human cost')
    ax.set_title(f'A. Cost ratio at T₅₀ = {base_cfg.t50_initial}h')
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.15)
    ax.set_ylim(0.01, 1e5)

    # Panel B: How firebreak evolves as T₅₀ grows
    ax = axes[1]
    t50s = [8, 16, 32, 64, 128]
    alphas = [1.0, 0.85, 0.7, 0.55, 0.4]

    for t50, alpha in zip(t50s, alphas):
        ratios_hi = np.array([
            min(cost_ratio(h, t50, 0.70, base_cfg.cost_per_step,
                base_cfg.steps_per_hour, base_cfg.human_rate), 1e5)
            for h in task_hours
        ])
        ax.plot(task_hours, ratios_hi, '-', alpha=alpha, linewidth=2,
                color=PALETTE['kappa_high'])

        ratios_lo = np.array([
            min(cost_ratio(h, t50, 0.37, base_cfg.cost_per_step,
                base_cfg.steps_per_hour, base_cfg.human_rate), 1e5)
            for h in task_hours
        ])
        ax.plot(task_hours, ratios_lo, '--', alpha=alpha, linewidth=1.5,
                color=PALETTE['kappa_low'])

    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)

    legend_elements = [
        Line2D([0], [0], color=PALETTE['kappa_high'], linewidth=2, label='κ = 0.70 (SOTA)'),
        Line2D([0], [0], color=PALETTE['kappa_low'], linewidth=1.5,
               linestyle='--', label='κ = 0.37 (human)'),
    ]
    for t50, alpha in zip(t50s, alphas):
        legend_elements.append(
            Line2D([0], [0], color='gray', linewidth=1, alpha=alpha,
                   label=f'T₅₀ = {t50}h'))
    ax.legend(handles=legend_elements, fontsize=7, framealpha=0.9, loc='upper left')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Task length (hours)')
    ax.set_ylabel('Agent cost / Human cost')
    ax.set_title('B. Firebreak erosion: κ = 0.70 (solid) vs κ = 0.37 (dashed)')
    ax.grid(alpha=0.15)
    ax.set_ylim(0.01, 1e5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig3_economic_firebreak.png'), dpi=180, bbox_inches='tight')
    plt.close()


def plot_fig4_regime_access(regime_results, output_dir):
    """Figure 4: Regime accessibility over time at different κ."""
    kappas_select = sorted(regime_results.keys())

    fig, axes = plt.subplots(1, len(kappas_select),
                             figsize=(4 * len(kappas_select), 4.5), sharey=True)

    for idx, k in enumerate(kappas_select):
        ax = axes[idx]
        data = regime_results[k]
        years = data['years']

        ax.plot(years, data['regime_a_p'], color=PALETTE['regime_a'],
                linewidth=2, label='Regime A (120h)')
        ax.plot(years, data['regime_b_p'], color=PALETTE['regime_b'],
                linewidth=2, label='Regime B (800h)')
        ax.plot(years, data['regime_c_p'], color=PALETTE['regime_c'],
                linewidth=1.5, linestyle='--', label='Regime C (5000h)')

        # Mark where each regime becomes economically viable
        for field, colour, rlabel in [
            ('viable_a', PALETTE['regime_a'], 'A'),
            ('viable_b', PALETTE['regime_b'], 'B'),
        ]:
            flags = data[field]
            for i, v in enumerate(flags):
                if v:
                    ax.axvline(x=years[i], color=colour, alpha=0.3, linewidth=1)
                    ax.text(years[i] + 0.1, 0.95, f'{rlabel}$',
                            fontsize=8, color=colour, fontweight='bold')
                    break

        ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.3)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f'κ = {k}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Year')
        if idx == 0:
            ax.set_ylabel('P(success)')
        ax.grid(alpha=0.12)
        if idx == 0:
            ax.legend(fontsize=7, loc='upper left', framealpha=0.9)

    plt.suptitle('Regime accessibility over time ($ = economically viable)',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig4_regime_access.png'), dpi=180, bbox_inches='tight')
    plt.close()


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    base_cfg = SimConfig()

    print("=" * 65)
    print("  Agent-Based Simulation of Scientific Knowledge Production")
    print("  Under Weibull Reliability Constraints")
    print("=" * 65)
    print(f"\n  Base parameters:")
    print(f"    T₅₀ (initial)     = {base_cfg.t50_initial}h (METR Feb 2026)")
    print(f"    T₅₀ doubling      = {base_cfg.t50_doubling_days} days")
    print(f"    κ (SOTA range)    = 0.6 – 0.9")
    print(f"    κ (human)         = 0.37")
    print(f"    β (fishing out)   = {base_cfg.beta}")
    print(f"    Human rate        = ${base_cfg.human_rate}/hr")
    print(f"    Agent cost/step   = ${base_cfg.cost_per_step}")

    # ── Experiment 1: κ sweep ──
    print("\n[1/4] κ sweep (0.30 → 1.00)...")
    kappas, kappa_results = experiment_kappa_sweep(base_cfg)
    plot_fig1_kappa_sweep(kappas, kappa_results, base_cfg, output_dir)
    print("  → fig1_kappa_sweep.png")

    print("\n  Final knowledge by κ:")
    for k in [0.37, 0.50, 0.60, 0.70, 0.85]:
        k = round(k, 2)
        if k in kappa_results:
            r = kappa_results[k]
            print(f"    κ = {k}: {r['final_knowledge_mean']:.2f} ± {r['final_knowledge_std']:.2f}")

    print("\n  Break-even task length at T₅₀ = {:.0f}h:".format(base_cfg.t50_initial))
    for k in [0.37, 0.50, 0.60, 0.70, 0.85]:
        be = find_breakeven(base_cfg.t50_initial, k, base_cfg)
        print(f"    κ = {k}: {be:.1f}h")

    # ── Experiment 2: κ × β interaction ──
    print("\n[2/4] κ × β interaction sweep...")
    ks, bs, fk, gr = experiment_kappa_beta_interaction(base_cfg)
    plot_fig2_kappa_beta(ks, bs, fk, gr, output_dir)
    print("  → fig2_kappa_beta.png")

    # ── Experiment 3: Economic firebreak ──
    print("\n[3/4] Economic firebreak plots...")
    plot_fig3_economic_firebreak(base_cfg, output_dir)
    print("  → fig3_economic_firebreak.png")

    # ── Experiment 4: Regime accessibility ──
    print("\n[4/4] Regime accessibility over time...")
    regime_results = experiment_regime_accessibility(base_cfg)
    plot_fig4_regime_access(regime_results, output_dir)
    print("  → fig4_regime_access.png")

    print("\n  Regime viability (first month):")
    for k in sorted(regime_results.keys()):
        data = regime_results[k]
        a_month = next((i for i, v in enumerate(data['viable_a']) if v), None)
        b_month = next((i for i, v in enumerate(data['viable_b']) if v), None)
        a_str = f"month {a_month} ({2026 + a_month/12:.1f})" if a_month is not None else "never (6yr)"
        b_str = f"month {b_month} ({2026 + b_month/12:.1f})" if b_month is not None else "never (6yr)"
        print(f"    κ = {k}: Regime A viable {a_str}, Regime B viable {b_str}")

    # ── Save results ──
    summary = {
        'parameters': {
            't50_initial': base_cfg.t50_initial,
            't50_doubling_days': base_cfg.t50_doubling_days,
            'beta': base_cfg.beta,
            'human_rate': base_cfg.human_rate,
            'cost_per_step': base_cfg.cost_per_step,
        },
        'kappa_sweep': {
            str(k): {
                'final_knowledge_mean': float(kappa_results[k]['final_knowledge_mean']),
                'final_knowledge_std': float(kappa_results[k]['final_knowledge_std']),
            } for k in sorted(kappa_results.keys())
        },
        'breakeven_at_t50_8h': {
            str(k): float(find_breakeven(base_cfg.t50_initial, k, base_cfg))
            for k in [0.37, 0.50, 0.60, 0.70, 0.85]
        },
    }

    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n  All outputs saved to {output_dir}/")
    print("=" * 65)


if __name__ == '__main__':
    main()
