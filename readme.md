# Kappa Autonomy Simulation

Agent-based simulation of scientific knowledge production under Weibull reliability constraints. Preliminary experiment accompanying a research proposal on measuring the transition from "fast tool" to "autonomous scientist."

## The model in plain language

This simulation connects two empirical findings that haven't been linked before.

**Finding 1: AI agents fail on long tasks in a specific, predictable way.** When you measure how reliably AI agents complete tasks of increasing length, their success rate follows a Weibull distribution with shape parameter κ < 1. This means agents that get through the early steps of a task become progressively *less* likely to fail per step, but their overall success probability still drops sharply with task length. Crucially, κ does not improve when you make the model bigger. It sits at roughly 0.6–0.9 for current frontier models (METR 2025, Hamilton 2026).

**Finding 2: Scientific discoveries are getting harder to make.** Bloom, Jones, Van Reenen & Webb (2020) showed that across semiconductors, agriculture, medicine, and firm-level R&D, the productivity of research effort is declining at 5–10% per year. It takes 18x more researchers today to sustain Moore's Law than it did in the 1970s. Growth only continues because we keep throwing more researchers at the problem. They capture this with an equation from the economic growth literature:

    Ȧ = α · S^λ · A^(1 - β)

where A is the stock of knowledge, S is research effort (number of effective researchers), λ captures diminishing returns to adding more researchers, and β is the "fishing out" exponent: how much harder each successive discovery gets as A grows. Bloom et al. estimate β at roughly 1–4 depending on the domain. At β = 0 (no fishing out), ideas build on ideas freely. At β = 2 (their central estimate), doubling the knowledge stock makes each new discovery roughly four times harder.

**What the simulation does:** It populates a research landscape with AI agents. Each timestep, agents attempt tasks across three difficulty regimes: recombination (120 human-equivalent hours), empirical extension (800h), and new theory (5000h). Whether each agent succeeds depends on the Weibull model: given the task length, the current T₅₀ (which grows on the observed METR trajectory), and κ.

Successful agents contribute to knowledge production via the Bloom et al. equation. As knowledge accumulates, the fishing-out penalty makes each subsequent discovery less valuable. The question is whether T₅₀ growth (which unlocks longer tasks and more valuable discoveries) can outrun this penalty.

We sweep across values of κ and β to find:

1. **Is there a critical κ threshold?** Below which knowledge production accelerates rather than decelerates. (Answer: yes, around κ ≈ 0.35–0.40.)

2. **How do κ and β interact?** β determines *how much* knowledge is ultimately produced; κ determines *when* production begins in earnest. They are independent bottlenecks.

3. **Where is the economic firebreak?** At what task length does an AI agent cost more than a human (accounting for retries due to failure)? This depends critically on κ. At SOTA κ (~0.7), break-even is around 51 hours. At human-level κ (~0.37), it extends to ~264 hours.

## What the simulation shows (and what it deliberately can't)

### What it shows

The simulation produces three concrete outputs.

**Economics.** The cost of deploying an agent on a task of length t scales as (compute per step) × (steps) / P(success). Because P(success) drops with task length (governed by κ), agent costs rise steeply for long tasks. At some task length, the agent costs more than a human. This is the "economic firebreak": the market itself prevents autonomous long-horizon deployment, not because of regulation, but because it is too expensive. The simulation computes this break-even precisely for each κ. At SOTA κ = 0.70 and current T₅₀ = 8h, the break-even sits at about 51 hours. At human-level κ = 0.37, it extends to about 264 hours. The difference is a 5x expansion of the range where autonomous deployment is economically rational.

**Knowledge dynamics.** Plugging AI agents into the Bloom et al. ideas production function lets us track cumulative knowledge over time. There is a genuine phase transition around κ ≈ 0.35–0.40: below this threshold, agents can access the long tasks where valuable discoveries are made, and knowledge accumulates faster in the early years. Above it, agents are confined to short tasks and contribute less.

**Independent bottlenecks.** The κ × β interaction surface (Fig 2) shows that these two parameters constrain knowledge production independently. β determines the ceiling (how much total knowledge the frontier can yield before fishing out dominates). κ determines the floor (how long agents must wait before T₅₀ grows enough to make long tasks viable). You can have low β (easy frontier) with high κ (unreliable agents) or vice versa, and the outcomes are very different.

### What it can't show (and why the experiment is needed)

The simulation has a critical limitation that is, in fact, the reason the proposed experiment exists.

T₅₀ doubles every 89 days on the observed METR trajectory. As T₅₀ grows, P(success) for any fixed-length task approaches 1.0. Given enough time and enough agents, the S^λ term (effective research effort) outpaces the A^(-β) fishing-out penalty, and knowledge production accelerates regardless of κ. In the simulation, final knowledge values across different κ converge by 2032. This is the "brute force" scenario: scale up deployment and you eventually outrun diminishing returns.

But this depends on a hidden assumption: **that completing a task equals generating genuine knowledge.** The simulation credits every successful 800-hour task completion as a Regime B discovery. In reality, an agent might complete 800 hours of sophisticated recombination — connecting known facts in known ways, very fast — without ever forming or testing a genuinely novel hypothesis.

If that is the case, the ideas production function needs a quality modifier:

    Ȧ = α · S^λ · A^(1 - β) · q(κ, regime)

where q is a "knowledge quality" multiplier that is close to 1 for recombination but potentially close to 0 for genuine novel inference. The simulation cannot estimate q because we do not know its value. That is what the proposed experiment (specifically Arm 2, the knowledge regime classification) is designed to measure.

The two scenarios look very different:

- **q ≈ 1 (completing long tasks = real science).** The simulation's optimistic projections hold. The path to autonomous science is primarily gated by κ and economics. Brute-force deployment eventually works. Safety concern: mainly about *when* and *how fast*.

- **q ≈ 0 for novel inference (completing long tasks = fast recombination).** No amount of deployment overcomes the barrier. You get a massive level boost in Regime A output (faster drug screening, faster engineering optimisation, faster literature synthesis) but a wall at Regimes B and C. Safety concern: qualitatively different, because "dangerous autonomous tool" and "autonomous scientist" are separated by a barrier that scaling alone does not cross.

The simulation motivates the experiment by showing that the economics and knowledge dynamics are tractable and interesting, but honestly cannot resolve the question that matters most: is AI-produced knowledge the real thing?

## Parameters

All grounded in empirical literature:

| Parameter | Value | Source |
|-----------|-------|--------|
| T₅₀ (initial) | 8h | METR TH1.1, Feb 2026 |
| T₅₀ doubling | 89 days | METR TH1.1 |
| κ (SOTA) | 0.6–0.9 | METR 2025, Hamilton 2026 |
| κ (human) | ~0.37 | Hamilton 2026 |
| β (fishing out) | 2.0 | Bloom et al. 2020 (central) |
| λ (returns to effort) | 0.6 | Standard in growth literature |
| Human rate | $150/hr | — |
| Agent cost | $0.15/step | Current API pricing |

## Running

```bash
pip install numpy matplotlib
python simulation.py
```

Outputs four figures and a results JSON to `output/`.

## Figures

- **fig1**: κ sweep -> knowledge trajectories, phase transition, break-even vs κ
- **fig2**: κ × β interaction heatmaps (total knowledge and growth acceleration)
- **fig3**: Economic firebreak -> cost ratio curves at different κ and T₅₀
- **fig4**: Regime accessibility -> when each knowledge regime becomes viable

## Context

This builds on:
- Stakenborg (2026), ["On Economics of A(S)I Agents"](https://forum.effectivealtruism.org/posts/dXsBcjCJAKX77Pgsd/on-economics-of-a-s-i-agents), EA Forum
- METR (2025, updated Feb 2026), ["Task-Completion Time Horizons of Frontier AI Models"](https://metr.org/time-horizons/)
- Bloom, Jones, Van Reenen & Webb (2020), "Are Ideas Getting Harder to Find?", *American Economic Review* 110(4), 1104–1144
- Jones (1995), "R&D-Based Models of Economic Growth", *Journal of Political Economy* 103(4), 759–784
- Hamilton (2026), "Peto's Paradox and AI Agents"
- Ord (2024), "Lessons from the Development of the Half-Life of AI"

