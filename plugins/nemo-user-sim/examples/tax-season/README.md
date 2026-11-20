<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Tax season sample: Demo Tax Helper

A sample agent and ten scenarios for trying NeMo UserSim on an agent that uses a real tool.

Demo Tax Helper answers everyday questions about 2026 US federal taxes over several turns. It looks current IRS guidance up with Tavily search. It's a demo, not a tax tool: it gives general information from public IRS guidance and isn't tax advice.

The scenarios are questions and life events that fit any adult: which credits might apply, whether you have to file at all, side gig income, moving states, a marriage or divorce, tips and overtime, the deduction for people 65 and older, school and student loans, retirement savings and owing more than you can pay. The simulated person supplies the details from who they are, so the same question plays out differently for a 24-year-old cashier and a 70-year-old retiree. Several have "that doesn't apply to you" as the honest answer, which catches an agent that assumes instead of asking.

## What you need

- NeMo Helix with the UserSim plugin and US personas (`nemo user-sim setup`).
- The Tavily plugin with a key in the workspace: `nemo tavily setup`. A free Tavily key is enough for small runs. Expect a few searches per conversation.

## Run it

```bash
nemo agents register --name demo-tax-helper --agent-config plugins/nemo-user-sim/examples/tax-season/agent.yaml
```

```bash
nemo agents deploy --agent demo-tax-helper
```

```bash
nemo tavily enable demo-tax-helper
```

```bash
nemo user-sim run --agent demo-tax-helper --scenarios plugins/nemo-user-sim/examples/tax-season --users 20 --use-agent-tools
```

`--use-agent-tools` keeps the agent's own tools, here Tavily search, during the run. Without it, the plugin cuts an agent off from its own tools, and the helper can only answer from memory. Turn it on only for tools that are safe to call for real, like search.

Then review the conversations in Studio, give them a thumbs up or down and export the good ones as training data.

## Files

- `agent.yaml`: the agent config. Tavily is attached with `nemo tavily enable`, not declared in the file.
- `topics.yaml`: the ten scenarios for the `general_open_ended` probe. Each names the IRS pages it was written from.
