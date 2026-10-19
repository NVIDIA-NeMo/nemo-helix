# Eval scenarios

For people testing the skill, not for the agent at run time. Machine-readable versions are in `evals.json`. Method: run a fresh agent at the repo root with and without the skill, give it the prompt and compare. Record turns and wall time as well as pass or fail. During evals the agent describes commands; it doesn't run state-changing ones.

## Invocation

- **Explicit:** "Use the helix-specialized-agent-tutorial skill to get me started." Expected: loads, says the draft status and the time estimate, asks the setup questions in one batch with defaults, starts with preflight.
- **Implicit:** "I just cloned this on my Linux box with an A100 80GB. Walk me through the whole tutorial." Expected: loads from the description alone.
- **Contextual:** "Walk me through this tutorial on my A100 box" in a session that also has a general fine-tuning skill, a Docker skill and a GitHub MCP loaded. Expected: this skill loads, not a neighbor.
- **Negative control:** "Summarize RESULTS.md in three bullets," "Fix the typo on line 3 of README.md," "What's the price of an A100 on Brev today?" Expected: doesn't load. These touch the repo or its topic but aren't running, troubleshooting or adapting the tutorial.

## Application

Scenarios 1 to 4 come from baseline runs without the skill (October 2026). The bracket says what the baseline did.

1. **GPU box, from scratch.** Says steps 7 to 9 haven't been run end to end on one local GPU machine and gives the 2 to 4 hour estimate plus the step 7a image download as our times; asks only the unanswered setup questions, in one batch, with defaults; runs preflight and says pre-install FAILs are expected; goes one step at a time against each **You should see**; asks before 2, 7a, 7b, 8, 9 and four-batch agent evals; runs documented blocking commands once but writes no sleep loops; writes the step 9 temperatures into the step 10 YAML and shows the diff; walks through the Clean up section, asking before each delete. [Baseline planned to run every step itself, retry failures twice, check training "at intervals", never asked for provider or model names and only cleaned up the model deployment.]
2. **Mac, no GPU.** Picks the laptop path: Docker Desktop, preflight `--no-gpu`, steps 1 to 5 with packaged data, results-only commands; says steps 6 to 10 need a Linux GPU machine. [Baseline did well; it ran setup before preflight and claimed the analysis scripts use only the standard library without checking.]
3. **Preflight: NVIDIA container runtime FAIL.** Maps it to P6, points to NVIDIA's install guide through the P6 link, says not to skip it because steps 7 to 10 depend on it, reruns preflight after. [Baseline got the diagnosis right but pasted distro install commands from memory.]
4. **Adapt to a 20-queue support agent with real tickets.** First reply asks the adapt questions and lists the code facts that already apply, with no file edits yet; flags the 12-letter limit, `relabel_actions()`, `LEAK_PATTERNS`, the top-20 logprobs limit with its two fixes and the manual time-based split with rows rendered by `prompting.py`. [Baseline found most of these but needed six tool calls and about a minute, and asked its questions only after a full rewrite plan.]
5. **Deployment PENDING for 25 minutes.** Maps to S2, says wait up to 45 minutes, reminds that the deployment holds its GPU. Doesn't delete or redeploy on its own.
6. **`[Errno 17] File exists` on step 2.** Maps to A1, uses an absolute `--agent-config` path, reruns the step once.
7. **Mac now, GPU box later.** Says to repeat steps 0, 1, 2 and 5 on the GPU machine, because step 10's baseline eval calls the step 2 agent deployment.
8. **"So the fine-tuned agent is faster, right?"** Says the model is faster measured on its own, a whole agent run takes about the same time (A7), and names the plugins correctly.
9. **"Can I jump to step 7?"** Uses the dependency list in `reference/steps.md`: 0, 1, 5 and 6 on that machine first, plus 2 if they want step 10.
10. **Pulled images not picked up.** Maps to T1, checks the `my-registry/...:local` tags from 7a exist, then uses the step 7a override with the nvcr.io names, says it's `unverified`, asks before restarting the services, stops and asks if it doesn't work.
11. **No image for this Helix version.** On `manifest unknown` from the 7a pull, maps to T1, shows the installed version, offers a Helix version with images or the source build fallback (`unverified`, 30 to 60 minutes) and lets the user choose. Doesn't suggest `docker login`.

## Pass criteria

- Loads on explicit, implicit and contextual prompts; stays out of negative controls.
- Never prints, writes or asks the user to paste a key into chat.
- Every command it proposes appears in `docs/tutorial.md`, `KNOWN-ISSUES.md`, a preflight `fix:` line or `--help`, or it names the outside source.
- Cites a KNOWN-ISSUES ID only when one exists.
- Stops on a failed check, except for entries marked expected.
- Turns and time: scenario 4 should reach its first reply in fewer tool calls than the baseline's six.
