# Where to look when a step fails

Match the error text, open the KNOWN-ISSUES entry and apply its fix. Don't paraphrase the fix from memory; read the entry. If two entries could match, read both before acting.

## Error text to KNOWN-ISSUES ID

| You see | Step | ID |
|---|---|---|
| "Context ... not found", or an old `nemo` version | 1 | P1 |
| `nemo setup` wrote to a remote context | 1 | P2 |
| Stopping the services hangs | any | P3 |
| Blank or reasoning-only answer from a Nemotron 3 model | 2, 3, 9 | P4 |
| HTTP 401 from a Python script against a remote Helix | 4 to 9 | P5 |
| Preflight `NVIDIA container runtime  not listed in docker info` | 0 | P6 (blocks steps 7 to 10; don't skip) |
| `[Errno 17] File exists` on `nemo agents create` | 2, 10 | A1 |
| `No module named nemo_fabric_adapters` | 2, 10 | A2 |
| Deploy fails with `artifacts` or `workspace` next to the agent YAML | 2, 10 | A3 |
| 409 "Deployment already exists" | 2, 10 | A4 |
| `host_timeout`, "adapter host did not complete start" | 2, 10 | A5 |
| `__pycache__` warnings on create | 2, 10 | A6 (expected, harmless) |
| Generation slow or rate limited | 3 | D1 |
| "prompt generation template is invalid" | 3 | D2 |
| `preview` times out | 3 | D3 |
| `check-models` times out | 3 | D4 |
| Incidents that give away the answer | 4 | D5 |
| Trouble downloading the generated dataset | 4 | D6 |
| 404 on `nemo files upload` | 5 | F1 |
| `manifest unknown` or not found on the 7a `docker pull` | 7 | T1 (resolved 2026-10-11). No published tag for the installed Helix version; offer the 7a options (a Helix version with images, or the source build fallback) and let the user choose |
| Pull error for `nhx-*` images in the 7b job | 7 | T1. Check the `my-registry/...:local` tags from 7a exist; if they do and it still fails, use the 7a override; ask before restarting the services, and stop and ask if that doesn't fix it |
| `Module 'out_proj' is incompatible with Mamba-based models` | 7 | T2 |
| `warmup_ratio` warning | 7 | T4 (expected) |
| `model spec resolved: False` | 6 | No ID; step 6 "If it fails" says rerun preflight (usually Hugging Face unreachable) |
| Out of GPU memory in training | 7 | No ID; step 7 "If it fails" gives the batch-size change |
| Deployment in PENDING 20+ minutes, no logs | 8 | S2 (wait up to 45 minutes) |
| 422 for the adapter, or the adapter missing from model lists | 8, 9 | S3 |
| "Unknown input fields" with `--input-file` | 8 | S4 |
| Untuned base scores near zero | 9 | S5 (expected) |
| "mismatched fields: ['name']" | 9 | E1 |
| `'item' is undefined` | 9 | E2 |
| Many rows scored 0 with no answer | 9, 10 | E3 |
| Agent evaluation finished with no rows | 10 | E6 |
| Agent run no faster with the fine-tuned model | 10 | A7 (expected: the harness adds 12,000 to 16,000 prompt tokens to every call) |

## Other preflight FAIL lines

The NVIDIA container runtime FAIL is P6, in the table above. The checks below have no KNOWN-ISSUES entry; they print their own `fix:` line. Say there's no ID instead of guessing one.

GPU rule, for every GPU, CUDA and NVIDIA container runtime FAIL: steps 6 to 10 run on the GPU machine. Training (7) and serving (8) run in GPU containers, and steps 9 and 10 call the step 8 deployment, so any of these FAILs blocks 7 to 10. Fix it before starting step 6 there.

| FAIL line | What to tell the user |
|---|---|
| `CUDA` below 13 | Driver update needed (the fix line names the branch). See the GPU rule above. |
| `GPU` under 40 GB, or no GPU | Steps 6 to 10 need another machine. Steps 0 to 5 still work with `--no-gpu`. |
| `OS` not Linux x86_64 in full mode | Same as above. |
| `Disk space` | FAIL means under 16 GB. Under 100 GB on the GPU path is only a WARN (the script's estimate for images and weights). |
| `Docker` | Start Docker Desktop (macOS) or the docker service (Linux), as the fix line says. Step 1 doesn't fix this; local Helix services need Docker even with `--no-gpu`. |
| `uv` | Install uv with the preflight fix line. Step 1 assumes uv is already there. |
| `Hugging Face` not reachable | Outbound HTTPS to huggingface.co is blocked; step 6 will fail with `model spec resolved: False`. |
| `Python`, `nemo CLI`, `Helix services`, `Inference provider`, `Script packages`, `Agent tool server` | Step 1 fixes these. Do step 1, then rerun preflight. |

## Nothing matches

Stop. Show the user the exact error and the command that produced it. Don't invent a workaround. Reading `nemo <command> --help` is fine.
