# DeepSeek V4.1 regression suite

A quick check (about 10 minutes on 8 GPUs) that a build still does what the fork's V4.1 work relies on:

| Section | Checks |
| --- | --- |
| `ops` | `test-backend-ops` against the CPU for the V4.1 ops: top-k and prefix masks, Lightning Indexer, HC, IQ and Q8_0 matmuls, 512-wide flash attention |
| `server` | one `llama-server` started with the short command (DSpark draft, vision, steering vectors; the server's V4.1 defaults set the rest: 4 slots in the shared KV pool): greedy tokens, decode speed and draft acceptance, 16K prompt tokens and prefill speed, the `/speculative` switch (same tokens without drafts), `/steering` (changes the output, unloading restores it), a second request forked from the prefix of a generating one, a generating slot that keeps going during a 16K prompt, three conversations at once in the pool and an idle slot that keeps its KV, an image question |
| `ui` | the web UI unit tests, and in a browser against the same server: the speculative toggle, the steering menu, a chat answer, page errors |

```sh
python3 tools/regress-v41/run.py                  # all sections
python3 tools/regress-v41/run.py --only ops,ui    # no model needed for ops and the unit tests
python3 tools/regress-v41/run.py --update         # accept the current results as the reference
```

The first run writes the reference (`reference.json`: prompts, tokens, speeds); later runs compare
against it. Tokens must match exactly, speeds must stay above `REGRESS_SPEED_TOL` (0.85) of the
reference. A change that is meant to change the numbers (a new layout from `--fit`, a kernel that
sums in a different order) shows up as token differences: score it, then run with `--update`. The
exit code is 1 if any check fails. Logs of every run are kept next to the reference.

## Configuration

| Variable | Default |
| --- | --- |
| `REGRESS_BUILD` | `build/bin` of this checkout |
| `REGRESS_MODEL` | `~/models/ud-iq2xxs/DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS-00001-of-00006.gguf` |
| `REGRESS_DRAFT` | the DSpark Q8_0 draft in the Hugging Face cache (no draft: those checks are skipped) |
| `REGRESS_MMPROJ`, `REGRESS_IMAGE`, `REGRESS_IMAGE_EXPECT` | `mmproj-F16.gguf` in the cache, `tools/mtmd/test-1.jpeg`, a regex for the answer |
| `REGRESS_STEER_DIR`, `REGRESS_STEER_VEC` | `~/dev/ds4/refusal-dir`, `glp27-l11-37.f32` |
| `REGRESS_SERVER_ARGS` | `-c 526000 -t 12 -tb 12` (the server's defaults set the rest) |
| `REGRESS_REF` | `~/.cache/llama-regress/v41` |
| `REGRESS_PORT`, `REGRESS_OPS_BACKEND`, `REGRESS_SPEED_TOL` | `18490`, `CUDA0`, `0.85` |

The server section needs the GPUs to itself: it stops if another `llama-server` is running. Build
`llama-server` and `test-backend-ops` first, and run `npm ci` in `tools/ui` for the UI section
(`npx playwright install chromium` for the browser checks).
