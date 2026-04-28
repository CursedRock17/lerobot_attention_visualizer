---
name: lerobot-context
description: Pull authoritative, version-correct info from Context7 (library docs) and the Hugging Face MCP server (Hub + docs + datasets/models). Invoke whenever a question involves lerobot/huggingface APIs, policy configs, VLA/diffusion/flow-matching behavior, or any comparison across library versions (e.g. v0.4.x vs v0.5.x). Also invoke when resolving a dataset/model repo id on the Hub.
---

# lerobot-context

This project depends on a fast-moving stack — `lerobot`, `huggingface_hub`, `transformers`, `datasets`, and VLA/diffusion policies (pi0, smolvla, etc.). Training-flag names, dataclass fields, module paths, and camera-key conventions change between minor tags. **Do not answer from memory** for these topics — use the MCP tools below first.

## When to use

Trigger on:
- Questions about lerobot APIs, training flags, policy configs (pi0, smolvla, pi05, act, diffusion, rtc).
- Anything that references a specific lerobot version, release, or upgrade path.
- Questions about a Hugging Face dataset or model repo (`lerobot/smolvla_base`, `${HF_USER}/...`, etc.).
- Dataset schema questions (feature keys like `observation.images.*`, `action`, `observation.state`).
- Compatibility questions between lerobot and its extras (`[pi]`, `[smolvla]`).

## How to use

**Step 1 — check the installed lerobot version first**, so the docs match the user's actual environment:

```bash
python -c "import lerobot; print(lerobot.__version__)"
# or, inside the vendored submodule:
git -C lerobot describe --tags --always
```

**Step 2 — pull docs:**

- `context7.resolve-library-id` → `huggingface/lerobot` (and transformers / huggingface_hub as needed), then `context7.get-library-docs` at the resolved id, filtered to the topic. Pass the version if Context7 supports it.
- `huggingface.documentation_semantic_search` for narrative questions ("how do I fine-tune smolvla with a custom camera layout?").
- `huggingface.hub_repository_details` for any `repo_id` the user mentions — confirm it exists, read the model/dataset card, check what camera keys or state dims it was trained on.
- `huggingface.model_search` / `dataset_search` when the user is fishing for a starting checkpoint.

**Step 3 — verify against the local tree before recommending.** Memories and docs can lag reality. If you'll recommend a symbol, `Grep` for it in the local lerobot checkout to confirm it exists at the user's version.

## What lives where

| Topic | Prefer |
|---|---|
| lerobot module paths, dataclass fields, CLI flags | Context7 (`huggingface/lerobot`) → then Grep the local checkout |
| Fine-tuning narrative / how-to | HF `documentation_semantic_search` |
| Model/dataset card, camera layout of a checkpoint | HF `hub_repository_details` |
| Cross-version breaking changes | Context7 with version pins + `git diff vA..vB` locally |
| transformers / huggingface_hub / peft / datasets APIs | Context7 (resolve → get-docs) |

## Auth — what the user must export

Both MCP servers are remote HTTP. `.mcp.json` references env vars; if either is missing the server will fail silently. Tell the user to export:

```bash
export CONTEXT7_API_KEY=...   # free tier at context7.com/dashboard; optional but raises rate limits
export HF_TOKEN=hf_...         # from https://huggingface.co/settings/tokens (Read scope is enough)
```

Persist them in `~/.bashrc` / `~/.zshrc` (or a project `.envrc` used with direnv). Do not write these into `.mcp.json` — that file is project-shared.

## Anti-patterns

- Recommending a training flag without first confirming it exists at the installed `lerobot.__version__`.
- Quoting a camera-key convention from memory — always verify against the target checkpoint's model card (HF) and the factory code that loads `input_features`.
- Citing v0.4.x module paths when the user is on v0.5.x (or vice-versa). The `so101_follower → so_follower` and `datasets.utils.hw_to_dataset_features → datasets.feature_utils.hw_to_dataset_features` moves are the canonical gotchas here.
