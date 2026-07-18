# Multi-Teacher Synthetic Data Distillation: Failure Modes in Small Model Fine-Tuning

**Training hardware:** NVIDIA RTX 5090 32GB, AMD Ryzen 7 9700X (8c/16t), 74GB RAM
**Software stack:** Ollama (local inference), Unsloth (QLoRA fine-tuning), HF Transformers, llama.cpp GGUF export
**Date:** July 2026

## Abstract

Distilling a suite of small (0.5B-1.5B parameter) specialist models from multiple large local "teacher" LLMs is a practical way to build narrow-domain models without cloud API costs, but the approach has several non-obvious failure modes that silently degrade training data quality without raising errors. This paper documents seven concrete bugs found while building a 7-model IT-operations diagnostic suite, three of which caused a *worse-performing* model to ship without any indication of failure — the pipeline completed successfully, produced a valid GGUF, and passed export, while the resulting model was measurably degraded or actively wrong.

## Key Finding #1: Template-Fill Failures Are Silent, Not Loud

**The Problem:** A synthetic-data generator that fills scenario templates with randomized values (`"SMART WARNING on {host}: {attr} = {val}"`) will silently write the literal placeholder string into training data if a referenced key is missing from the fill-value dictionary — there is no error, no exception, just a malformed training example that looks superficially fine in a spot-check.

```python
FILL_VALUES = {"host": [...], "val": [...]}  # "attr" missing
template = "SMART WARNING on {host}: {attr} = {val}"
# Silent result: "SMART WARNING on DC-01: {attr} = 47"
```

**Impact measured:** In one domain, 949 of 2,003 generated pairs (47%) contained at least one unfilled placeholder. These pairs still passed length and JSON-validity filters downstream, because the filters checked *shape*, not *content*.

**The Fix:** A permanent regex guard at the dataset-build stage, independent of the generator:

```python
PLACEHOLDER_RE = re.compile(r"\{[a-z_]+\}")
if PLACEHOLDER_RE.search(human) or PLACEHOLDER_RE.search(gpt):
    continue  # reject silently-contaminated rows
```

**Implication:** Any template-based synthetic generation pipeline needs a *downstream* content-level guard, not just generator-level testing. The generator can be fixed and re-verified, but stale contaminated data from before the fix will still exist on disk and needs an independent filter to catch it on every build, not just at generation time.

## Key Finding #2: Fuzzy Deduplication Can Collapse Distinct Examples

**The Problem:** A common deduplication strategy hashes only the first N words of an example to catch near-duplicates cheaply. For short-form outputs (e.g., structured JSON classifications), this over-collapses:

```python
def fuzzy_hash(text):
    normalized = " ".join(text.lower().split()[:30])  # BUG: truncates
    return hashlib.md5(normalized.encode()).hexdigest()
```

**Impact measured:** For a short-alert classification domain, this collapsed 2,003 raw pairs to 323 unique examples (84% loss) — most of the loss was not genuine duplication, it was the hash function ignoring everything past word 30 of already-short inputs and thus treating structurally different examples as identical.

**The Fix:** Hash the full input+output pair, not a truncated prefix:

```python
def fuzzy_hash(text):
    normalized = " ".join(text.lower().split())  # full text, no truncation
    return hashlib.md5(normalized.encode()).hexdigest()

h = fuzzy_hash(human_turn + "||" + model_turn)  # include output, not just input
```

This alone recovered the collapse from 84% to 75% loss on the same dataset — improvement, but not a full fix, revealing that the underlying issue was compounded by limited template diversity (see Finding #4).

**Implication:** For short-output domains (classification, structured extraction), prefer full-string or output-inclusive hashing. Word-prefix truncation is a reasonable heuristic for long-form text but actively harmful for terse outputs.

## Key Finding #3: Teacher Model Selection Matters More Than Volume

**The Problem:** When generating "diversity" data by mixing multiple teacher models for stylistic variation, it is tempting to treat all available local models interchangeably — pick whichever is fast and has spare capacity. This breaks down badly for code generation specifically: a general-purpose chat model asked to write PowerShell/Bash produces *plausible-looking but factually incorrect* code, because it lacks the tool-specific training a dedicated code model has.

**Concrete example** — a general chat model, asked to check SMART disk health via WMI, produced:

```powershell
ReallocatedSectorCount = ($Drive | Get-WmiObject Win32_DiskPartition).DiskSize `
                          - ($Drive | Get-WmiObject Win32_LogicalDisk).FreeSpace
```

This computes a disk-space delta, which has no relationship to SMART reallocated sector counts. It is not a stylistic weakness — it is fabricated logic that would pass a superficial code review (correct PowerShell syntax, plausible variable names, follows the expected structure) while being functionally nonsensical.

**Measured impact:** With 35% of a code-generation training set sourced from the general chat model, the resulting fine-tuned model scored 0/3 on a functional smoke test. Replacing that 35% with a dedicated code-agent model (keeping everything else — data volume, base model, training config — identical) brought the same smoke test to 3/3.

**Implication:** For narrow technical domains (code generation, structured data extraction, domain-specific tool usage), teacher model *fit* dominates teacher model *scale* or generation speed. A smaller, task-matched teacher will out-teach a larger general-purpose one on tasks requiring precise, tool-specific correctness.

## Key Finding #4: Checkpoint Export Selection Can Silently Ship an Undertrained Model

**The Problem:** A common pattern for exporting "the best checkpoint" reads `best_model_checkpoint` from a training checkpoint's `trainer_state.json`. If the code iterates checkpoint directories with an unsorted filesystem glob and returns the first match, it can return an *early* checkpoint's view of "best," not the final training record:

```python
# BUG: glob() order is filesystem-dependent, not sorted
for state_file in checkpoints_dir.glob("checkpoint-*/trainer_state.json"):
    state = json.loads(state_file.read_text())
    if state.get("best_model_checkpoint"):
        return state["best_model_checkpoint"]  # could be from checkpoint-100 of a 285-step run
```

**Measured impact:** This caused a completed 285-step training run (final eval_loss 0.952, steadily improving from 1.184) to export checkpoint-100 (eval_loss 1.049) instead — a materially undertrained model shipped silently, with no error at any pipeline stage. The export script logged "success," GGUF conversion succeeded, and the model registered in the inference server without complaint.

**The Fix:** Read the trainer state from the deterministically-latest checkpoint (highest step count), which holds the authoritative final training record:

```python
checkpoints = sorted(checkpoints_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
latest = checkpoints[-1]
state = json.loads((latest / "trainer_state.json").read_text())
best = state.get("best_model_checkpoint")
return best if best and Path(best).exists() else str(latest)
```

**Implication:** Any pipeline stage that reads training metadata for a "which checkpoint is best" decision should explicitly sort by step count before trusting any single checkpoint's self-reported state. This class of bug is particularly dangerous because every downstream step succeeds — there is no failure signal without independently re-verifying the exported model's quality.

## Results: Cumulative Impact of Fixes

| Fix applied | Before | After |
|---|---|---|
| Template-fill guard | 47% of one domain's data silently contaminated | 0% (contamination rejected at build time) |
| Full-string dedup hash | 84% dedup collapse | 75% dedup collapse (partial — template diversity is the remaining bottleneck) |
| Teacher model swap (code domains) | 0/3, 1/3 smoke test | 3/3, 3/3 smoke test (same volume, same base model) |
| Checkpoint selection fix | Undertrained checkpoint shipped silently | Correct final checkpoint shipped, verified via loss re-check |

## Reproduction

```python
# Minimal reproduction of the checkpoint-selection bug class
from pathlib import Path
import json

def find_best_checkpoint_BUGGY(checkpoints_dir: Path):
    for state_file in checkpoints_dir.glob("checkpoint-*/trainer_state.json"):
        state = json.loads(state_file.read_text())
        best = state.get("best_model_checkpoint")
        if best and Path(best).exists():
            return best  # first match wins — order is undefined
    return None

def find_best_checkpoint_FIXED(checkpoints_dir: Path):
    checkpoints = sorted(
        checkpoints_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    if not checkpoints:
        return None
    latest = checkpoints[-1]
    state_file = latest / "trainer_state.json"
    if state_file.exists():
        state = json.loads(state_file.read_text())
        best = state.get("best_model_checkpoint")
        if best and Path(best).exists():
            return best
    return str(latest)
```

## Conclusion

Multi-teacher synthetic data distillation is a viable, low-cost way to build small specialist models from local infrastructure, but every stage of the pipeline — template filling, deduplication, teacher selection, and checkpoint export — has a failure mode that produces a *valid-looking but degraded* artifact rather than a hard error. None of these bugs would be caught by monitoring pipeline exit codes or log output; each requires an independent content-level check (regex guards on training data, full-string hashing, functional smoke tests per teacher model, and step-sorted checkpoint verification) to catch. The unifying lesson: in an unattended multi-stage ML pipeline, "the script ran without error" and "the pipeline succeeded" are not the same claim, and treating them as equivalent is how undertrained or contaminated models ship silently.
