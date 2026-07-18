# Does Domain-Specialist Fine-Tuning Beat a Generalist? An Evaluation Methodology Fix and a Domain-Dependent Answer

**Training hardware:** NVIDIA RTX 5090 32GB, AMD Ryzen 7 9700X (8c/16t), 74GB RAM
**Software stack:** Ollama (local inference + judge model), Unsloth (QLoRA), Qwen2.5-1.5B-Instruct base models
**Date:** July 2026

## Abstract

A common assumption in small-model fine-tuning is that narrowing a model's training data to a single domain will outperform a generalist model of the same size on in-domain tasks. Testing this directly across four IT-operations diagnostic domains (Windows, Linux, networking, hardware) at 1.5B parameters produced two distinct results depending on evaluation depth: a shallow test set (single-hop, surface-level scenarios, matching the style of the original training data) found no statistically detectable difference between specialist and generalist models. A deeper follow-up test set (multi-hop scenarios requiring layered domain reasoning) revealed a real effect — but one that is domain-dependent rather than universal: the specialist model won clearly on Windows, the generalist won clearly on networking, and the other two domains were statistical ties. This paper documents the evaluation methodology work required to get a trustworthy signal at all (exact-keyword matching produced confidently wrong conclusions before being fixed), and the more nuanced, less satisfying conclusion that emerged once the methodology was solid enough to trust: specialization's effect on model quality is real in this data regime, but not consistent across domains, and the reason why is not yet isolated.

## Key Finding #1: Exact-Keyword Test Matching Produces False Negatives on Correct Answers

**The Problem:** A common fast-evaluation pattern checks whether a model's response contains a fixed set of expected keywords (e.g., `["repadmin", "Get-ADDomainController"]` for an Active Directory diagnostic script). This conflates "used our expected specific implementation" with "is correct" — and technical domains routinely have multiple valid approaches to the same problem.

**Concrete example:** A fine-tuned model, asked to check Active Directory replication status, produced a technically correct, well-structured script using the modern PowerShell AD module:

```powershell
$replStatus = Get-ADReplicationPartnerMetadata -TargetServer $DC.HostName
# ...proper error handling, correct Event IDs (1085, 2097) for replication failures
```

This scored **0/3 (fail)** against a test expecting the literal string `"repadmin"` — the legacy CLI tool, not the modern cmdlet-based approach the model correctly used instead.

**Measured impact:** Two models in a 7-model suite scored 0/3 and 1/3 on their functional smoke tests. Manual inspection of the actual model outputs found both were producing valid, working code; the test methodology was penalizing correct alternative implementations, not detecting real failures.

**The Fix:** Support "match any of these equivalent alternatives" groups instead of requiring every exact string:

```python
"expect_contains": [
    ["repadmin", "Get-ADReplicationPartnerMetadata", "Get-ADReplicationFailure"],  # any one
    "Get-ADDomainController",  # still required
]

def kw_matches(kw):
    if isinstance(kw, list):
        return any(alt.lower() in response_lower for alt in kw)
    return kw.lower() in response_lower
```

Applying this fix alone — with **no changes to the model** — flipped both smoke tests to 3/3.

**Implication:** Before concluding a fine-tuned model underperforms, verify the failure is a real capability gap by reading the actual output, not just the test's pass/fail verdict. A "failing" score on a rigid exact-match test is evidence of nothing except that the test didn't anticipate one valid answer.

## Key Finding #2: Small Sample Sizes Produce Unstable, Non-Reproducible Pass Rates

**The Problem:** Running the identical test input against the identical model on separate occasions produced different pass/fail outcomes purely from LLM sampling variance — even at low temperature (0.1). A 3-question-per-domain smoke test has enough noise that a model's "score" is not a stable property of the model; it is a property of that specific sampling run.

**Measured impact:** The same network-diagnostic model scored 2/3 in one run and 3/3 in a re-run minutes later with an unchanged model and unchanged input, differing only in whether a specific word ("routing" vs. an equally-valid synonym) appeared in that generation's phrasing.

**The Fix:** Two changes together produce a statistically honest result:
1. Increase sample size (3 → 10 test cases per domain in this work; literature suggests 50-200 for a workload-representative evaluation).
2. Report a confidence interval on the pass rate, not a bare fraction, using the Wilson score interval (more reliable than a normal approximation at small n):

```python
def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))
```

**Implication:** "2/3" and "3/3" read as meaningfully different results but, at n=3, the underlying uncertainty is large enough that they may not represent a real difference in model quality. Any small-sample evaluation should report an interval, not a point estimate, or it will systematically overstate confidence in noisy results.

## Key Finding #3: LLM-as-Judge Scoring Recovers Semantic Correctness That Keyword Matching Cannot

**The Problem:** Even with OR-group keyword matching, exact-string tests cannot evaluate holistic correctness — whether a diagnosis correctly identifies root cause and proposes a *reasonable* remediation, independent of specific phrasing.

**The Fix:** Use an independent, larger local model as a judge with a single-dimension rubric, explicitly instructed not to penalize valid alternative approaches:

```
Score on a 1-5 scale:
5 = Correct root cause, valid and complete remediation
4 = Correct root cause, remediation is valid but minor gaps
3 = Partially correct — right area but missed key detail or an alternative valid approach
2 = Mostly wrong — touches the right topic but the core diagnosis is off
1 = Wrong or irrelevant

Do not penalize the response for using a different but technically valid
tool/cmdlet/approach than you might expect — judge correctness, not phrasing or style.
```

This single-dimension design follows established guidance that combining multiple quality dimensions (correctness, tone, completeness) into one rubric produces inconsistent scoring; each rubric should assess exactly one property.

## Key Finding #4: Shallow Test Scenarios Cannot Detect a Real Effect, Even When One Exists

**The Problem:** The initial n=10/domain evaluation used single-hop scenarios ("SMART WARNING: reallocated sectors exceed threshold X") — matching the style of the synthetic templates used to generate training data in the first place. This creates a blind spot: if a model's specialization advantage lies in deeper, multi-step reasoning (the kind a real senior engineer applies — recognizing that a symptom's *obvious* cause is wrong, or tracing a failure through several dependent systems), a test built from the same shallow template family as the training data cannot surface it, because neither the specialist nor the generalist was meaningfully differentiated by that style of question.

**The Fix:** Doubled the evaluation set (10 → 20 cases/domain) by adding ten genuinely multi-hop scenarios per domain — e.g. an Active Directory forest-trust SID-filtering issue requiring the model to recognize why group membership alone doesn't explain the symptom, or a Linux cgroup v2 memory-accounting mismatch where the obvious explanation (a low memory limit) is a red herring and the real cause is a parent-slice limit interacting with a container runtime's own limit.

**Measured impact:** Re-running the comparison with the harder test set produced a materially different result — not just tighter confidence intervals on the same conclusion, but an actual reversal in direction for one domain (see Results below).

**Implication:** A test set's *depth*, not just its *size*, determines whether an evaluation can detect a real specialization effect. A larger sample of shallow questions narrows your confidence interval around the wrong number if the questions themselves aren't sensitive to the capability difference you're trying to measure.

## Results: Specialist vs. Generalist, LLM-as-Judge Evaluation (Wilson 95% CI)

### Shallow scenarios (n=10/domain — single-hop, template-style questions)

| Domain | Specialist avg score / pass rate (95% CI) | Generalist avg score / pass rate (95% CI) |
|---|---|---|
| Windows | 3.7/5, 60% [31%, 83%] | 3.7/5, 70% [40%, 89%] |
| Linux | 3.8/5, 60% [31%, 83%] | 3.7/5, 60% [31%, 83%] |
| Networking | 3.9/5, 70% [40%, 89%] | 3.7/5, 60% [31%, 83%] |
| Hardware | 3.8/5, 60% [31%, 83%] | 3.7/5, 50% [24%, 76%] |

Every confidence interval overlaps; average judge scores cluster tightly (3.7-3.9/5) regardless of specialist vs. generalist. No detectable difference at this test depth.

### Deep scenarios (n=20/domain — added 10 multi-hop cases per domain)

| Domain | Specialist avg / pass rate (95% CI) | Generalist avg / pass rate (95% CI) | Result |
|---|---|---|---|
| Windows | 3.6/5, 55% [34%, 74%] | 3.1/5, 30% [14%, 52%] | **Specialist ahead** — CIs barely overlap |
| Linux | 3.65/5, 55% [34%, 74%] | 3.0/5, 45% [26%, 66%] | Specialist ahead — CIs overlap substantially |
| Networking | 3.55/5, 50% [30%, 70%] | **3.65/5, 70% [48%, 86%]** | **Generalist ahead** — CIs barely overlap |
| Hardware | 3.2/5, 50% [30%, 70%] | 3.55/5, 50% [30%, 70%] | Tie — identical pass rate |

**The deeper test set reveals a real, domain-dependent effect that the shallow test set could not detect at all** — including a reversal (networking) that argues against a simple "specialization always helps a little" story. Windows shows the clearest specialist advantage; networking shows the opposite; linux and hardware are inconclusive/tied.

## Reproduction

```python
# Core evaluation loop — judge-scored, confidence-interval-reported
import requests

def query_model(model, prompt):
    r = requests.post("http://localhost:11434/api/chat", json={
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 600},
    })
    return r.json()["message"]["content"].strip()

def judge_score(problem_input, response, judge_model="qwen3-30b"):
    rubric = f"""Score 1-5: does the response correctly identify root cause and
propose valid remediation? Do not penalize valid alternative approaches.
PROBLEM: {problem_input}
RESPONSE: {response}
Respond with ONLY a single digit 1-5."""
    raw = query_model(judge_model, rubric)
    import re
    m = re.search(r"[1-5]", raw)
    return int(m.group(0)) if m else None
```

## Conclusion

This work set out to test whether narrow domain fine-tuning outperforms a generalist at small model scale (1.5B parameters, 2,000-4,000 training pairs per domain). The answer changed twice as the evaluation methodology improved, and the final answer is less clean than either intermediate one: an initial exact-keyword test produced confidently-wrong "failing" scores on models that were actually correct; a corrected but shallow judge-based test (n=10, single-hop questions) found no detectable difference at all; a deeper judge-based test (n=20, multi-hop questions requiring layered reasoning) revealed a real effect that is domain-dependent — a clear specialist win on Windows, a clear generalist win on networking, and ties elsewhere. None of these three answers should be trusted in isolation; only by fixing the scoring methodology *and* then interrogating whether the test itself was capable of detecting the effect in question did a trustworthy — if unsatisfying — picture emerge.

The domain-dependence itself is now the open question this work could not resolve: is it explained by uneven per-domain data volume (networking had the smallest raw dataset of the four), by real-world documentation density varying by topic (Windows/AD ecosystems may simply have richer, more internally-consistent training signal available than fragmented multi-vendor networking troubleshooting), or by an unverified confound in the specific generalist model used as baseline? Any of these would require controlled, single-variable experiments — not yet run — to isolate. What this work does establish with confidence: (1) evaluation methodology for small-model comparisons requires real engineering investment before any A/B result should be trusted, and (2) test *depth* is as important as test *size* — a larger sample of shallow questions will confidently measure the wrong thing if the questions themselves can't distinguish the capability being tested for.
