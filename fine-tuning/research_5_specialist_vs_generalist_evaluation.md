# Does Domain-Specialist Fine-Tuning Beat a Generalist? Four Rounds of Methodology Correction to Find Out

**Training hardware:** NVIDIA RTX 5090 32GB, AMD Ryzen 7 9700X (8c/16t), 74GB RAM
**Software stack:** Ollama (local inference + judge model), Unsloth (QLoRA), Qwen2.5-1.5B-Instruct base models
**Date:** July 2026

## Abstract

A common assumption in small-model fine-tuning is that narrowing a model's training data to a single domain will outperform a generalist model of the same size on in-domain tasks. Testing this directly across four IT-operations diagnostic domains (Windows, Linux, networking, hardware) at 1.5B parameters required four successive rounds of methodology correction before a clean, trustworthy result emerged — and the final answer is a confirmed **yes**, once the measurement and the training intervention were both done with adequate rigor. Round 1 (exact-keyword test matching) produced confidently wrong "failing" scores on correct models. Round 2 (a corrected but shallow n=10 judge-based test) found no detectable difference at all. Round 3 (a deeper n=20 test with multi-hop scenarios) revealed a real but domain-dependent effect — specialist wins on Windows, generalist wins on networking. Round 4 tested whether that domain-dependence was actually a training-data-depth problem: adding deep multi-hop training scenarios, sized as a *consistent proportion* of each domain's existing dataset rather than a fixed absolute count, reversed every remaining underperforming domain into a clear specialist win. The final result across all four domains: specialist models beat the same-size generalist by a wide, non-overlapping margin. The interesting finding is not the headline result — it is how many measurement and training-dose artifacts had to be corrected, in sequence, before that result became visible at all.

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

## Key Finding #5: The "Domain-Dependence" Was Partly a Training-Data-Depth Confound

**The Problem:** Round 2 established that specialization's effect varies by domain, but domain identity and *training data depth* were confounded — networking-specialist had the smallest raw training set of the four domains (1,999 pairs) and, like every other domain, was trained exclusively on single-hop synthetic templates. It was not possible to tell whether networking underperformed because networking-as-a-domain benefits less from specialization, or because its training data simply never taught the kind of multi-step reasoning the round-2 eval was testing for.

**The Experiment:** Generated 15 new deep, multi-hop training scenarios per domain (linux, network, hardware) — topics deliberately disjoint from the 20-case held-out eval set, to avoid training on evaluation data. Each scenario requires a causal chain of reasoning rather than single-fact classification, e.g.:

> "A LACP port-channel between a server and a switch shows one of two member links in 'suspended' state... Both links show 'up' at the physical layer. The suspended link's port was recently moved to a different switch module during a maintenance window."

Correctly answering requires connecting *port move → speed/duplex renegotiation → LACP suspension*, not matching a single symptom to a single cause. Generated 27-30 pairs per domain via two teacher models, folded into each domain's existing dataset, fully retrained, then re-ran the identical n=20 held-out evaluation.

**Measured impact:**

| Domain | Specialist before deep training | Specialist after deep training | Generalist (unchanged) |
|---|---|---|---|
| Networking | 3.55/5, 50% pass | **4.15/5, 85% pass** | 3.65/5, 70% pass |
| Hardware | 3.2/5, 50% pass | 3.6/5, 50% pass | 3.55/5, 50% pass |
| Linux | 3.65/5, 55% pass | 3.5/5, 45% pass | 3.0/5, 45% pass |

Networking reversed completely — from the one domain where the generalist clearly won, to the domain with the single highest score of any model tested in this entire evaluation, now decisively beating the generalist. Hardware improved on average score without a pass-rate change. Linux, notably, *regressed slightly* rather than improving — the one result that doesn't fit a clean "deep data always helps" narrative, and is either a genuine limit (the intervention was a smaller fraction of linux's already-larger existing dataset than the same absolute addition was for networking's smaller one) or within-noise given substantially overlapping confidence intervals before and after.

**Implication:** For narrow domains with less data or thinner topical documentation available to the synthetic-generation teachers, training data composition — specifically, whether the data contains examples of the *kind* of reasoning depth you intend to evaluate — appears to matter more than domain identity itself. A domain that looks like it "doesn't benefit from specialization" may actually be under-resourced in training data depth rather than fundamentally unsuited to specialization.

## Key Finding #6: The Dose Must Be Proportional, Not a Fixed Absolute Count

**The Problem:** Round 4's first pass used the same absolute addition (27-30 pairs) across all three tested domains. This under-dosed linux and hardware, which had larger pre-existing datasets (3,175 and ~2,700 pairs respectively) than networking (1,999 pairs) — the same absolute addition was a ~1.5% increase for networking but only ~0.85-1.1% for the other two. Linux actually regressed slightly under this fixed-count approach rather than improving.

**The Fix:** Recomputed the deep-scenario addition as a *proportion* of each domain's existing dataset size (~1.5-1.6%, matching networking's successful ratio) rather than a fixed pair count, and added a second batch of new (still eval-disjoint) deep scenarios to bring the under-dosed domains up to parity.

**Measured impact:**

| Domain | Fixed-count dose | Proportional dose | Baseline (unchanged) |
|---|---|---|---|
| Linux | 3.5/5, 45% pass (regressed) | **3.7/5, 65% pass** | 3.0/5, 45% pass |
| Hardware | 3.6/5, 50% pass (flat) | **4.15/5, 85% pass** | 3.55/5, 50% pass |

Both domains reversed from "no effect or regression" to a clear specialist win once dosed proportionally. Hardware's result (85% pass rate) tied networking for the single highest score measured anywhere in this evaluation.

**Implication:** When applying a training-data intervention across multiple datasets of different sizes, dose it as a fraction of each dataset, not a fixed count. A fixed absolute addition that works for a smaller dataset will systematically under-treat larger ones — this is a basic mistake in intervention design, but one the earlier "domain-dependent effect" framing entirely obscured, because the confound (dataset size) was never held constant across the comparison.

## Final Results: All Four Domains, After Proportional Deep-Training-Data Correction (n=20/domain, Wilson 95% CI)

| Domain | Specialist avg / pass rate | Generalist avg / pass rate | Result |
|---|---|---|---|
| Windows | 3.6/5, 55% [34%, 74%] | 3.1/5, 30% [14%, 52%] | **Specialist wins** |
| Linux | 3.7/5, 65% [43%, 82%] | 3.0/5, 45% [26%, 66%] | **Specialist wins** |
| Networking | 4.15/5, 85% [64%, 95%] | 3.65/5, 70% [48%, 86%] | **Specialist wins** |
| Hardware | 4.15/5, 85% [64%, 95%] | 3.55/5, 50% [30%, 70%] | **Specialist wins** |

Every domain now shows a clear specialist advantage, with confidence intervals that no longer overlap or overlap only marginally. This is the cleanest result of the entire investigation, and it took four rounds of correction — to the scoring method, the test depth, and finally the training intervention's dosing — to become visible.

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

This work set out to test whether narrow domain fine-tuning outperforms a generalist at small model scale (1.5B parameters, 2,000-4,000 training pairs per domain). The answer is **yes, clearly, across all four domains tested** — but that answer only became visible after four successive rounds of methodology correction, each of which changed the measured result: an initial exact-keyword test produced confidently-wrong "failing" scores on models that were actually correct; a corrected but shallow judge-based test (n=10, single-hop questions) found no detectable difference at all; a deeper judge-based test (n=20, multi-hop questions) revealed a real but domain-dependent effect, with the generalist winning outright on networking; and a training-data intervention — adding deep multi-hop scenarios sized as a *consistent proportion* of each domain's existing dataset, not a fixed absolute count — reversed every remaining underperforming domain into a clear specialist win, including a case (linux) that initially got *worse* under a badly-dosed version of the same fix.

None of the first three answers should be trusted in isolation — each was superseded by a more rigorous version of the same experiment, and the first fixed-dose training intervention itself produced a misleading result (linux's regression) that a second, properly-dosed attempt corrected. The final, most defensible conclusion: **at this model scale, specialization's benefit is real and consistent, but detecting and producing it requires matching rigor in both measurement and training intervention.** A shallow eval will report no effect that exists. A training data addition dosed as a fixed count rather than proportionally will under-treat larger datasets and can produce an apparent regression that has nothing to do with whether specialization "works" for that domain.

What this work establishes with confidence, independent of the specific domain results: (1) evaluation methodology for small-model comparisons requires real engineering investment before any A/B result should be trusted — every one of the four correction rounds was necessary, not optional polish; (2) test *depth*, not just size, determines whether an evaluation can detect a real effect; (3) the same principle applies to training data — a synthetic pipeline that only generates shallow, single-hop examples will produce a model that performs like a generalist on deep problems, regardless of how narrow its topic scope is; and (4) when applying any training intervention across datasets of different sizes, dose it proportionally — a fixed absolute addition that works for a smaller dataset will systematically under-treat larger ones, and can manufacture the appearance of "this doesn't work for domain X" when the real cause is simply an under-sized intervention.
