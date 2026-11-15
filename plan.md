# Research plan: separating knowledge from reasoning with a maskable memory

CSCI 2951X project. Aidan Flaherty, Adyant Ranjan, Louis Geer, Eric Xu.

## 1. Problem and motivation

**One-sentence version.** We want a language model whose facts live in an addressable memory and whose remaining weights hold only language and reasoning, so that removing a block of memory removes the corresponding knowledge and nothing else.

**Why this is not already solved.** Several 2025-26 papers move facts out of the base model, but each leaves the same gap:

| Work | What it does | What it leaves open |
|---|---|---|
| Hierarchical Memories (Apple, arXiv 2510.02375) | FFN-attached memory blocks fetched by document cluster; plain next-token loss | Nothing stops the base model from also learning the facts. Only a preliminary "block 1/16 of the bank" test. |
| LMLM (arXiv 2505.15962), Co-LMLM (arXiv 2607.07707) | Facts go to an external database queried by lookup calls; fact values are excluded from the loss | Behaviour when the lookup fails is untrained (left to future work). No test of whether deleted facts can be relearned quickly. |
| Gradient routing (arXiv 2410.04332), SGTM (arXiv 2512.05648) | Routes gradients for labelled data into a designated weight region, then ablates it | The region is ordinary weights: not addressable per fact, no editing story. |
| Neural Knowledge Bank (arXiv 2208.00399) | Extra FFN key-value slots trained with the base frozen | Never evaluated with the bank masked. |
| Knowledge Neurons (arXiv 2104.08696), ROME/MEMIT | Find and edit facts in a normally trained model | Localization is discovered after the fact and is weak (34% edit success in Knowledge Neurons). |

**Our contribution.** Train the separation in rather than hoping for it: a learned key-value memory in the FFN path, plus an objective with two sides, one that rewards recall when memory is present and one that removes recall when memory is masked. Then test whether the knowledge is really absent, not just suppressed.

The honest framing: "externalizing knowledge at pretraining enables unlearning" is no longer novel. The defensible niche is the combination of (a) from-scratch training, (b) latent FFN-path memory, (c) an explicit masked-memory objective, and (d) relearning-attack evaluation, in a controlled synthetic setting where every fact is enumerable.

### Research questions

- **RQ1 (separation).** Does a two-sided objective reduce the factual knowledge held in base weights below what architecture alone (memory + plain loss) achieves?
- **RQ2 (absence vs. suppression).** With memory masked, is the knowledge gone, or has the model learned to detect the mask and stay quiet? Measured by relearning speed and probing against a model that never saw the facts.
- **RQ3 (no collateral damage).** Do language modelling and reasoning from in-context information stay the same with memory masked?
- **RQ4 (editability).** Can facts be added, changed and deleted by touching memory only, with the base frozen?
- **RQ5 (scaling).** The proposal argues that a larger base model simply absorbs more facts. Does our objective hold the leak flat as the base grows?

### What must stay in the base model

Type-level knowledge stays: "born in" is followed by a city, and universities have university-like names. Instance-level knowledge moves: which city this person was born in. All metrics below are defined against that line, so "chance" means guessing from the attribute's prior, not uniform over the vocabulary.

## 2. Data

Real-world QA (Natural Questions, TriviaQA) is the wrong starting point: a small from-scratch model cannot learn it, facts are not enumerable, and there is no clean forget/retain split. Use synthetic facts in a natural-language carrier.

### Primary corpus

| Component | Source | Role |
|---|---|---|
| **Fact corpus** | Synthetic biographies in the style of bioS (Allen-Zhu & Li, arXiv 2309.14316): N fictional people, 6 attributes each (birth date, birth city, university, major, employer, work city) | The knowledge to be stored in memory |
| **Language corpus** | TinyStories (arXiv 2305.07759) or SimpleStories (arXiv 2504.09184) | Knowledge-free language ability; perplexity measured here |
| **Closed-book QA** | Templated questions over the biographies | Knowledge-dependent task |
| **Open-book QA** | The same questions with the biography placed in the prompt, using people who never appear in training | Reasoning over context with no memory needed |
| **Reasoning probe** | bAbI-style or ProofWriter-style (arXiv 2012.13048) deduction, or modular arithmetic | Knowledge-independent task |

Start with N around 10-20k people and grow to 100k once the pipeline works.

### Details that matter

- **Augmentation.** Allen-Zhu & Li show a model can memorize biographies yet fail to answer questions about them unless the biographies are augmented. Generate several templates per sentence, shuffle sentence order, and repeat the full name instead of pronouns.
- **Held-out QA people.** Train on biographies of all people but QA pairs for only half. Report closed-book accuracy on the other half. This measures extraction rather than memorized question strings.
- **Fact-span labels.** The generator emits a mask over attribute-value tokens. The objective and the metrics both need it.
- **Splits for forgetting.** Reserve a forget set (a group of people, and separately a whole attribute) and a retain set. Reserve a further set of people who appear in no training run, for the oracle and for insertion experiments.
- **Open-book QA is the key control.** A model that answers "I don't know" to everything would pass a naive masking test. Open-book accuracy with memory masked shows it can still read and reason.

### Secondary, only after the main results

- TOFU (arXiv 2401.06121), for a number comparable to LMLM.
- FictionalQA (arXiv 2506.05639), for leak-free natural text.
- A Wikipedia subset with entity spans tagged by NER, as in Neural Knowledge Bank and LMLM.

## 3. Model

### Current repo state

- `model.py` is a correct Llama-3-style decoder (RMSNorm, RoPE, SwiGLU, grouped-query attention): 12 layers, width 1024, about 238M parameters, of which about 131M are the 128k-token embedding table.
- `train.py` trains on Natural Questions and evaluates on the first 100 training examples, so the score measures training-set memorization.
- The "exact match" check is `pred in answer`, a substring test. An empty generation counts as correct.
- There is no memory, no masking and no second loss.

### Recommendation: keep the architecture, shrink it, replace the tokenizer

"Llama 3 from scratch" is only an architecture choice, and the existing code is fine. The size and vocabulary are the problem.

| Setting | Now | Proposed base |
|---|---|---|
| Layers / width | 12 / 1024 | 8 / 512 |
| FFN hidden | 2048 | about 1400 |
| Vocabulary | 128,256 (Llama tokenizer) | 8-16k BPE trained on our corpus |
| Parameters | about 238M | about 30-35M |
| Context | 512 | 512 |

A small model trains in hours, allows several seeds per condition, and makes a retrain-from-scratch oracle affordable. For RQ5, add two more sizes (roughly half and double the width).

### Memory module

A bank of slots, each with a learned key and a learned value, read in parallel with the FFN in a few middle layers:

```
q      = W_q · norm(h)                # query from the residual stream
scores = q · K                        # K: slot keys; masked slots get -inf
w      = sparse top-k softmax(scores)
h      = h + FFN(h) + Σ w_i · V_i     # V: slot values
```

Build it in three stages, each a separate experiment:

1. **M0, oracle routing.** Each person owns a small block of slots, fetched by entity ID (as Apple fetches by precomputed cluster ID). Masking a person means not fetching their block. This isolates the question "does the objective work?" from "can retrieval be learned?".
2. **M1, learned retrieval.** One global bank; the model learns queries and keys; top-k read. Clusters are found afterwards from access statistics. Product keys (Lample et al., arXiv 1907.05242) are only needed if the bank outgrows exhaustive search.
3. **M2, hierarchical.** Path retrieval down a learned hierarchy, as in the proposal. Stretch goal.

The encoder-decoder variant from the proposal is the lowest priority; it changes too many things at once.

### Alternatives considered

- **Frozen pretrained backbone plus memory** (SmolLM, Qwen 0.5B, Llama 3.2 1B), as Apple and Neural Knowledge Bank do post hoc. It is cheap and gives fluent language, but the base already holds world knowledge, so it can only answer RQ4 on fictional facts, not RQ1-2. Worth one stretch experiment.
- **nanoGPT / GPT-2 small.** Equivalent for our purposes; no reason to switch.

## 4. Training objective

Each batch is run twice, memory on and memory masked.

**Memory on:** next-token loss on everything.

**Memory masked, non-fact tokens:** next-token loss as usual, so the base model is self-sufficient for language, reasoning and open-book QA.

**Memory masked, fact tokens:** four variants to compare. The proposal's sign-flipped loss is excluded because it is unbounded and damages fluency.

| Variant | Masked-memory target on fact tokens | Note |
|---|---|---|
| **A. Abstain** | QA answer is replaced by "I don't know" | The proposal's option; indirect gradient (cf. R-Tuning, arXiv 2311.09677) |
| **B. Prior matching** | KL to the attribute's prior distribution (e.g. the marginal over cities) | Bounded; keeps type-level knowledge; grammar tokens untouched |
| **C. Gradient routing** | No masked loss; the memory-on loss on fact tokens updates memory parameters only | Structural, no adversarial term (Cloud et al.) |
| **D. Memory swap** | Fetch another person's memory; the target is that person's attribute | The instructor's "flip the bits" suggestion: output must follow memory |

Variants A and B can be satisfied by a model that keeps the facts and learns to detect the mask. C and D make that harder, because C never gives the base a gradient for the fact and D makes any base-stored answer wrong half the time. Expect C+D or B+D to be the strongest; this is the central comparison of the project.

**Instructor suggestion to test as an ablation:** a narrow bottleneck in the layers above the memory, to limit how much the upper layers can store.

## 5. Baselines

| ID | Model | What it tells us |
|---|---|---|
| B0 | Vanilla transformer, parameter count matched to base + memory | Standard entangled model |
| B1 | Base only, no memory | Language upper bound; fact capacity of the base alone |
| B2 | Memory architecture, plain next-token loss | How much separation architecture gives for free (Apple / NKB setting) |
| B3 | Gradient routing into designated FFN neurons, no retrieval | Strongest "localize in weights" competitor |
| B4 | Vanilla model with the biography in context | Open-book upper bound on separation (RAG / LMLM-like) |
| B5 | **Oracle:** same model trained without the forget set | Gold standard for "knowledge absent" |
| B6 | Post-hoc unlearning on B0 (fine-tune to abstain, gradient ascent) | What you get without designing for forgetting |
| B7 | Fine-tuning B0 on new or changed facts | Editing baseline; ROME/MEMIT if time allows |

B2 and B5 are the two that matter most. B2 shows whether the objective adds anything; B5 is what every "is it really gone" metric is compared against.

## 6. Metrics

Report every metric with memory on and memory masked, over 3 seeds, with bootstrap intervals.

- **Recall.** Closed-book QA accuracy on held-out-QA people. Use true exact match on the attribute value.
- **Leakage.** With memory masked: accuracy above prior-guessing, and bits leaked per fact = entropy of the attribute prior minus masked-memory cross-entropy on the answer. This is a lower-bound estimate of the mutual information between base weights and facts, which the instructor asked for.
- **Utility.** Story perplexity, reasoning accuracy, open-book QA accuracy; the on-versus-masked gap should be near zero.
- **Abstention quality** (variant A). Abstain rate on masked facts versus false abstention on open-book and retained facts.
- **Relearning attack** (Deeb & Roger, arXiv 2410.08827). Fine-tune the masked model on some forgotten facts; measure accuracy on the other forgotten facts. Compare the curve with the oracle B5.
- **Probing.** Linear probe for each attribute from the residual stream at the name token, memory masked, compared with the oracle.
- **Cross-contamination.** Language held by memory: perplexity change on stories when memory is masked. Facts held by base: the leakage metric. Both directions, as the instructor suggested.
- **Access behaviour** (M1 onward). Slot overlap across paraphrases of the same fact; read weight on text that needs no knowledge.
- **Editing.** Efficacy, paraphrase generalization, locality on unrelated facts, change in story perplexity.

## 7. Experiments

### Phase 0: infrastructure

- Biography and QA generator with fact-span masks and fixed splits.
- BPE tokenizer trained on the corpus; shrunken model config.
- Evaluation harness covering all metrics in section 6, with a held-out split and real exact match.
- Config files, seeds, logging, checkpointing.

### Phase 1: getting started and sanity checks

- **E0a.** Overfit a few hundred biographies to confirm the pipeline.
- **E0b.** Vanilla model (B0) on the full mix. Confirm closed-book QA works on held-out-QA people, and that it fails without augmentation (a miniature replication of Allen-Zhu & Li).
- **E0c.** Capacity curve: facts stored versus base size, for B1. This sets how many people the main experiments need for memory to be necessary.
- **E0d.** Train the oracle (B5) and record its leakage, probe and relearning numbers. These define "absent".

### Phase 2: core result (RQ1-3)

- **E1.** B2, memory with plain loss, oracle routing. Mask memory and measure the drop. Expect partial separation.
- **E2.** Objective variants A-D and the promising pairs, oracle routing. Full metric table against B0, B2, B3, B5.
- **E3.** Robustness of the best two variants: relearning attack, probes, paraphrased and reordered questions.

### Phase 3: learned memory and editing (RQ4)

- **E4.** Learned retrieval (M1) with the best objective. Does separation survive? Report access metrics; mask discovered clusters.
- **E5.** Editing with the base frozen: insert new people by training new slots only, change an attribute, delete a person and an attribute. Compare with B7.

### Phase 4: ablations and scaling (RQ5)

- **E6.** Base size at three widths, with and without the objective: does leakage grow with size under B2 and stay flat under ours?
- **E7.** Memory layer placement, slots per person, top-k, masked-loss weight, upper-layer bottleneck.

### Stretch

- Hierarchical memory (M2).
- TOFU or FictionalQA.
- Frozen pretrained backbone with fictional facts.

### Minimum viable project

Phases 0-2 plus E5. That answers RQ1-4 on synthetic data with oracle routing and is a complete story.

## 8. Risks

| Risk | Sign | Response |
|---|---|---|
| Model hides knowledge instead of losing it | Low leakage but fast relearning or high probe accuracy | Prefer variants C and D; report against the oracle |
| Base ignores memory | Masking changes nothing | Raise the fact load past base capacity (E0c); warm up with memory-on training |
| Model abstains on everything | Open-book QA drops with memory masked | Weight the open-book and story loss up in the masked pass |
| Learned retrieval collapses onto a few slots | Skewed access counts | Keep oracle routing as the main result; add key normalization or a usage penalty |
| Synthetic-only result looks narrow | Reviewer objection | One natural-text stretch experiment |

## 9. Open items

- **Compute.** We run on Brown's OSCAR GPU cluster (Slurm). Consequences for the plan:
  - Runs are batch jobs with a wall-time limit, so Phase 0 must include checkpoint-and-resume and an `sbatch` script per experiment config.
  - Seeds and objective variants are independent, so E2, E6 and E7 should be submitted as Slurm job arrays rather than run in sequence.
  - GPU type varies by node; log the GPU model with each run and keep bf16 autocast behind a capability check.
  - Training time per run is unmeasured and will come out of E0b. Our GPU-hour allocation and per-job time limit are still to be confirmed, and they decide whether E6 (three base sizes) fits.
- **Citation check.** The proposal's "Constructing Efficient Fact Storing Transformers" link is arXiv 2512.00207 (Dugan et al.). A possibly retitled later version, "MLPs are Hebbians" (arXiv 2607.10034), turned up in search but was not opened.
- **Unverified details.** Which layers Apple attaches FFN memories to; whether Co-LMLM reports a knowledge-base-off ablation.

## 10. References

Closest work:
- Pouransari et al., Pretraining with Hierarchical Memories, arXiv 2510.02375. Code: github.com/apple/ml-memory-pretraining
- Zhao et al., Pre-training Limited Memory Language Models, arXiv 2505.15962
- Feldman et al., Co-LMLM, arXiv 2607.07707
- Cloud et al., Gradient Routing, arXiv 2410.04332
- Shilov et al., Beyond Data Filtering (SGTM), arXiv 2512.05648
- Ghosal et al., Memorization Sinks, arXiv 2507.09937
- Dai et al., Neural Knowledge Bank, arXiv 2208.00399
- Dai et al., Knowledge Neurons, arXiv 2104.08696
- Dugan et al., Constructing Efficient Fact-Storing MLPs, arXiv 2512.00207

Memory layers:
- Lample et al., Large Memory Layers with Product Keys, arXiv 1907.05242
- Berges et al., Memory Layers at Scale, arXiv 2412.09764. Code: github.com/facebookresearch/memory
- Lin et al., Sparse Memory Finetuning, arXiv 2510.15103
- Yu et al., ExplicitLM, arXiv 2511.01581
- Verga et al., Facts as Experts, arXiv 2007.00849

Background and motivation:
- Geva et al., FFN Layers Are Key-Value Memories, arXiv 2012.14913
- Nichani et al., Factual Recall via Associative Memories, arXiv 2412.06538
- Hase et al., Does Localization Inform Editing?, arXiv 2301.04213
- Han et al., General Reasoning Requires Learning to Reason from the Get-go, arXiv 2502.19402
- Mahowald et al., Dissociating Language and Thought in LLMs, arXiv 2301.06627

Data and evaluation:
- Allen-Zhu & Li, Physics of Language Models Part 3.1, arXiv 2309.14316
- Eldan & Li, TinyStories, arXiv 2305.07759
- Maini et al., TOFU, arXiv 2401.06121
- Kirchenbauer et al., FictionalQA, arXiv 2506.05639
- Deeb & Roger, Do Unlearning Methods Remove Information from Weights?, arXiv 2410.08827
- Lynch et al., Eight Methods to Evaluate Robust Unlearning, arXiv 2402.16835
- Meng et al., ROME, arXiv 2202.05262; MEMIT, arXiv 2210.07229
