# Proof-stage changes to `cxt` manuscript

All changes below are paragraph shortenings made to fit the 12-page limit.
Line numbers are `.tex` source line numbers (not compiled-PDF margin numbers).
**Old** lines refer to `cxtkit_manuscript/main.tex`; **New** lines refer to `Research_report.tex`.

---

## 1. Transformer / GPT introduction paragraph

### Old (`main.tex` lines 238–250)

> More recently, transformer-based language models (LMs) have been introduced
> that enhance prediction capabilities through the learning of stochastic processes,
> rather than scenario-constrained parameter estimation \cite{nait_saada_inference_2023}.
> Transformer LMs are perhaps best known in the context of pretrained generative models for human
> language such as GPT \cite{Radford2018ImprovingLU}. These models are pretrained in
> that they are first trained on a "pretext" task, such as next token prediction, that
> requires the LM to learn context among observations (tokens) in a sequence locally
> and their relation to higher-level associations in the space of language.
> Modern large LMs are in part wildly successful due to their massive numbers of parameters (e.g. hundreds of billions) that enable them
> to effectively learn rich representations and flexible conditional prediction across contexts and thus to generalize.
> Here we describe a pretrained generative model for population genetics that
> leverages a small language model (10-20M parameters) to effectively translate from mutational patterns across chromosomes to coalescent time estimates.
> This opens the door to a generalizable (non-task specific) model paradigm: pretraining a flexible transformer on a diverse range of coalescent simulations, followed by fine-tuning for specific evolutionary tasks, if necessary.

### New (`Research_report.tex` lines 122–129)

> More recently, transformer-based language models (LMs) have been introduced
> that enhance prediction capabilities through the learning of stochastic processes,
> rather than scenario-constrained parameter estimation \cite{nait_saada_inference_2023}.
> Inspired by pretrained generative models such as GPT \cite{Radford2018ImprovingLU},
> we leverage this paradigm with a small model (10–20M parameters) that translates
> mutational patterns across chromosomes to coalescent time estimates,
> opening the door to a generalizable approach: pretraining on diverse coalescent simulations,
> followed by fine-tuning for specific evolutionary tasks if necessary.

---

## 2. Next-coalescence prediction definition

### Old (`main.tex` lines 259–260)

> To enable this inference, we define a novel task, analogous to *next-token prediction* in language modeling, which we term *next-coalescence prediction*. In this formulation, the model predicts the next coalescence time along a sequence of previously predicted events, across discrete sequence space, conditioned on local mutation densities within a fixed context window. While this does not reconstruct full genealogical topologies, it enables a structured prediction task that translates between observed mutation patterns and pairwise coalescence times — akin to approaches such as PSMC or SMC++, the latter being the method to which cxt is conceptually a close neural analog, in the sense that prediction happens on a pivot (or distinguished pair), while sharing information across haplotypes through the SFS. Thus, the model offers a way to navigate between the observed distribution of mutation densities and the latent pairwise genealogical structure by means of a sequence of coalescence event prediction.

### New (`Research_report.tex` lines 136–142)

> To enable this inference, we define a novel task analogous to *next-token prediction*,
> which we term *next-coalescence prediction*: the model autoregressively predicts
> the next pairwise coalescence time conditioned on local mutation densities within a fixed
> context window. While this does not reconstruct full genealogical topologies, it provides
> a structured translation from observed mutation patterns to pairwise coalescence times.
> Conceptually, cxt is a close neural analog of SMC++: prediction is centered on a
> pivot (distinguished) pair, with information shared across haplotypes through the SFS.

---

## 3. Right panel of Figure 1 description

### Old (`main.tex` lines 369–374)

> The right panel of Figure 1 shows a concrete example of the model's output.
> Leveraging our GPU-enabled method, we infer pairwise coalescence times for all C(50,2)
> pairs of a sample of 50 haploid chromosomes in parallel by placing them in a batch of size 1225 (although note this batch size is dependent on GPU memory, and can be adjusted to available GPU memory).
> We then run the generative process multiple times, typically 15 replicates, and average the results to obtain stable and robust estimates.
> This generative process provides samples from an approximate posterior distribution of pair coalescence times across the sequence (where the prior is implicit via the simulated training set; see Calibration).
> In the right panel of Figure 1, we apply our cxt to a constant population size scenario with a population size of 2×10⁴ and approximately equal mutation and recombination rates (see Datasets below). In that figure true coalescent times are shown in black, the replicate predictions in light blue, and the average prediction in dark blue. In this simple setting, where the model is well-specified, cxt produces highly accurate predictions. Importantly, cxt can predict coalescent times for *all* pivot pairs concurrently in approximately five minutes of computation on an NVIDIA A100 GPU with 80GB memory.

### New (`Research_report.tex` lines 218–226)

> The right panel of Figure 1 illustrates cxt on a constant-size scenario
> (N_e = 2×10⁴, roughly equal mutation and recombination rates; see Datasets).
> All C(50,2) pivot pairs of 50 haploid chromosomes are inferred in parallel;
> for each pair the generative process is repeated 15 times, yielding samples from an approximate
> posterior over pairwise coalescence times along the sequence
> (where the prior is implicit via the simulated training set; see Calibration).
> Averaging across replicates produces stable estimates.
> In this well-specified setting cxt is highly accurate,
> and inference for all pairs completes in approximately five minutes on a single NVIDIA A100 GPU.

---

## 4. Sample-size adapter paragraph

### Old (`main.tex` lines 420–428)

> As cxt is trained at a specific sample size, the user has a few options should
> they want to infer coalescent times in a new sample of different size:
> 1) they could retrain the model starting from a new set of simulations that condition on their sample size,
> 2) if the sample size is larger than the trained model, they could subsample down to the appropriate size,
> or 3) if the sample is smaller than the trained model we have created a light-weight adapter, which quickly fine-tunes
> a larger sample size trained cxt model to a smaller sample size. In SI Appendix, Fig. S4, we show
> representative results from the adapter applied to samples of size N=5 diploids
> from a cxt model trained at sample size N=25 diploids,
> and then used for inference of a constant sized simulation (left panel) or the sawtooth simulation (right panel).

### New (`Research_report.tex` lines 262–266)

> Because cxt is trained at a fixed sample size, users with larger samples can subsample,
> while for smaller samples we provide a lightweight adapter that fine-tunes the pretrained model
> to the target cohort size.
> SI Appendix, Fig. S4 shows adapter results transferring from N=25 to N=5 diploids
> on both the constant-size and sawtooth scenarios.

---

## 5. Interpolation / extrapolation across parameter regimes

### Old (`main.tex` lines 549–557)

> In SI Appendix, Fig. S12, we assess the model's ability to interpolate and extrapolate beyond the training
> distribution with respect to mutation and recombination rates, holding Ne fixed at 20,000. We define a grid of mutation
> and recombination rates chosen to encompass most stdpopsim species (panel A in SI Appendix, Fig. S12);
> the distribution of species across this grid is shown in SI Appendix, Fig. S12 (B). SI Appendix, Fig. S12 (A)
> also highlights a range of recombination-to-mutation rate ratios SI Appendix, Fig. S12 (B), while SI Appendix,
> Fig. S12 (C) and (D) report performance over the grid in terms of MSE and KL divergence, respectively. As
> expected, error is highest in the low-mutation, high-recombination region (bottom right of SI Appendix,
> Fig. S12 (C)) where the signal-to-noise ratio is the least favorable for TMRCA inference. SI Appendix, Fig. S12
> provides representative examples of TMRCA predictions from the four corners of the grid. Together, these results
> show that cxt can generalize beyond the specific regions of parameter space represented in the training data.

### New (`Research_report.tex` lines 368–371)

> We assessed cxt's ability to interpolate and extrapolate
> beyond the training distribution by evaluating prediction
> accuracy across a grid of mutation and recombination
> rates at fixed N_e = 20,000 (SI Appendix, Fig. S12).
> Panel A defines the grid, chosen to span most stdpopsim
> species, with their distribution shown in panel B.
> Panels C and D report MSE and KL divergence,
> respectively; as expected, error is highest in the
> low-mutation, high-recombination regime where the
> signal-to-noise ratio is least favorable.
> Representative TMRCA predictions from the four corners of the grid
> confirm that cxt generalizes beyond the parameter
> regions represented in training.

---

## 6. Demographic inferences from short sequences

### Old (`main.tex` lines 637–645)

> It is worth noting that the demographic inferences shown in Figure 4 and SI Appendix, Fig. S10
> are based on relatively short simulated sequences.
> For computational efficiency, we restrict these demonstrations to 10 Mb of
> simulated genome.
> Because shorter sequences increase the variance of recent-time coalescence-rate estimates by truncating the longest tracts of recent ancestry,
> we expect that applying the same procedure to full-length chromosomes would yield substantially improved accuracy.
> Together, these results show that cxt's local TMRCA predictions can be
> aggregated into accurate genome-scale summaries of demography, with performance
> expected to improve further at realistic chromosome lengths.

### New (`Research_report.tex` lines 427–430)

> These demographic inferences are based on 10 Mb of simulated sequence (Figure 4 and SI Appendix, Fig. S10);
> because shorter sequences truncate the longest tracts of recent ancestry, increasing variance in recent-time rate estimates,
> we expect substantially improved accuracy at full chromosome lengths.
> Overall, cxt's local TMRCA predictions aggregate into accurate genome-scale demographic summaries.

---

## 7. Rdl locus investigation / comparison panel

### Old (`main.tex` lines 723–733)

> To investigate selective dynamics at the *Rdl* locus across multiple *A. gambiae* populations,
> we constructed a comparison panel (SI Appendix, Fig. S11) comprising
> (i) coalescent-time landscapes inferred by cxt (left column),
> (ii) corresponding estimates from Singer+Polegon (middle column), and (iii) SMC++ decodings (right column).
> Analyzing Ag1000G data poses two practical challenges.
> First, large effective population sizes imply high N_e-scaled mutation and recombination rates,
> which increase genealogical heterogeneity and the number of recombination events,
> substantially slowing MCMC-based approaches such as Singer+Polegon.
> Second, missingness varies widely across genomic regions,
> requiring methods that can accommodate irregular observation patterns
> (density of missing data is shown beneath the upper and lower panels as a track in SI Appendix, Fig. S11).
> To improve readability in the main text, we restrict these figures to cxt (SI Appendix, Fig. S14);
> full comparisons with Singer+Polegon and other approaches are shown in the SI Appendix, where these analyses are presented in detail.

### New (`Research_report.tex` lines 498–504)

> To investigate selective dynamics at the *Rdl* locus across multiple *A. gambiae* populations,
> we constructed a comparison panel (SI Appendix, Fig. S11) showing coalescent-time landscapes
> from cxt, Singer+Polegon, and SMC++.
> The challenges outlined above are particularly acute here;
> high N_e-scaled rates substantially slow MCMC-based approaches such as Singer+Polegon.
> For readability, main-text figures show only cxt results (SI Appendix, Fig. S14);
> full method comparisons appear in the SI Appendix.

---

## 8. SMC++ blockwise representation at Rdl

### Old (`main.tex` lines 795–804)

> SMC++ also detects a noticeable reduction in TMRCA around the *Rdl* sweep,
> but its blockwise representation limits how precisely this signal can be localized along the genome.
> Even after scaling the SMC++ output to match the true genomic span of the region
> and applying manual calibration so that its genome-wide diversity patterns align with the cxt and Singer+Polegon panels,
> the inferred trough remains discretized and slightly shifted.
> This behavior is expected:
> SMC++ models the genome using coarse blocks under a global demographic process,
> so narrow sweeps can be detected, but their boundaries are blurred by the block structure.
> As a result, SMC++ can identify strong selective events,
> but it is less able to resolve their precise genomic breakpoints.

### New (`Research_report.tex` lines 549–555)

> SMC++ also detects a noticeable reduction in TMRCA around the *Rdl* sweep,
> but its blockwise representation limits localization.
> Even after scaling the output to match the true genomic span and calibrating
> genome-wide diversity to the cxt and Singer+Polegon panels,
> the inferred trough remains discretized and slightly shifted.
> This is expected: SMC++ fits a global demographic process using coarse blocks,
> so narrow sweeps can be detected but their genomic breakpoints are less precisely resolved.

---
---

# Comprehensive change log: `cxtkit_manuscript/main.tex` → `Research_report.tex`

Every substantive text, number, reference, and structural change between the two versions.
The paragraph shortenings (§1–§8 above) are included by reference and not repeated in full.

---

## A. Template and formatting

| Item | Old | New |
|---|---|---|
| Document class | `\documentclass{article}` | `\documentclass[9pt,twocolumn,twoside]{pnas-new}` |
| Bibliography back-end | `biblatex` / `biber` (`\parencite`, `\textcite`) | `natbib`-style (`\cite`) via PNAS class |
| Line numbers | `\linenumbers` (present) | Removed |
| Page style | `\fancyhdr` with custom header | PNAS template default |
| Geometry | `a4paper, margin=1in` | PNAS two-column layout |
| Significance statement | Inline `\section*{Significance statement}` | PNAS `\significancestatement{…}` environment |
| Author contributions / declarations | Not present as front-matter | Added via `\authorcontributions`, `\authordeclaration`, `\correspondingauthor` |
| Figures | Inline `\begin{figure}` throughout text | Figures removed from body; collected **Figure Legends** section at end |
| Supplementary | Inline supplementary after `\beginsupplement` | Moved to separate SI Appendix; all refs changed to "SI Appendix, Fig. S*N*" |

---

## B. Title

No change. The rendered title was already "A Language Model for Population Genetics" in both versions. (The old `\title{}` macro contained a stale subtitle — "Leveraging large language models…" — but the actual typeset title block already matched the new version.)

---

## C. Figure numbering and relocation

Several figures were renumbered or moved to the SI Appendix:

| Old reference | Content | New reference |
|---|---|---|
| Figure 1 (`fig:schematic`) | Model schematic + constant-size demo | **Figure 1** (unchanged) |
| Figure 2 (`fig:comp`) | Benchmark comparisons (cxt-narrow / broad / Singer / SMC++) | **Figure 2** (unchanged) |
| Figure 3 (`fig:margdist_nomap`) | Marginal coalescence distributions, stdpopsim v0.2 (no map) | **SI Appendix, Fig. S15** (moved to SI) |
| Supp. Fig. `fig:margdist_map` | v0.2 distributions with genetic map | **SI Appendix, Fig. S7** |
| Figure 4 (`fig:margdist_stdpop3`) | Out-of-sample stdpopsim v0.3 | **Figure 3** (renumbered) |
| Figure 5 (`fig:iicr`) | IICR demography (H. sapiens / B. taurus / A. thaliana) | **Figure 4** (renumbered) |
| Figure `fig:iicr_cross` | Cross-coalescence rate (OutOfAfrica_2T12) | **SI Appendix, Fig. S10** (was already supplementary) |
| Figure 6 (`fig:chr2_6_corr`) | Human chr 2 & 6 (LCT / HLA) | **Figure 5** (renumbered) |
| Supp. Fig. `fig:tmrca_recom` | TMRCA vs recombination breakpoints | **SI Appendix, Fig. S13** |
| Figure 7 (`fig:anopheles_main`) | Ag1000G Rdl + chr2L cxt results | Main-text **Figure 7** kept but referenced via "SI Appendix, Fig. S14" in some passages |
| Supp. Fig. `fig:anopheles` | Full comparison panel (cxt / Singer / SMC++) | **SI Appendix, Fig. S11** |
| Figure `fig:inv_interior` | Inversion interior coalescence | **Figure 6** (renumbered; was already main text) |
| Figure `fig:anogam_w200` | AnoGam window-size trade-off | **SI Appendix, Fig. S5** |
| Supp. Fig. `fig:adapter` | Sample-size adapter results | **SI Appendix, Fig. S4** |
| Supp. Fig. `fig:time_benchmark` | Runtime benchmark | **SI Appendix, Fig. S6** |
| Supp. Fig. `fig:interp_corr` | Interpolation/extrapolation grid | **SI Appendix, Fig. S12** |
| Table `tab:mses` | MSE summary table | **SI Appendix, Table S6** |
| Table `tab:model-config` | Model hyperparameters | **SI Appendix, Table S1** |
| Algorithm `alg:alg1` | Genotype processing | **SI Appendix, Algorithm 1** |
| Algorithm `alg:projEmbed` | Projection embedding | **SI Appendix, Algorithm 2** |
| Figure `fig:proc` | Preprocessing schematic | **SI Appendix, Fig. S1** |

---

## D. All MSE numbers changed (Figure 2 / Benchmark Comparisons)

Every reported MSE value in the benchmark section was updated (presumably from re-runs):

| Model / Scenario | Old MSE | New MSE |
|---|---|---|
| **cxt-narrow**, constant | 0.2447 | **0.2531** |
| **cxt-narrow**, sawtooth | 0.6431 | **0.7496** |
| **cxt-broad**, sawtooth | 0.1807 | **0.1796** |
| **Singer+Polegon**, constant | 0.2464 | **0.2470** |
| **Singer+Polegon**, sawtooth | 0.2430 | **0.2129** |
| **SMC++**, constant | 0.9032 | **0.8685** |
| **SMC++**, sawtooth | 2.7716 | **1.7919** |

---

## E. Introduction changes (beyond §1 already documented)

1. **Section heading removed**: Old had `\section*{Introduction}`; new drops it per PNAS convention ("no heading per PNAS convention" comment retained).

2. **Citation prefixes removed**: Several `\citep[e.g.][]{…}` changed to plain `\cite{…}`:
   - Old l.203: `\citep[e.g.][]{sellinger_inference_2020,…}` → New l.91: `\cite{sellinger_inference_2020,…}`
   - Old l.211: `\citep[e.g.][]{beaumont_approximate_2002}` → New l.98: `\cite{beaumont_approximate_2002}`
   - Old l.227: `\citep[as reviewed in][]{korfmann_deep_2023}` → New l.112: `\cite{korfmann_deep_2023}`
   - Old l.234: `\citep[e.g.][]{mo2023domain}` → New l.119: `\cite{mo2023domain}`
   - Old l.240: `\citep[e.g.][]{nait_saada_inference_2023}` → New l.124: `\cite{nait_saada_inference_2023}`

3. **Em-dash to comma**: Old l.219 "settings—exemplified by PSMC-like models—" → New l.105 "settings, exemplified by PSMC-like models,"

4. **`\Parasplit` macro**: Old l.198 uses `\Parasplit` after PSMC sentence. New l.86 keeps it.

5. **Paragraph on GPT / LMs** (§1 above): Shortened from 8 lines to 4; content on "pretext task", "hundreds of billions of parameters", GPT context removed.

6. **Next-coalescence prediction** (§2 above): Full paragraph collapsed from free-form description into tighter definition.

---

## F. Results section — Fast and Accurate Inference

1. **Schematic reference**: Old "Figure~\ref{fig:schematic}A" → New "the left panel of Figure~1".

2. **Right-panel description** (§3 above): Heavily shortened. Removed mention of "batch of size 1225", "80GB memory", and the sentence-by-sentence description of colors. New version merges these into a compact paragraph.

---

## G. Results section — Benchmark Comparisons

1. **MSE values**: All seven values updated (see §D above).

2. **Formatting**: Old uses em-dash `---` for parenthetical; new uses comma.
   - Old l.393: `"narrow model''---a \cxtkit{} model` → New l.235: `"narrow model,'' a \cxtkit{} model`
   - Old l.398: `domain---constant population size` → New l.239: `domain, constant population size`
   - Old l.405: `broad model''---` → New l.248: `broad model,''`

3. **Sample-size adapter paragraph** (§4 above): Three options reduced to two sentences. "Supplementary Figure~\ref{fig:adapter}" → "SI Appendix, Fig.~S4".

4. **Table reference**: Old "Table~\ref{tab:mses}" → New "SI Appendix, Table~S6".

---

## H. Results section — Computational Efficiency

1. **Supplementary reference**: Old "Supplementary Figure~\ref{fig:time_benchmark}" → New "SI Appendix, Fig.~S6".
2. **Text**: Old "Supplementary Material" → New "SI Appendix".

---

## I. Results section — Towards a generalizable model

1. **Marginal distribution figure**: Old "Figure~\ref{fig:margdist_nomap}" → New "SI Appendix, Fig.~S15" (figure moved to SI).
2. **Map figure**: Old "Supplementary Figure~\ref{fig:margdist_map}" → New "SI Appendix, Fig.~S7".
3. **Evaluations reference**: Old "Figures~\ref{fig:margdist_nomap} and \ref{fig:margdist_map}" → New "SI Appendix, Figs.~S15 and S7".
4. **Window-size trade-off figure**: Old "Figure~\ref{fig:anogam_w200}" → New "SI Appendix, Fig.~S5".
5. **Fine-tune reference**: Old "Figure~\ref{fig:anogam_w200} right" → New "SI Appendix, Fig.~S5, right".
6. **Formatting**: Old uses em-dash `---through many learnable parameters---`; new uses comma.

---

## J. Results section — Out-of-sample tests

1. **Figure reference**: Old "Figure~\ref{fig:margdist_stdpop3}" → New "Figure~3" (renumbered).

---

## K. Results section — Generalization across parameter regimes

1. **Entire paragraph shortened** (§5 above): Multi-sentence description of panels A–D collapsed.
2. **Figure references**: All "Supplementary Figure~\ref{fig:interp_corr}" → "SI Appendix, Fig.~S12"; "Figure \ref{fig:interp2_corr}" removed (folded into same reference).

---

## L. Results section — Calibration

1. **Reference**: Old "Supplementary Material~\ref{fig:posterior-calibration}" → New "SI Appendix" (generic).

---

## M. Results section — Demography estimation

1. **Figure references**: Old "Figure~\ref{fig:iicr}" → New "Figure~4"; Old "Figure~\ref{fig:iicr_cross}" → New "SI Appendix, Fig.~S10".
2. **Cross-coalescence paragraph**: Old "Figure~\ref{fig:iicr_cross} summarizes these results" → New "SI Appendix, Fig.~S10 summarizes these results".
3. **Short-sequences paragraph** (§6 above): Shortened from 6 sentences to 2.
4. Old: "Figures \ref{fig:iicr} and \ref{fig:iicr_cross}" → New: "Figure 4 and SI Appendix, Fig.~S10".

---

## N. Results section — Application to empirical data

1. **Human figures**: Old "Figure~\ref{fig:chr2_6_corr}" → New "Figure~5".
2. **Recombination validation**: Old "Fig.~\ref{fig:tmrca_recom}" → New "SI Appendix, Fig.~S13".
3. **Anopheles comparison panel** (§7 above): Old described a three-column panel inline; new shortened.
   - Old "Figure \ref{fig:anopheles}" → New "SI Appendix, Fig.~S11".
   - Old "Figure \ref{fig:anopheles_main}" → New "SI Appendix, Fig.~S14".
4. **Singer missing-data reference**: Old "see Figure \ref{fig:anopheles}" → New "see SI Appendix, Fig.~S11".
5. **SMC++ paragraph** (§8 above): Shortened.

---

## O. Discussion changes

1. **Em-dash formatting**: Old "likelihood---and its enforced first-order dependence---" → New "likelihood, and its enforced first-order dependence,".
2. **Table reference**: Old "Table \ref{tab:mses}" → New "SI Appendix, Table~S6".
3. **Figure references**: Old "Figure~\ref{fig:chr2_6_corr}, top right" → New "Figure~5, top right"; Old "Figure \ref{fig:anopheles}, bottom panels" → New "SI Appendix, Fig.~S14, bottom panels".
4. **Known-outliers em-dash**: Old "---known coalescent time outliers" → New "known coalescent time outliers," (em-dash removed).
5. **Species em-dash**: Old "---with potentially different mutation" → New "with potentially different mutation" (em-dash to comma in preceding sentence).
6. All comments (`% what this paper does...`, `% NSP:`, `% ADK:`, `% KK:`) removed from new version.

---

## P. Methods changes

1. **Introductory paragraph removed**: Old had a 4-sentence overview ("We start by providing a detailed overview…"); new version cuts straight to subsections.

2. **Typo fixes**:
   - Old l.1061: "commenly" → New l.719: "commonly"
   - Old l.1112: "these these" → New: "these" (duplicate word removed)
   - Old l.1114: "opperations" → New: "operations"
   - Old l.1250: "unaccessible" → New l.849: "inaccessible" (×2 occurrences)

3. **Algorithm/figure/table references updated**:
   - Old "Algorithm~\ref{alg:alg1}" → New "SI Appendix, Algorithm~1"
   - Old "Figure~\ref{fig:proc}" → New "SI Appendix, Fig.~S1"
   - Old "Algorithm~\ref{alg:projEmbed}" → New "SI Appendix, Algorithm~2"
   - Old "Table~\ref{tab:model-config}" → New "SI Appendix, Table~S1"

4. **324 intervals phrasing**: Old "324; Table~\ref{tab:model-config}) discrete intervals" → New "324 discrete intervals (SI Appendix, Table~S1)".

5. **Calibration reference**: Old "Supplementary Materials~\ref{app:clock}" → New "SI Appendix".

6. **Ne estimation reference**: Old "Appendix~\ref{app:ne}" → New "SI Appendix".

7. **Datasets reference**: Old "Supplementary Material and online manual" → New "SI Appendix and in the online manual".

8. **Environmental Considerations**: Section title capitalization changed: old "Environmental considerations" → new "Environmental Considerations".

---

## Q. Acknowledgments

| Old | New |
|---|---|
| Three separate sentences; no reviewer thanks | Added: "We thank the two anonymous reviewers who provided constructive feedback on our manuscript." |

---

## R. Data Availability / Competing Interests

- Old: separate `\section*{Data and Code Availability}` and `\section*{Competing Interests}`.
- New: PNAS `\dataavail{…}` and `\authordeclaration{…}` front-matter macros.

---

## S. Supplementary material

- Old: Supplementary material appended inline after `\beginsupplement` (tables S2–S5, supplementary figures, algorithms, "Things we tried" section).
- New: All supplementary content removed from `Research_report.tex` and placed in a separate `cxt_supplementary/` directory.

---

## T. Figure legend text changes

### Figure 1 legend
- Old: "(A)" and "panel~(B)" → New: "(Left)" and "In the right panel"
- Old: "%(and the computation can be distributed across multiple GPUs)" comment removed

### Figure 2 legend
- Identical text.

### Figure 3 legend (new; was old Figure for stdpopsim v0.3)
- Old caption described v0.3 evaluation inline. New legend is rewritten for PNAS format, adds: "Top Rows: cxt; Middle Rows: Singer+Polegon; Bottom Rows: SMC++".

### Figure 4 legend (was old Figure 5 / IICR)
- Identical text.

### Figure 5 legend (was old Figure 6 / human chr2&6)
- Old: "The top-left panel shows chromosome~2, and the top-right panel shows chromosome~6. The LCT region (bottom-left) and HLA region (bottom-right) are zoomed respectively."
- New: "The top panel shows chromosome-wide TMRCA landscapes for chromosomes~2 (left) and~6 (right). The bottom panel zooms into the LCT region on chromosome~2 (left) and the HLA region on chromosome~6 (right)."

### Figure 6 legend (inversion interior)
- Old: caption was inline. New: moved to Figure Legends section with added note about Ghana sample size and adapter.

---

## U. `\begin{comment}` blocks

All `\begin{comment}…\end{comment}` blocks present in the old manuscript (containing alternative/draft text) have been removed from the new version.
