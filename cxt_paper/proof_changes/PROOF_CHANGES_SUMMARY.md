# Summary of proof-stage changes

Last changes made at proof stage when converting to PNAS formatting.

---

## 1. Formatting 

The manuscript was reformatted from a single-column article to the PNAS two-column format. A separate SI Appendix file was created and internalreferences were updated accordingly (e.g. "Supplementary Figure X" -> "SI Appendix, Fig. SX").

## 2. Text shortenings for page limit

In every case the scientific content and conclusions are preserved; only redundant phrasing, procedural detail, and exposition were trimmed.

### 2a. Transformer / GPT introduction (Introduction)
Old `main.tex` lines 238–250 -> New `Research_report.tex` lines 122–129.

```diff
  More recently, transformer-based language models (LMs) have been introduced
  that enhance prediction capabilities through the learning of stochastic processes,
  rather than scenario-constrained parameter estimation.
- Transformer LMs are perhaps best known in the context of pretrained generative
- models for human language such as GPT. These models are pretrained in that they
- are first trained on a "pretext" task, such as next token prediction, that
- requires the LM to learn context among observations (tokens) in a sequence
- locally and their relation to higher-level associations in the space of language.
- Modern large LMs are in part wildly successful due to their massive numbers of
- parameters (e.g. hundreds of billions) that enable them to effectively learn
- rich representations and flexible conditional prediction across contexts and
- thus to generalize.
- Here we describe a pretrained generative model for population genetics that
- leverages a small language model (10-20M parameters) to effectively translate
- from mutational patterns across chromosomes to coalescent time estimates.
- This opens the door to a generalizable (non-task specific) model paradigm:
- pretraining a flexible transformer on a diverse range of coalescent simulations,
- followed by fine-tuning for specific evolutionary tasks, if necessary.
+ Inspired by pretrained generative models such as GPT, we leverage this paradigm
+ with a small model (10-20M parameters) that translates mutational patterns
+ across chromosomes to coalescent time estimates, opening the door to a
+ generalizable approach: pretraining on diverse coalescent simulations, followed
+ by fine-tuning for specific evolutionary tasks if necessary.
```

### 2b. Next-coalescence prediction definition (Introduction)
Old `main.tex` lines 259–260 -> New `Research_report.tex` lines 136–142.

```diff
- To enable this inference, we define a novel task, analogous to next-token
- prediction in language modeling, which we term next-coalescence prediction. In
- this formulation, the model predicts the next coalescence time along a sequence
- of previously predicted events, across discrete sequence space, conditioned on
- local mutation densities within a fixed context window. While this does not
- reconstruct full genealogical topologies, it enables a structured prediction
- task that translates between observed mutation patterns and pairwise coalescence
- times -- akin to approaches such as PSMC or SMC++, the latter being the method
- to which cxt is conceptually a close neural analog, in the sense that
- prediction happens on a pivot (or distinguished pair), while sharing information
- across haplotypes through the SFS. Thus, the model offers a way to navigate
- between the observed distribution of mutation densities and the latent pairwise
- genealogical structure by means of a sequence of coalescence event prediction.
+ To enable this inference, we define a novel task analogous to next-token
+ prediction, which we term next-coalescence prediction: the model
+ autoregressively predicts the next pairwise coalescence time conditioned on
+ local mutation densities within a fixed context window. While this does not
+ reconstruct full genealogical topologies, it provides a structured translation
+ from observed mutation patterns to pairwise coalescence times. Conceptually,
+ cxt is a close neural analog of SMC++: prediction is centered on a pivot
+ (distinguished) pair, with information shared across haplotypes through the SFS.
```

### 2c. Right panel of Figure 1 description (Fast and Accurate Inference)
Old `main.tex` lines 369–374 -> New `Research_report.tex` lines 218–226.

```diff
- The right panel of Figure 1 shows a concrete example of the model's output.
- Leveraging our GPU-enabled method, we infer pairwise coalescence times for all
- C(50,2) pairs of a sample of 50 haploid chromosomes in parallel by placing them
- in a batch of size 1225 (although note this batch size is dependent on GPU
- memory, and can be adjusted to available GPU memory). We then run the generative
- process multiple times, typically 15 replicates, and average the results to
- obtain stable and robust estimates. This generative process provides samples
- from an approximate posterior distribution of pair coalescence times across the
- sequence (where the prior is implicit via the simulated training set; see
- Calibration). In the right panel of Figure 1, we apply our cxt to a constant
- population size scenario with a population size of 2x10^4 and approximately
- equal mutation and recombination rates (see Datasets below). In that figure
- true coalescent times are shown in black, the replicate predictions in light
- blue, and the average prediction in dark blue. In this simple setting, where the
- model is well-specified, cxt produces highly accurate predictions. Importantly,
- cxt can predict coalescent times for all pivot pairs concurrently in
- approximately five minutes of computation on an NVIDIA A100 GPU with 80GB memory.
+ The right panel of Figure 1 illustrates cxt on a constant-size scenario
+ (N_e = 2x10^4, roughly equal mutation and recombination rates; see Datasets).
+ All C(50,2) pivot pairs of 50 haploid chromosomes are inferred in parallel;
+ for each pair the generative process is repeated 15 times, yielding samples
+ from an approximate posterior over pairwise coalescence times along the sequence
+ (where the prior is implicit via the simulated training set; see Calibration).
+ Averaging across replicates produces stable estimates.
+ In this well-specified setting cxt is highly accurate,
+ and inference for all pairs completes in approximately five minutes on a single
+ NVIDIA A100 GPU.
```

### 2d. Sample-size adapter (Benchmark Comparisons)
Old `main.tex` lines 420–428 -> New `Research_report.tex` lines 262–266.

```diff
- As cxt is trained at a specific sample size, the user has a few options should
- they want to infer coalescent times in a new sample of different size:
- 1) they could retrain the model starting from a new set of simulations that
- condition on their sample size, 2) if the sample size is larger than the
- trained model, they could subsample down to the appropriate size, or 3) if the
- sample is smaller than the trained model we have created a light-weight adapter,
- which quickly fine-tunes a larger sample size trained cxt model to a smaller
- sample size. In SI Appendix, Fig. S4, we show representative results from the
- adapter applied to samples of size N=5 diploids from a cxt model trained at
- sample size N=25 diploids, and then used for inference of a constant sized
- simulation (left panel) or the sawtooth simulation (right panel).
+ Because cxt is trained at a fixed sample size, users with larger samples can
+ subsample, while for smaller samples we provide a lightweight adapter that
+ fine-tunes the pretrained model to the target cohort size.
+ SI Appendix, Fig. S4 shows adapter results transferring from N=25 to N=5
+ diploids on both the constant-size and sawtooth scenarios.
```

### 2e. Interpolation / extrapolation across parameter regimes (Generalization)
Old `main.tex` lines 549–557 -> New `Research_report.tex` lines 368–371.

```diff
- In SI Appendix, Fig. S12, we assess the model's ability to interpolate and
- extrapolate beyond the training distribution with respect to mutation and
- recombination rates, holding Ne fixed at 20,000. We define a grid of mutation
- and recombination rates chosen to encompass most stdpopsim species (panel A);
- the distribution of species across this grid is shown in panel B. Panel A also
- highlights a range of recombination-to-mutation rate ratios (B), while panels C
- and D report performance over the grid in terms of MSE and KL divergence,
- respectively. As expected, error is highest in the low-mutation,
- high-recombination region (bottom right of C) where the signal-to-noise ratio
- is the least favorable for TMRCA inference. Fig. S12 provides representative
- examples of TMRCA predictions from the four corners of the grid. Together,
- these results show that cxt can generalize beyond the specific regions of
- parameter space represented in the training data.
+ We assessed cxt's ability to interpolate and extrapolate beyond the training
+ distribution by evaluating prediction accuracy across a grid of mutation and
+ recombination rates at fixed N_e = 20,000 (SI Appendix, Fig. S12). Panel A
+ defines the grid, chosen to span most stdpopsim species, with their
+ distribution shown in panel B. Panels C and D report MSE and KL divergence,
+ respectively; as expected, error is highest in the low-mutation,
+ high-recombination regime where the signal-to-noise ratio is least favorable.
+ Representative TMRCA predictions from the four corners of the grid confirm
+ that cxt generalizes beyond the parameter regions represented in training.
```

### 2f. Demographic inferences from short sequences (Demography estimation)
Old `main.tex` lines 637–645 -> New `Research_report.tex` lines 427–430.

```diff
- It is worth noting that the demographic inferences shown in Figure 4 and SI
- Appendix, Fig. S10 are based on relatively short simulated sequences. For
- computational efficiency, we restrict these demonstrations to 10 Mb of
- simulated genome. Because shorter sequences increase the variance of
- recent-time coalescence-rate estimates by truncating the longest tracts of
- recent ancestry, we expect that applying the same procedure to full-length
- chromosomes would yield substantially improved accuracy. Together, these
- results show that cxt's local TMRCA predictions can be aggregated into
- accurate genome-scale summaries of demography, with performance expected to
- improve further at realistic chromosome lengths.
+ These demographic inferences are based on 10 Mb of simulated sequence (Figure 4
+ and SI Appendix, Fig. S10); because shorter sequences truncate the longest
+ tracts of recent ancestry, increasing variance in recent-time rate estimates,
+ we expect substantially improved accuracy at full chromosome lengths. Overall,
+ cxt's local TMRCA predictions aggregate into accurate genome-scale demographic
+ summaries.
```

### 2g. Rdl comparison panel description (Application to Ag1000G)
Old `main.tex` lines 723–733 -> New `Research_report.tex` lines 498–504.

```diff
  To investigate selective dynamics at the Rdl locus across multiple
  A. gambiae populations, we constructed a comparison panel (SI Appendix,
- Fig. S11) comprising (i) coalescent-time landscapes inferred by cxt (left
- column), (ii) corresponding estimates from Singer+Polegon (middle column),
- and (iii) SMC++ decodings (right column).
- Analyzing Ag1000G data poses two practical challenges.
- First, large effective population sizes imply high N_e-scaled mutation and
- recombination rates, which increase genealogical heterogeneity and the number
- of recombination events, substantially slowing MCMC-based approaches such as
- Singer+Polegon. Second, missingness varies widely across genomic regions,
- requiring methods that can accommodate irregular observation patterns (density
- of missing data is shown beneath the upper and lower panels as a track in SI
- Appendix, Fig. S11).
- To improve readability in the main text, we restrict these figures to cxt (SI
- Appendix, Fig. S14); full comparisons with Singer+Polegon and other approaches
- are shown in the SI Appendix, where these analyses are presented in detail.
+ Fig. S11) showing coalescent-time landscapes from cxt, Singer+Polegon,
+ and SMC++.
+ The challenges outlined above are particularly acute here; high N_e-scaled
+ rates substantially slow MCMC-based approaches such as Singer+Polegon.
+ For readability, main-text figures show only cxt results (SI Appendix,
+ Fig. S14); full method comparisons appear in the SI Appendix.
```

### 2h. SMC++ blockwise representation at Rdl (Application to Ag1000G)
Old `main.tex` lines 795–804 -> New `Research_report.tex` lines 549–555.

```diff
  SMC++ also detects a noticeable reduction in TMRCA around the Rdl sweep,
- but its blockwise representation limits how precisely this signal can be
- localized along the genome.
- Even after scaling the SMC++ output to match the true genomic span of the
- region and applying manual calibration so that its genome-wide diversity
- patterns align with the cxt and Singer+Polegon panels,
+ but its blockwise representation limits localization.
+ Even after scaling the output to match the true genomic span and calibrating
+ genome-wide diversity to the cxt and Singer+Polegon panels,
  the inferred trough remains discretized and slightly shifted.
- This behavior is expected:
- SMC++ models the genome using coarse blocks under a global demographic process,
- so narrow sweeps can be detected, but their boundaries are blurred by the
- block structure. As a result, SMC++ can identify strong selective events,
- but it is less able to resolve their precise genomic breakpoints.
+ This is expected: SMC++ fits a global demographic process using coarse blocks,
+ so narrow sweeps can be detected but their genomic breakpoints are less
+ precisely resolved.
```

## 3. Updated benchmark numbers (Figure 2)

All MSE values in the benchmark comparison were recomputed as part of standardizing the reproducible evaluation pipeline:

| Model / Scenario | Old | New |
|---|---|---|
| cxt-narrow, constant | 0.2447 | 0.2531 |
| cxt-narrow, sawtooth | 0.6431 | 0.7496 |
| cxt-broad, sawtooth | 0.1807 | 0.1796 |
| Singer+Polegon, constant | 0.2464 | 0.2470 |
| Singer+Polegon, sawtooth | 0.2430 | 0.2129 |
| SMC++, constant | 0.9032 | 0.8685 |
| SMC++, sawtooth | 2.7716 | 1.7919 |

In the course of building the fully reproducible pipeline, we identified and corrected minor bookkeeping errors in the evaluation scripts for the baseline methods. These were small issues in the seeds of the post-hoc correction approach; no qualitative conclusion is affected.


## 4. Figure renumbering

Figures were renumbered to accommodate the PNAS main-text figure limit. The stdpopsim v0.2 marginal-distribution figure (old Fig. 3) was moved to the SI Appendix (now Fig. S15), and the mosquito Rdl figure (old Fig. 7) was moved to the SI Appendix (now Fig. S14). Subsequent figures were renumbered: old Fig. 4 -> Fig. 3, old Fig. 5 -> Fig. 4, old Fig. 6 -> Fig. 5, old Fig. 8 -> Fig. 6. The inversion-interior figure remains in the main text as Fig. 6.

## 5. Revised Singer+Polegon comparison language (reviewer response)

The original manuscript claimed cxt "performs on par with Singer+Polegon" in out-of-sample testing. A reviewer pointed out that the out-of-sample figure shows cxt has higher error in nearly every case, by as much as ~50%, and that practitioners are better off using Singer+Polegon if willing to wait. The following edits address it.

### Abstract (main.tex line 150)

```diff
- We show that cxt performs on par with state-of-the-art MCMC-based likelihood
- models across a broad range of demographic scenarios, including both
- in-distribution and out-of-distribution settings.
+ We show that cxt performs competitively with state-of-the-art MCMC-based
+ likelihood models across a broad range of demographic scenarios, matching their
+ accuracy in-distribution and approaching it in out-of-distribution settings,
+ with the potential for further improvement via targeted fine-tuning.
```

### Introduction, results summary (main.tex lines 269–275)

```diff
  Using extensive simulations and applications to human and mosquito genomic
  variation data,
- we show that cxt provides competitive, state-of-the-art performance for
+ we show that cxt provides competitive performance for
  inferring local coalescence times across recombining chromosomes,
  ...
- cxt maintains strong out-of-sample performance and can be further improved
- via targeted fine-tuning.
+ cxt maintains reasonable out-of-sample performance, though with some loss in
+ accuracy relative to likelihood-based methods, and can be further improved via
+ targeted fine-tuning.
```

### Benchmark conclusion (main.tex lines 438–439)

```diff
- Our results demonstrate that the language modeling approach implemented in cxt
- is highly competitive with the most recent theory-driven methods, whether
- ARG-based (as in Singer) or SMC-based (as in SMC++).
+ Our results demonstrate that the language modeling approach implemented in cxt
+ is competitive with the most recent theory-driven methods, whether ARG-based
+ (as in Singer) or SMC-based (as in SMC++), in these well-specified settings.
```

### Out-of-sample results (main.tex line 539)

```diff
- Across these v0.3 species, cxt performs on par with Singer+Polegon and
- substantially better than the SMC++ decodings.
+ Across these v0.3 species, cxt generalizes reasonably well, although
+ Singer+Polegon achieves lower error in most scenarios, in some cases by a
+ substantial margin. Both methods substantially outperform the SMC++ decodings.
```

### Discussion (main.tex lines 914–918)

```diff
- Comparisons to SMC++ and Singer+Polegon show that cxt
- consistently outperforms SMC++ in terms of accuracy of TMRCA inference
- (Table S6), and matches the state-of-the-art ARG-based method Singer+Polegon,
- a strong benchmark given its joint use of all samples.
- Achieving comparable performance without explicit ARG modeling highlights
- the efficiency of the language-model approach.
+ Comparisons to SMC++ and Singer+Polegon show that cxt
+ consistently outperforms SMC++ in terms of accuracy of TMRCA inference
+ (Table S6), and performs competitively with the state-of-the-art ARG-based
+ method Singer+Polegon---matching it in well-specified settings, though with
+ higher error in some out-of-distribution scenarios.
+ Singer+Polegon remains the more accurate choice when computational cost is
+ not a constraint, but achieving competitive performance without explicit ARG
+ modeling highlights the efficiency of the language-model approach.
```

---

## 6. Minor corrections

- Four typos fixed: "commenly" -> "commonly", "these these" -> "these", "opperations" -> "operations", "unaccessible" -> "inaccessible".
- Citation prefixes (e.g. "[e.g.]", "[as reviewed in]") removed to match PNAS citation style.
- Acknowledgments updated to thank the two anonymous reviewers.

## 7. Phasing clarification (reviewer response)

A reviewer noted that the description of the genotype matrix (`G in Z^{N x M}`) appeared to describe unphased data, yet the method operates on haplotype pairs. The following edit to the "Processing of Coalescent Simulations" subsection clarifies (a) that the matrix contains 2N haploid sequences, (b) that all empirical analyses use within-individual pairs for which phasing is not required, and (c) that cross-individual pairs (used only in simulated benchmarks) would require phased haplotypes.

Old `main.tex` lines 1079-1084 -> New `Research_report.tex` lines 729-742.

```diff
- We begin with a genotype matrix G in Z^{N x M}
- containing N diploid samples across M sites,
- from coalescent simulations (here msprime).
- Phasing is not required for real data,
- as using both haplotypes per individual yields the same pivot-pair
- conditioning used throughout.
+ We begin with a genotype matrix G in Z^{2N x M}
+ containing 2N haploid sequences (from N diploid individuals) across M sites,
+ obtained from coalescent simulations (here msprime) or empirical data.
+ All empirical analyses in this work use within-individual pivot pairs
+ (the two haplotypes of one diploid, analogous to the PSMC setting),
+ for which phasing is not required:
+ heterozygous and homozygous sites are determined by the genotype alone,
+ so the XOR/XNOR features and SFS weights that constitute the model input
+ are identical whether or not phase is known.
+ Cross-individual pivot pairs, used here only in simulated benchmarks
+ where phase is known by construction,
+ would require phased haplotypes;
+ in that setting, standard statistical phasing tools
+ can be applied as a routine upstream step.
```
