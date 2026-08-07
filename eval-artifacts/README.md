# Evaluation artifacts — the evidence chain behind BENCH.md's FLEURS-fr numbers

Committed because the paper's numbers must be reproducible from the repository alone.
Score any hyp file with:

    python3 tools/score_wer.py eval-artifacts/test.tsv eval-artifacts/<hyp>.tsv --ci
    # A-vs-B claims: add --compare <other>.tsv  (paired bootstrap, p-value)

`test.tsv` is the FLEURS fr_fr test manifest (google/fleurs, CC-BY-4.0), 676 utterances,
17 997 reference words. Audio itself is fetched by `tools/fleurs_prepare.py` (peak-norm
to −3 dBFS, protocol stated there). Sweep drivers live in `tools/eval-scripts/`
(they expect the gitignored `eval-data/` layout).

## Full test set (676)

| file | config | WER | note |
|---|---|---|---|
| `hyp_ref.tsv` | kyutai reference PyTorch F32, fresh state/utt, tail=7 | 12.62 % | driven by `tools/ref_eval.py` |
| `hyp_f16_norm.tsv` | port F16, deployment mode, tail=14, linear resample | 11.29 % | CI [10.38, 12.26]; vs ref: −1.33 pt, p=0.005 |
| `hyp_q4k_norm.tsv` | port Q4_K, same protocol | **11.61 %** | ⚠ this is the post-crash RESUMED chain. The unbroken chain scored 11.40 % (BENCH.md) but its hyp file was not preserved — the ±0.2 pt gap is the measured session-segmentation variance. vs F16: +0.32 pt, p=0.10 (n.s.) |
| `hyp_combo676.tsv` | Q4_K + tail16 + prefix6 + AGC | 11.29 % | vs q4k base: −0.32 pt, p=0.32 (n.s. alone; deletions −23 % is the causal signal) |

## 113-file subset (⚠ also the tuning set — every 6th file of test, 2 979 ref words)

All serving/ensemble choices were selected on these files; treat subset WER as
development numbers. The full-676 confirmation is `tools/eval-scripts/full_sub10.sh`.

| file | config | WER |
|---|---|---|
| `s2_t16p8.tsv` | Q4_K, sinc, tail16 prefix8 (primary) | 10.31 % |
| `s2_t16p6_f16.tsv` | F16, sinc, tail16 prefix6 | 10.44 % |
| `s3_t8_f16.tsv` | F16, sinc, tail8 | 10.74 % |
| `s3_t8p4.tsv` | Q4_K, sinc, tail8 prefix4 | 10.84 % |
| `s2_t16p4.tsv` | Q4_K, sinc, tail16 prefix4 | 10.51 % |
| `rover_C.tsv` | 5-way vote, HISTORICAL (pre-fix rover.py: non-deterministic tie-break + head-insertion vote bug) | 9.67 % |
| `rover_C_fixed.tsv` | 5-way vote, deterministic rover.py | 9.70 % (vs primary: −0.60 pt, p=0.06) |
| `conf_t16p8.tsv` | primary + CONF margin lines (MOSHI_MARGIN=1) | feeds `tools/gated_ensemble.py` |
| `fleurs_hyp_device.tsv` | device (Poco F8 Ultra) run of the subset | 11.58 % |

Member diversity (tools/ensemble_diversity.py): pairwise error-count correlation
0.81–0.92, hypothesis disagreement 2.8–6.3 % of words, oracle-per-utterance 8.09 % —
i.e. a perturbation ensemble of one greedy decoder, not five systems; say so when
writing it up.
