# DFlash online-hybrid vs offline baseline (ckpt0-2)

Acceptance is measured with vLLM 0.28 on GSM8K/HumanEval/MT-Bench (temperature 0, 1,563 prompts). `w/c` is the fraction of offline-correct draft tokens that cannot be accepted because an earlier prefix token is wrong. `cond` is the conditional acceptance rate P(pos i | pos i-1).

## Epoch 0


| model | AL | AR | reach | offline correct | P(waste) | waste/correct | pos6 waste/correct |
|---|---|---|---|---|---|---|---|
| dflash_hybrid | 3.7259 | 38.94% | 0.5955 | 0.5505 | 0.0715 | 0.1299 | 0.3701 |
| dflash_offline | 3.5939 | 37.06% | 0.5797 | 0.5599 | 0.0986 | 0.1761 | 0.4654 |

| pos | pos0 | pos1 | pos2 | pos3 | pos4 | pos5 | pos6 |
|---|---|---|---|---|---|---|---|
| dflash_hybrid acc | 0.802 | 0.596 | 0.447 | 0.336 | 0.247 | 0.177 | 0.122 |
| dflash_hybrid cond | 0.802 | 0.743 | 0.749 | 0.752 | 0.736 | 0.718 | 0.686 |
| dflash_hybrid w/c | 0.000 | 0.050 | 0.105 | 0.161 | 0.222 | 0.292 | 0.370 |
| dflash_offline acc | 0.784 | 0.577 | 0.425 | 0.314 | 0.228 | 0.159 | 0.107 |
| dflash_offline cond | 0.784 | 0.736 | 0.735 | 0.740 | 0.725 | 0.699 | 0.673 |
| dflash_offline w/c | 0.000 | 0.064 | 0.141 | 0.222 | 0.299 | 0.381 | 0.465 |

## Epoch 1


| model | AL | AR | reach | offline correct | P(waste) | waste/correct | pos6 waste/correct |
|---|---|---|---|---|---|---|---|
| dflash_hybrid | 4.0044 | 42.92% | 0.6338 | 0.5953 | 0.0694 | 0.1166 | 0.3045 |
| dflash_offline | 3.8505 | 40.72% | 0.6148 | 0.5988 | 0.0958 | 0.1599 | 0.4050 |

| pos | pos0 | pos1 | pos2 | pos3 | pos4 | pos5 | pos6 |
|---|---|---|---|---|---|---|---|
| dflash_hybrid acc | 0.821 | 0.630 | 0.489 | 0.382 | 0.295 | 0.224 | 0.163 |
| dflash_hybrid cond | 0.821 | 0.768 | 0.776 | 0.781 | 0.772 | 0.760 | 0.729 |
| dflash_hybrid w/c | 0.000 | 0.044 | 0.094 | 0.142 | 0.193 | 0.245 | 0.305 |
| dflash_offline acc | 0.804 | 0.613 | 0.467 | 0.356 | 0.269 | 0.199 | 0.142 |
| dflash_offline cond | 0.804 | 0.762 | 0.762 | 0.762 | 0.757 | 0.741 | 0.713 |
| dflash_offline w/c | 0.000 | 0.058 | 0.126 | 0.196 | 0.265 | 0.329 | 0.405 |

## Epoch 2


| model | AL | AR | reach | offline correct | P(waste) | waste/correct | pos6 waste/correct |
|---|---|---|---|---|---|---|---|
| dflash_hybrid | 4.4360 | 49.09% | 0.6837 | 0.6519 | 0.0633 | 0.0971 | 0.2311 |
| dflash_offline | 4.3394 | 47.71% | 0.6734 | 0.6610 | 0.0859 | 0.1300 | 0.3129 |

| pos | pos0 | pos1 | pos2 | pos3 | pos4 | pos5 | pos6 |
|---|---|---|---|---|---|---|---|
| dflash_hybrid acc | 0.849 | 0.680 | 0.550 | 0.450 | 0.368 | 0.300 | 0.238 |
| dflash_hybrid cond | 0.849 | 0.801 | 0.809 | 0.818 | 0.818 | 0.814 | 0.793 |
| dflash_hybrid w/c | 0.000 | 0.037 | 0.078 | 0.115 | 0.155 | 0.189 | 0.231 |
| dflash_offline acc | 0.840 | 0.672 | 0.541 | 0.437 | 0.352 | 0.280 | 0.218 |
| dflash_offline cond | 0.840 | 0.801 | 0.805 | 0.808 | 0.806 | 0.797 | 0.777 |
| dflash_offline w/c | 0.000 | 0.045 | 0.096 | 0.153 | 0.207 | 0.258 | 0.313 |

