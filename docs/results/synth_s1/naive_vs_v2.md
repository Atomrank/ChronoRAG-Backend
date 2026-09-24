# Synth seed-1: naive vs kaalkram_v2 (llm extractor)

gold: `synth_s1.jsonl` (cleaned questions, same for both)
naive: `run_20260924_232553_naive_765a1c`
v2_llm: `run_20260924_231311_kaalkram_v2_a51196`

| metric | naive | v2_llm |
|---|---:|---:|
| overall acc | 0.461 | 0.565 |
| overall coverage | 0.695 | 0.526 |
| selective acc | 0.607 | 0.963 |
| aligned acc | 0.478 | 0.574 |
| inverted acc | 0.000 | 0.000 |
| unordered acc | 0.600 | 0.900 |
| aligned cov | 0.735 | 0.588 |
| inverted cov | 0.375 | 0.000 |
| unordered cov | 0.400 | 0.100 |
| hit@5 | 0.779 | 0.253 |
| mrr | 0.560 | 0.167 |
| recall@5 | 0.744 | 0.146 |
| ndcg@10 | 0.625 | 0.258 |

## Paired compare (v2_llm as A, naive as B)

```json
{
  "n": 154,
  "mcnemar": {
    "a_only": 41,
    "b_only": 25,
    "p_value": 0.06401750413722564
  },
  "accuracy_diff": {
    "diff": 0.1038961038961039,
    "lo": 0.0,
    "hi": 0.2077922077922078,
    "p_one_sided": 0.0268,
    "n": 154
  },
  "aligned": {
    "n": 136,
    "mcnemar": {
      "a_only": 37,
      "b_only": 24,
      "p_value": 0.12373144253809217
    }
  },
  "inverted": {
    "n": 8,
    "mcnemar": {
      "a_only": 0,
      "b_only": 0,
      "p_value": 1.0
    }
  },
  "unordered": {
    "n": 10,
    "mcnemar": {
      "a_only": 4,
      "b_only": 1,
      "p_value": 0.375
    }
  }
}
```
