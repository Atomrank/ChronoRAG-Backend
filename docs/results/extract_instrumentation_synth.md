# Extract instrumentation (synth)

doc_id: `doc_b281c9bd70bedd34`
extract_llm_call count: **30**
extract_output_truncated count: **1**
json_salvaged count: **0**

## finish_reason histogram

- `stop`: 29
- `length`: 1

## by pipeline / phase

- `kaalkram_v1` / `pass1`: 1
- `kaalkram_v2` / `extract`: 29

## Truncation verdict

**CONFIRMED**: 1 call(s) hit finish_reason length/max_tokens (hard error; partial output not accepted).

## Per-call table

| pipe | phase | win | start-end | in_chars | max_tok | finish | out_tok | n_items | salvage |
|------|-------|-----|-----------|----------|---------|--------|---------|---------|---------|
| kaalkram_v2 | extract | 0 | 0-514 | 813 | 12000 | stop | 806 | 7 | False |
| kaalkram_v2 | extract | 1 | 516-1151 | 1145 | 12000 | stop | 848 | 7 | False |
| kaalkram_v2 | extract | 2 | 1153-1905 | 1607 | 12000 | stop | 314 | 2 | False |
| kaalkram_v2 | extract | 2 | 1153-1556 | 962 | 12000 | stop | 437 | 4 | False |
| kaalkram_v2 | extract | 2 | 1558-1905 | 1051 | 12000 | stop | 660 | 4 | False |
| kaalkram_v2 | extract | 3 | 1907-2548 | 1723 | 12000 | stop | 606 | 5 | False |
| kaalkram_v2 | extract | 4 | 2550-3341 | 1742 | 12000 | stop | 493 | 3 | False |
| kaalkram_v2 | extract | 4 | 2550-2852 | 1006 | 12000 | stop | 550 | 5 | False |
| kaalkram_v2 | extract | 4 | 2854-3341 | 1191 | 12000 | stop | 700 | 4 | False |
| kaalkram_v2 | extract | 5 | 3343-3879 | 1864 | 12000 | stop | 747 | 7 | False |
| kaalkram_v2 | extract | 6 | 3881-4499 | 1786 | 12000 | stop | 408 | 3 | False |
| kaalkram_v2 | extract | 6 | 3881-4149 | 1229 | 12000 | stop | 427 | 4 | False |
| kaalkram_v2 | extract | 6 | 4151-4499 | 1309 | 12000 | stop | 553 | 4 | False |
| kaalkram_v2 | extract | 7 | 4501-5158 | 2194 | 12000 | stop | 997 | 8 | False |
| kaalkram_v2 | extract | 8 | 5160-5757 | 2683 | 12000 | stop | 922 | 8 | False |
| kaalkram_v2 | extract | 9 | 5759-6375 | 2859 | 12000 | stop | 1232 | 8 | False |
| kaalkram_v2 | extract | 10 | 6377-6932 | 3257 | 12000 | stop | 897 | 8 | False |
| kaalkram_v2 | extract | 11 | 6934-7439 | 3217 | 12000 | stop | 868 | 8 | False |
| kaalkram_v2 | extract | 12 | 7441-7899 | 3168 | 12000 | stop | 866 | 8 | False |
| kaalkram_v2 | extract | 13 | 7901-8406 | 3219 | 12000 | stop | 953 | 8 | False |
| kaalkram_v2 | extract | 14 | 8408-8884 | 3186 | 12000 | stop | 641 | 6 | False |
| kaalkram_v2 | extract | 15 | 8886-9356 | 3187 | 12000 | stop | 216 | 2 | False |
| kaalkram_v2 | extract | 15 | 8886-9118 | 2726 | 12000 | stop | 426 | 4 | False |
| kaalkram_v2 | extract | 15 | 9120-9356 | 2730 | 12000 | stop | 445 | 4 | False |
| kaalkram_v2 | extract | 16 | 9358-9813 | 3178 | 12000 | stop | 856 | 8 | False |
| kaalkram_v2 | extract | 17 | 9815-10429 | 3334 | 12000 | stop | 1222 | 7 | False |
| kaalkram_v2 | extract | 18 | 10431-11069 | 3288 | 12000 | stop | 951 | 9 | False |
| kaalkram_v2 | extract | 19 | 11071-11726 | 3273 | 12000 | stop | 805 | 7 | False |
| kaalkram_v2 | extract | 20 | 11728-11904 | 2639 | 12000 | stop | 214 | 2 | False |
| kaalkram_v1 | pass1 | 6 | 1-6 | 11958 | 1600 | length | 1600 | None | None |

## Truncated calls (errors)

- `{"kind": "extract_output_truncated", "finish_reason": "length", "max_tokens": 1600, "input_chars": 11958, "completion_tokens": 1600, "pipeline": "kaalkram_v1", "phase": "pass1", "doc_id": "doc_b281c9bd70bedd34", "window_id": "win_1_6", "window_index": 6, "window_start": 1, "window_end": 6, "window_chars": 11802, "max_tokens_setting": 1600, "attempt": 0}`
