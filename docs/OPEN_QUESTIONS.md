# Open questions

## Notes

- Sabha Parva ingested as `doc_30ae2bd6556705ff` (`structure_source=font`; see
  `docs/ingest_report_sabha.txt`). No empty pages; no paragraphs >5k chars.
- The Old Man and the Sea PDF is not present under `data/books/` (copyright);
  Sabha + synthetic only for this run.
- `GOLD_FULLTEXT_JUDGE` and `LOCAL_LLM_*` unset: Sabha silver gold will be
  before/after only; unordered/made-up-order from synthetic.
