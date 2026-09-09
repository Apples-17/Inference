# Benchmark results

`final_mt_bench/` contains the completed Tesla T4 run reported in the project
README. The most useful entry points are:

- `final_report.md`: compact human-readable result;
- `final_report.json`: complete machine-readable summary;
- `pass_sweep/summary.csv`: one row per optimization pass;
- `full_gate/summary.json`: full-corpus selection decisions; and
- `final/comparison.json`: detailed custom-model/vLLM comparison.

Raw per-pass and final engine JSON files are retained so the reported tables can
be audited without rerunning the GPU benchmark. Runtime logs are not committed.

