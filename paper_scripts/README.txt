Scripts that regenerate the numbers of the paper (run from this folder, repository root is the parent).
  python -B paper_numbers.py   -> paper_numbers.json  (every value quoted in the text and tables)
  python -B mc_framework.py    -> mc_framework.json   (Monte Carlo framework, robustness, variance decomposition; seed 42)
  python -B ../run_model.py    -> baseline outputs in ../results
Tests: python -m pytest ../tests
The paper_numbers.json and mc_framework.json shipped here are the outputs used in the manuscript.
