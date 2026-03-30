# Latest Status

Updated: 2026-03-30T13:10:45.683813+00:00

## Current Best

- Run: `incremental_resdil_step1`
- Model: `resdil_cnn1d`
- Best validation Q8: `0.7268`
- Test Q8 on CB513: `0.6878`
- Test loss: `0.8872`

## Runner-up

- Run: `cnn1d_optuna_final`
- Model: `cnn1d`
- Best validation Q8: `0.7203`
- Test Q8 on CB513: `0.6820`
- Test loss: `0.8967`

## Notes

- Incremental research is active and every experiment is appended to `outputs/reports/run_ledger.csv`.
- The current best run lowered test loss and improved test Q8 over the tuned CNN1D baseline.
- Detailed artifacts live under `artifacts/`.
