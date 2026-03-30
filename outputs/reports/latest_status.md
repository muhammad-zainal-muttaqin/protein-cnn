# Latest Status

Updated: 2026-03-30T12:51:36.192006+00:00

## Current Best

- Model: `cnn1d`
- Phase: `tune_final`
- Best validation Q8: `0.7203`
- Test Q8 on CB513: `0.6820`

## Runner-up

- Model: `cnn2d`
- Best validation Q8: `0.6914`
- Test Q8 on CB513: `0.6526`

## Notes

- Baseline and tuning artifacts are stored under `artifacts/`.
- Full experiment ledger is stored in `outputs/reports/run_ledger.csv`.
- `cnn1d` remains the recommended architecture for next-step work.
