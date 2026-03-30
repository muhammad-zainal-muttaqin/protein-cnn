# Latest Status

Updated: 2026-03-30T13:20:31.220024+00:00

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
- Test loss: `inf`

## Notes

- Search is running sequentially and recording every experiment.
- Full ledger is stored in `outputs/reports/run_ledger.csv`.
- Detailed experiment summaries are stored in `outputs/reports/research_summary.json`.
