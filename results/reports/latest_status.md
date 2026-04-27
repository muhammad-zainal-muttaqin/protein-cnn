# Latest Status

Updated: 2026-03-30T17:39:01.797484+00:00

## Current Best

- Run: `p4_07_resdil_b42_ce_none_c320_e24_seed7`
- Model: `resdil_cnn1d`
- Best validation Q8: `0.7354`
- Test Q8 on CB513: `0.6926`
- Test loss: `0.9242`

## Runner-up

- Run: `s2_01_s1_06_resdil_cnn1d_baseline42_ce_none_c320`
- Model: `resdil_cnn1d`
- Best validation Q8: `0.7303`
- Test Q8 on CB513: `0.6919`
- Test loss: `0.9327`

## Notes

- Ledger final berisi `101` eksperimen (`102` record CSV termasuk header).
- Best final berasal dari `research_phase4` dan mengungguli baseline CNN 1D sebesar `+0.0231` absolute pada test Q8.
- Full ledger tersedia di `outputs/reports/run_ledger.csv`.
