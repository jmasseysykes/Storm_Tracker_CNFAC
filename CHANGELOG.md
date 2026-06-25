# Changelog

All notable changes to the CNFAC Storm Tracker D-Size Estimator.

## [2.0.0] - 2026-06-24

### Major Changes
- **Unified Start Zone Method**: Combined the previous "Quick" and "Detailed" tabs into a single, cleaner "Start Zone Method".
  - Geometry: Slab dimensions (width × length) with optional polygon area override.
  - Density options via radio:
    - Layer density + grain type (default: Simple hand hardness + grain; optional Detailed multi-layer profile)
    - SWE-based (manual or SNOTEL)
  - Optional entrainment module with choice of method (grain-based or SWE-based).
- **Runout/Debris Method**: Now explicitly volume-first (Jamieson 2024 bins). Custom density is optional for mass reference only.
- Introduced `schema_version = "2.0"` in all new records for easy filtering of V2 data.
- `method` field is now `"start_zone"` or `"runout_debris"` (clean values for research queries). Human-friendly labels remain in the UI.

### Database & Research Improvements
- New fields for better method tracking:
  - `density_profile`
  - `swe_source`
  - `area_overridden`
  - `entrainment_method`
- Consistent snake_case naming across V2 fields.
- On the "View Saved Avalanches" page and CSV export, columns are presented with clean, consistent names (via display mapping). Underlying DB schema is unchanged to preserve historical data.
- Migration logic updated and verified to handle both old and new records safely (no column renames or data loss).
- Added `tests/generate_v2_test_data.py` + `tests/v2_test_data.csv` — a synthetic dataset covering all major input combinations (simple vs detailed, SWE, with/without entrainment, runout variants, area override, etc.).

### UI / Usability
- V2.0 messaging is low-profile (in captions, not the main title).
- Per-section uncertainty expanders remain.
- Removed redundant captions and improved output clarity.
- Fixed various Streamlit key duplication issues.

### Other
- Updated README and added CHANGELOG.
- All core calculation tests pass.
- App deployed on Render (auto-deploys on push to main).

### Migration Notes
- Existing records are preserved. New saves will have `schema_version = "2.0"`.
- The D-size migration (`dsize_mass_midpoint`, `dsize_volume_midpoint`, etc.) runs automatically on app start.
- To analyze only V2 data: `WHERE schema_version = '2.0'`

[2.0.0]: https://github.com/jmasseysykes/Storm_Tracker_CNFAC/compare/v1...v2.0.0
