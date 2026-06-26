# Spreadsheet → Web App Import Spec

**Status:** On hold — waiting for Zach's go-ahead before implementing.  
**Source file:** `D size study_v3.xlsx - Guy _CO Case Studies.csv`  
**Last updated:** 2026-06-26

## Goal

Bulk-import Guy/CO case studies from the original spreadsheet into Supabase, using **raw inputs only** and letting the web app calculation logic produce mass, volume, and D-size (consistent with manual entry). Use spreadsheet calculated values for **QC comparison only** — they may not align perfectly.

---

## Dataset summary

| Source | Total rows | Meaningful rows |
|--------|------------|-----------------|
| Spreadsheet CSV | 112 | **33** (have Location + Observation link) |
| Supabase (as of Jun 2026) | 41 | 41 |

79 spreadsheet rows are empty template rows (formulas/zeros) — skip them.

---

## Overlapping cases — SKIP on import

These 7 spreadsheet cases were **already entered manually** by Zach. Do not re-import. Verify D-size (and ideally area) match as a sanity check:

| Spreadsheet location | Supabase id | Supabase location |
|---------------------|-------------|-------------------|
| Baldy Nirvana Bowl | 5 | Elk Mountain, CO - Baldy Nirvana Bowl 12/31/21 |
| West Beckwith | 15 | Elk Mountains, CO - West Beckwith 12/31/21 |
| Elk Basin | 13 | Elk Mountains, CO - Elk Basin 2/25/25 |
| Carbon Peak | 11 | Elk Mountains, CO - Carbon Peak 2/25/25 |
| Schuykill Ridge Great Wide Open | 8 | Elk Mountains, CO - Schuylkill Ridge GWO 12/31/21 |
| Climax | 19 | Elk Mountains, CO - Climax Chutes 11/27/24 |
| Ruby Peak Fatality | 17 | Elk Mountains, CO - Ruby Fatality 1/21/16 |

**~26 spreadsheet cases** are new relative to this tab.

34 Supabase records are not in this spreadsheet tab (AK/MT/WY cases, duplicates, etc.).

---

## Key design decisions (from team)

1. **Skip the 7 overlaps** — already in DB; confirm sizes match.
2. **SWE adjustment** is a **value difference**, not a % slider:
   - `SWE at SNOTEL since WL burial` → raw → `slab_swe_mm` (inches × 25.4)
   - `Start zone SWE (inches)` or `(cm)` → locally adjusted → `adjusted_swe_mm` (inches × 25.4 or cm × 10)
3. **`Avalanche Coding`** column → `field_assessed_d_size` (parse D-size from OBS string, e.g. `HS-N-R4-D4-O` → `D4`).
4. **Do NOT import** pre-calculated D-size, mass, or total weight as authoritative — feed raw inputs and run `avalanche_calcs` / web app logic.
5. Spreadsheet totals (`Start Zone Weight`, `Total Weight`, `D Size estimation`) are for **QC double-check only**.

---

## Column mapping

### Import as raw inputs (reliable)

| Spreadsheet column | Web app column | Conversion |
|--------------------|----------------|------------|
| Location | location | direct |
| Observation link | report_link | first URL if comma-separated |
| Site visit date | timestamp | parse date |
| Start zone area (m^2) | area_m2 | direct |
| Crown thickness (cm) | depth_m, crown_depth_direct_m | ÷ 100 |
| Estimated slab density | density_kgm3 | parse `270 kg/m^3` → 270 |
| SWE at SNOTEL since WL burial | slab_swe_mm | inches → mm (× 25.4) |
| Start zone SWE (inches/cm) | adjusted_swe_mm | inches × 25.4 or cm × 10 |
| Representative Snotel(s) | snotel_station | parse name |
| Weak layer | weak_layer_date | fuzzy parse (needs year from context) |
| Track area (m^2) | entr_area_m2 | direct |
| Track zone SWE (cm) | entr_swe_mm | cm × 10 |
| Avalanche Coding | field_assessed_d_size | parse D-size from OBS code |
| Crown details + Profile notes + SWE Notes | notes | concatenate |

### Let web app calculate (do not import as stored results)

- mass_tonnes, entrainment_mass, total_mass
- volume_m3
- calculated_d_size, dsize_mass_midpoint, dsize_volume_midpoint
- unc_low, unc_high

### Defaults on import

- method = `start_zone`
- schema_version = `2.0`
- include_entrainment = 1 if track area > 0
- density_mode = `SWE based density estimate` when SWE columns present, else `Layer density + grain type` when density given

### Low reliability / manual review

- hardness, grain — free text in Crown details; may stay in notes
- release_date — rarely explicit in spreadsheet
- weak_layer_date — month/day only (`Dec 8`, `Nov 25`)

### No DB column (append to notes if useful)

- Elevation, run length, alpha angle, com1DFA simulation
- Most representative profile (Google Drive URL)
- Redundant unit columns (cm², cm³, kg, lbs)

---

## QC plan (when implementing)

1. **Pilot:** import ~5 non-overlapping cases; compare computed vs spreadsheet mass/D-size.
2. **Flag** rows where computed total mass differs from `Total Weight (tons)` by >10–15%.
3. **Skip** the 7 overlap IDs listed above.
4. Produce a sidecar CSV: `spreadsheet_location`, `db_id` (if matched), `qc_mass_diff_pct`, `qc_dsize_match`.

---

## Suggested prompt when resuming

> "Please implement the spreadsheet import per `docs/spreadsheet_import_spec.md`. Start with a 5-case pilot and QC report."

No need to paste prior chat — point Grok at this file and the CSV path.