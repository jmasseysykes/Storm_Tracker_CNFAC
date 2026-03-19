# Storm Tracker CNFAC

Avalanche forecasting tool that visualizes historic and current storm totals from SNOTEL sites, computes rolling SWE deltas (1/3/7/10-day), and compares current values against historical distributions with percentile histograms.

**Live app**: https://avystormtracker.com

![Screenshot of Storm Tracker app showing histogram and current storm totals](https://github.com/jmasseysykes/Storm_Tracker_CNFAC/raw/main/screenshot.png)

## Features
- Fetches daily SNOTEL data directly from NRCS Report Generator (public API).
- Computes positive storm totals with proper gap detection (no forward-fill).
- Overlays current storm values on historical percentiles (50th–99th).
- Supports manual CSV upload for any daily weather station data.
- Clean matplotlib histograms with dynamic percentile coloring and record emphasis.
- Docker-ready for easy deployment (e.g., Render.com).

## Installation & Local Run

1. Clone the repo:
git clone https://github.com/jmasseysykes/Storm_Tracker_CNFAC.git
cd Storm_Tracker_CNFAC

2. Install dependencies:
pip install -r requirements.txt

3. Run the app:
streamlit run storm_tracker_app.py

## Docker (for Render or local testing)
docker build -t storm-tracker .
docker run -p 8501:8501 storm-tracker
Access at http://localhost:8501

## Data Sources
-SNOTEL data: NRCS Report Generator (public, no API key needed) https://wcc.sc.egov.usda.gov/reportGenerator/
-Station list: SNOTEL_station_list.csv (update periodically from NRCS or CUAHSI).

## Licesnse
This project is licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)

-You may share and adapt the material for non-commercial purposes with attribution.
-Any derivatives must use the same license.
-No commercial use allowed (e.g., no selling access or incorporating in paid products).

Full license: LICENSE | https://creativecommons.org/licenses/by-nc-sa/4.0/

## Contributing
Issues, feature requests, and pull requests are welcome!
Fork → branch → PR. Please keep non-commercial intent in mind.

## Acknowledgements
Original Storm Tracker spreadsheet developed by Jason Konigsberg and Spencer Logan at Colorado Avalanche Information Center and adapted for Chugach National Forest Avalnache Center by Mik Dalpes and Andrew Schauer. Adapted for web app by John Sykes. 

-Built for CNFAC (Chugach National Forest Avalanche Center) avalanche forecasting.
-Powered by public NRCS SNOTEL data.

Questions or feedback? Open an issue or reach out via GitHub.
