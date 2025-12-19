import requests
import pandas as pd
from datetime import datetime

# Your CSV link (use your triplet)
url = "https://wcc.sc.egov.usda.gov/reportGenerator/view_csv/customSingleStationReport/daily/954:AK:SNTL%7Cid=%22%22%7Cname/POR_BEGIN,POR_END/WTEQ::value,PREC::value,TMAX::value,TMIN::value,TAVG::value,PRCP::value"

print(f"Fetching SNOTEL CSV from {url}...")

try:
    response = requests.get(url)
    response.raise_for_status()

    # Read CSV, skip comment lines starting with '#'
    df = pd.read_csv(pd.io.common.StringIO(response.text), comment='#')

    # Rename columns if needed (NRCS CSV may have extra headers)
    df.columns = ['Date', 'SWE', 'PREC', 'TMAX', 'TMIN', 'TAVG', 'PRCP']  # Adjust based on actual columns

    df['Date'] = pd.to_datetime(df['Date'])
    df['SWE'] = pd.to_numeric(df['SWE'], errors='coerce')
    df = df.dropna(subset=['SWE'])

    print(f"Fetched {len(df)} days of data.")
    print("Date range:", df['Date'].min().date(), "to", df['Date'].max().date())
    print("First 5 rows:")
    print(df.head())
    print("Last 5 rows:")
    print(df.tail())

except Exception as e:
    print(f"Error: {e}")
    print("Try a different URL or check the NRCS site.")