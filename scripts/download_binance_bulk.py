"""
BINANCE BULK DOWNLOADER (GCP OPTIMIZED)
=======================================

Downloads historical 1-minute klines directly from Binance Vision
and uploads them to Google Cloud Storage.

Usage:
    python scripts/download_binance_bulk.py --bucket cift-historical-data --years 2023 2024

"""

import os
import sys
import requests
import zipfile
import io
import pandas as pd
from datetime import datetime
from google.cloud import storage
import argparse
from concurrent.futures import ThreadPoolExecutor

# Constants
BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT"]
INTERVAL = "1m"

def download_and_upload(args):
    symbol, year, month, bucket_name = args
    month_str = f"{month:02d}"
    filename = f"{symbol}-{INTERVAL}-{year}-{month_str}.zip"
    url = f"{BASE_URL}/{symbol}/{INTERVAL}/{filename}"
    
    print(f"Processing {filename}...")
    
    try:
        # 1. Download into memory (don't save to disk)
        response = requests.get(url)
        if response.status_code != 200:
            print(f"  ❌ Not found: {url}")
            return

        # 2. Extract CSV from ZIP in memory
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            csv_filename = z.namelist()[0]
            with z.open(csv_filename) as csv_file:
                # 3. Read to Pandas
                df = pd.read_csv(csv_file, header=None)
                # Binance columns: Open time, Open, High, Low, Close, Volume, ...
                df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                              'close_time', 'quote_asset_volume', 'trades', 
                              'taker_buy_base', 'taker_buy_quote', 'ignore']
                
                # Optimize types
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'trades']]
                
                # 4. Convert to Parquet (Much smaller than CSV)
                parquet_buffer = io.BytesIO()
                df.to_parquet(parquet_buffer)
                parquet_buffer.seek(0)
                
                # 5. Upload to GCS
                client = storage.Client()
                bucket = client.bucket(bucket_name)
                blob_path = f"processed/{symbol}/{year}/{month_str}.parquet"
                blob = bucket.blob(blob_path)
                blob.upload_from_file(parquet_buffer)
                
                print(f"  ✅ Uploaded gs://{bucket_name}/{blob_path}")
                
    except Exception as e:
        print(f"  ❌ Error {filename}: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket', required=True, help='GCS Bucket Name')
    parser.add_argument('--years', nargs='+', required=True, help='Years to download (e.g. 2023 2024)')
    args = parser.parse_args()
    
    tasks = []
    for symbol in SYMBOLS:
        for year in args.years:
            for month in range(1, 13):
                # Skip future months
                if int(year) == datetime.now().year and month >= datetime.now().month:
                    continue
                tasks.append((symbol, int(year), month, args.bucket))
    
    # Parallel processing
    print(f"Starting download of {len(tasks)} files...")
    with ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(download_and_upload, tasks)

if __name__ == "__main__":
    main()
