import os
import sys
import time
import pandas as pd
import numpy as np

def run_quick_scan():
    t0 = time.time()
    print("Loading twcs.csv...")
    df = pd.read_csv(
        'dataset/twcs.csv',
        usecols=['tweet_id', 'author_id', 'inbound', 'in_response_to_tweet_id', 'response_tweet_id'],
        dtype={
            'tweet_id': 'int64',
            'author_id': 'string',
            'inbound': 'bool',
        },
        low_memory=False
    )
    print(f"Loaded {len(df):,} rows in {time.time()-t0:.2f}s")
    
    # Check brands (inbound == False)
    brands = df[~df['inbound']]['author_id'].value_counts()
    print("Top 20 outbound brands by volume:")
    print(brands.head(20))

if __name__ == '__main__':
    run_quick_scan()
