import os
import re
import time
import json
import math
from collections import defaultdict, Counter
import pandas as pd
import numpy as np

# Create output directories
os.makedirs('analysis', exist_ok=True)
os.makedirs('analysis/candidate_examples', exist_ok=True)

def load_and_reconstruct():
    t0 = time.time()
    print("Step 1: Loading twcs.csv...")
    
    # Load dataset
    df = pd.read_csv(
        'dataset/twcs.csv',
        dtype={
            'tweet_id': 'int64',
            'author_id': 'string',
            'inbound': 'bool',
            'created_at': 'string',
            'text': 'string',
            'response_tweet_id': 'string',
        },
        low_memory=False
    )
    print(f"Loaded {len(df):,} tweets in {time.time()-t0:.2f}s")
    
    # Data Quality check on raw data
    print("\nStep 2: Checking Raw Data Quality...")
    raw_dq = {
        'total_rows': len(df),
        'missing_text': int(df['text'].isna().sum()),
        'empty_text': int((df['text'].str.strip() == '').sum()),
        'duplicate_tweet_ids': int(df['tweet_id'].duplicated().sum()),
        'missing_parent_id': int(df['in_response_to_tweet_id'].isna().sum()),
        'missing_response_id': int(df['response_tweet_id'].isna().sum()),
        'inbound_count': int(df['inbound'].sum()),
        'outbound_count': int((~df['inbound']).sum()),
    }
    print("Raw Data Quality Summary:", raw_dq)
    
    # Clean in_response_to_tweet_id to numeric/int where present
    df['in_response_to_tweet_id_clean'] = pd.to_numeric(df['in_response_to_tweet_id'], errors='coerce')
    
    # Build fast parent map: child_tweet_id -> parent_tweet_id
    print("\nStep 3: Building conversation graphs...")
    t1 = time.time()
    
    # Map tweet_id to parent
    tweet_id_arr = df['tweet_id'].values
    parent_id_arr = df['in_response_to_tweet_id_clean'].values
    
    parent_map = {}
    for tid, pid in zip(tweet_id_arr, parent_id_arr):
        if not np.isnan(pid):
            parent_map[tid] = int(pid)
            
    # Also handle response_tweet_id to capture forward links if parent link was missing
    # (TWCS typically has consistent bidirectional links)
    
    # Disjoint Set / Union-Find or Root finding to group into conversations
    # For each tweet, traverse up parent pointers until root (a tweet with no parent in parent_map)
    root_cache = {}
    def find_root(tid):
        path = []
        curr = tid
        while curr in parent_map:
            if curr in root_cache:
                curr = root_cache[curr]
                break
            path.append(curr)
            curr = parent_map[curr]
            if len(path) > 50: # prevent cyclic loops if any
                break
        for node in path:
            root_cache[node] = curr
        root_cache[tid] = curr
        return curr

    print("Grouping tweets into conversation roots...")
    conv_roots = [find_root(tid) for tid in tweet_id_arr]
    df['conv_id'] = conv_roots
    print(f"Conversation grouping completed in {time.time()-t1:.2f}s. Unique conversations: {df['conv_id'].nunique():,}")
    
    # Group tweets by conv_id
    print("\nStep 4: Analyzing conversation structure and brand attribution...")
    t2 = time.time()
    
    # Pre-aggregate conversation metadata
    # For each conversation, we need:
    # - brand(s) involved
    # - number of tweets
    # - inbound tweet count, outbound tweet count
    # - order of inbound vs outbound
    # - tweets list
    
    # Filter to brand accounts (outbound tweets have author_id as brand name)
    outbound_df = df[~df['inbound']]
    brand_counts = outbound_df['author_id'].value_counts()
    
    # We will focus on candidate brands with significant volume
    top_brands = brand_counts.head(40).index.tolist()
    # Add specific candidates mentioned by user if not in top 40
    user_candidates = [
        'MicrosoftHelps', 'AskAmex', 'DellCares', 'nationalrailenq', 'Tesco', 
        'SW_Help', 'VerizonSupport', 'AmazonHelp', 'GWRHelp', 'XboxSupport', 
        'AppleSupport', 'Uber_Support', 'SpotifyCares', 'hulu_support', 
        'sprintcare', 'TMobileHelp', 'British_Airways', 'SouthwestAir', 'Delta', 'comcastcares'
    ]
    candidate_set = set(top_brands + user_candidates)
    
    return df, candidate_set

if __name__ == '__main__':
    df, candidates = load_and_reconstruct()
