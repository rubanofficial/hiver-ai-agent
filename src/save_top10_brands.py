import os
import time
import pandas as pd
import numpy as np

os.makedirs('analysis', exist_ok=True)

def compute_brand_conversation_metrics():
    t0 = time.time()
    print("Loading structural columns only (no text)...", flush=True)
    df = pd.read_csv(
        'dataset/twcs.csv',
        usecols=['tweet_id', 'author_id', 'inbound', 'in_response_to_tweet_id'],
        dtype={
            'tweet_id': 'int64',
            'author_id': 'string',
            'inbound': 'bool'
        },
        low_memory=False
    )
    print(f"Loaded in {time.time()-t0:.2f}s", flush=True)
    
    # Fast parent map
    t1 = time.time()
    p_series = pd.to_numeric(df['in_response_to_tweet_id'], errors='coerce')
    parent_map = dict(zip(df['tweet_id'][p_series.notna()], p_series[p_series.notna()].astype(int)))
    
    # Find roots
    root_cache = {}
    def get_root(tid):
        curr = tid
        path = []
        while curr in parent_map:
            if curr in root_cache:
                curr = root_cache[curr]
                break
            path.append(curr)
            curr = parent_map[curr]
            if len(path) > 40:
                break
        for node in path:
            root_cache[node] = curr
        root_cache[tid] = curr
        return curr

    df['conv_id'] = [get_root(tid) for tid in df['tweet_id'].values]
    print(f"Graph resolved in {time.time()-t1:.2f}s", flush=True)
    
    # Target top candidates
    target_brands = [
        'MicrosoftHelps', 'AskAmex', 'DellCares', 'nationalrailenq', 'Tesco',
        'SW_Help', 'VerizonSupport', 'AmazonHelp', 'GWRHelp', 'XboxSupport',
        'AppleSupport', 'Uber_Support'
    ]
    
    # Filter to conversations containing both customer (inbound=True) and brand (inbound=False)
    # 1. Identify which conv_ids have inbound
    inbound_convs = set(df[df['inbound']]['conv_id'].unique())
    
    # 2. Outbound tweets
    outbound_df = df[(~df['inbound']) & (df['author_id'].isin(target_brands)) & (df['conv_id'].isin(inbound_convs))]
    
    # Map conv_id to brand
    # If multiple brands, map to first/dominant brand
    conv_to_brand = outbound_df.groupby('conv_id')['author_id'].first().to_dict()
    valid_conv_ids = set(conv_to_brand.keys())
    
    # Filter df to valid convs to get total lengths
    subset_df = df[df['conv_id'].isin(valid_conv_ids)]
    conv_lengths = subset_df.groupby('conv_id').size().to_dict()
    
    # Collect lengths per brand
    brand_lengths = {b: [] for b in target_brands}
    for cid, brand in conv_to_brand.items():
        if brand in brand_lengths:
            brand_lengths[brand].append(conv_lengths[cid])
            
    # Calculate stats
    rows = []
    for brand in target_brands:
        lens = brand_lengths[brand]
        n = len(lens)
        if n == 0:
            continue
        arr = np.array(lens)
        avg_len = round(float(np.mean(arr)), 2)
        pct_3plus = round(float(np.sum(arr >= 3) / n * 100), 2)
        pct_4plus = round(float(np.sum(arr >= 4) / n * 100), 2)
        pct_5plus = round(float(np.sum(arr >= 5) / n * 100), 2)
        
        rows.append({
            'brand': brand,
            'unique_conversations': n,
            'avg_conv_length': avg_len,
            'pct_3plus_msgs': pct_3plus,
            'pct_4plus_msgs': pct_4plus,
            'pct_5plus_msgs': pct_5plus
        })
        
    res_df = pd.DataFrame(rows)
    # Sort by pct_3plus_msgs descending or unique_conversations
    res_df = res_df.sort_values(by='pct_3plus_msgs', ascending=False).reset_index(drop=True)
    
    # Save CSV
    out_csv = 'analysis/candidate_brands.csv'
    res_df.to_csv(out_csv, index=False)
    print(f"Saved to {out_csv} in {time.time()-t0:.2f}s total!", flush=True)
    print("\nTop Candidate Brands:")
    print(res_df.to_string(index=False), flush=True)

if __name__ == '__main__':
    compute_brand_conversation_metrics()
