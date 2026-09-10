"""
Extract 10 clean MicrosoftHelps multi-turn conversations (C -> B -> C -> B)
and inspect actions, redirections, and resolutions.
"""

import sys
import re
import json
import random
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
random.seed(101)

print("Loading dataset...", flush=True)
df = pd.read_csv(
    'dataset/twcs.csv',
    usecols=['tweet_id', 'author_id', 'inbound', 'in_response_to_tweet_id', 'text'],
    dtype={
        'tweet_id': 'int64',
        'author_id': 'string',
        'inbound': 'bool',
        'text': 'string',
    },
    low_memory=False
)

# Parent map
p_num = pd.to_numeric(df['in_response_to_tweet_id'], errors='coerce')
valid = p_num.notna()
parent_map = dict(zip(df.loc[valid, 'tweet_id'], p_num[valid].astype(int)))

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

# Find MicrosoftHelps tweet roots
ms_tweets = df[df['author_id'] == 'MicrosoftHelps']
ms_roots = set(get_root(tid) for tid in ms_tweets['tweet_id'])

# Filter df to just these roots
ms_df = df[df['tweet_id'].apply(lambda x: get_root(x) in ms_roots)].copy()
ms_df['conv_id'] = ms_df['tweet_id'].apply(get_root)
ms_df = ms_df.sort_values(['conv_id', 'tweet_id'])

# Extract conversations that match C -> B -> C -> B
candidates = []
for cid, group in ms_df.groupby('conv_id'):
    tweets = []
    for row in group.itertuples(index=False):
        tweets.append({
            'tweet_id': int(row.tweet_id),
            'author_id': str(row.author_id),
            'inbound': bool(row.inbound),
            'text': str(row.text) if pd.notna(row.text) else ''
        })
    
    # Check if there's customer and brand
    roles = [('C' if t['inbound'] else ('B' if t['author_id'] == 'MicrosoftHelps' else 'O')) for t in tweets]
    
    # We want at least C -> B -> C -> B pattern
    seq = []
    last_role = None
    for r in roles:
        if r != last_role and r in ('C', 'B'):
            seq.append(r)
            last_role = r
    
    # Check if 'C', 'B', 'C', 'B' is a subsequence
    is_cbcb = False
    c_idx = 0
    target_seq = ['C', 'B', 'C', 'B']
    for s in seq:
        if s == target_seq[c_idx]:
            c_idx += 1
            if c_idx == len(target_seq):
                is_cbcb = True
                break
    
    if is_cbcb and 4 <= len(tweets) <= 8:
        # Check that customer messages have substantial content (> 20 chars)
        c_texts = [t['text'] for t in tweets if t['inbound']]
        if all(len(txt) > 20 for txt in c_texts):
            candidates.append({'conv_id': cid, 'tweets': tweets})

print(f"Found {len(candidates)} candidate CBCB conversations for MicrosoftHelps")

# Sample 10 diverse conversations
sample_convs = random.sample(candidates, 10)

output_data = []
for idx, item in enumerate(sample_convs, 1):
    cid = item['conv_id']
    tweets = item['tweets']
    
    # Filter to the core 4 turns C -> B -> C -> B
    # Find sequence of alternating C, B, C, B
    turns = []
    state = 'WAIT_C1'
    for t in tweets:
        role = 'C' if t['inbound'] else ('B' if t['author_id'] == 'MicrosoftHelps' else 'O')
        if state == 'WAIT_C1' and role == 'C':
            turns.append(('Customer', t['text']))
            state = 'WAIT_B1'
        elif state == 'WAIT_B1' and role == 'B':
            turns.append(('MicrosoftHelps', t['text']))
            state = 'WAIT_C2'
        elif state == 'WAIT_C2' and role == 'C':
            turns.append(('Customer', t['text']))
            state = 'WAIT_B2'
        elif state == 'WAIT_B2' and role == 'B':
            turns.append(('MicrosoftHelps', t['text']))
            state = 'DONE'
            break
            
    if len(turns) == 4:
        output_data.append({'id': cid, 'turns': turns})

print(f"Selected {len(output_data)} strictly structured conversations.")
with open('analysis/microsoft_10_validation.json', 'w', encoding='utf-8') as f:
    json.dump(output_data, f, indent=2)

for i, c in enumerate(output_data, 1):
    print(f"\n--- CONVERSATION {i} (conv_id: {c['id']}) ---")
    for role, text in c['turns']:
        print(f"[{role}]: {text}")
