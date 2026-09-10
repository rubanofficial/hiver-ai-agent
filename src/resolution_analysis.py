"""
Historical Resolution Quality Analysis
For: MicrosoftHelps, DellCares, VerizonSupport, AmazonHelp

Loads dataset once. Reconstructs conversations. Filters to true multi-turn
C->B->C->B threads. Computes resolution quality metrics and prints examples.
Saves to analysis/resolution_analysis.csv
"""

import os
import re
import sys
import time
import json
import random
import pandas as pd
import numpy as np
from collections import defaultdict

os.makedirs('analysis', exist_ok=True)
os.makedirs('analysis/candidate_examples', exist_ok=True)
random.seed(42)
sys.stdout.reconfigure(encoding='utf-8')

TARGET_BRANDS = ['MicrosoftHelps', 'DellCares', 'VerizonSupport', 'AmazonHelp']

# ── Signals ───────────────────────────────────────────────────────────────────
ACTION_RE = re.compile(
    r'\b(please\s+(try|check|go|visit|click|tap|select|open|send|reply|follow|'
    r'restart|reset|update|remove|add|make\s+sure|ensure|confirm|verify|'
    r'turn\s+off|turn\s+on|clear)|'
    r'step[s]?\s+\d|step[s]?:|'
    r'settings\b|restart\b|reboot\b|reinstall\b|clear\s+cache|'
    r'factory\s+reset|power\s+cycle|troubleshoot|'
    r'https?://\S+|bit\.ly|t\.co|'
    r'guide\b|instructions\b|article\b|help\s+page|support\s+page|'
    r'click\s+here|tap\s+on|go\s+to\s+settings|'
    r'follow\s+these|here\s+is\s+how|you\s+can\s+do\s+this)',
    re.I
)

DM_RE = re.compile(
    r'\b(dm\b|direct\s+message|send\s+us\s+a\s+dm|reach\s+out\s+via|'
    r'private\s+message|send\s+a\s+private|via\s+dm|in\s+our\s+dm)',
    re.I
)

RESOLVED_RE = re.compile(
    r'\b(thank\s+you|thanks\b|that\s+worked|it\s+worked|fixed\b|sorted\b|'
    r'works\s+now|appreciate\s+it|all\s+set\b|resolved\b|great\s+help|'
    r'solved\b|problem\s+solved|issue\s+(is\s+)?fixed|no\s+longer|'
    r'finally\s+working|back\s+up|back\s+online|got\s+it\s+working)',
    re.I
)

PHONE_RE = re.compile(
    r'\b(call\s+us|give\s+us\s+a\s+call|please\s+call|our\s+number|'
    r'phone\s+number|\d{3}[-.\s]\d{3}[-.\s]\d{4})',
    re.I
)


def load_and_reconstruct():
    t0 = time.time()
    print("Loading dataset (structural + text)...", flush=True)
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
    print(f"Loaded {len(df):,} rows in {time.time()-t0:.2f}s", flush=True)

    # Build parent map
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

    t1 = time.time()
    df['conv_id'] = [get_root(tid) for tid in df['tweet_id'].values]
    print(f"Conversation tree built in {time.time()-t1:.2f}s", flush=True)
    return df


def classify_conv(tweets_sorted, brand):
    """
    Analyze a single conversation (list of tweet dicts sorted by tweet_id).
    Returns a dict of metrics for this conversation.
    """
    cust_turns  = [t for t in tweets_sorted if t['inbound']]
    brand_turns = [t for t in tweets_sorted if not t['inbound'] and t['author_id'] == brand]

    # Count turn transitions (C->B or B->C)
    transitions = 0
    prev_role = None
    for t in tweets_sorted:
        role = 'C' if t['inbound'] else 'B'
        if prev_role and role != prev_role:
            transitions += 1
        prev_role = role

    # Check for true C->B->C->B pattern (at least 3 transitions = C B C B)
    roles_seq = ['C' if t['inbound'] else 'B' for t in tweets_sorted]
    is_cbcb = False
    # Look for at least C then B then C then B in sequence (not necessarily adjacent)
    c_seen, b_after_c, c2_after_b, b2_after_c2 = False, False, False, False
    for r in roles_seq:
        if r == 'C' and not c_seen:
            c_seen = True
        elif r == 'B' and c_seen and not b_after_c:
            b_after_c = True
        elif r == 'C' and b_after_c and not c2_after_b:
            c2_after_b = True
        elif r == 'B' and c2_after_b and not b2_after_c2:
            b2_after_c2 = True
    is_cbcb = b2_after_c2

    # Brand response metrics
    brand_reply_lens = [len(t['text']) for t in brand_turns]
    avg_brand_len = np.mean(brand_reply_lens) if brand_reply_lens else 0

    has_action   = any(ACTION_RE.search(t['text']) for t in brand_turns)
    has_dm       = any(DM_RE.search(t['text'])     for t in brand_turns)
    has_phone    = any(PHONE_RE.search(t['text'])   for t in brand_turns)

    # Resolution: look for resolution signal in last customer message
    resolved = False
    if len(cust_turns) >= 2:
        last_cust = cust_turns[-1]['text']
        if RESOLVED_RE.search(last_cust):
            resolved = True

    return {
        'is_cbcb':          is_cbcb,
        'num_brand_turns':  len(brand_turns),
        'num_cust_turns':   len(cust_turns),
        'total_turns':      len(tweets_sorted),
        'avg_brand_len':    avg_brand_len,
        'has_action':       has_action,
        'has_dm':           has_dm,
        'has_phone':        has_phone,
        'resolved':         resolved,
    }


def analyze_brand(df, brand, example_count=5):
    print(f"\n{'='*60}", flush=True)
    print(f"  Analyzing: {brand}", flush=True)
    print(f"{'='*60}", flush=True)

    # Conversations where brand posted outbound
    brand_conv_ids = set(df[(~df['inbound']) & (df['author_id'] == brand)]['conv_id'].unique())

    # Also need inbound tweets in those conversations
    inbound_conv_ids = set(df[df['inbound']]['conv_id'].unique())
    valid_ids = brand_conv_ids & inbound_conv_ids
    print(f"  Candidate conversations (both sides): {len(valid_ids):,}", flush=True)

    # Build conversation index
    conv_df = df[df['conv_id'].isin(valid_ids)][['tweet_id', 'author_id', 'inbound', 'text', 'conv_id']].copy()
    conv_df = conv_df.sort_values(['conv_id', 'tweet_id'])

    # Aggregate metrics
    total_convs     = 0
    cbcb_convs      = 0
    b2plus_convs    = 0
    b3plus_convs    = 0
    action_convs    = 0
    dm_convs        = 0
    phone_convs     = 0
    resolved_convs  = 0
    brand_lens_all  = []

    # Good examples buffer (CBCB, actionable, resolved)
    good_examples   = []
    ok_examples     = []   # CBCB but not necessarily resolved

    for cid, group in conv_df.groupby('conv_id'):
        tweets = [
            {
                'tweet_id':  int(row.tweet_id),
                'author_id': str(row.author_id),
                'inbound':   bool(row.inbound),
                'text':      str(row.text) if pd.notna(row.text) else '',
            }
            for row in group.itertuples(index=False)
        ]

        m = classify_conv(tweets, brand)
        total_convs += 1

        if m['num_brand_turns'] >= 1:
            brand_lens_all.append(m['avg_brand_len'])

        if m['num_brand_turns'] >= 2:
            b2plus_convs += 1
        if m['num_brand_turns'] >= 3:
            b3plus_convs += 1
        if m['is_cbcb']:
            cbcb_convs  += 1
        if m['has_action']:
            action_convs += 1
        if m['has_dm']:
            dm_convs     += 1
        if m['has_phone']:
            phone_convs  += 1
        if m['resolved']:
            resolved_convs += 1

        # Collect examples
        if m['is_cbcb'] and len(tweets) >= 4:
            entry = {'conv_id': cid, 'tweets': tweets, 'metrics': m}
            if m['resolved'] and m['has_action']:
                good_examples.append(entry)
            else:
                ok_examples.append(entry)

    # Pick best examples (prioritise resolved+actionable)
    selected = (good_examples + ok_examples)
    random.shuffle(selected)
    examples = selected[:example_count]

    # Print metrics
    pct = lambda n: f"{n/total_convs*100:.1f}%" if total_convs else "N/A"
    avg_brand_len = np.mean(brand_lens_all) if brand_lens_all else 0

    print(f"\n  RESOLUTION METRICS", flush=True)
    print(f"  Total conversations (both-sided):     {total_convs:>8,}", flush=True)
    print(f"  C->B->C->B multi-turn:                {cbcb_convs:>8,}  ({pct(cbcb_convs)})", flush=True)
    print(f"  Convs with 2+ brand responses:        {b2plus_convs:>8,}  ({pct(b2plus_convs)})", flush=True)
    print(f"  Convs with 3+ brand responses:        {b3plus_convs:>8,}  ({pct(b3plus_convs)})", flush=True)
    print(f"  Avg brand response length (chars):    {avg_brand_len:>8.1f}", flush=True)
    print(f"  Convs with actionable guidance:       {action_convs:>8,}  ({pct(action_convs)})", flush=True)
    print(f"  Convs with DM redirect:               {dm_convs:>8,}  ({pct(dm_convs)})", flush=True)
    print(f"  Convs with phone redirect:            {phone_convs:>8,}  ({pct(phone_convs)})", flush=True)
    print(f"  Convs with resolution signal:         {resolved_convs:>8,}  ({pct(resolved_convs)})", flush=True)

    # Print examples
    print(f"\n  --- {example_count} REAL CONVERSATION EXAMPLES ---", flush=True)
    for i, ex in enumerate(examples, 1):
        print(f"\n  Example {i}  (conv_id: {ex['conv_id']}, tweets: {len(ex['tweets'])})", flush=True)
        print(f"  Resolved: {ex['metrics']['resolved']}  |  Actionable: {ex['metrics']['has_action']}", flush=True)
        print(f"  {'─'*55}", flush=True)
        for t in ex['tweets']:
            role  = 'CUSTOMER  ' if t['inbound'] else f'BRAND ({brand})'
            # Truncate long texts for readability
            text  = t['text'].replace('\n', ' ').strip()[:220]
            ellip = '...' if len(t['text']) > 220 else ''
            print(f"  [{role}] {text}{ellip}", flush=True)

    # Write examples to markdown
    md_lines = [f"# Resolution Examples: {brand}\n\n"]
    for i, ex in enumerate(examples, 1):
        md_lines.append(f"### Example {i} (conv_id: {ex['conv_id']}, length: {len(ex['tweets'])})\n")
        md_lines.append(f"Resolved: `{ex['metrics']['resolved']}` | Actionable: `{ex['metrics']['has_action']}`\n\n")
        for t in ex['tweets']:
            role = 'Customer' if t['inbound'] else f'Brand ({brand})'
            md_lines.append(f"**{role}**:\n> {t['text'].strip()}\n\n")
        md_lines.append("---\n\n")

    md_path = f"analysis/candidate_examples/{brand}_resolution_examples.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.writelines(md_lines)
    print(f"\n  Saved examples -> {md_path}", flush=True)

    return {
        'brand':                      brand,
        'total_conversations':        total_convs,
        'cbcb_multi_turn':            cbcb_convs,
        'pct_cbcb':                   round(cbcb_convs / total_convs * 100, 2) if total_convs else 0,
        'convs_2plus_brand_resp':     b2plus_convs,
        'pct_2plus_brand_resp':       round(b2plus_convs / total_convs * 100, 2) if total_convs else 0,
        'convs_3plus_brand_resp':     b3plus_convs,
        'pct_3plus_brand_resp':       round(b3plus_convs / total_convs * 100, 2) if total_convs else 0,
        'avg_brand_reply_chars':      round(avg_brand_len, 1),
        'convs_with_action_guidance': action_convs,
        'pct_action_guidance':        round(action_convs / total_convs * 100, 2) if total_convs else 0,
        'convs_dm_redirect':          dm_convs,
        'pct_dm_redirect':            round(dm_convs / total_convs * 100, 2) if total_convs else 0,
        'convs_phone_redirect':       phone_convs,
        'pct_phone_redirect':         round(phone_convs / total_convs * 100, 2) if total_convs else 0,
        'convs_resolved_signal':      resolved_convs,
        'pct_resolved_signal':        round(resolved_convs / total_convs * 100, 2) if total_convs else 0,
    }


def main():
    t_start = time.time()
    df = load_and_reconstruct()

    all_rows = []
    for brand in TARGET_BRANDS:
        row = analyze_brand(df, brand, example_count=5)
        all_rows.append(row)

    # Save CSV
    result_df = pd.DataFrame(all_rows)
    result_df.to_csv('analysis/resolution_analysis.csv', index=False)
    print(f"\n\nSaved -> analysis/resolution_analysis.csv", flush=True)

    # Summary table
    print("\n" + "="*90, flush=True)
    print("RESOLUTION QUALITY SUMMARY TABLE", flush=True)
    print("="*90, flush=True)
    cols = ['brand', 'total_conversations', 'cbcb_multi_turn', 'pct_cbcb',
            'pct_2plus_brand_resp', 'pct_3plus_brand_resp',
            'avg_brand_reply_chars', 'pct_action_guidance',
            'pct_dm_redirect', 'pct_resolved_signal']
    print(result_df[cols].to_string(index=False), flush=True)
    print(f"\nTotal time: {time.time()-t_start:.1f}s", flush=True)


if __name__ == '__main__':
    main()
