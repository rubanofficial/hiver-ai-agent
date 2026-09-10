"""
Intent Diversity Analysis for 6 Candidate Brands
Uses keyword-based classification on sampled customer messages from multi-turn conversations.
No LLM used. Reads dataset once, writes analysis/intent_analysis.csv.
"""

import os
import re
import time
import json
import random
import pandas as pd
import numpy as np
from collections import defaultdict, Counter

os.makedirs('analysis', exist_ok=True)
random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Target brands
# ─────────────────────────────────────────────────────────────────────────────
TARGET_BRANDS = [
    'MicrosoftHelps',
    'Tesco',
    'VerizonSupport',
    'AmazonHelp',
    'XboxSupport',
    'DellCares',
]

# ─────────────────────────────────────────────────────────────────────────────
# 2. Intent keyword patterns  (order matters – first match wins if exclusive)
#    Each pattern is tried on the entire customer message text.
# ─────────────────────────────────────────────────────────────────────────────
INTENT_PATTERNS = [
    ('Account & Login / Auth',
     re.compile(
         r'\b(log.?in|sign.?in|sign.?out|log.?out|password|reset\s+password|'
         r'locked.?out|account\s+access|2fa|two.factor|verification\s+code|'
         r'auth|username|pin\b|security\s+code|credentials|account\s+blocked|'
         r'can.?t\s+(log|sign)|forgot\s+password|account\s+suspended|'
         r'access\s+account|email\s+address|change\s+email)\b', re.I)),

    ('Billing, Charges & Payments',
     re.compile(
         r'\b(bill\b|billing|charge[ds]?\b|charged\s+twice|payment|invoice|'
         r'receipt|fee\b|credit\s+card|debit\s+card|double\s+charg|bank\b|'
         r'overcharg|unauthorized\s+charge|transaction|refund|refunded|'
         r'money\s+back|reimburs|wrong\s+amount|incorrect\s+charge|'
         r'subscri[pb]tion\s+charge|auto.?renew|charged\s+wrong)\b', re.I)),

    ('Order, Delivery & Tracking',
     re.compile(
         r'\b(order\b|orders\b|deliver[yed]+|delivery|package\b|parcel\b|'
         r'shipment|shipped|shipping|tracking|track\s+my|arrived|not\s+arrived|'
         r'not\s+delivered|missing\s+(item|order)|late\s+delivery|'
         r'dispatch[ed]*|estimated\s+(delivery|arrival)|where\s+is\s+my\s+order|'
         r'carrier|courier|ups\b|usps\b|fedex\b|dhl\b|royal\s+mail)\b', re.I)),

    ('Cancellation & Subscription',
     re.compile(
         r'\b(cancel\b|cancell|cancellation|unsubscri|stop\s+subscription|'
         r'end\s+subscription|close\s+(account|service)|terminate|'
         r'delete\s+account|cancel\s+(order|plan|membership)|'
         r'auto.?renew|downgrade|free\s+trial|trial\s+ended|'
         r'membership\s+cancel|subscription\s+cancel|how\s+do\s+i\s+cancel)\b', re.I)),

    ('Technical Error & Troubleshooting',
     re.compile(
         r'\b(error\b|bug\b|crash[es]*|crashing|glitch|freeze|freezing|'
         r'not\s+working|won\'t\s+work|doesn\'t\s+work|broke[n]?|broken\b|'
         r'black\s+screen|blue\s+screen|reboot|restart|slow\b|lagging|'
         r'update\b|install\b|reinstall|firmware|driver|patch\b|'
         r'app\s+(crash|not\s+open|not\s+load|won\'t\s+start)|'
         r'server\s+(down|error)|down\b|outage|not\s+respond|'
         r'connection\s+(drop|fail|issue|problem)|disconnected)\b', re.I)),

    ('Network, Internet & Connectivity',
     re.compile(
         r'\b(wifi|wi.?fi|internet|broadband|connection\b|signal\b|'
         r'no\s+service|no\s+signal|network\b|coverage\b|data\b|'
         r'mobile\s+data|4g\b|5g\b|lte\b|connect[ing]+\s+to|'
         r'can\'t\s+connect|router\b|modem\b|hotspot\b|speed\b|bandwidth|'
         r'data\s+(plan|limit|usage)|roaming\b|slow\s+(internet|data|wifi))\b', re.I)),

    ('Damaged, Wrong or Missing Product',
     re.compile(
         r'\b(damaged|broken\s+(product|item|box)|defective|faulty|'
         r'wrong\s+item|wrong\s+product|incorrect\s+(item|product|order)|'
         r'missing\s+(item|product|part)|not\s+as\s+described|'
         r'poor\s+quality|spoiled|expired|contaminated|'
         r'received\s+wrong|item\s+not\s+working|faulty\s+product)\b', re.I)),

    ('Flight, Transit & Booking',
     re.compile(
         r'\b(flight\b|flights\b|train\b|trains\b|ticket\b|tickets\b|'
         r'seat\b|seats\b|boarding|platform\b|timetable|schedule[d]*|'
         r'delay[ed]*\b|disruption|cancell[ed]*\s+flight|'
         r'baggage\b|luggage\b|check.?in|reservation|booking\b|'
         r'trip\b|journey\b|station\b|airport\b|gate\b|coach\b|'
         r'rail\b|railway)\b', re.I)),

    ('Product/Feature How-To & General Inquiry',
     re.compile(
         r'\b(how\s+do\s+i|how\s+to|how\s+can\s+i|'
         r'is\s+it\s+possible|can\s+i\b|where\s+(can\s+i|do\s+i)|'
         r'what\s+is\b|guide\b|tutorial|instructions|set\s+up|setup|'
         r'configure|help\s+with|need\s+help|assistance|'
         r'does\s+it\s+work|compatible|support[s]?\s+for|feature\b)\b', re.I)),

    ('Complaint & Feedback',
     re.compile(
         r'\b(disappoint[ed]*|terrible|awful|worst|horrible|disgusting|'
         r'unacceptable|outrageous|poor\s+service|bad\s+service|'
         r'this\s+is\s+(a\s+)?joke|scam\b|fraud\b|lied|mislead|'
         r'waste\s+of\s+(time|money)|never\s+again|lost\s+(a\s+)?customer|'
         r'complain[t]*|feedback\b|review\b|report\b|'
         r'no\s+response|ignored|escalate)\b', re.I)),

    ('Store, Pickup & In-Person',
     re.compile(
         r'\b(store\b|stores\b|branch\b|shop\b|click\s*(and|\&)\s*collect|'
         r'pickup\b|pick.up|collect\s+from|open\b|opening\s+hours|'
         r'closed\b|location\b|nearby\b|in.store|self.?checkout|'
         r'staff\b|manager\b|instore)\b', re.I)),

    ('Warranty, Repair & Replacement',
     re.compile(
         r'\b(warrant[yied]+|repair\b|replacement\b|replace\b|'
         r'fix\b|technician|engineer|service\s+centre|'
         r'send\s+back|return\s+for\s+repair|booked\s+repair|'
         r'under\s+warrant|out\s+of\s+warrant|cover[ed]?\s+under)\b', re.I)),
]

def classify_intent(text):
    """Return the first matching intent label, or 'Other / Uncategorized'."""
    for label, pattern in INTENT_PATTERNS:
        if pattern.search(text):
            return label
    return 'Other / Uncategorized'


def load_data():
    t0 = time.time()
    print("Loading dataset (structural + text columns)...", flush=True)
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
    return df


def reconstruct_convs(df):
    t0 = time.time()
    print("Building conversation tree...", flush=True)
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

    df['conv_id'] = [get_root(tid) for tid in df['tweet_id'].values]
    print(f"Tree built in {time.time()-t0:.2f}s — {df['conv_id'].nunique():,} conversations", flush=True)
    return df


def analyze_brand_intents(df, brand, sample_size=2000):
    """
    Extract customer messages from multi-turn conversations for a brand,
    classify by intent, and return summary rows + example messages.
    """
    print(f"\n--- Analyzing {brand} ---", flush=True)

    # Conversations where brand posted an outbound tweet
    brand_conv_ids = set(df[(~df['inbound']) & (df['author_id'] == brand)]['conv_id'].unique())

    # Filter to conversations with BOTH sides (multi-turn: len >= 3)
    conv_size = df[df['conv_id'].isin(brand_conv_ids)].groupby('conv_id').size()
    multi_turn_ids = set(conv_size[conv_size >= 3].index)

    # Customer messages in those conversations
    cust_msgs = df[
        df['inbound'] &
        df['conv_id'].isin(multi_turn_ids)
    ][['conv_id', 'tweet_id', 'text']].dropna(subset=['text'])

    cust_msgs = cust_msgs[cust_msgs['text'].str.strip().str.len() >= 20]
    total_msgs = len(cust_msgs)
    print(f"  Multi-turn convs: {len(multi_turn_ids):,}, customer messages: {total_msgs:,}", flush=True)

    # Sample
    if total_msgs > sample_size:
        cust_msgs = cust_msgs.sample(sample_size, random_state=42)

    # Classify intents
    cust_msgs = cust_msgs.copy()
    cust_msgs['intent'] = cust_msgs['text'].apply(classify_intent)

    intent_counts = cust_msgs['intent'].value_counts()
    sampled_total = len(cust_msgs)

    # Build per-intent rows with 2 examples each
    rows = []
    examples_store = {}
    for intent_label, count in intent_counts.items():
        pct = round(count / sampled_total * 100, 1)
        examples = (
            cust_msgs[cust_msgs['intent'] == intent_label]['text']
            .dropna()
            .str.strip()
            .sample(min(2, count), random_state=42)
            .tolist()
        )
        examples_store[intent_label] = examples
        rows.append({
            'brand': brand,
            'intent': intent_label,
            'count_in_sample': int(count),
            'pct_of_sample': pct,
            'example_1': examples[0] if len(examples) > 0 else '',
            'example_2': examples[1] if len(examples) > 1 else '',
        })

    return rows, examples_store


def main():
    t_start = time.time()
    df = load_data()
    df = reconstruct_convs(df)

    all_rows = []
    all_examples = {}

    for brand in TARGET_BRANDS:
        rows, examples = analyze_brand_intents(df, brand, sample_size=2000)
        all_rows.extend(rows)
        all_examples[brand] = examples

    # Save CSV
    result_df = pd.DataFrame(all_rows)
    result_df.to_csv('analysis/intent_analysis.csv', index=False)
    print("\n✓ Saved analysis/intent_analysis.csv", flush=True)

    # Save examples as JSON for later use
    with open('analysis/intent_examples.json', 'w', encoding='utf-8') as f:
        json.dump(all_examples, f, indent=2, ensure_ascii=False)
    print("✓ Saved analysis/intent_examples.json", flush=True)

    # Print summary per brand
    print("\n" + "="*90, flush=True)
    print("INTENT DIVERSITY SUMMARY", flush=True)
    print("="*90, flush=True)
    for brand in TARGET_BRANDS:
        brand_rows = [r for r in all_rows if r['brand'] == brand]
        print(f"\n{'─'*60}\n{brand}\n{'─'*60}", flush=True)
        for r in brand_rows:
            bar = '█' * max(1, round(r['pct_of_sample'] / 2))
            print(f"  {r['intent']:<42} {r['count_in_sample']:>5}  ({r['pct_of_sample']:>5.1f}%)  {bar}", flush=True)

    print(f"\nDone in {time.time()-t_start:.1f}s", flush=True)


if __name__ == '__main__':
    main()
