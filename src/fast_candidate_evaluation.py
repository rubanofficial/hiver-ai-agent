import os
import re
import sys
import time
import json
import math
from collections import defaultdict, Counter
import pandas as pd
import numpy as np

# Ensure analysis folders exist
os.makedirs('analysis', exist_ok=True)
os.makedirs('analysis/candidate_examples', exist_ok=True)

def run_evaluation():
    t0 = time.time()
    print("="*80, flush=True)
    print("HIVER AI AGENT - OPTIMIZED TWCS CANDIDATE EVALUATION", flush=True)
    print("="*80, flush=True)
    
    # -------------------------------------------------------------
    # Step 1: Load Data
    # -------------------------------------------------------------
    print("\n[1/6] Loading twcs.csv...", flush=True)
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
    print(f"Loaded {len(df):,} tweets in {time.time()-t0:.2f}s", flush=True)
    
    # -------------------------------------------------------------
    # Step 2: Global Data Quality
    # -------------------------------------------------------------
    print("\n[2/6] Computing Data Quality Metrics...", flush=True)
    total_tweets = len(df)
    missing_text_cnt = int(df['text'].isna().sum())
    empty_text_cnt = int((df['text'].fillna('').str.strip() == '').sum())
    duplicate_ids_cnt = int(df['tweet_id'].duplicated().sum())
    missing_parent_cnt = int(df['in_response_to_tweet_id'].isna().sum())
    missing_response_cnt = int(df['response_tweet_id'].isna().sum())
    
    text_clean = df['text'].fillna('').str.strip()
    short_msgs = int((text_clean.str.len() < 15).sum())
    url_only_regex = re.compile(r'^(https?://\S+|t\.co/\S+|\s+)+$', re.IGNORECASE)
    # Fast regex match on sample / vectorize
    url_only_cnt = int(text_clean.str.match(r'^(https?://\S+|t\.co/\S+|\s+)+$', na=False).sum())
    
    global_dq = {
        'total_rows': total_tweets,
        'missing_text_count': missing_text_cnt,
        'empty_text_count': empty_text_cnt,
        'duplicate_tweet_ids': duplicate_ids_cnt,
        'missing_parent_id_count': missing_parent_cnt,
        'missing_parent_id_pct': round(missing_parent_cnt / total_tweets * 100, 2),
        'missing_response_id_count': missing_response_cnt,
        'short_messages_lt_15_chars': short_msgs,
        'short_messages_pct': round(short_msgs / total_tweets * 100, 2),
        'url_only_messages': url_only_cnt,
        'inbound_tweets': int(df['inbound'].sum()),
        'outbound_tweets': int((~df['inbound']).sum())
    }
    with open('analysis/global_data_quality.json', 'w') as f:
        json.dump(global_dq, f, indent=2)
    print("Global DQ:", global_dq, flush=True)
    
    # -------------------------------------------------------------
    # Step 3: Fast Thread Reconstruction
    # -------------------------------------------------------------
    t1 = time.time()
    print("\n[3/6] Reconstructing Conversation Trees...", flush=True)
    df['in_response_clean'] = pd.to_numeric(df['in_response_to_tweet_id'], errors='coerce')
    
    # Build array parent map
    tid_vals = df['tweet_id'].values
    pid_vals = df['in_response_clean'].values
    
    parent_dict = {}
    for tid, pid in zip(tid_vals, pid_vals):
        if not np.isnan(pid):
            parent_dict[tid] = int(pid)
            
    root_cache = {}
    def get_root(tid):
        curr = tid
        path = []
        while curr in parent_dict:
            if curr in root_cache:
                curr = root_cache[curr]
                break
            path.append(curr)
            curr = parent_dict[curr]
            if len(path) > 50:
                break
        for node in path:
            root_cache[node] = curr
        root_cache[tid] = curr
        return curr

    df['conv_id'] = [get_root(tid) for tid in tid_vals]
    print(f"Reconstructed conversations in {time.time()-t1:.2f}s. Unique conversations: {df['conv_id'].nunique():,}", flush=True)
    
    # -------------------------------------------------------------
    # Step 4: Candidate Selection & Fast Subsetting
    # -------------------------------------------------------------
    t2 = time.time()
    print("\n[4/6] Filtering candidate brand conversations...", flush=True)
    
    target_brands = [
        'AmazonHelp', 'AppleSupport', 'Uber_Support', 'SpotifyCares', 'Delta',
        'Tesco', 'AmericanAir', 'TMobileHelp', 'comcastcares', 'British_Airways',
        'SouthwestAir', 'VirginTrains', 'Ask_Spectrum', 'XboxSupport', 'sprintcare',
        'hulu_support', 'sainsburys', 'GWRHelp', 'AskPlayStation', 'ChipotleTweets',
        'MicrosoftHelps', 'AskAmex', 'DellCares', 'nationalrailenq', 'SW_Help', 'VerizonSupport'
    ]
    target_set = set(target_brands)
    
    # Outbound tweets for target brands
    outbound_target = df[(~df['inbound']) & (df['author_id'].isin(target_set))]
    target_conv_ids = set(outbound_target['conv_id'].unique())
    print(f"Found {len(target_conv_ids):,} conversations involving target brands.", flush=True)
    
    # Filter df to only these conversations
    cdf = df[df['conv_id'].isin(target_conv_ids)].copy()
    cdf = cdf.sort_values(by=['conv_id', 'tweet_id']).reset_index(drop=True)
    print(f"Sub-dataframe shape: {cdf.shape} in {time.time()-t2:.2f}s", flush=True)
    
    # -------------------------------------------------------------
    # Step 5: High-Performance Per-Conversation Aggregation
    # -------------------------------------------------------------
    t3 = time.time()
    print("\n[5/6] Analyzing conversation quality, intents, and actionability...", flush=True)
    
    intent_patterns = {
        'Account & Authentication': re.compile(r'\b(login|log in|log-in|sign in|signin|password|reset password|locked out|account access|2fa|verification code|auth|username|pin|credentials)\b', re.I),
        'Billing & Payments': re.compile(r'\b(bill|billing|charge|charged|payment|invoice|receipt|fee|credit card|debit|double charge|bank|paid|overcharge|unauthorized charge|transaction)\b', re.I),
        'Refunds & Returns': re.compile(r'\b(refund|refunded|refunding|return|returns|money back|reimbursement|exchange|return item)\b', re.I),
        'Shipping & Delivery': re.compile(r'\b(ship|shipping|delivery|package|parcel|courier|tracking|delivered|dispatch|order status|arrive|arrived|late delivery|tracking number|ups|usps|fedex|carrier|delayed delivery)\b', re.I),
        'Cancellation': re.compile(r'\b(cancel|cancelling|cancellation|close account|stop service|unsubscribe|terminate|delete account|cancel order|cancel subscription)\b', re.I),
        'Technical & Troubleshooting': re.compile(r'\b(error|bug|crash|crashes|crashing|glitch|freeze|freezes|freezing|slow|broken|not working|wont work|black screen|reboot|wifi|connect|connection|app update|install|firmware|hardware|server|down|outage)\b', re.I),
        'Product & Service Defect': re.compile(r'\b(damaged|broken|faulty|defective|poor quality|terrible service|bad quality|missing item|wrong item|spoiled|expired|disappointed)\b', re.I),
        'Subscription & Membership': re.compile(r'\b(subscription|membership|renew|renewal|auto-renew|trial|free trial|prime|plan|upgrade|tier|premium|family plan)\b', re.I),
        'Booking, Transit & Schedule': re.compile(r'\b(flight|flights|train|trains|ticket|tickets|seat|seats|boarding|delay|delayed|timetable|schedule|platform|coach|ride|driver|pickup|trip|fare|station|airport|gate)\b', re.I),
        'General Inquiry & How-To': re.compile(r'\b(how do i|how to|question|help with|where can i|can i|is there a way|policy|store hours|available|wondering if)\b', re.I)
    }
    
    dm_regex = re.compile(r'\b(dm|direct message|dm us|send us a dm|send a dm|private message|pm us|reach out via dm)\b', re.I)
    action_regex = re.compile(r'\b(settings|restart|turn off|clear cache|reinstall|steps|click|tap|select|press|browser|update|link|help page|article|guide|try this|go to|menu|following)\b', re.I)
    resolution_signal_regex = re.compile(r'\b(thank you|thanks|that worked|it worked|fixed|sorted|works now|appreciate it|all set|resolved|great help|solved)\b', re.I)
    
    # Iterate conversation groups efficiently
    # We group by conv_id
    conv_grouped = cdf.groupby('conv_id')
    
    brand_convs = defaultdict(list)
    brand_intent_counts = defaultdict(Counter)
    
    for cid, group in conv_grouped:
        # Check if conversation has both inbound and outbound
        has_inbound = group['inbound'].any()
        outbound_rows = group[~group['inbound']]
        if not has_inbound or outbound_rows.empty:
            continue
            
        brands_in_group = outbound_rows['author_id'].unique()
        
        # Build lightweight conversation structure
        # Extract row lists: (tweet_id, author_id, inbound, created_at, text)
        t_records = []
        for row in group.itertuples(index=False):
            t_records.append({
                'tweet_id': row.tweet_id,
                'author_id': str(row.author_id) if pd.notna(row.author_id) else '',
                'inbound': bool(row.inbound),
                'created_at': str(row.created_at) if pd.notna(row.created_at) else '',
                'text': str(row.text) if pd.notna(row.text) else '',
                'conv_id': cid
            })
            
        for b in brands_in_group:
            if b in target_set:
                brand_convs[b].append(t_records)
                
    print(f"Grouped conversations for {len(brand_convs)} brands in {time.time()-t3:.2f}s", flush=True)
    
    # -------------------------------------------------------------
    # Step 6: Compute Full Metrics per Brand
    # -------------------------------------------------------------
    t4 = time.time()
    brand_stats = []
    
    for brand, conv_list in brand_convs.items():
        n_conv = len(conv_list)
        if n_conv < 100:
            continue
            
        lengths = []
        ge3 = 0
        ge4 = 0
        ge5 = 0
        cust_then_brand = 0
        multi_turn = 0
        dm_convs = 0
        action_convs = 0
        res_convs = 0
        brand_lens = []
        
        for conv in conv_list:
            clen = len(conv)
            lengths.append(clen)
            if clen >= 3: ge3 += 1
            if clen >= 4: ge4 += 1
            if clen >= 5: ge5 += 1
            
            # Turns and interaction
            first_cust_txt = None
            last_cust_txt = None
            has_cb = False
            prev_role = None
            turns = 0
            conv_has_dm = False
            conv_has_action = False
            
            for idx, t in enumerate(conv):
                role = 'C' if t['inbound'] else 'B'
                if t['inbound']:
                    if first_cust_txt is None:
                        first_cust_txt = t['text']
                    last_cust_txt = t['text']
                elif t['author_id'] == brand:
                    brand_lens.append(len(t['text']))
                    if dm_regex.search(t['text']):
                        conv_has_dm = True
                    if action_regex.search(t['text']):
                        conv_has_action = True
                        
                if prev_role == 'C' and role == 'B':
                    has_cb = True
                if prev_role is not None and role != prev_role:
                    turns += 1
                prev_role = role
                
            if has_cb: cust_then_brand += 1
            if turns >= 3: multi_turn += 1
            if conv_has_dm: dm_convs += 1
            if conv_has_action: action_convs += 1
            if clen >= 3 and last_cust_txt and last_cust_txt != first_cust_txt:
                if resolution_signal_regex.search(last_cust_txt):
                    res_convs += 1
                    
            # Intent classification
            if first_cust_txt:
                matched = False
                for cat, pat in intent_patterns.items():
                    if pat.search(first_cust_txt):
                        brand_intent_counts[brand][cat] += 1
                        matched = True
                if not matched:
                    brand_intent_counts[brand]['Other / Uncategorized'] += 1
                    
        # Compute Stats
        avg_l = np.mean(lengths) if lengths else 0
        med_l = np.median(lengths) if lengths else 0
        pct_ge3 = round(ge3 / n_conv * 100, 2)
        pct_ge4 = round(ge4 / n_conv * 100, 2)
        pct_ge5 = round(ge5 / n_conv * 100, 2)
        pct_cb = round(cust_then_brand / n_conv * 100, 2)
        pct_mt = round(multi_turn / n_conv * 100, 2)
        avg_blen = round(float(np.mean(brand_lens)), 1) if brand_lens else 0
        dm_rate = round(dm_convs / n_conv * 100, 2)
        action_rate = round(action_convs / n_conv * 100, 2)
        res_rate = round(res_convs / n_conv * 100, 2)
        
        # Entropy
        cat_counts = {k: v for k, v in brand_intent_counts[brand].items() if k != 'Other / Uncategorized'}
        tot_cats = sum(cat_counts.values())
        if tot_cats > 0:
            probs = [c / tot_cats for c in cat_counts.values()]
            entropy = round(-sum(p * math.log2(p) for p in probs if p > 0), 3)
        else:
            entropy = 0.0
            
        top_int = brand_intent_counts[brand].most_common(1)[0][0] if brand_intent_counts[brand] else 'None'
        top_int_pct = round(brand_intent_counts[brand].most_common(1)[0][1] / n_conv * 100, 1) if brand_intent_counts[brand] else 0
        distinct_int = len([k for k, v in cat_counts.items() if v >= 10])
        
        brand_stats.append({
            'brand': brand,
            'reconstructed_conversations': n_conv,
            'avg_conv_length': round(float(avg_l), 2),
            'median_conv_length': round(float(med_l), 2),
            'pct_ge_3_msgs': pct_ge3,
            'pct_ge_4_msgs': pct_ge4,
            'pct_ge_5_msgs': pct_ge5,
            'cust_then_brand_convs': cust_then_brand,
            'pct_cust_then_brand': pct_cb,
            'multi_turn_convs': multi_turn,
            'pct_multi_turn': pct_mt,
            'avg_brand_reply_chars': avg_blen,
            'dm_pivot_pct': dm_rate,
            'actionable_guidance_pct': action_rate,
            'resolution_signal_pct': res_rate,
            'intent_entropy': entropy,
            'top_intent': top_int,
            'top_intent_pct': top_int_pct,
            'distinct_intent_categories': distinct_int
        })
        
    stats_df = pd.DataFrame(brand_stats)
    
    # -------------------------------------------------------------
    # Step 7: Scoring Formula (Transparent Weights)
    # -------------------------------------------------------------
    # 1. Volume & Golden Set Viability (15 pts): log-scaled (4,000+ = 15 pts)
    stats_df['score_volume'] = stats_df['reconstructed_conversations'].apply(
        lambda v: min(15.0, (math.log10(max(v, 1)) / math.log10(50000)) * 15.0)
    ).round(2)
    
    # 2. Conversation Depth (30 pts): % >=3 msgs (15 pts) + % Multi-turn (15 pts)
    stats_df['score_depth'] = (
        (stats_df['pct_ge_3_msgs'] / 100.0 * 15.0) +
        (stats_df['pct_multi_turn'] / 70.0 * 15.0).clip(upper=15.0)
    ).round(2)
    
    # 3. Intent Richness & Entropy (25 pts): Entropy (15 pts) + Distinct intents (10 pts)
    stats_df['score_intent'] = (
        (stats_df['intent_entropy'] / 3.0 * 15.0).clip(upper=15.0) +
        (stats_df['distinct_intent_categories'] / 9.0 * 10.0).clip(upper=10.0)
    ).round(2)
    
    # 4. Actionability & Grounding (20 pts): Actionable guidance (10 pts) + Verbosity (5 pts) + Non-DM (5 pts)
    stats_df['score_grounding'] = (
        (stats_df['actionable_guidance_pct'] / 60.0 * 10.0).clip(upper=10.0) +
        (stats_df['avg_brand_reply_chars'] / 140.0 * 5.0).clip(upper=5.0) +
        ((100.0 - stats_df['dm_pivot_pct']) / 100.0 * 5.0)
    ).round(2)
    
    # 5. Data Hygiene & Interactive Flow (10 pts): % cust-then-brand
    stats_df['score_hygiene'] = (stats_df['pct_cust_then_brand'] / 100.0 * 10.0).round(2)
    
    # Composite Score
    stats_df['composite_score'] = (
        stats_df['score_volume'] +
        stats_df['score_depth'] +
        stats_df['score_intent'] +
        stats_df['score_grounding'] +
        stats_df['score_hygiene']
    ).round(2)
    
    stats_df = stats_df.sort_values(by='composite_score', ascending=False).reset_index(drop=True)
    stats_df['rank'] = stats_df.index + 1
    
    # Save CSVs
    stats_df.to_csv('analysis/candidate_brands.csv', index=False)
    stats_df.to_csv('analysis/conversation_stats.csv', index=False)
    
    # Save Intent Distribution CSV
    intent_rows = []
    for brand, dist in brand_intent_counts.items():
        r = {'brand': brand}
        r.update(dist)
        intent_rows.append(r)
    intent_df = pd.DataFrame(intent_rows).fillna(0)
    intent_df.to_csv('analysis/intent_samples.csv', index=False)
    
    print("\n" + "="*80, flush=True)
    print("FINAL CANDIDATE RANKING TABLE (TOP 15)", flush=True)
    print("="*80, flush=True)
    disp_cols = ['rank', 'brand', 'reconstructed_conversations', 'avg_conv_length', 'pct_ge_3_msgs', 'pct_multi_turn', 'actionable_guidance_pct', 'intent_entropy', 'composite_score']
    print(stats_df[disp_cols].head(15).to_string(index=False), flush=True)
    
    # -------------------------------------------------------------
    # Step 8: Extract Real Exemplars for Top Candidates
    # -------------------------------------------------------------
    top_candidates_to_extract = stats_df.head(5)['brand'].tolist()
    candidate_examples_data = {}
    
    for brand in top_candidates_to_extract:
        conv_list = brand_convs[brand]
        # Pick 3 rich multi-turn examples
        valid_exemplars = []
        for c in conv_list:
            # Starts with customer, length between 3 and 7, contains brand
            if len(c) >= 3 and c[0]['inbound']:
                # count brand responses
                brand_resp = [t for t in c if not t['inbound'] and t['author_id'] == brand]
                cust_resp = [t for t in c if t['inbound']]
                if len(brand_resp) >= 1 and len(cust_resp) >= 2:
                    valid_exemplars.append(c)
                    
        valid_exemplars.sort(key=lambda x: len(x), reverse=True)
        selected = valid_exemplars[:3]
        
        ex_list = []
        for i, c in enumerate(selected):
            ex_obj = {
                'example_id': f"{brand}_ex_{i+1}",
                'conv_id': int(c[0]['conv_id']),
                'length': len(c),
                'turns': []
            }
            for t in c:
                ex_obj['turns'].append({
                    'tweet_id': int(t['tweet_id']),
                    'author': 'Customer' if t['inbound'] else f"Brand ({t['author_id']})",
                    'inbound': t['inbound'],
                    'created_at': t['created_at'],
                    'text': t['text']
                })
            ex_list.append(ex_obj)
        candidate_examples_data[brand] = ex_list
        
        # Write Markdown file
        with open(f'analysis/candidate_examples/{brand}_examples.md', 'w', encoding='utf-8') as f:
            f.write(f"# Real Reconstructed Conversation Examples: {brand}\n\n")
            for ex in ex_list:
                f.write(f"### Example {ex['example_id']} (Length: {ex['length']} tweets, Conv ID: {ex['conv_id']})\n\n")
                for turn in ex['turns']:
                    f.write(f"**{turn['author']}** (`tweet_id: {turn['tweet_id']}`):\n> {turn['text']}\n\n")
                f.write("---\n\n")
                
    with open('analysis/candidate_examples.json', 'w', encoding='utf-8') as f:
        json.dump(candidate_examples_data, f, indent=2)
        
    print(f"\nCompleted all evaluations in {time.time()-t0:.2f}s successfully!", flush=True)

if __name__ == '__main__':
    run_evaluation()
