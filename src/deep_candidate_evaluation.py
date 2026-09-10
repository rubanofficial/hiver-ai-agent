import os
import re
import time
import json
import math
from collections import defaultdict, Counter
import pandas as pd
import numpy as np

# Ensure analysis folders exist
os.makedirs('analysis', exist_ok=True)
os.makedirs('analysis/candidate_examples', exist_ok=True)

def analyze_all():
    t0 = time.time()
    print("="*80)
    print("HIVER AI AGENT - TWCS CANDIDATE BRAND EVALUATION")
    print("="*80)
    
    print("\n[1/6] Loading twcs.csv...")
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
    
    # -------------------------------------------------------------
    # Step 1: Overall Data Quality
    # -------------------------------------------------------------
    print("\n[2/6] Calculating Global Data Quality...")
    total_tweets = len(df)
    missing_text_cnt = int(df['text'].isna().sum())
    empty_text_cnt = int((df['text'].fillna('').str.strip() == '').sum())
    duplicate_ids_cnt = int(df['tweet_id'].duplicated().sum())
    missing_parent_cnt = int(df['in_response_to_tweet_id'].isna().sum())
    missing_response_cnt = int(df['response_tweet_id'].isna().sum())
    
    # Short messages and URL only
    text_clean = df['text'].fillna('').str.strip()
    short_msgs = int((text_clean.str.len() < 15).sum())
    url_only_regex = re.compile(r'^(https?://\S+|t\.co/\S+|\s+)+$', re.IGNORECASE)
    url_only_cnt = int(text_clean.apply(lambda x: bool(url_only_regex.match(x.strip())) if x else False).sum())
    
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
    print(f"Global Data Quality saved to analysis/global_data_quality.json")
    
    # -------------------------------------------------------------
    # Step 2: Build Conversation Graph
    # -------------------------------------------------------------
    print("\n[3/6] Reconstructing Conversation Threads...")
    df['in_response_to_clean'] = pd.to_numeric(df['in_response_to_tweet_id'], errors='coerce')
    
    # Fast parent dictionary
    parent_map = {}
    for tid, pid in zip(df['tweet_id'].values, df['in_response_to_clean'].values):
        if not np.isnan(pid):
            parent_map[tid] = int(pid)
            
    # Find root for each tweet
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
            if len(path) > 60: # loop prevention
                break
        for node in path:
            root_cache[node] = curr
        root_cache[tid] = curr
        return curr

    df['conv_id'] = [find_root(tid) for tid in df['tweet_id'].values]
    print(f"Discovered {df['conv_id'].nunique():,} unique conversation clusters.")
    
    # -------------------------------------------------------------
    # Step 3: Brand Identification and Conversation Filtering
    # -------------------------------------------------------------
    print("\n[4/6] Processing Conversations per Brand...")
    
    # Outbound tweets define brand accounts
    outbound_df = df[~df['inbound']]
    brand_vol = outbound_df['author_id'].value_counts()
    
    # Candidates to evaluate
    top_candidates = [
        'AmazonHelp', 'AppleSupport', 'Uber_Support', 'SpotifyCares', 'Delta',
        'Tesco', 'AmericanAir', 'TMobileHelp', 'comcastcares', 'British_Airways',
        'SouthwestAir', 'VirginTrains', 'Ask_Spectrum', 'XboxSupport', 'sprintcare',
        'hulu_support', 'sainsburys', 'GWRHelp', 'AskPlayStation', 'ChipotleTweets',
        'MicrosoftHelps', 'AskAmex', 'DellCares', 'nationalrailenq', 'SW_Help', 'VerizonSupport'
    ]
    candidate_set = set(top_candidates)
    
    # Create tweet mapping for fast lookup
    # Convert df to records / indexed dict for fast access
    print("Indexing tweets for thread traversal...")
    # Store tweet metadata in dicts for fast iteration
    tweet_dict = {}
    for row in df[['tweet_id', 'author_id', 'inbound', 'created_at', 'text', 'in_response_to_clean', 'conv_id']].itertuples(index=False):
        tweet_dict[row.tweet_id] = {
            'tweet_id': row.tweet_id,
            'author_id': str(row.author_id) if pd.notna(row.author_id) else '',
            'inbound': bool(row.inbound),
            'created_at': str(row.created_at) if pd.notna(row.created_at) else '',
            'text': str(row.text) if pd.notna(row.text) else '',
            'parent_id': int(row.in_response_to_clean) if not np.isnan(row.in_response_to_clean) else None,
            'conv_id': row.conv_id
        }
        
    # Group tweet IDs by conv_id
    conv_groups = defaultdict(list)
    for tid, tmeta in tweet_dict.items():
        conv_groups[tmeta['conv_id']].append(tid)
        
    print(f"Indexed {len(conv_groups):,} conversation groups.")
    
    # Intent categorization lexicons
    intent_patterns = {
        'Account & Authentication': re.compile(r'\b(login|log in|log-in|sign in|signin|password|reset password|locked out|account access|2fa|verification code|auth|username|pin|security code|credentials)\b', re.I),
        'Billing & Payments': re.compile(r'\b(bill|billing|charge|charged|payment|invoice|receipt|fee|credit card|debit|double charge|bank|paid|overcharge|unauthorized charge|transaction)\b', re.I),
        'Refunds & Returns': re.compile(r'\b(refund|refunded|refunding|return|returns|money back|reimbursement|exchange|return item)\b', re.I),
        'Shipping & Delivery': re.compile(r'\b(ship|shipping|delivery|package|parcel|courier|tracking|delivered|dispatch|order status|arrive|arrived|late delivery|tracking number|ups|usps|fedex|carrier|delayed delivery)\b', re.I),
        'Cancellation': re.compile(r'\b(cancel|cancelling|cancellation|close account|stop service|unsubscribe|terminate|delete account|cancel order|cancel subscription)\b', re.I),
        'Technical & Troubleshooting': re.compile(r'\b(error|bug|crash|crashes|crashing|glitch|freeze|freezes|freezing|slow|broken|not working|wont work|black screen|reboot|wifi|connect|connection|app update|install|firmware|hardware|server|down|outage)\b', re.I),
        'Product & Service Defect': re.compile(r'\b(damaged|broken|faulty|defective|poor quality|terrible service|bad quality|missing item|wrong item|spoiled|expired|scam|disappointed)\b', re.I),
        'Subscription & Membership': re.compile(r'\b(subscription|membership|renew|renewal|auto-renew|trial|free trial|prime|plan|upgrade|tier|premium|family plan)\b', re.I),
        'Booking, Transit & Schedule': re.compile(r'\b(flight|flights|train|trains|ticket|tickets|seat|seats|boarding|delay|delayed|timetable|schedule|platform|coach|ride|driver|pickup|trip|fare|station|airport|gate)\b', re.I),
        'General Inquiry & How-To': re.compile(r'\b(how do i|how to|question|help with|where can i|can i|is there a way|policy|store hours|available|wondering if)\b', re.I)
    }
    
    # DM and Actionability regexes
    dm_regex = re.compile(r'\b(dm|direct message|dm us|send us a dm|send a dm|private message|pm us|reach out via dm)\b', re.I)
    action_regex = re.compile(r'\b(settings|restart|turn off|clear cache|reinstall|steps|click|tap|select|press|browser|update|link|help page|article|guide|try this|go to|menu|following)\b', re.I)
    resolution_signal_regex = re.compile(r'\b(thank you|thanks|that worked|it worked|fixed|sorted|works now|appreciate it|all set|resolved|great help|solved)\b', re.I)
    
    # Process stats for all candidates
    brand_stats = []
    brand_intent_dist = {}
    brand_conversations = defaultdict(list)
    
    print("\n[5/6] Evaluating candidate brands...")
    
    # Assign conversations to brands
    # A conversation belongs to brand B if brand B posted at least one outbound message in it.
    for cid, tids in conv_groups.items():
        tweets = [tweet_dict[tid] for tid in tids]
        # Check if contains both inbound (customer) and outbound (brand)
        inbound_tweets = [t for t in tweets if t['inbound']]
        outbound_tweets = [t for t in tweets if not t['inbound']]
        
        if not inbound_tweets or not outbound_tweets:
            continue # skip one-sided
            
        # Find brand(s)
        brands_in_conv = set(t['author_id'] for t in outbound_tweets)
        for b in brands_in_conv:
            if b in candidate_set:
                brand_conversations[b].append(tweets)
                
    # Now calculate detailed metrics per candidate
    for brand in candidate_set:
        conv_list = brand_conversations.get(brand, [])
        num_convs = len(conv_list)
        if num_convs < 100:
            continue # skip negligible candidates
            
        lengths = []
        ge3_cnt = 0
        ge4_cnt = 0
        ge5_cnt = 0
        cust_then_brand_cnt = 0
        multi_turn_cnt = 0 # >= 2 turns (C -> B -> C -> B)
        brand_reply_lens = []
        dm_pivot_convs = 0
        actionable_convs = 0
        resolved_convs = 0
        
        # Intent counts
        intent_counts = Counter()
        first_cust_msgs = []
        
        # DQ metrics
        brand_empty_short_cnt = 0
        brand_url_only_cnt = 0
        
        for tweets in conv_list:
            # Sort tweets chronologically if possible, or by id (TWCS tweet_ids are sequentially ordered)
            tweets_sorted = sorted(tweets, key=lambda x: x['tweet_id'])
            
            clen = len(tweets_sorted)
            lengths.append(clen)
            if clen >= 3: ge3_cnt += 1
            if clen >= 4: ge4_cnt += 1
            if clen >= 5: ge5_cnt += 1
            
            # Check interaction pattern
            # Count turns: transitions between customer and brand
            turn_transitions = 0
            prev_role = None
            cust_first = False
            has_cust_then_brand = False
            first_cust_text = None
            last_cust_text = None
            
            has_dm = False
            has_action = False
            
            for idx, t in enumerate(tweets_sorted):
                curr_role = 'C' if t['inbound'] else 'B'
                
                # Brand response properties
                if not t['inbound'] and t['author_id'] == brand:
                    brand_reply_lens.append(len(t['text']))
                    if dm_regex.search(t['text']):
                        has_dm = True
                    if action_regex.search(t['text']):
                        has_action = True
                    if len(t['text']) < 15:
                        brand_empty_short_cnt += 1
                    if url_only_regex.match(t['text'].strip()):
                        brand_url_only_cnt += 1
                else:
                    if first_cust_text is None:
                        first_cust_text = t['text']
                    last_cust_text = t['text']
                
                if idx == 0 and curr_role == 'C':
                    cust_first = True
                if prev_role == 'C' and curr_role == 'B':
                    has_cust_then_brand = True
                if prev_role is not None and curr_role != prev_role:
                    turn_transitions += 1
                prev_role = curr_role
                
            if has_cust_then_brand:
                cust_then_brand_cnt += 1
            if turn_transitions >= 3: # C -> B -> C -> B is >= 3 transitions
                multi_turn_cnt += 1
            if has_dm:
                dm_pivot_convs += 1
            if has_action:
                actionable_convs += 1
                
            # Resolution signal in customer follow-up
            if clen >= 3 and last_cust_text and last_cust_text != first_cust_text:
                if resolution_signal_regex.search(last_cust_text):
                    resolved_convs += 1
                    
            # Intent classification on customer problem (first customer message)
            if first_cust_text:
                first_cust_msgs.append(first_cust_text)
                matched_intents = []
                for cat, pat in intent_patterns.items():
                    if pat.search(first_cust_text):
                        matched_intents.append(cat)
                if not matched_intents:
                    intent_counts['Other / Uncategorized'] += 1
                else:
                    for cat in matched_intents:
                        intent_counts[cat] += 1

        # Intent distribution & Shannon Entropy
        total_intent_hits = sum(intent_counts.values()) if intent_counts else 1
        intent_proportions = {k: v / total_intent_hits for k, v in intent_counts.items()}
        # Shannon entropy of intent distribution (higher = more balanced diversity across intents)
        # Exclude Other for entropy calculation
        cat_probs = [v for k, v in intent_proportions.items() if k != 'Other / Uncategorized']
        s = sum(cat_probs)
        if s > 0:
            norm_probs = [p / s for p in cat_probs]
            entropy = -sum(p * math.log2(p) for p in norm_probs if p > 0)
        else:
            entropy = 0.0
            
        brand_intent_dist[brand] = dict(intent_counts)
        
        # Compute summary metrics
        avg_len = np.mean(lengths) if lengths else 0
        med_len = np.median(lengths) if lengths else 0
        pct_ge3 = (ge3_cnt / num_convs * 100) if num_convs else 0
        pct_ge4 = (ge4_cnt / num_convs * 100) if num_convs else 0
        pct_ge5 = (ge5_cnt / num_convs * 100) if num_convs else 0
        pct_multi_turn = (multi_turn_cnt / num_convs * 100) if num_convs else 0
        pct_cust_then_brand = (cust_then_brand_cnt / num_convs * 100) if num_convs else 0
        avg_brand_reply_len = np.mean(brand_reply_lens) if brand_reply_lens else 0
        dm_rate = (dm_pivot_convs / num_convs * 100) if num_convs else 0
        action_rate = (actionable_convs / num_convs * 100) if num_convs else 0
        res_rate = (resolved_convs / num_convs * 100) if num_convs else 0
        
        brand_stats.append({
            'brand': brand,
            'reconstructed_conversations': num_convs,
            'avg_conv_length': round(float(avg_len), 2),
            'median_conv_length': round(float(med_len), 2),
            'pct_ge_3_msgs': round(pct_ge3, 2),
            'pct_ge_4_msgs': round(pct_ge4, 2),
            'pct_ge_5_msgs': round(pct_ge5, 2),
            'cust_then_brand_convs': cust_then_brand_cnt,
            'pct_cust_then_brand': round(pct_cust_then_brand, 2),
            'multi_turn_convs': multi_turn_cnt,
            'pct_multi_turn': round(pct_multi_turn, 2),
            'avg_brand_reply_chars': round(float(avg_brand_reply_len), 1),
            'dm_pivot_pct': round(dm_rate, 2),
            'actionable_guidance_pct': round(action_rate, 2),
            'resolution_signal_pct': round(res_rate, 2),
            'intent_entropy': round(entropy, 3),
            'top_intent': intent_counts.most_common(1)[0][0] if intent_counts else 'None',
            'top_intent_pct': round(intent_counts.most_common(1)[0][1] / len(conv_list) * 100, 1) if intent_counts else 0,
            'distinct_intent_categories': len([k for k in intent_counts if k != 'Other / Uncategorized' and intent_counts[k] >= 10])
        })
        
    stats_df = pd.DataFrame(brand_stats)
    
    # -------------------------------------------------------------
    # Step 4: Transparent Multi-Criteria Scoring Model
    # -------------------------------------------------------------
    # Transparent formula weights (Sum = 100):
    # 1. Volume & Viability (15 pts): log-scaled volume (>= 2000 is max 15 pts, minimum threshold for golden set)
    # 2. Conversation Depth (30 pts): Combination of % >= 3 msgs (15 pts) and % Multi-turn >= 2 turns (15 pts)
    # 3. Intent Richness & Balance (25 pts): Combination of intent entropy (15 pts) and # of distinct active intents (10 pts)
    # 4. Actionability & Grounding Quality (20 pts): Substantive guidance vs blind DM redirects (Actionable rate + non-DM balance + brand response verbosity)
    # 5. Data Hygiene & Interactive Structure (10 pts): % Cust-then-brand and low short/empty message rate
    
    print("\n[6/6] Computing Composite Ranking Scores...")
    
    # Component 1: Volume Score (0 - 15)
    # 1000 convs is good (10 pts), 4000+ is max (15 pts)
    stats_df['score_volume'] = stats_df['reconstructed_conversations'].apply(
        lambda v: min(15.0, (math.log10(max(v, 1)) / math.log10(50000)) * 15.0)
    )
    
    # Component 2: Depth Score (0 - 30)
    # pct_ge_3_msgs (0-15) + pct_multi_turn (0-15)
    # pct_ge_3_msgs range typically 30% to 85%
    stats_df['score_depth'] = (
        (stats_df['pct_ge_3_msgs'] / 100.0 * 15.0) +
        (stats_df['pct_multi_turn'] / 70.0 * 15.0).clip(upper=15.0)
    )
    
    # Component 3: Intent Diversity Score (0 - 25)
    # entropy (0-15 based on max theoretical ~3.32 for 10 classes) + distinct intents (0-10)
    stats_df['score_intent'] = (
        (stats_df['intent_entropy'] / 3.0 * 15.0).clip(upper=15.0) +
        (stats_df['distinct_intent_categories'] / 9.0 * 10.0).clip(upper=10.0)
    )
    
    # Component 4: Grounding & Actionability Score (0 - 20)
    # High actionable rate (0-10) + reasonable reply length >= 120 chars (0-5) + non-DM rate (0-5)
    stats_df['score_grounding'] = (
        (stats_df['actionable_guidance_pct'] / 60.0 * 10.0).clip(upper=10.0) +
        (stats_df['avg_brand_reply_chars'] / 140.0 * 5.0).clip(upper=5.0) +
        ((100.0 - stats_df['dm_pivot_pct']) / 100.0 * 5.0)
    )
    
    # Component 5: Data Hygiene Score (0 - 10)
    # High cust_then_brand percentage
    stats_df['score_hygiene'] = (stats_df['pct_cust_then_brand'] / 100.0 * 10.0)
    
    # Total Composite Score
    stats_df['composite_score'] = (
        stats_df['score_volume'] +
        stats_df['score_depth'] +
        stats_df['score_intent'] +
        stats_df['score_grounding'] +
        stats_df['score_hygiene']
    ).round(2)
    
    # Sort by composite score
    stats_df = stats_df.sort_values(by='composite_score', ascending=False).reset_index(drop=True)
    stats_df['rank'] = stats_df.index + 1
    
    # Save results to CSV
    stats_df.to_csv('analysis/candidate_brands.csv', index=False)
    stats_df.to_csv('analysis/conversation_stats.csv', index=False)
    
    # Save Intent Distribution to CSV
    intent_df_rows = []
    for brand, dist in brand_intent_dist.items():
        row = {'brand': brand}
        row.update(dist)
        intent_df_rows.append(row)
    intent_df = pd.DataFrame(intent_df_rows).fillna(0)
    intent_df.to_csv('analysis/intent_samples.csv', index=False)
    
    print("\nTop 15 Ranked Brands:")
    display_cols = ['rank', 'brand', 'reconstructed_conversations', 'avg_conv_length', 'pct_ge_3_msgs', 'pct_multi_turn', 'actionable_guidance_pct', 'intent_entropy', 'composite_score']
    print(stats_df[display_cols].head(15).to_string(index=False))
    
    # -------------------------------------------------------------
    # Step 5: Extract Real Conversation Examples for Top Candidates
    # -------------------------------------------------------------
    top_5_brands = stats_df.head(5)['brand'].tolist()
    print(f"\nExtracting real multi-turn conversation examples for top 5 candidates: {top_5_brands}...")
    
    candidate_examples_data = {}
    
    for brand in top_5_brands:
        conv_list = brand_conversations[brand]
        # Filter for high quality multi-turn conversations (length between 3 and 7, starts with Customer)
        exemplars = []
        for tweets in conv_list:
            sorted_tweets = sorted(tweets, key=lambda x: x['tweet_id'])
            if len(sorted_tweets) >= 3 and sorted_tweets[0]['inbound']:
                # Check for customer follow-up
                roles = [('Customer' if t['inbound'] else brand) for t in sorted_tweets]
                if 'Customer' in roles[1:] and brand in roles:
                    exemplars.append(sorted_tweets)
                    
        # Sort exemplars by length and clarity
        exemplars.sort(key=lambda x: len(x), reverse=True)
        
        # Pick 3 diverse examples per brand
        selected = exemplars[:3]
        brand_examples = []
        for c_idx, conv in enumerate(selected):
            conv_obj = {
                'example_id': f"{brand}_ex_{c_idx+1}",
                'conv_id': conv[0]['conv_id'],
                'length': len(conv),
                'turns': []
            }
            for t in conv:
                conv_obj['turns'].append({
                    'tweet_id': t['tweet_id'],
                    'author': 'Customer' if t['inbound'] else f"Brand ({t['author_id']})",
                    'inbound': t['inbound'],
                    'created_at': t['created_at'],
                    'text': t['text']
                })
            brand_examples.append(conv_obj)
            
        candidate_examples_data[brand] = brand_examples
        
        # Write individual markdown file for each top candidate
        with open(f'analysis/candidate_examples/{brand}_examples.md', 'w', encoding='utf-8') as f:
            f.write(f"# Real Multi-Turn Conversation Examples: {brand}\n\n")
            for ex in brand_examples:
                f.write(f"### Example {ex['example_id']} (Length: {ex['length']} tweets, Conv ID: {ex['conv_id']})\n\n")
                for turn in ex['turns']:
                    f.write(f"**{turn['author']}** (`tweet_id: {turn['tweet_id']}`):\n> {turn['text']}\n\n")
                f.write("---\n\n")
                
    with open('analysis/candidate_examples.json', 'w', encoding='utf-8') as f:
        json.dump(candidate_examples_data, f, indent=2)
        
    print("Candidate examples saved successfully to analysis/candidate_examples/")
    print(f"Evaluation completed in {time.time()-t0:.2f}s.")

if __name__ == '__main__':
    analyze_all()
