import numpy as np
from tqdm import tqdm

def compute_alignment_cost(cue_start, cue_end, group_start, group_end, 
                           duration_penalty_weight, gap_penalty_weight, gap,
                           mismatch_total, mismatch_weight):
    cue_duration = cue_end - cue_start
    group_duration = group_end - group_start
    return (abs(cue_start - group_start) +
            abs(cue_end - group_end) +
            duration_penalty_weight * abs(cue_duration - group_duration) +
            gap_penalty_weight * gap +
            mismatch_weight * mismatch_total)

def compute_mismatch_array(signs, cue_text):
    """
    For a list of sign segments (signs), compute an array where for each segment:
      - If seg['subtitle'] is non-empty and differs from cue_text, penalty = mismatch_weight.
      - Otherwise, penalty = 0.
    Returns a list of penalty values.
    """
    return [1 if (seg.get('subtitle', '') and seg.get('subtitle', '') != cue_text) else 0
            for seg in signs]

def dp_align_subtitles_to_signs(cues, sign_segments, 
                                duration_penalty_weight=0.4, gap_penalty_weight=2.0, 
                                window_size=40, max_gap=8.0, mismatch_weight=1000):
    M = len(cues)
    N = len(sign_segments)
    if M == 0 or N == 0:
        return
    original_cue_timings = [(c['start'], c['end']) for c in cues]
    sign_mids = [(seg['start'] + seg['end']) / 2 for seg in sign_segments]
    
    # Precompute gap costs.
    gap_cost = np.zeros((N+1, N+1))
    for i in range(N):
        current_gap = 0
        for j in range(i+1, N):
            gap = sign_segments[j]['start'] - sign_segments[j-1]['end']
            current_gap += max(0, gap)
            gap_cost[i][j] = current_gap
            
    dp = [[float('inf')] * (N + 1) for _ in range(M + 1)]
    prev = [[-1] * (N + 1) for _ in range(M + 1)]
    dp[0][0] = 0

    # Check if any segment has a non-empty "subtitle".
    cslr_exists = any(seg.get('subtitle', '') for seg in sign_segments)

    for i in tqdm(range(1, M + 1), desc="Aligning cues"):
        cue = cues[i - 1]
        cue_mid = (cue['start'] + cue['end']) / 2
        cue_text = cue['text']
        diffs = [abs(mid - cue_mid) for mid in sign_mids]
        candidate_indices = np.argsort(diffs)[:window_size]
        lower_bound = int(min(candidate_indices))
        upper_bound = int(max(candidate_indices))
        j_lower = max(i, lower_bound + 1)
        j_upper = min(N, upper_bound + 1)
        if j_lower > j_upper:
            j_lower, j_upper = i, N
        
        if cslr_exists:
            # Precompute mismatch penalty array for the candidate window.
            local_signs = sign_segments[lower_bound:upper_bound+1]
            mismatch_array = compute_mismatch_array(local_signs, cue_text)
            mismatch_cumsum = np.concatenate(([0], np.cumsum(mismatch_array)))
        
        for j in range(j_lower, j_upper + 1):
            for k in range(i - 1, j):
                group_start = sign_segments[k]['start']
                group_end = sign_segments[j-1]['end']
                total_gap = gap_cost[k][j-1] if k < j-1 else 0
                if cslr_exists:
                    # Map global indices k and j to local indices.
                    local_k = k - lower_bound
                    local_j = j - lower_bound
                    if local_k < 0:
                        local_k = 0
                    if local_j >= len(mismatch_cumsum):
                        local_j = len(mismatch_cumsum) - 1
                    mismatch_total = mismatch_cumsum[local_j] - mismatch_cumsum[local_k]
                else:
                    mismatch_total = 0
                candidate = dp[i-1][k] + compute_alignment_cost(cue['start'], cue['end'],
                                                                  group_start, group_end,
                                                                  duration_penalty_weight,
                                                                  gap_penalty_weight,
                                                                  total_gap,
                                                                  mismatch_total,
                                                                  mismatch_weight)
                if candidate < dp[i][j]:
                    dp[i][j] = candidate
                    prev[i][j] = k

    best_j = None
    best_cost = float('inf')
    for j in range(M, N + 1):
        if dp[M][j] < best_cost:
            best_cost = dp[M][j]
            best_j = j
    if best_j is None or best_cost == float('inf'):
        return
    boundaries = [0] * (M + 1)
    boundaries[M] = best_j
    cur = best_j
    for i in range(M, 0, -1):
        k = prev[i][cur]
        boundaries[i - 1] = k
        cur = k

    # Post-processing: For each cue, further split the assigned group if gaps exceed max_gap.
    for i in range(M):
        original_group = sign_segments[boundaries[i]:boundaries[i+1]]
        if not original_group:
            continue
        original_start, original_end = original_cue_timings[i]
        sub_groups = []
        current_group = [original_group[0]]
        sub_group_gaps = [0]
        for seg in original_group[1:]:
            gap = seg['start'] - current_group[-1]['end']
            if gap <= max_gap:
                current_group.append(seg)
                sub_group_gaps[-1] += gap
            else:
                sub_groups.append((current_group, sub_group_gaps[-1]))
                current_group = [seg]
                sub_group_gaps.append(0)
        sub_groups.append((current_group, sub_group_gaps[-1]))
        min_cost = float('inf')
        best_sub_group = None
        for sg, sg_gap in sub_groups:
            sg_start = sg[0]['start']
            sg_end = sg[-1]['end']
            # Compute mismatch total for the subgroup.
            mismatch_total = sum(compute_mismatch_array(sg, cues[i]['text'])) if cslr_exists else 0
            cost = compute_alignment_cost(original_start, original_end, sg_start, sg_end,
                                          duration_penalty_weight, gap_penalty_weight, sg_gap,
                                          mismatch_total, mismatch_weight)
            if cost < min_cost:
                min_cost = cost
                best_sub_group = sg
        if best_sub_group:
            cues[i]['start'] = best_sub_group[0]['start']
            cues[i]['end'] = best_sub_group[-1]['end']
            cues[i]['mid'] = (cues[i]['start'] + cues[i]['end']) / 2

from fastdtw import fastdtw

def dp_align_subtitles_to_signs_dtw(cues, sign_segments):
    """
    A simple DTW-based alignment function.
    
    For each cue and sign segment, we form a tuple (mid, id) where:
      - For a cue, mid = cue['mid'] (or (start+end)/2) and id is a unique numeric ID for cue['text'] (0 if empty).
      - For a sign segment, mid = seg['mid'] (or computed) and id is a unique numeric ID for seg.get('subtitle', '') (0 if empty).
    
    The DTW distance function is defined as follows:
      - If both IDs are nonzero and equal, the distance is 0.
      - If both IDs are nonzero and different, the distance is abs(mid_cue - mid_seg) + 1000.
      - Otherwise, the distance is abs(mid_cue - mid_seg).
      
    After DTW, each cue is updated so that its start is the minimum start and its end is the maximum end among all sign segments assigned to it.
    """
    # Build a set of all non-empty text values from cues and sign segments.
    texts = set()
    for c in cues:
        if c['text']:
            texts.add(c['text'])
    for s in sign_segments:
        text = s.get('subtitle', '')
        if text:
            texts.add(text)
    # Map each non-empty text to a unique ID starting from 1; use 0 for empty.
    text_to_id = {text: idx+1 for idx, text in enumerate(sorted(texts))}
    
    # Build the sequences for DTW.
    cue_seq = [((c['mid'] if isinstance(c['mid'], float) else (c['start'] + c['end'])/2), 
                 text_to_id[c['text']] if c['text'] in text_to_id else 0)
               for c in cues]
    seg_seq = [((s['mid'] if isinstance(s['mid'], float) else (s['start'] + s['end'])/2),
                 text_to_id[s.get('subtitle', '')] if s.get('subtitle', '') in text_to_id else 0)
               for s in sign_segments]
    
    # Define the distance function.
    def dtw_dist(x, y):
        # x = (cue_mid, cue_id), y = (seg_mid, seg_id)
        if x[1] != 0 and y[1] != 0:
            if x[1] == y[1]:
                return 0
            else:
                return abs(x[0] - y[0]) + 1000
        return abs(x[0] - y[0])
    
    # Compute DTW alignment.
    distance, path = fastdtw(cue_seq, seg_seq, dist=dtw_dist)
    
    # Build assignments: cue index -> list of sign segment indices.
    assignments = {i: [] for i in range(len(cues))}
    for i, j in path:
        assignments[i].append(j)
    
    # Update each cue boundaries based on assigned sign segments.
    for i, cue in enumerate(cues):
        if not assignments[i]:
            continue
        assigned_segments = [sign_segments[j] for j in assignments[i]]
        new_start = min(seg['start'] for seg in assigned_segments)
        new_end = max(seg['end'] for seg in assigned_segments)
        cue['start'] = new_start
        cue['end'] = new_end
        cue['mid'] = (new_start + new_end) / 2
