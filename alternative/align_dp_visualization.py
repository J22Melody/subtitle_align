import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def visualize_similarity_heatmap(sim_matrix, cues, sign_segments, gt_cues=None, new_cues=None, fps=25, start_time_window=0, end_time_window=256):
    """Visualize similarity matrix with aligned and ground truth cues at frame-level resolution.
    
    [Documentation omitted for brevity]
    """

    def format_time_full(total_seconds):
        if total_seconds < 0:
            total_seconds = 0
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    time_axis = np.arange(start_time_window, end_time_window + 1/fps, 1/fps)
    seg_indices_in_window = []
    for idx, seg in enumerate(sign_segments):
        if seg['start'] is not None and seg['end'] is not None:
            if seg['end'] >= start_time_window and seg['start'] <= end_time_window:
                seg_indices_in_window.append(idx)

    cue_indices_in_window = []
    for idx, cue in enumerate(cues):
        if cue['start'] is not None and cue['end'] is not None:
            if cue['end'] >= start_time_window and cue['start'] <= end_time_window:
                cue_indices_in_window.append(idx)
    M_filtered = len(cue_indices_in_window)

    heatmap_data = np.full((M_filtered, len(time_axis)), np.nan)
    for j in seg_indices_in_window:
        seg = sign_segments[j]
        if seg['start'] is None or seg['end'] is None:
            continue
        indices = np.where((time_axis >= seg['start']) & (time_axis <= seg['end']))[0]
        if len(indices) > 0:
            for r, cue_idx in enumerate(cue_indices_in_window):
                heatmap_data[r, indices] = sim_matrix[cue_idx, j]

    masks_list = []
    normalized_heatmap = heatmap_data.copy()
    window_size = 40
    filtered_seg_mid_times = []
    for j in seg_indices_in_window:
        seg = sign_segments[j]
        mid_t = (seg['start'] + seg['end']) / 2
        filtered_seg_mid_times.append(mid_t)
    filtered_seg_mid_times = np.array(filtered_seg_mid_times)

    for row_idx, cue_idx in enumerate(cue_indices_in_window):
        cue = cues[cue_idx]
        cue_mid = (cue['start'] + cue['end']) / 2
        diffs = np.abs(filtered_seg_mid_times - cue_mid)
        candidate_order = np.argsort(diffs)[:window_size]
        
        valid_candidates = [c for c in candidate_order if c < len(seg_indices_in_window)]
        mask = np.zeros(len(time_axis), dtype=bool)
        for candidate in valid_candidates:
            seg_global_idx = seg_indices_in_window[candidate]
            seg = sign_segments[seg_global_idx]
            if seg['start'] is None or seg['end'] is None:
                continue
            candidate_mask = (time_axis >= seg['start']) & (time_axis <= seg['end'])
            mask = mask | candidate_mask
        masks_list.append(mask)
        if np.any(mask):
            local_vals = heatmap_data[row_idx, mask]
            # local_vals = softmax_normalize(local_vals, axis=0, tau=10)
            normalized_heatmap[row_idx, mask] = local_vals

    global_min = np.nanmin(normalized_heatmap)
    for row_idx in range(M_filtered):
        mask = masks_list[row_idx]
        normalized_heatmap[row_idx, ~mask] = global_min

    heatmap_data = normalized_heatmap

    duration = end_time_window - start_time_window
    fig_width = duration * 1.5

    plt.figure(figsize=(fig_width, 10))
    plt.subplots_adjust(left=10/fig_width)
    im = plt.imshow(heatmap_data, aspect='auto', origin='upper',
                    interpolation='nearest', cmap='inferno',
                    extent=(start_time_window, end_time_window, M_filtered, 0))
    plt.colorbar(im, label='Similarity')
    
    tick_positions = np.arange(start_time_window, end_time_window + 1, 1)
    plt.xticks(tick_positions, [format_time_full(t) for t in tick_positions], rotation=45)

    y_labels = []
    for idx in cue_indices_in_window:
        cue = cues[idx]
        cue_text = (cue['text'] or "").replace("\n", " ")
        if len(cue_text) > 20:
            cue_text = cue_text[:20] + "..."
        y_labels.append(f"{format_time_full(cue['start'])}-{format_time_full(cue['end'])} {cue_text}")
    plt.yticks(np.arange(0.5, M_filtered + 0.5), y_labels)

    ax = plt.gca()

    for i, cue_idx in enumerate(cue_indices_in_window[::-1]):
        cue = cues[cue_idx]
        box_start = max(cue['start'], start_time_window)
        box_end = min(cue['end'], end_time_window)
        if box_end > box_start:
            y_box = M_filtered - i - 1
            rect = plt.Rectangle((box_start, y_box), box_end-box_start, 1,
                                 edgecolor='red', facecolor='none', linewidth=2)
            ax.add_patch(rect)
    
    if gt_cues:
        gt_text_map = {}
        for gt_cue in gt_cues:
            if gt_cue['text']:
                clean_text = gt_cue['text'].strip().replace("\n", " ")[:50]
                gt_text_map[clean_text] = gt_cue
        
        for i, cue_idx in enumerate(cue_indices_in_window[::-1]):
            cue = cues[cue_idx]
            clean_text = cue['text'].strip().replace("\n", " ")[:50]
            gt_cue = gt_text_map.get(clean_text)
            if gt_cue and gt_cue['start'] and gt_cue['end']:
                box_start = max(gt_cue['start'], start_time_window)
                box_end = min(gt_cue['end'], end_time_window)
                if box_end > box_start:
                    y_box = M_filtered - i - 1
                    rect = plt.Rectangle((box_start, y_box), box_end-box_start, 1,
                                         edgecolor='lime', facecolor='none', 
                                         linewidth=2)
                    ax.add_patch(rect)

    # New block for new_cues (yellow dotted boxes)
    if new_cues:
        new_text_map = {}
        for new_cue in new_cues:
            if new_cue['text']:
                clean_text = new_cue['text'].strip().replace("\n", " ")[:50]
                new_text_map[clean_text] = new_cue
        
        for i, cue_idx in enumerate(cue_indices_in_window[::-1]):
            cue = cues[cue_idx]
            clean_text = (cue['text'] or "").strip().replace("\n", " ")[:50]
            new_cue = new_text_map.get(clean_text)
            if new_cue and new_cue['start'] and new_cue['end']:
                box_start = max(new_cue['start'], start_time_window)
                box_end = min(new_cue['end'], end_time_window)
                if box_end > box_start:
                    y_box = M_filtered - i - 1
                    rect = plt.Rectangle((box_start, y_box), box_end-box_start, 1,
                                         edgecolor='lime', facecolor='none', 
                                         linewidth=2, linestyle='--')
                    ax.add_patch(rect)

    plt.xlabel("Time")
    plt.ylabel("Subtitle Cue")
    plt.title("Similarity Heatmap ({} - {})".format(format_time_full(start_time_window), format_time_full(end_time_window)))
    plt.tight_layout()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(script_dir, "heatmap.png"))
    plt.close()
    print("Saved heatmap to heatmap.png")
