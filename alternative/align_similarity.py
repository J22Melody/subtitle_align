import numpy as np

def compute_similarity_matrix(cues, sign_segments, similarity_measure, subtitle_embedding=None, subtitle_embedding_tokenized=None, segmentation_embedding=None, tokenize_text_embedding=False, text_embedding_pooling='max'):
    """
    Compute a similarity matrix between cues and sign segments along with a cumulative sum.
    
    [Documentation omitted for brevity]
    """
    M = len(cues)
    N = len(sign_segments)
    sim_matrix = None

    if similarity_measure in ["cslr_subtitle", "cslr_text"]:
        sim_matrix = np.zeros((M, N))
        for i in tqdm(range(M), desc="Precomputing similarity matrix (cslr_subtitle)"):
            cue_text = cues[i]['text']
            for j in range(N):
                if similarity_measure == "cslr_subtitle":
                    seg_sub = sign_segments[j].get('subtitle', '')
                    if seg_sub:
                        sim_matrix[i, j] = 1 if seg_sub == cue_text else -1
                    else:
                        sim_matrix[i, j] = 0
                elif similarity_measure == "cslr_text":
                    seg_text = sign_segments[j].get('text', '')
                    if seg_text:
                        sim_matrix[i, j] = -1
                        seg_texts = seg_text.split('/')
                        for seg_text in seg_texts:
                            if seg_text.lower() in cue_text.lower():
                                probs = sign_segments[j].get('probs', 1)
                                sim_matrix[i, j] = probs
                    else:
                        sim_matrix[i, j] = 0

    elif similarity_measure == "text_embedding":
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Encode sign texts once.
        sign_texts = [(seg.get('text') or "").strip() for seg in sign_segments]
        sign_embeddings = model.encode(sign_texts, show_progress_bar=True)
        
        if tokenize_text_embedding:
            # Initialize an empty similarity matrix.
            sim_matrix = np.zeros((M, N))
            for i, cue in enumerate(cues):
                cue_text = (cue.get('text') or "").strip()
                # Tokenize the cue text into words (customize tokenization as needed)
                tokens = cue_text.split()
                if tokens:
                    # Compute embeddings for each token.
                    token_embeddings = model.encode(tokens, show_progress_bar=False)
                    for j, sign_embedding in enumerate(sign_embeddings):
                        # If the sign text is empty, assign 0 similarity.
                        if not sign_texts[j]:
                            sim_matrix[i, j] = 0
                        else:
                            # Compute the similarity (dot product) between each token and the sign embedding.
                            token_similarities = np.dot(token_embeddings, sign_embedding)
                            # Pool the token similarities based on the text_embedding_pooling method.
                            if text_embedding_pooling == "mean":
                                sim_matrix[i, j] = np.mean(token_similarities)
                            elif text_embedding_pooling == "max":
                                sim_matrix[i, j] = np.max(token_similarities)
                            else:
                                raise ValueError("Invalid text_embedding_pooling value. Use 'mean' or 'max'.")
                else:
                    # If there are no tokens, set similarity to zero.
                    sim_matrix[i, :] = 0
        else:
            # Original behavior: compute embeddings for the full cue texts.
            cue_texts = [cue.get('text') or "" for cue in cues]
            cue_embeddings = model.encode(cue_texts, show_progress_bar=True)
            sim_matrix = np.dot(cue_embeddings, sign_embeddings.T)
            # Zero out columns corresponding to empty sign texts.
            for j, seg in enumerate(sign_segments):
                if not (seg.get('text') or "").strip():
                    sim_matrix[:, j] = 0

    elif similarity_measure == "sign_clip_embedding":
        if tokenize_text_embedding:
            # Ensure we have one tokenized embedding per cue.
            if subtitle_embedding_tokenized is None or len(subtitle_embedding_tokenized) != M:
                raise ValueError(f"Subtitle embedding tokenized mismatch: expected {M} elements, got {len(subtitle_embedding_tokenized) if subtitle_embedding_tokenized is not None else 'None'}")
            # Initialize an empty similarity matrix.
            sim_matrix = np.zeros((M, N))
            for i in range(M):
                token_embeddings = subtitle_embedding_tokenized[i]  # shape: (num_tokens, embedding_dim)
                if token_embeddings.size == 0 or token_embeddings.shape[0] == 0:
                    sim_matrix[i, :] = 0
                else:
                    for j in range(N):
                        sign_embedding = segmentation_embedding[j]  # shape: (embedding_dim,)
                        token_similarities = np.dot(token_embeddings, sign_embedding)
                        if text_embedding_pooling == "mean":
                            sim_matrix[i, j] = np.mean(token_similarities)
                        elif text_embedding_pooling == "max":
                            sim_matrix[i, j] = np.max(token_similarities)
                        else:
                            raise ValueError("Invalid text_embedding_pooling value. Use 'mean' or 'max'.")
        else:
            if subtitle_embedding.shape[0] != M:
                raise ValueError(f"Subtitle embedding mismatch: expected {M} rows, got {subtitle_embedding.shape[0]}")
            if segmentation_embedding.shape[0] != N:
                raise ValueError(f"Segmentation embedding mismatch: expected {N} rows, got {segmentation_embedding.shape[0]}")
            sim_matrix = np.dot(subtitle_embedding, segmentation_embedding.T)
    
    else:
        raise ValueError(f"Unsupported similarity_measure: {similarity_measure}")

    return sim_matrix