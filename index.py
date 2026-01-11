"""
index_accurate.py - Builds high-accuracy multi-centroid index with adaptive thresholds
Balances accuracy with safety - routes more tickets correctly while avoiding wrong assignments
"""

import os
import re
import json
import logging
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import faiss
import joblib

# ----------------- CONFIG -----------------
MODEL_NAME = "BAAI/bge-small-en-v1.5"  # More accurate than MiniLM
OUT_DIR = "accurate_models"
CSV_PATH = "subject.csv"  # Columns: subject, assignment_group
RANDOM_STATE = 42
TEST_SIZE = 0.15  # More data for training
EMBEDDING_DIM = 384  # For bge-small-en-v1.5

# Multi-centroid configuration
MULTICENTROID = True
K_RULES = [
    (2000, 8),  # Very large teams: 8 centroids
    (800, 5),   # Large teams: 5 centroids
    (300, 3),   # Medium teams: 3 centroids
    (100, 2),   # Small teams: 2 centroids
    (30, 1)     # Very small teams: 1 centroid
]

# Threshold learning
FPR_TARGET = 0.01  # Allow 1% false positive rate (wrong assignments)
RECALL_TARGET = 0.85  # Aim for 85% recall (tickets correctly assigned)

# Reranker settings
TRAIN_RERANKER = True
RERANKER_C = 0.5  # Regularization - lower for simpler model

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("index_build.log"),
        logging.StreamHandler()
    ]
)
os.makedirs(OUT_DIR, exist_ok=True)


# ----------------- UTILITIES -----------------
def clean_text(s: str) -> str:
    """Enhanced text cleaning"""
    if not isinstance(s, str):
        s = str(s)
    
    # Remove common email artifacts
    s = re.sub(r"\nOn .*wrote:\n", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\S+@\S+\.\S+", " ", s)
    s = re.sub(r"https?://\S+", " ", s)
    
    # Remove special characters but keep important ones
    s = re.sub(r"[^\w\s\-\.\#\/]", " ", s)
    
    # Normalize whitespace and case
    s = re.sub(r"\s+", " ", s).strip()
    
    # Intelligent lowercasing (keep acronyms)
    words = s.split()
    cleaned_words = []
    for word in words:
        if len(word) <= 3 or word.isupper():
            cleaned_words.append(word)
        else:
            cleaned_words.append(word.lower())
    
    return " ".join(cleaned_words)


def choose_k(n_examples: int) -> int:
    """Choose number of centroids based on sample size"""
    if not MULTICENTROID:
        return 1
    
    for thresh, k in K_RULES:
        if n_examples >= thresh:
            return k
    return 1


def build_label_centroids(embs: np.ndarray, labels: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Build centroids for a label, ensuring quality"""
    if len(embs) < k:
        k = max(1, len(embs) // 2)
    
    if k == 1 or len(embs) <= 3:
        # Single centroid with weighted average
        centroid = np.mean(embs, axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
        return np.array([centroid]), np.array([len(embs)])  # centroid, cluster sizes
    
    # Use K-means with multiple initializations
    best_inertia = float('inf')
    best_centroids = None
    best_cluster_sizes = None
    
    for attempt in range(3):  # Multiple attempts
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE + attempt, n_init=5)
        kmeans.fit(embs)
        
        if kmeans.inertia_ < best_inertia:
            best_inertia = kmeans.inertia_
            best_centroids = kmeans.cluster_centers_
            best_cluster_sizes = np.bincount(kmeans.labels_)
    
    # Normalize centroids
    normalized_centroids = []
    for c in best_centroids:
        c_norm = c / (np.linalg.norm(c) + 1e-12)
        normalized_centroids.append(c_norm)
    
    return np.array(normalized_centroids), best_cluster_sizes


def find_optimal_thresholds(true_sims: np.ndarray, false_sims: np.ndarray, 
                           fpr_target: float = FPR_TARGET) -> Dict:
    """Find thresholds that maximize recall while controlling FPR"""
    
    if len(true_sims) == 0 or len(false_sims) == 0:
        return {
            "high": 0.7,
            "low": 0.4,
            "ood": 0.3,
            "recall": 0.7,
            "fpr": 0.05
        }
    
    # Sort similarities for threshold search
    true_sims_sorted = np.sort(true_sims)
    false_sims_sorted = np.sort(false_sims)
    
    thresholds = np.linspace(0.1, 0.9, 81)  # Fine-grained search
    best_score = -1
    best_thresholds = {
        "high": 0.7,
        "low": 0.4,
        "ood": 0.3,
        "recall": 0.7,
        "fpr": 0.05
    }
    
    for thresh in thresholds:
        # Calculate metrics at this threshold
        recall = np.mean(true_sims >= thresh)
        fpr = np.mean(false_sims >= thresh) if len(false_sims) > 0 else 0
        
        # Score function: prioritize recall while controlling FPR
        if fpr <= fpr_target:
            score = recall * 100 - fpr * 10
            if score > best_score:
                best_score = score
                
                # Set thresholds based on percentile
                if np.any(true_sims >= thresh):
                    high_thresh = np.percentile(true_sims[true_sims >= thresh], 20)
                else:
                    high_thresh = thresh + 0.1
                
                low_thresh = thresh
                ood_thresh = np.percentile(true_sims, 10) if len(true_sims) > 0 else 0.3
                
                best_thresholds = {
                    "high": float(max(0.6, min(0.9, high_thresh))),
                    "low": float(max(0.3, min(0.7, low_thresh))),
                    "ood": float(max(0.2, min(0.5, ood_thresh))),
                    "recall": float(recall),
                    "fpr": float(fpr)
                }
    
    return best_thresholds


# ----------------- MAIN INDEX BUILDING -----------------
def build_accurate_index(csv_path: str = CSV_PATH) -> Dict:
    """Build high-accuracy index with optimized thresholds"""
    
    # Use local variable for reranker flag to avoid scope issues
    train_reranker_flag = TRAIN_RERANKER
    
    logging.info("=" * 70)
    logging.info("BUILDING HIGH-ACCURACY TICKET ROUTING INDEX")
    logging.info("=" * 70)
    
    # 1. Load and prepare data
    logging.info("Loading CSV: %s", csv_path)
    df = pd.read_csv(csv_path, dtype=str)
    df=df.dropna()
    required_cols = ["subject", "assignment_group"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV must contain columns: {required_cols}")
    
    # Clean text
    df["clean_subject"] = df["subject"].fillna("").apply(clean_text)
    df = df[df["clean_subject"].str.strip().astype(bool)].reset_index(drop=True)
    
    # Check label distribution
    label_counts = df["assignment_group"].value_counts()
    logging.info("Label distribution:")
    for label, count in label_counts.items():
        logging.info("  %s: %d tickets", label, count)
    
    # 2. Create embeddings
    logging.info("Loading model: %s", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)
    
    texts = df["clean_subject"].tolist()
    logging.info("Encoding %d texts...", len(texts))
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    ).astype("float32")
    
    # Save embeddings and labels
    np.save(os.path.join(OUT_DIR, "embeddings.npy"), embeddings)
    np.save(os.path.join(OUT_DIR, "labels.npy"), df["assignment_group"].values)
    
    # 3. Train/Validation split with stratification
    logging.info("Creating train/validation split...")
    train_idx, val_idx = train_test_split(
        np.arange(len(df)),
        test_size=TEST_SIZE,
        stratify=df["assignment_group"],
        random_state=RANDOM_STATE
    )
    
    train_embs = embeddings[train_idx]
    train_labels = df["assignment_group"].iloc[train_idx].values
    val_embs = embeddings[val_idx]
    val_labels = df["assignment_group"].iloc[val_idx].values
    
    # 4. Build multi-centroid representations
    logging.info("Building multi-centroid representations...")
    unique_labels = np.unique(train_labels)
    
    all_centroids = []
    centroid_labels = []
    label_centroid_map = defaultdict(list)
    centroid_weights = []  # Weight by cluster size
    
    for label in unique_labels:
        mask = train_labels == label
        label_embs = train_embs[mask]
        
        if len(label_embs) == 0:
            logging.warning("Label %s has no training examples", label)
            continue
        
        # Choose k based on sample size
        k = choose_k(len(label_embs))
        logging.info("  %s: %d examples -> %d centroids", label, len(label_embs), k)
        
        # Build centroids
        centroids, cluster_sizes = build_label_centroids(label_embs, train_labels[mask], k)
        
        for i, centroid in enumerate(centroids):
            all_centroids.append(centroid)
            centroid_labels.append(label)
            label_centroid_map[label].append(len(all_centroids) - 1)
            if i < len(cluster_sizes):
                centroid_weights.append(cluster_sizes[i])
            else:
                centroid_weights.append(1)
    
    centroids_array = np.vstack(all_centroids).astype("float32")
    centroid_labels_array = np.array(centroid_labels)
    if len(centroid_weights) > 0:
        centroid_weights_array = np.array(centroid_weights) / np.sum(centroid_weights)
    else:
        centroid_weights_array = np.ones(len(centroids_array)) / len(centroids_array)
    
    # Save centroids
    np.save(os.path.join(OUT_DIR, "centroids.npy"), centroids_array)
    np.save(os.path.join(OUT_DIR, "centroid_labels.npy"), centroid_labels_array)
    np.save(os.path.join(OUT_DIR, "centroid_weights.npy"), centroid_weights_array)
    
    logging.info("Built %d centroids for %d labels", len(centroids_array), len(unique_labels))
    
    # 5. Build FAISS index
    logging.info("Building FAISS index...")
    d = embeddings.shape[1]
    
    # Create index with inner product (cosine similarity)
    index = faiss.IndexFlatIP(d)
    index.add(embeddings.astype("float32"))
    faiss.write_index(index, os.path.join(OUT_DIR, "faiss.index"))
    
    logging.info("FAISS index built with %d vectors", index.ntotal)
    
    # 6. Optimize thresholds on validation set
    logging.info("Optimizing thresholds on validation set...")
    
    # Collect similarity scores
    per_label_true_sims = defaultdict(list)
    per_label_false_sims = defaultdict(list)
    
    for emb, true_label in zip(val_embs, val_labels):
        # Similarity to all centroids
        sims = centroids_array @ emb
        
        # True label similarities
        true_centroid_idxs = label_centroid_map.get(true_label, [])
        if true_centroid_idxs:
            true_max_sim = float(np.max(sims[true_centroid_idxs]))
            per_label_true_sims[true_label].append(true_max_sim)
        
        # False label similarities
        for label in unique_labels:
            if label == true_label:
                continue
            false_centroid_idxs = label_centroid_map.get(label, [])
            if false_centroid_idxs:
                false_max_sim = float(np.max(sims[false_centroid_idxs]))
                per_label_false_sims[label].append(false_max_sim)
    
    # Find optimal global thresholds
    all_true_sims = np.concatenate([s for s in per_label_true_sims.values()]) if per_label_true_sims else np.array([])
    all_false_sims = np.concatenate([s for s in per_label_false_sims.values()]) if per_label_false_sims else np.array([])
    
    global_thresholds = find_optimal_thresholds(all_true_sims, all_false_sims)
    
    # Find per-label thresholds
    per_label_thresholds = {}
    for label in unique_labels:
        if label in per_label_true_sims and len(per_label_true_sims[label]) >= 5:
            true_sims = np.array(per_label_true_sims[label])
            # Threshold is 10th percentile of true similarities
            label_thresh = np.percentile(true_sims, 10)
            per_label_thresholds[label] = float(max(0.25, min(0.6, label_thresh)))
        else:
            per_label_thresholds[label] = global_thresholds["low"]
    
    # 7. Train reranker with calibrated probabilities
    reranker_info = {}
    if train_reranker_flag and len(unique_labels) > 1:
        logging.info("Training calibrated reranker...")
        
        try:
            # Prepare data
            le = LabelEncoder()
            y = le.fit_transform(df["assignment_group"].values)
            
            # Add centroid similarities as features
            centroid_sim_features = []
            for emb in embeddings:
                sims = centroids_array @ emb
                centroid_sim_features.append(sims)
            
            centroid_features = np.vstack(centroid_sim_features)
            X_enhanced = np.hstack([embeddings, centroid_features])
            
            # Train with calibration
            X_train, X_val, y_train, y_val = train_test_split(
                X_enhanced, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
            )
            
            # Calculate class weights
            label_to_idx = {label: idx for idx, label in enumerate(le.classes_)}
            class_weights = {}
            for label, idx in label_to_idx.items():
                if label in label_counts:
                    class_weights[idx] = 1.0 / label_counts[label]
                else:
                    class_weights[idx] = 1.0
            
            clf = LogisticRegression(
                max_iter=1000,
                multi_class="multinomial",
                solver="saga",
                C=RERANKER_C,
                class_weight=class_weights,
                random_state=RANDOM_STATE
            )
            clf.fit(X_train, y_train)
            
            # Evaluate
            val_pred = clf.predict(X_val)
            val_proba = clf.predict_proba(X_val)
            
            # Find optimal probability threshold
            best_thresh = 0.5
            best_f1 = 0
            for thresh in np.linspace(0.3, 0.9, 13):
                # Apply threshold
                thresholded_pred = np.argmax(val_proba, axis=1)
                mask = np.max(val_proba, axis=1) >= thresh
                thresholded_pred[~mask] = -1  # Reject
                
                # Calculate metrics (excluding rejected)
                valid_mask = thresholded_pred != -1
                if np.sum(valid_mask) > 0:
                    # Simple accuracy for valid predictions
                    accuracy = np.mean(thresholded_pred[valid_mask] == y_val[valid_mask])
                    coverage = np.mean(valid_mask)
                    f1 = 2 * accuracy * coverage / (accuracy + coverage + 1e-12)
                    
                    if f1 > best_f1:
                        best_f1 = f1
                        best_thresh = thresh
            
            reranker_info = {
                "model": "LogisticRegression",
                "val_accuracy": float(clf.score(X_val, y_val)),
                "optimal_threshold": float(best_thresh),
                "classes": le.classes_.tolist(),
                "coverage_at_threshold": float(np.mean(np.max(val_proba, axis=1) >= best_thresh))
            }
            
            # Save models
            joblib.dump(clf, os.path.join(OUT_DIR, "reranker.joblib"))
            joblib.dump(le, os.path.join(OUT_DIR, "label_encoder.joblib"))
            
            logging.info("Reranker trained (val_acc=%.3f, optimal_thresh=%.3f)", 
                        reranker_info["val_accuracy"], reranker_info["optimal_threshold"])
            
        except Exception as e:
            logging.error("Failed to train reranker: %s", e)
            # If reranker fails, set flag to False
            train_reranker_flag = False
    
    # 8. Save metadata
    metadata = {
        "model_name": MODEL_NAME,
        "embedding_dim": int(d),
        "n_examples": int(len(df)),
        "n_centroids": int(len(centroids_array)),
        "labels": unique_labels.tolist(),
        "label_counts": label_counts.to_dict(),
        
        # Thresholds
        "global_thresholds": global_thresholds,
        "per_label_thresholds": per_label_thresholds,
        
        # Model info
        "multicentroid": MULTICENTROID,
        "k_rules": K_RULES,
        "reranker": reranker_info if train_reranker_flag else {},
        
        # Performance targets
        "fpr_target": FPR_TARGET,
        "recall_target": RECALL_TARGET,
        
        # Timestamp and version
        "build_date": pd.Timestamp.now().isoformat(),
        "version": "2.0_high_accuracy_fixed"
    }
    
    with open(os.path.join(OUT_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    # 9. Final validation report
    logging.info("\n" + "=" * 70)
    logging.info("VALIDATION REPORT")
    logging.info("=" * 70)
    
    # Simulate routing on validation set
    correct = 0
    total = len(val_embs)
    
    for emb, true_label in zip(val_embs, val_labels):
        # Simple routing simulation
        sims = centroids_array @ emb
        predicted_label = centroid_labels_array[np.argmax(sims)]
        
        if predicted_label == true_label:
            correct += 1
    
    if total > 0:
        accuracy = correct / total
        logging.info("Baseline accuracy on validation set: %.2f%%", accuracy * 100)
    else:
        logging.info("No validation data available")
    
    logging.info("Global thresholds: high=%.3f, low=%.3f, ood=%.3f",
                global_thresholds["high"], global_thresholds["low"], global_thresholds["ood"])
    
    if "recall" in global_thresholds and "fpr" in global_thresholds:
        logging.info("Expected recall: %.1f%% at FPR: %.1f%%",
                    global_thresholds.get("recall", 0) * 100,
                    global_thresholds.get("fpr", 0) * 100)
    
    logging.info("\nIndex build complete!")
    logging.info("Output directory: %s", OUT_DIR)
    
    return metadata


def main():
    """Main function to build index"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build high-accuracy ticket routing index")
    parser.add_argument("--data", type=str, default=CSV_PATH, help="Path to CSV file")
    parser.add_argument("--model", type=str, default="BAAI/bge-small-en-v1.5", help="Model name")
    parser.add_argument("--no-reranker", action="store_true", help="Skip reranker training")
    
    args = parser.parse_args()
    
    # Update config
    if args.model:
      #  global MODEL_NAME
        MODEL_NAME = args.model
    
    if args.no_reranker:
        global TRAIN_RERANKER
        TRAIN_RERANKER = False
    
    # Build index
    try:
        metadata = build_accurate_index(args.data)
        
        print("\n" + "=" * 70)
        print("✅ HIGH-ACCURACY INDEX BUILT SUCCESSFULLY")
        print("=" * 70)
        print(f"Output directory: {OUT_DIR}/")
        print(f"Model: {metadata['model_name']}")
        print(f"Centroids: {metadata['n_centroids']} (multi-centroid: {metadata['multicentroid']})")
        print(f"Labels: {', '.join(metadata['labels'])}")
        if 'recall' in metadata['global_thresholds']:
            print(f"Expected performance: {metadata['global_thresholds']['recall']*100:.1f}% recall")
        
        print(f"\nFiles created:")
        print(f"  - embeddings.npy: All ticket embeddings")
        print(f"  - labels.npy: Corresponding labels")
        print(f"  - centroids.npy: {metadata['n_centroids']} centroid vectors")
        print(f"  - centroid_labels.npy: Team for each centroid")
        print(f"  - faiss.index: FAISS search index")
        if metadata['reranker']:
            print(f"  - reranker.joblib: Trained reranker model")
            print(f"    (Accuracy: {metadata['reranker']['val_accuracy']*100:.1f}%, Threshold: {metadata['reranker']['optimal_threshold']:.3f})")
        print(f"  - metadata.json: Configuration and thresholds")
        
        print("\nNext steps:")
        print(f"1. Test routing: python route_accurate.py --subject 'Your ticket subject here'")
        print(f"2. Batch process: python route_accurate.py --file new_tickets.csv --output results.csv")
        
    except Exception as e:
        print(f"\n❌ Error building index: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
