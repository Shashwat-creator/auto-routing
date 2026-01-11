"""
test_ood.py - Strict OOD detection that forces truly novel tickets to DEFAULT
"""

import numpy as np
import pandas as pd
import faiss
import json
import joblib
import argparse
import sys
from sentence_transformers import SentenceTransformer
import re

class StrictOODRouter:
    def __init__(self, model_dir="accurate_models"):
        """Initialize with strict OOD detection"""
        self.model_dir = model_dir
        self.load_models()
        
        # Ultra-strict parameters
        self.NEIGHBOR_THRESHOLD = 0.65  # Max similarity to any neighbor
        self.MIN_NEIGHBORS_ABOVE_THRESH = 5  # Need 5+ neighbors above threshold
        self.TOP_K = 50  # Check more neighbors
        
        print("=" * 70)
        print("STRICT OOD TICKET ROUTER")
        print("=" * 70)
        print(f"OOD Detection: Tickets must have {self.MIN_NEIGHBORS_ABOVE_THRESH} neighbors above {self.NEIGHBOR_THRESHOLD:.3f}")
        
    def load_models(self):
        """Load models"""
        try:
            with open(f'{self.model_dir}/metadata.json', 'r') as f:
                self.meta = json.load(f)
            
            self.embeddings = np.load(f'{self.model_dir}/embeddings.npy')
            self.labels = np.load(f'{self.model_dir}/labels.npy', allow_pickle=True)
            self.centroids = np.load(f'{self.model_dir}/centroids.npy')
            self.centroid_labels = np.load(f'{self.model_dir}/centroid_labels.npy', allow_pickle=True)
            self.faiss_index = faiss.read_index(f'{self.model_dir}/faiss.index')
            
            print(f"✓ Model loaded: {len(self.embeddings)} training tickets")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            sys.exit(1)
    
    def clean_text(self, text):
        """Clean text"""
        if not isinstance(text, str):
            text = str(text)
        return text.lower().strip()
    
    def is_novel_ticket(self, query_embedding):
        """
        Check if ticket is truly novel (not in distribution)
        Returns: (is_ood, max_similarity, neighbor_count_above_thresh)
        """
        # Search for nearest neighbors
        query_array = np.expand_dims(query_embedding, axis=0)
        distances, indices = self.faiss_index.search(query_array, self.TOP_K)
        similarities = distances[0]
        
        # Count neighbors above threshold
        neighbors_above_thresh = np.sum(similarities >= self.NEIGHBOR_THRESHOLD)
        max_similarity = np.max(similarities) if len(similarities) > 0 else 0
        
        # Check if we have enough similar neighbors
        is_ood = neighbors_above_thresh < self.MIN_NEIGHBORS_ABOVE_THRESH
        
        return is_ood, max_similarity, neighbors_above_thresh
    
    def route_ticket(self, subject):
        """
        Strict routing: If ticket is OOD (novel), force to DEFAULT
        """
        clean_subject = self.clean_text(subject)
        
        # Encode
        model = SentenceTransformer(self.meta['model_name'])
        query_embedding = model.encode([clean_subject], 
                                      convert_to_numpy=True,
                                      normalize_embeddings=True)[0].astype('float32')
        
        # Step 1: OOD Detection - Is this ticket similar to training data?
        is_ood, max_sim, neighbor_count = self.is_novel_ticket(query_embedding)
        
        if is_ood:
            return {
                "assigned_group": "DEFAULT",
                "confidence": round(max_sim * 0.2, 3),
                "method": "ood_detected",
                "max_similarity": round(max_sim, 3),
                "neighbors_above_thresh": int(neighbor_count),
                "required_neighbors": self.MIN_NEIGHBORS_ABOVE_THRESH,
                "similarity_threshold": self.NEIGHBOR_THRESHOLD,
                "reason": f"OOD detected: Only {neighbor_count} neighbors above {self.NEIGHBOR_THRESHOLD:.3f} (need {self.MIN_NEIGHBORS_ABOVE_THRESH})"
            }
        
        # Step 2: Get centroid similarities (only if NOT OOD)
        centroid_sims = self.centroids @ query_embedding
        
        # Group by team
        team_centroid_sims = {}
        for i, label in enumerate(self.centroid_labels):
            label_str = str(label)
            if label_str not in team_centroid_sims:
                team_centroid_sims[label_str] = []
            team_centroid_sims[label_str].append(centroid_sims[i])
        
        # Get max centroid similarity per team
        team_max_sim = {label: max(sims) for label, sims in team_centroid_sims.items()}
        
        # Get best team
        sorted_teams = sorted(team_max_sim.items(), key=lambda x: x[1], reverse=True)
        best_team, best_sim = sorted_teams[0]
        second_team, second_sim = sorted_teams[1] if len(sorted_teams) > 1 else ("none", 0)
        
        # Step 3: Apply strict thresholds
        HIGH_THRESH = 0.75
        LOW_THRESH = 0.65
        
        if best_sim < LOW_THRESH:
            return {
                "assigned_group": "DEFAULT",
                "confidence": round(best_sim * 0.3, 3),
                "method": "low_centroid_similarity",
                "best_team": best_team,
                "best_score": round(best_sim, 3),
                "threshold": LOW_THRESH,
                "reason": f"Centroid similarity {best_sim:.3f} below threshold {LOW_THRESH:.3f}"
            }
        
        # Step 4: Check margin
        margin = best_sim - second_sim
        if margin < 0.1:
            return {
                "assigned_group": "DEFAULT",
                "confidence": round(best_sim * 0.4, 3),
                "method": "ambiguous_margin",
                "best_team": best_team,
                "best_score": round(best_sim, 3),
                "second_team": second_team,
                "margin": round(margin, 3),
                "reason": f"Low margin ({margin:.3f}) between {best_team} and {second_team}"
            }
        
        # Step 5: Final confidence calculation
        if best_sim >= HIGH_THRESH:
            confidence = 0.8 + 0.2 * (best_sim - HIGH_THRESH) / (1.0 - HIGH_THRESH)
        else:
            confidence = 0.5 + 0.3 * (best_sim - LOW_THRESH) / (HIGH_THRESH - LOW_THRESH)
        
        return {
            "assigned_group": best_team,
            "confidence": round(confidence, 3),
            "method": "strict_assignment",
            "score": round(best_sim, 3),
            "margin": round(margin, 3),
            "neighbors_above_thresh": int(neighbor_count),
            "second_best": second_team,
            "reason": f"Strict assignment: similarity={best_sim:.3f}, neighbors={neighbor_count}, margin={margin:.3f}"
        }
    
    def analyze_ticket(self, subject):
        """Detailed analysis of why a ticket gets assigned"""
        clean_subject = self.clean_text(subject)
        model = SentenceTransformer(self.meta['model_name'])
        query_embedding = model.encode([clean_subject], 
                                      convert_to_numpy=True,
                                      normalize_embeddings=True)[0].astype('float32')
        
        # Get nearest neighbors
        query_array = np.expand_dims(query_embedding, axis=0)
        distances, indices = self.faiss_index.search(query_array, 10)
        
        print(f"\n🔍 Analysis for: '{subject}'")
        print("-" * 50)
        
        # Show top neighbors
        print(f"Top 10 most similar tickets in training data:")
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if i < 10:
                similar_ticket = self.labels[idx]
                print(f"  {i+1:2}. Similarity: {dist:.3f} → Team: {similar_ticket}")
        
        # OOD check
        is_ood, max_sim, neighbor_count = self.is_novel_ticket(query_embedding)
        print(f"\nOOD Analysis:")
        print(f"  Max similarity to any training ticket: {max_sim:.3f}")
        print(f"  Neighbors above {self.NEIGHBOR_THRESHOLD:.3f}: {neighbor_count}/{self.MIN_NEIGHBORS_ABOVE_THRESH}")
        print(f"  OOD verdict: {'YES' if is_ood else 'NO'}")
        
        if is_ood:
            print(f"  → Will be assigned to DEFAULT (novel ticket)")
        else:
            print(f"  → Will be considered for team assignment")
        
        # Get routing result
        result = self.route_ticket(subject)
        
        print(f"\nRouting Decision:")
        print(f"  Assigned to: {result['assigned_group']}")
        print(f"  Method: {result['method']}")
        print(f"  Reason: {result['reason']}")
        
        return result


def main():
    parser = argparse.ArgumentParser(description='Strict OOD ticket router')
    
    parser.add_argument('--subject', type=str, help='Single ticket subject')
    parser.add_argument('--analyze', type=str, help='Analyze a ticket in detail')
    parser.add_argument('--file', type=str, help='CSV file with tickets')
    parser.add_argument('--output', type=str, default='ood_results.csv', help='Output file')
    parser.add_argument('--model-dir', type=str, default='accurate_models', help='Model directory')
    
    args = parser.parse_args()
    
    router = StrictOODRouter(model_dir=args.model_dir)
    
    if args.analyze:
        router.analyze_ticket(args.analyze)
    elif args.subject:
        result = router.route_ticket(args.subject)
        
        print("\n" + "=" * 70)
        print("STRICT OOD ROUTING RESULT")
        print("=" * 70)
        print(f"Subject: {args.subject}")
        print(f"Assigned to: {result['assigned_group']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Method: {result['method']}")
        print(f"Reason: {result['reason']}")
        
        if 'max_similarity' in result:
            print(f"Max similarity to training: {result['max_similarity']:.3f}")
        if 'neighbors_above_thresh' in result:
            print(f"Neighbors above threshold: {result['neighbors_above_thresh']}")
        
        if result['assigned_group'] == 'DEFAULT':
            print(f"\n✅ Correct: Novel ticket assigned to DEFAULT")
        else:
            print(f"\n⚠️  Assigned to {result['assigned_group']} (not novel)")
        
    elif args.file:
        # Batch processing
        print(f"Processing {args.file}...")
        
        df = pd.read_csv(args.file)
        if 'subject' not in df.columns:
            print("✗ Error: CSV must have 'subject' column")
            return
        
        results = []
        for idx, row in df.iterrows():
            result = router.route_ticket(row['subject'])
            result.update(row.to_dict())
            results.append(result)
            
            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx + 1}/{len(df)}")
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(args.output, index=False)
        
        # Statistics
        print(f"\n✅ Saved {len(results)} results to {args.output}")
        
        # Analyze DEFAULT assignments
        default_tickets = results_df[results_df['assigned_group'] == 'DEFAULT']
        print(f"\nDEFAULT assignments: {len(default_tickets)}/{len(results)} ({len(default_tickets)/len(results)*100:.1f}%)")
        
        if len(default_tickets) > 0:
            print(f"\nSample DEFAULT tickets:")
            for _, ticket in default_tickets.head(5).iterrows():
                method = ticket.get('method', 'unknown')
                reason = ticket.get('reason', '')[:80]
                print(f"  - '{ticket['subject'][:50]}...' ({method})")
        
    else:
        # Test specific cases
        test_cases = [
            "Backup not configured for Azure",
            "Daily Storage Server Backup On One-Drive",
            "i am mohan",
            "Server hardware failure",
            "Password reset request",
            "Sibel report error",
            "Azure VM backup configuration",
            "OneDrive storage quota exceeded",
            "Cloud backup retention policy",
            "Hello team, need assistance"
        ]
        
        print("Testing various ticket types:")
        print("-" * 70)
        
        for subject in test_cases:
            result = router.route_ticket(subject)
            print(f"\n{subject[:50]:<50}")
            print(f"  → {result['assigned_group']:8} ({result['method']})")
            print(f"     Reason: {result['reason']}")


if __name__ == "__main__":
    main()