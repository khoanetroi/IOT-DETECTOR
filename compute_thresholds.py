import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
from scipy.spatial.distance import cosine

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.dataset import (
    load_flow_data,
    clean_and_normalise,
    create_windows,
    FlowClassificationDataset,
)
from modules.models import FlowClassifier, FlowTransformerEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print("Loading data...")
    df_dict = load_flow_data("flows")
    df_all, feature_cols, scaler = clean_and_normalise(df_dict)
    
    # Do not scale again, we just want raw windows to pass through model (model uses scaler from checkpoint in inference, 
    # but here we can just use the scaled data from clean_and_normalise)
    # Actually wait, clean_and_normalise DOES scale. Let's just do it exactly like finetune.
    
    windows, labels = create_windows(df_all, feature_cols, window_size=10)
    
    device_names = df_all["device_id"].unique().tolist()
    idx_to_device = {i: name for i, name in enumerate(device_names)}
    
    dataset = FlowClassificationDataset(windows, labels)
    loader = DataLoader(dataset, batch_size=512, shuffle=False)

    print("Loading model...")
    best_path = "experiments/finetune_best.pth"
    ckpt = torch.load(best_path, map_location=device, weights_only=False)

    encoder = FlowTransformerEncoder(
        input_dim=len(feature_cols),
        d_model=128,
        nhead=4,
        num_layers=3,
        dropout=0.1
    )
    model = FlowClassifier(encoder=encoder, num_classes=len(device_names)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    from collections import defaultdict
    class_embeddings = defaultdict(list)

    print("Extracting embeddings...")
    with torch.no_grad():
        for x, y_batch in loader:
            x = x.to(device)
            emb = model.encoder.encode(x)
            for i in range(len(y_batch)):
                label_name = idx_to_device[int(y_batch[i])]
                class_embeddings[label_name].append(emb[i].cpu().numpy())

    print("Computing thresholds...")
    centroids = {}
    class_thresholds = {}
    
    # Calculate centroids
    for name, embs in class_embeddings.items():
        centroids[name] = np.mean(embs, axis=0).tolist()
    
    # Calculate thresholds
    for name, embs in class_embeddings.items():
        centroid = np.array(centroids[name])
        distances = [float(cosine(emb, centroid)) for emb in embs]
        # Use 90th percentile to get the core cluster, ignoring the long tail
        p90_dist = np.percentile(distances, 90)
        
        # Add a 15% margin to the 90th percentile, clamp between 0.15 and 0.65
        dynamic_thresh = p90_dist * 1.15
        dynamic_thresh = max(0.15, min(dynamic_thresh, 0.65))
        
        class_thresholds[name] = float(dynamic_thresh)
        print(f"  {name}: threshold = {class_thresholds[name]:.4f}")

    ckpt["centroids"] = centroids
    ckpt["class_thresholds"] = class_thresholds
    
    torch.save(ckpt, best_path)
    print(f"\nSaved centroids and adaptive thresholds to {best_path}!")

if __name__ == "__main__":
    main()
