import torch
import os
import numpy as np
import json
import transformer.transformer as tr
from utils.config import TrainingConfig
from utils.modelSelector import list_trained_models, select_model_interactive
import utils.dumpPreprocessor as dpp

# Label mapping (must match dumpPreprocessor.py)
LABEL_NAMES = {
    0: "BINARY_TEXT",
    1: "BINARY_IMAGE", 
    2: "BINARY_OTHER",
    3: "ENCRYPTED",
    4: "DECODED",
    5: "BASE64",
    6: "COMPRESSED",
    7: "SYSTEM",
    8: "NOISE"
}

def predict_dump_segment(segment_data: bytes, model, device, config, normalize=True):
    """
    Predict the type of a RAM dump segment.
    
    Args:
        segment_data: Raw bytes of the segment
        model: Loaded transformer model
        device: torch device
        config: TrainingConfig
        normalize: Whether to normalize byte values
    
    Returns:
        (predicted_class_idx, confidence_score, class_name)
    """
    # Convert bytes to numpy array
    data = np.frombuffer(segment_data, dtype=np.uint8).astype(np.float32)
    
    if normalize:
        data = data / 255.0
    
    # Pad or truncate to sequence_length
    if len(data) < config.sequence_length:
        padded = np.zeros(config.sequence_length, dtype=np.float32)
        padded[:len(data)] = data
        data = padded
    elif len(data) > config.sequence_length:
        data = data[:config.sequence_length]
    
    # Convert to tensor and add batch dimension
    tensor = torch.from_numpy(data).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        score, predicted = torch.max(probs, 1)
    
    pred_idx = predicted.item()
    confidence = score.item() * 100
    class_name = LABEL_NAMES.get(pred_idx, f"UNKNOWN_{pred_idx}")
    
    return pred_idx, confidence, class_name


def test_model_on_dump(dump_path: str, metadata_path: str, model_path: str, config: TrainingConfig):
    """
    Test the model on a RAM dump file.
    
    Args:
        dump_path: Path to the .bin dump file
        metadata_path: Path to the metadata .json file
        model_path: Path to the trained model .pth file
        config: Training configuration
    """
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() and not config.use_cpu else "cpu")
    print(f"Testing on device: {device}")
    
    # Load model
    model = tr.Transformer(
        sequenceLength=config.sequence_length,
        kernelSize=config.patch_size,
        inChannels=1,
        embedDim=config.embed_dim, 
        dropout=config.dropout, 
        depth=config.depth, 
        heads=config.heads,
        numClasses=config.num_classes
    )
    
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        return
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return
    
    model.eval()
    model.to(device)
    
    # Load dump data
    with open(dump_path, 'rb') as f:
        dump_data = f.read()
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # Filter out noise segments
    segments = [m for m in metadata if m['type'] != 'NOISE']
    
    print(f"\nTesting {len(segments)} segments from {dump_path}")
    print("=" * 60)
    
    # Statistics
    total = 0
    correct = 0
    per_class_stats = {name: {'total': 0, 'correct': 0} for name in LABEL_NAMES.values()}
    confidences_all = []
    confidences_correct = []
    
    for entry in segments:
        true_label = entry['type']
        if true_label not in dpp.LABEL_MAPPING:
            continue
            
        true_idx = dpp.LABEL_MAPPING[true_label]
        
        # Extract segment
        start = entry['data_start']
        end = entry['data_end']
        segment = dump_data[start:end]
        
        # Predict
        pred_idx, confidence, pred_name = predict_dump_segment(segment, model, device, config)
        
        total += 1
        per_class_stats[true_label]['total'] += 1
        confidences_all.append(confidence)
        
        if pred_idx == true_idx:
            correct += 1
            per_class_stats[true_label]['correct'] += 1
            confidences_correct.append(confidence)
            if config.debug:
                print(f"✓ Correct: {true_label} (conf: {confidence:.2f}%)")
        else:
            print(f"✗ Wrong: True={true_label}, Pred={pred_name} (conf: {confidence:.2f}%)")
    
    # Print summary
    print("\n" + "=" * 60)
    print("           MODEL EVALUATION SUMMARY")
    print("=" * 60)
    
    if total > 0:
        accuracy = (correct / total) * 100
        avg_conf_global = sum(confidences_all) / len(confidences_all)
        avg_conf_correct = sum(confidences_correct) / len(confidences_correct) if confidences_correct else 0
        
        print(f"\nOverall Accuracy: {accuracy:.2f}% ({correct}/{total})")
        print(f"Average Confidence (all): {avg_conf_global:.2f}%")
        print(f"Average Confidence (correct): {avg_conf_correct:.2f}%")
        
        print("\nPer-class Accuracy:")
        print("-" * 40)
        for class_name, stats in per_class_stats.items():
            if stats['total'] > 0:
                class_acc = (stats['correct'] / stats['total']) * 100
                print(f"  {class_name:15}: {class_acc:.1f}% ({stats['correct']}/{stats['total']})")
    else:
        print("No segments found for evaluation.")
    
    print("=" * 60)


if __name__ == "__main__":
    
    # Model selection
    print("Searching for trained models...\n")
    models_found = list_trained_models("output")
    
    if not models_found:
        print("No trained models found. Please train a model first.")
        exit(1)
    
    # Interactive selection
    model_path, config_path = select_model_interactive(models_found)
    
    if model_path is None:
        print("No model selected. Exiting.")
        exit(0)
    
    # Load model configuration
    if config_path and os.path.exists(config_path):
        config = TrainingConfig(config_path)
        print(f"Configuration loaded: {config}\n")
    else:
        print("No config found, using default config.\n")
        config = TrainingConfig('config.cfg')
    
    # Default dump location
    dump_dir = config.data_dir if hasattr(config, 'data_dir') else "output"
    dump_path = os.path.join(dump_dir, "ram_dump.bin")
    metadata_path = os.path.join(dump_dir, "metadata.json")
    
    if not os.path.exists(dump_path):
        print(f"Error: Dump file not found at {dump_path}")
        print("Please generate a dump first using dataSetGenerator.py")
        exit(1)
    
    if not os.path.exists(metadata_path):
        print(f"Error: Metadata file not found at {metadata_path}")
        exit(1)
    
    # Run evaluation
    test_model_on_dump(dump_path, metadata_path, model_path, config)