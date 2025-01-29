import time
import numpy as np

def analyze_predictions(predictions: np.ndarray, probabilities: np.ndarray) -> None:
    total_preds = len(predictions)
    positive_preds = np.sum(predictions == 1)
    negative_preds = np.sum(predictions == 0)
    
    mean_confidence = np.mean([max(p) for p in probabilities])
    high_confidence_preds = np.sum([max(p) >= 0.9 for p in probabilities])
    very_high_confidence_preds = np.sum([max(p) >= 0.95 for p in probabilities])
    
    print("\n=== Prediction Analysis ===")
    print(f"\nTotal predictions: {total_preds}")
    print(f"Duration: {total_preds} seconds ({total_preds//60}min {total_preds%60}s)")
    
    print("\n--- Predictions Distribution ---")
    print(f"Positive predictions: {positive_preds} ({(positive_preds/total_preds)*100:.1f}%)")
    print(f"Negative predictions: {negative_preds} ({(negative_preds/total_preds)*100:.1f}%)")
    
    print("\n--- Confidence Analysis ---")
    print(f"Mean confidence rate: {mean_confidence:.3f}")
    print(f"High confidence predictions (≥0.90): {high_confidence_preds} ({(high_confidence_preds/total_preds)*100:.1f}%)")
    print(f"Very high confidence (≥0.95): {very_high_confidence_preds} ({(very_high_confidence_preds/total_preds)*100:.1f}%)")
    
    print("\n--- Confidence Distribution ---")
    confidence_ranges = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 0.95), (0.95, 1.0)]
    for low, high in confidence_ranges:
        count = np.sum([(low <= max(p) < high) for p in probabilities])
        print(f"{low:.1f} - {high:.1f}: {count:3d} ({(count/total_preds)*100:5.1f}%)")

def measure_inference_performance(model, test_sample, block_size=1):
    total_samples = len(test_sample)
    predictions_list = []
    probabilities_list = []
    inference_times = []

    print("\n=== Inference Performance ===")
    print(f"Total samples to process: {total_samples}")
    
    for i in range(0, total_samples, block_size):
        block = test_sample[i:i + block_size]
        
        start_time = time.perf_counter()
        pred, prob = model.predict(block)
        end_time = time.perf_counter()
        
        inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
        inference_times.append(inference_time)
        
        predictions_list.extend(pred)
        probabilities_list.extend(prob)

    avg_inference = np.mean(inference_times)
    max_inference = np.max(inference_times)
    min_inference = np.min(inference_times)
    
    print("\n--- Timing Statistics ---")
    print(f"Average inference time per block: {avg_inference/1000:.3f}s")
    print(f"Maximum inference time: {max_inference/1000:.5f}s")
    print(f"Minimum inference time: {min_inference/1000:.5f}s")
    print(f"Predictions per second: {1000/avg_inference:.1f}")

    
    return np.array(predictions_list), np.array(probabilities_list)

