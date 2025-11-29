import os
import json
import argparse
from typing import List, Tuple

import pandas as pd
import torch
from torch import nn
import torchvision.transforms as T
from torchvision import models

from PIL import Image
import matplotlib.pyplot as plt


DEFAULT_WEIGHTS_PATH = "best_model.pth"
DEFAULT_CLASS_NAMES_JSON = "class_names.json"
DEFAULT_CSV_PATH = "labels_clean.csv"


def _get_eval_transform():
	imagenet_mean = [0.485, 0.456, 0.406]
	imagenet_std = [0.229, 0.224, 0.225]
	return T.Compose([
		T.Resize((224, 224)),
		T.ToTensor(),
		T.Normalize(mean=imagenet_mean, std=imagenet_std),
	])


def _build_model_for_inference(num_classes: int) -> nn.Module:
	model = models.densenet121(weights=None)
	in_features = model.classifier.in_features
	model.classifier = nn.Linear(in_features, num_classes)
	model.eval()
	return model


def load_model(weights_path: str, class_names_path: str) -> (nn.Module, List[str], torch.device):
	with open(class_names_path, "r", encoding="utf-8") as f:
		class_names = json.load(f)
	checkpoint = torch.load(weights_path, map_location="cpu")
	model = _build_model_for_inference(num_classes=len(class_names))
	model.load_state_dict(checkpoint["model_state_dict"])
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)
	model.eval()
	return model, class_names, device


@torch.no_grad()
def predict_diseases(image_path: str, weights_path: str = DEFAULT_WEIGHTS_PATH, class_names_path: str = DEFAULT_CLASS_NAMES_JSON, threshold: float = 0.5) -> List[str]:
	model, class_names, device = load_model(weights_path, class_names_path)
	transform = _get_eval_transform()
	image = Image.open(image_path).convert("RGB")
	image_t = transform(image).unsqueeze(0).to(device)
	logits = model(image_t)
	probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
	preds = [name for name, p in zip(class_names, probs) if p >= threshold]
	if not preds:
		# If nothing passes threshold, choose the highest probability as a fallback
		idx = int(probs.argmax())
		preds = [class_names[idx]]
	return preds


def get_ground_truth_labels(image_path: str, csv_path: str, class_names: List[str]) -> List[str]:
	"""Get ground truth labels from CSV"""
	df = pd.read_csv(csv_path)
	# Get filename from full path
	filename = os.path.basename(image_path)
	
	# Find matching row
	if "Image Index" in df.columns:
		matches = df[df["Image Index"] == filename]
	else:
		matches = df[df["filename"] == filename]
	
	if matches.empty:
		return None
	
	label_str = str(matches.iloc[0]["Finding Labels"]).strip()
	if not label_str or label_str == "No Finding":
		return ["No Finding"] if "No Finding" in class_names else []
	
	# Parse pipe-separated labels
	labels = []
	for lab in label_str.split("|"):
		lab = lab.strip()
		if lab in class_names:
			labels.append(lab)
	
	return labels if labels else ["No Finding"]


def visualize_prediction(image_path: str, predictions: List[str], ground_truth: List[str] = None, class_names: List[str] = None):
	img = Image.open(image_path).convert("RGB")
	
	# Create figure with more space for text
	fig = plt.figure(figsize=(10, 9))
	plt.imshow(img, cmap="gray")
	plt.axis("off")
	
	# Display predictions
	pred_text = ", ".join(predictions) if predictions else "No Finding"
	plt.figtext(0.5, 0.08, f"Predicted Diseases: {pred_text}", ha="center", fontsize=14, weight='bold', color='blue')
	
	# Display ground truth if available
	if ground_truth:
		gt_text = ", ".join(ground_truth) if ground_truth else "No Finding"
		plt.figtext(0.5, 0.03, f"Actual Labels: {gt_text}", ha="center", fontsize=14, weight='bold', color='green')
		
		# Check if prediction is correct
		pred_set = set(predictions)
		gt_set = set(ground_truth)
		
		if pred_set == gt_set:
			status = "✓ CORRECT"
			color = 'green'
		else:
			# Find matches and misses
			correct = pred_set & gt_set
			missed = gt_set - pred_set
			false_pos = pred_set - gt_set
			
			status_parts = []
			if correct:
				status_parts.append(f"Correct: {', '.join(correct)}")
			if missed:
				status_parts.append(f"Missed: {', '.join(missed)}")
			if false_pos:
				status_parts.append(f"False: {', '.join(false_pos)}")
			
			status = "✗ " + " | ".join(status_parts)
			color = 'red'
		
		plt.figtext(0.5, 0.01, status, ha="center", fontsize=12, weight='bold', color=color)
	
	plt.tight_layout(rect=[0, 0.10, 1, 1])
	plt.show()


def parse_args():
	parser = argparse.ArgumentParser(description="Predict and visualize chest X-ray diseases")
	parser.add_argument("image", type=str, help="Path to an X-ray image")
	parser.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS_PATH, help="Path to trained model weights")
	parser.add_argument("--classes_json", type=str, default=DEFAULT_CLASS_NAMES_JSON, help="Path to class names JSON")
	parser.add_argument("--csv", type=str, default=DEFAULT_CSV_PATH, help="Path to labels CSV file")
	parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for positive labels")
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	
	# Load model and class names
	with open(args.classes_json, "r", encoding="utf-8") as f:
		class_names = json.load(f)
	
	# Get predictions
	preds = predict_diseases(args.image, args.weights, args.classes_json, args.threshold)
	
	# Get ground truth if CSV file exists
	ground_truth = None
	if os.path.exists(args.csv):
		ground_truth = get_ground_truth_labels(args.image, args.csv, class_names)
	
	# Visualize with comparison
	visualize_prediction(args.image, preds, ground_truth=ground_truth, class_names=class_names)


