import os
import json
import argparse
from typing import List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
from torchvision import models

from sklearn.metrics import f1_score
from tqdm import tqdm


DEFAULT_IMAGE_DIR = "resized"
DEFAULT_LABELS_CSV = "labels_clean.csv"
DEFAULT_WEIGHTS_PATH = "best_model.pth"
DEFAULT_CLASS_NAMES_JSON = "class_names.json"


class ChestXrayDataset(Dataset):
	def __init__(self, csv_path: str, image_dir: str, class_names: List[str], transform=None):
		self.df = pd.read_csv(csv_path)
		self.image_dir = image_dir
		self.class_names = class_names
		self.transform = transform

		# Normalize filename column name
		if "filename" not in self.df.columns:
			if "Image Index" in self.df.columns:
				self.df = self.df.rename(columns={"Image Index": "filename"})
			else:
				raise ValueError("CSV must contain a 'filename' (or 'Image Index') column with image filenames")

		# If explicit one-hot columns are not present, expect 'Finding Labels' with pipe-separated labels
		missing_cols = [c for c in class_names if c not in self.df.columns]
		self.uses_multilabel_string = False
		if missing_cols:
			if "Finding Labels" not in self.df.columns:
				raise ValueError(
					"CSV must contain either one-hot disease columns or a 'Finding Labels' column"
				)
			self.uses_multilabel_string = True

	def __len__(self) -> int:
		return len(self.df)

	def __getitem__(self, idx: int):
		row = self.df.iloc[idx]
		image_path = os.path.join(self.image_dir, str(row["filename"]))
		image = Image.open(image_path).convert("RGB")

		if self.transform is not None:
			image = self.transform(image)

		if self.uses_multilabel_string:
			labels_vec = np.zeros(len(self.class_names), dtype=np.float32)
			label_str = str(row.get("Finding Labels", "")).strip()
			if label_str:
				if label_str == "No Finding":
					# Only mark 'No Finding'
					if "No Finding" in self.class_names:
						labels_vec[self.class_names.index("No Finding")] = 1.0
				else:
					for lab in label_str.split("|"):
						lab = lab.strip()
						if lab in self.class_names:
							labels_vec[self.class_names.index(lab)] = 1.0
			labels = torch.tensor(labels_vec, dtype=torch.float32)
		else:
			labels = torch.tensor(row[self.class_names].astype(float).values, dtype=torch.float32)

		return image, labels


def get_transforms(train: bool = True):
	# ImageNet normalization
	imagenet_mean = [0.485, 0.456, 0.406]
	imagenet_std = [0.229, 0.224, 0.225]

	if train:
		return T.Compose([
			T.Resize((224, 224)),
			T.RandomHorizontalFlip(p=0.5),
			T.RandomRotation(degrees=7),
			T.ToTensor(),
			T.Normalize(mean=imagenet_mean, std=imagenet_std),
		])
	else:
		return T.Compose([
			T.Resize((224, 224)),
			T.ToTensor(),
			T.Normalize(mean=imagenet_mean, std=imagenet_std),
		])


def build_model(num_classes: int) -> nn.Module:
	model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
	# Replace classifier for multi-label outputs
	in_features = model.classifier.in_features
	model.classifier = nn.Linear(in_features, num_classes)
	return model


@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device, class_names: List[str]) -> Tuple[float, float, List[float]]:
	model.eval()
	all_targets = []
	all_probs = []
	for images, targets in dataloader:
		images = images.to(device, non_blocking=True)
		targets = targets.to(device, non_blocking=True)
		logits = model(images)
		probs = torch.sigmoid(logits)
		all_targets.append(targets.detach().cpu())
		all_probs.append(probs.detach().cpu())

	all_targets = torch.cat(all_targets, dim=0).numpy()
	all_probs = torch.cat(all_probs, dim=0).numpy()
	preds = (all_probs >= 0.5).astype(np.float32)

	# Subset accuracy: exact match of all labels per sample
	subset_acc = (preds == all_targets).all(axis=1).mean().item()

	# Per-class F1 (binary for each class)
	per_class_f1 = []
	for i in range(len(class_names)):
		f1 = f1_score(all_targets[:, i], preds[:, i], zero_division=0)
		per_class_f1.append(f1)
	macro_f1 = float(np.mean(per_class_f1))

	return float(subset_acc), macro_f1, per_class_f1


def train(
	csv_path: str,
	image_dir: str,
	output_weights: str,
	class_names_path: str,
	epochs: int = 10,
	batch_size: int = 32,
	val_split: float = 0.1,
	learning_rate: float = 1e-4,
	workers: int = 4,
):
	class_names = [
		"Atelectasis",
		"Consolidation",
		"Edema",
		"Effusion",
		"Infiltration",
		"Mass",
		"Nodule",
		"Pneumothorax",
		"No Finding",
	]

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	full_dataset = ChestXrayDataset(
		csv_path=csv_path,
		image_dir=image_dir,
		class_names=class_names,
		transform=get_transforms(train=True),
	)

	val_len = int(len(full_dataset) * val_split)
	train_len = len(full_dataset) - val_len
	train_dataset, val_dataset = random_split(full_dataset, [train_len, val_len])
	# Validation must not use augmentation beyond resize/normalize
	val_dataset.dataset.transform = get_transforms(train=False)

	train_loader = DataLoader(
		train_dataset,
		batch_size=batch_size,
		shuffle=True,
		num_workers=workers,
		pin_memory=True,
	)
	val_loader = DataLoader(
		val_dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=workers,
		pin_memory=True,
	)

	model = build_model(num_classes=len(class_names)).to(device)
	criterion = nn.BCEWithLogitsLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

	# Save class names for inference
	with open(class_names_path, "w", encoding="utf-8") as f:
		json.dump(class_names, f, indent=2)

	best_macro_f1 = -1.0
	for epoch in range(1, epochs + 1):
		model.train()
		running_loss = 0.0
		for images, targets in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} - Training", leave=False):
			images = images.to(device, non_blocking=True)
			targets = targets.to(device, non_blocking=True)
			optimizer.zero_grad(set_to_none=True)
			logits = model(images)
			loss = criterion(logits, targets)
			loss.backward()
			optimizer.step()
			running_loss += loss.item() * images.size(0)

		train_loss = running_loss / train_len if train_len > 0 else 0.0
		subset_acc, macro_f1, per_class_f1 = evaluate(model, val_loader, device, full_dataset.class_names)

		print(f"Epoch {epoch}/{epochs} | TrainLoss: {train_loss:.4f} | ValSubsetAcc: {subset_acc:.4f} | ValMacroF1: {macro_f1:.4f}")
		print("Per-class F1:")
		for name, f1 in zip(full_dataset.class_names, per_class_f1):
			print(f"  {name}: {f1:.4f}")

		if macro_f1 > best_macro_f1:
			best_macro_f1 = macro_f1
			torch.save({
				"model_state_dict": model.state_dict(),
				"num_classes": len(full_dataset.class_names),
				"class_names_path": class_names_path,
				"arch": "densenet121",
			}, output_weights)
			print(f"Saved new best model to {output_weights} (ValMacroF1={macro_f1:.4f})")

	print("Training complete.")
	print(f"Best Val Macro-F1: {best_macro_f1:.4f}")


def parse_args():
	parser = argparse.ArgumentParser(description="Train DenseNet121 for multi-label chest X-ray classification")
	parser.add_argument("--csv", type=str, default=DEFAULT_LABELS_CSV, help="Path to labels CSV")
	parser.add_argument("--images", type=str, default=DEFAULT_IMAGE_DIR, help="Path to images directory")
	parser.add_argument("--out", type=str, default=DEFAULT_WEIGHTS_PATH, help="Output path for best model weights")
	parser.add_argument("--classes_json", type=str, default=DEFAULT_CLASS_NAMES_JSON, help="Path to save class names JSON")
	parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
	parser.add_argument("--batch", type=int, default=32, help="Batch size")
	parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
	parser.add_argument("--val_split", type=float, default=0.1, help="Validation split fraction")
	parser.add_argument("--workers", type=int, default=4, help="Number of DataLoader workers")
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	train(
		csv_path=args.csv,
		image_dir=args.images,
		output_weights=args.out,
		class_names_path=args.classes_json,
		epochs=args.epochs,
		batch_size=args.batch,
		val_split=args.val_split,
		learning_rate=args.lr,
		workers=args.workers,
	)


