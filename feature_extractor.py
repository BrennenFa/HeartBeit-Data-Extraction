import wfdb
import scipy.io
import numpy as np
import glob
import os
from scipy.signal import resample
import torch
import torch.nn as nn
import pandas as pd
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.preprocessing import MultiLabelBinarizer
import ast
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


# ML Model - changed in the future
class SimpleECGCNN(nn.Module):
    def __init__(self, num_classes=1):  # change num_classes depending on your task
        super(SimpleECGCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 64 * 64, 128)  # assuming input image is resized to 256x256
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # -> [B, 16, 128, 128]
        x = self.pool(F.relu(self.conv2(x)))  # -> [B, 32, 64, 64]
        x = x.view(-1, 32 * 64 * 64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



# pytorch usable format
transform = transforms.Compose([
    transforms.Grayscale(),  # ensure it's 1 channel
    transforms.Resize((256, 256)),
    transforms.ToTensor(),  # Converts to [C, H, W] and scales to [0,1]
    transforms.Normalize([0.5], [0.5])
])

# load the image
def load_image(filepath):
    img = Image.open(filepath).convert("RGB")
    return transform(img)

# create the image dataset
class ECGImageDataset(Dataset):
    def __init__(self, image_paths, labels):
        self.image_paths = image_paths
        self.labels = labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = load_image(self.image_paths[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img, label


# train the model




def train_model(model, train_loader, val_loader, epochs=5, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

        # validation
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).unsqueeze(1)
                outputs = model(images)
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
            print(f"Validation Accuracy: {100 * correct / total:.2f}%")

# =============================== new code??

# linear, no longer need
# # get the ecg signal
# def load_ecg_record(record_path):
#     record = wfdb.rdrecord(record_path)
#     signal = record.p_signal
#     return signal


# no longer need --- coordinate points
# def preprocess_signal(signal, original_fs=500, target_fs=250, duration=10):
#     signal = signal[:original_fs * duration]  # clip to 10s
#     signal = resample(signal, int(target_fs * duration))
#     signal = (signal - np.mean(signal)) / np.std(signal)
#     return signal  # shape: (time, channels)


# def create_sequence_windows(signal, window_size=250, step=125):
#     windows = []
#     for start in range(0, signal.shape[0] - window_size + 1, step):
#         window = signal[start:start+window_size]
#         windows.append(window)
#     return np.stack(windows)  # shape: (seq_len, window_size, channels)

def train_ecg(labelPairs,
              batchSize=32,
              epochs=10,
              lr=1e-3
            ):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


  mlb = MultiLabelBinarizer()
  allLabels = [labels for _, labels in labelPairs]
  binary_labels = mlb.fit_transform(allLabels)

  # # Save this so you can inverse-transform predictions later
  # np.save("mlb_classes.npy", mlb.classes_)

  img_paths = [pair[0] for pair in labelPairs]
  X_train, X_val, y_train, y_val = train_test_split(img_paths, binary_labels, test_size=0.2, random_state=42)

  train_dataset = ECGImageDataset(X_train, y_train)
  val_dataset = ECGImageDataset(X_val, y_val)

  train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=batchSize)

  num_classes = len(mlb.classes_)

  model = SimpleECGCNN(num_classes=num_classes)
  criterion = nn.BCEWithLogitsLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)

  # Training loop
  for epoch in range(epochs):
      model.train()
      total_loss = 0
      for images, labels in train_loader:
          images, labels = images.to(device), labels.to(device)

          optimizer.zero_grad()
          outputs = model(images)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()
          total_loss += loss.item()

      print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

      # Validation
      model.eval()
      with torch.no_grad():
          correct = 0
          total = 0
          for images, labels in val_loader:
              images, labels = images.to(device), labels.to(device)
              outputs = model(images)
              preds = (torch.sigmoid(outputs) > 0.5).float()
              correct += (preds == labels).sum().item()
              total += labels.numel()
          print(f"Validation Accuracy: {100 * correct / total:.2f}%")

  return model


def extract_ecg_features(record_path, featureModel):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load the image
    image = load_image(record_path).unsqueeze(0)
    image = image.to(device)


    featureModel.eval()
    
    with torch.no_grad():
        x = featureModel.pool(F.relu(featureModel.conv1(image)))
        x = featureModel.pool(F.relu(featureModel.conv2(x)))
        x = x.view(-1, 32 * 64 * 64)
        features = F.relu(featureModel.fc1(x))
        return features.squeeze(0).cpu().numpy()

ecgFeatures = []
ids = []

# find all prexisiting images
files = glob.glob('ecg_images/**/*.png', recursive=True)
print(files)
print("FILES  ^")

# generate labels
labels = pd.read_csv("records_w_diag_icd10.csv")
labels['all_diag_hosp'] = labels['all_diag_hosp'].apply(ast.literal_eval)
diagnosisDictionary = dict(zip(labels['study_id'], labels['all_diag_hosp']))


labelPairs = []
for path in files:
    study_id = os.path.splitext(os.path.basename(path))[0]
    if study_id in diagnosisDictionary and diagnosisDictionary[study_id]:  # has labels
        labelPairs.append((path, diagnosisDictionary[study_id]))


featureModel = train_ecg(labelPairs)
# feature extraction --- note, need to train model first
for file in files:
    record_path = os.path.splitext(file)[0]
    # record_id = os.path.basename(record_path)
    record_id = os.path.splitext(os.path.basename(file))[0]


    print(f"Processing: {record_path}")
    try:
        features = extract_ecg_features(record_path, featureModel)
        ecgFeatures.append(features)
        ids.append(record_id)
    except Exception as e:
        print(f"Failed to process {record_path}: {e}")

df = pd.DataFrame(ecgFeatures)
df.insert(0, "record_id", ids)
df.to_csv("ecg_features_all.csv", index=False)
