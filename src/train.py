
import kagglehub

import torch
from torch.utils.data import Dataset, DataLoader

import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd

import random
import torch
import numpy as np

import deataset
import models
import utils

def train_val(training_technique,num_epochs, learning_rate, attr_loss_weight, landmark_loss_weight, patience):
    # Hyperparameters
    num_epochs = num_epochs
    learning_rate = learning_rate

    # Loss weights (initial values)
    attr_loss_weight = attr_loss_weight
    landmark_loss_weight = landmark_loss_weight

    # Early stopping patience
    best_val_loss = float('inf')
    patience = patience
    patience_counter = 0 # Initialize patience counter

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model, loss functions, and optimizer
    if training_technique == 'mtl':
      model = models.MultiTaskCelebAModel(3).to(device)
      optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:
      modelAttr = models.AttrTaskCelebAModel(3).to(device)
      modelLandmark = models.LandmarkTaskCelebAModel().to(device)
      optimizerAttr = optim.Adam(modelAttr.parameters(), lr=learning_rate)
      optimizerLandmark = optim.Adam(modelLandmark.parameters(), lr=learning_rate)

    # Loss functions
    attr_loss_fn = nn.BCEWithLogitsLoss()
    landmark_loss_fn = nn.MSELoss()

    # Initialize lists to store losses
    train_losses = []
    val_losses = []
    train_attr_losses = []
    val_attr_losses = []
    train_landmark_losses = []
    val_landmark_losses = []


    if training_technique == 'mtl':
      text_model = 'Multi-Task Learning'
    else:
      text_model = 'Single Model'
    print(f"\nStarting {text_model} training with Early Stopping...")

    for epoch in range(num_epochs):
        # --- Training Phase ---
        if training_technique == 'mtl':
          model.train()
          current_optimizer = optimizer
        else:
          modelAttr.train()
          modelLandmark.train()
          current_optimizer_attr = optimizerAttr
          current_optimizer_landmark = optimizerLandmark


        running_train_loss = 0.0
        running_train_attr_loss = 0.0
        running_train_landmark_loss = 0.0
        for batch_idx, (images, attributes, landmarks) in enumerate(train_loader):
            images, attributes, landmarks = images.to(device), attributes.to(device), landmarks.to(device)

            # Forward pass
            if training_technique == 'mtl':
              current_optimizer.zero_grad()
              attr_logits, landmark_preds = model(images)

              # Calculate individual task losses
              loss_attr = attr_loss_fn(attr_logits, attributes)
              loss_landmark = landmark_loss_fn(landmark_preds, landmarks)

              # Combine losses
              total_loss = (attr_loss_weight * loss_attr) + (landmark_loss_weight * loss_landmark)

              # Backward pass and optimize
              total_loss.backward()
              current_optimizer.step()

              # Accumulate training loss
              running_train_loss += total_loss.item()
              running_train_attr_loss += loss_attr.item()
              running_train_landmark_loss += loss_landmark.item()

            else: # Single models training
              # Train Attribute Model
              current_optimizer_attr.zero_grad()
              attr_logits = modelAttr(images)
              loss_attr = attr_loss_fn(attr_logits, attributes)
              loss_attr.backward()
              current_optimizer_attr.step()
              running_train_attr_loss += loss_attr.item()

              # Train Landmark Model
              current_optimizer_landmark.zero_grad()
              landmark_preds = modelLandmark(images)
              loss_landmark = landmark_loss_fn(landmark_preds, landmarks)
              loss_landmark.backward()
              current_optimizer_landmark.step()
              running_train_landmark_loss += loss_landmark.item()
              total_loss = (attr_loss_weight * loss_attr) + (landmark_loss_weight * loss_landmark)
              running_train_loss += total_loss.item()

        # Calculate average training loss for the epoch
        avg_train_attr_loss = running_train_attr_loss / len(train_loader)
        avg_train_landmark_loss = running_train_landmark_loss / len(train_loader)
        avg_train_loss = running_train_loss / len(train_loader)

        # Store training losses
        train_losses.append(avg_train_loss)
        train_attr_losses.append(avg_train_attr_loss)
        train_landmark_losses.append(avg_train_landmark_loss)

        # Validation (similar loop, but call model.eval() and no gradient updates)
        # --- Validation Phase ---

        if training_technique == 'mtl':
          model.eval()
        else:
          modelAttr.eval()
          modelLandmark.eval()

        running_val_loss = 0.0
        running_val_attr_loss = 0.0
        running_val_landmark_loss = 0.0
        with torch.no_grad():
          for batch_idx, (images, attributes, landmarks) in enumerate(val_loader):
            images, attributes, landmarks = images.to(device), attributes.to(device), landmarks.to(device)

            # Forward pass
            if training_technique == 'mtl':
              attr_logits, landmark_preds = model(images)
            else:
              attr_logits = modelAttr(images)
              landmark_preds = modelLandmark(images)


            # Calculate individual task losses
            loss_attr = attr_loss_fn(attr_logits, attributes)
            loss_landmark = landmark_loss_fn(landmark_preds, landmarks)

            # Accumulate validation loss
            running_val_attr_loss += loss_attr.item()
            running_val_landmark_loss += loss_landmark.item()
            total_loss = (attr_loss_weight * loss_attr) + (landmark_loss_weight * loss_landmark)
            running_val_loss += total_loss.item()


        # Calculate average validation loss for the epoch
        avg_val_attr_loss = running_val_attr_loss / len(val_loader)
        avg_val_landmark_loss = running_val_landmark_loss / len(val_loader)
        avg_val_loss = running_val_loss / len(val_loader)


        # Store validation losses
        val_attr_losses.append(avg_val_attr_loss)
        val_landmark_losses.append(avg_val_landmark_loss)
        val_losses.append(avg_val_loss)


        print(f"Epoch {epoch+1}/{num_epochs}")
        if training_technique == 'mtl':
            print(f"Train Loss: {avg_train_loss:.4f}, Attr Loss: {avg_train_attr_loss:.4f}, Landmark Loss: {avg_train_landmark_loss:.4f}")
            print(f"Val Loss: {avg_val_loss:.4f}, Attr Loss: {avg_val_attr_loss:.4f}, Landmark Loss: {avg_val_landmark_loss:.4f}")
        else:
            print(f"Train Loss: {avg_train_loss:.4f}, Train Attr Loss: {avg_train_attr_loss:.4f}, Train Landmark Loss: {avg_train_landmark_loss:.4f}")
            print(f"Val Loss: {avg_val_loss:.4f},Val Attr Loss: {avg_val_attr_loss:.4f}, Val Landmark Loss: {avg_val_landmark_loss:.4f}")

        # early stop
        if avg_val_loss < best_val_loss:
          best_val_loss = avg_val_loss
          patience_counter = 0
        else:
          patience_counter += 1
          if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    print("\nTraining complete.")

    loss_history = {'train_losses':train_losses,
                 'val_losses':val_losses,
                 'train_attr_losses':train_attr_losses,
                 'val_attr_losses':val_attr_losses,
                 'train_landmark_losses':train_landmark_losses,
                 'val_landmark_losses':val_landmark_losses
                 }

    if training_technique == 'mtl':
      return model, loss_history
    else:
      return modelAttr, modelLandmark, loss_history


if __name__ == "__main__":
    # data prepared
    path = kagglehub.dataset_download("jessicali9530/celeba-dataset")
    use_columns = ["image_id","High_Cheekbones","Mouth_Slightly_Open","Smiling"]

    img_dir = path+'/img_align_celeba/img_align_celeba/'
    landmarks_df = pd.read_csv(path + '/list_landmarks_align_celeba.csv')
    attrs_df = pd.read_csv(path+'/list_attr_celeba.csv',usecols= use_columns )
    partition_df = pd.read_csv(path+'/list_eval_partition.csv')

    train_dataset = deataset.CelebAMultiTaskDataset(img_dir, attrs_df, landmarks_df, partition_df, 'train', 162770)
    val_dataset = deataset.CelebAMultiTaskDataset(img_dir, attrs_df, landmarks_df, partition_df, 'val', 19962)
    test_dataset = deataset.CelebAMultiTaskDataset(img_dir, attrs_df, landmarks_df, partition_df, 'test', 19867)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    # model training
    torch.manual_seed(77)
    np.random.seed(77)
    random.seed(77)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(77)

    multiTask_model, multi_loss_history = train_val(training_technique = 'mtl',
                                            num_epochs = 10,
                                            learning_rate = 0.001,
                                            attr_loss_weight= 50.0,
                                            landmark_loss_weight= 100.0,
                                            patience= 2)

    modelAttr, modelLandmark, single_loss_history = train_val(training_technique = 'single',
                                                    num_epochs = 10,
                                                    learning_rate = 0.01,
                                                    attr_loss_weight= 50,
                                                    landmark_loss_weight= 100,
                                                    patience= 2)

    utils.plot_training_curves(title = 'multi-task learning model',data = multi_loss_history)
    utils.plot_training_curves(title = 'single model',data = single_loss_history)

    #Evaluation Model
    utils.evaluate_model('mtl')
    utils.evaluate_model('single')

