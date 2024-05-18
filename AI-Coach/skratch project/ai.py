import numpy as np
import cv2
import mediapipe as mp
import os
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import LabelEncoder

dataset = 'All data'
NUM_CLASSES = 4

def load_dataset(dataset_path):
    x = []
    y = []
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Loop through dataset directory
    for exercise_folder in os.listdir(dataset_path):
        exercise_label = exercise_folder
        exercise_folder_path = os.path.join(dataset_path, exercise_folder)  #/content/Data1/crunches

        # Loop through video files in exercise folder
        for video_file in os.listdir(exercise_folder_path):
            video_path = os.path.join(exercise_folder_path, video_file)
            cap = cv2.VideoCapture(video_path)
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    break
                # Process image using MediaPipe Pose Detection
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                result = pose.process(image_rgb)
                if result.pose_landmarks:
                    # Extract pose landmarks
                    landmarks =np.array([[lm.x, lm.y] for lm in result.pose_landmarks.landmark]).flatten()
                    x.append(landmarks)
                    y.append(exercise_label)
            cap.release()
    return np.array(x), np.array(y)

# Define the path to your dataset directory
dataset_path = 'All data'

# Load the dataset
x_data, y_data = load_dataset(dataset_path)

# Print the shapes of the loaded data
print("Shape of x_data (pose landmarks):", x_data.shape)
print("Shape of y_data (exercise labels):", y_data.shape)

m = 0
for i in y_data:
  if (i == 'squat true'):
    m+=1
print(m)

print(y_data)

for i in range(y_data.size):
  if y_data[i] == 'shouldertaps true':
    y_data[i] = 0
  elif y_data[i] == 'shouldertaps false':
    y_data[i] = 1
  elif y_data[i] == 'lunges true':
    y_data[i] = 2
  elif y_data[i] == 'lunges false':
    y_data[i] = 3
  else:
    y_data[i] = -1

    print( y_data)


# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

# Print the shapes of the training and testing sets
print("Shape of x_train (training data):", x_train.shape)
print("Shape of y_train (training labels):", y_train.shape)
print("Shape of x_test (testing data):", x_test.shape)
print("Shape of y_test (testing labels):", y_test.shape)

# Data Preprocessing
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)

x_train = pd.get_dummies(x_train)

# After loading the dataset
x_train = x_train.reshape(x_train.shape[0], -1)  # Reshape to flatten each data point
x_test = x_test.reshape(x_test.shape[0], -1)

# Define a simple neural network model using PyTorch
class ExerciseClassifier(nn.Module):
    def __init__(self):
        super(ExerciseClassifier, self).__init__()
        self.fc1 = nn.Linear(x_train.shape[1], 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x
    
    # Initialize the model
model = ExerciseClassifier()

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Convert data to PyTorch tensors
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

# Train the model
for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(x_train_tensor)
    loss = criterion(outputs, y_train_tensor.view(-1, 1))
    loss.backward()
    optimizer.step()

    # Calculate accuracy
    predicted = (outputs > 0.5).float()  # Convert outputs to binary predictions
    correct = (predicted == y_train_tensor.view(-1, 1)).float().sum()  # Count correct predictions
    accuracy = correct / len(y_train_tensor)  # Calculate accuracy

    print(f'Epoch [{epoch+1}/10], Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.4f}')
    
  # Save the trained model
torch.save(model.state_dict(), 'exercise_classifier.pth')

# Load the saved model
model = ExerciseClassifier()  # Assuming ExerciseClassifier is your model class
model.load_state_dict(torch.load('exercise_classifier.pth'))
model.eval()  # Set the model to evaluation mode

# Convert test data to PyTorch tensor
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)

# Perform inference with the model
with torch.no_grad():
    outputs = model(x_test_tensor)
    predictions = (outputs > 0.5).float()

# Calculate accuracy
correct = (predictions == y_test.reshape(-1, 1)).float().sum()
accuracy = correct / len(y_test)
print(f'Test Accuracy: {accuracy.item():.4f}')