import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import time
import os
from datetime import datetime
import csv

class SimpleNet(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim_1=512):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class SimpleCNN(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim_1=512):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.feature_dim = 8192

        self.fc1 = nn.Linear(self.feature_dim, hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.view(x.size(0), 3, 32, 32)
        elif x.dim() == 4:
            pass
        else:
            raise ValueError(f"Expected input with 2 or 4 dimensions, got {x.shape} dimensions")
        
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
class SimpleLinear(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimpleLinear, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.fc(x)
    
def weighted_cross_entropy_loss(outputs, targets, weights):
    log_probs = F.log_softmax(outputs, dim=1)
    one_hot = F.one_hot(targets, num_classes=outputs.size(1)).float()
    weighed_loss = -torch.sum(weights[:, None] * one_hot * log_probs) / torch.sum(weights)
    return weighed_loss
    
class BoostModel(nn.Module):
    def __init__(self, input_dim, num_classes, num_estimators=20, batch_size=128):
        super(BoostModel, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.n_estimators = num_estimators
        self.models = nn.ModuleList()
        self.model_weights = nn.Parameter(torch.ones(num_estimators), requires_grad=False)
        self.alphas = []
        self.batch_size = batch_size

        self.training_history = {
            'train_loss': [],
            'train_accuracy': [],
            'test_loss': [],
            'test_accuracy': [],
            'error_rates': [],
            'alphas': []
        }

    def fit(self, X, y, epochs, device, test_dataloader=None, weak_learner=None):
        n_samples = X.shape[0]
        if weak_learner is None:
            print("No weak learner provided")
            return

        # Check if input is 2D or 3D and reshape accordingly
        if weak_learner == 'cnn':
            if X.dim() == 2:
                X = X.view(n_samples, 3, 32, 32)  # Assuming input is 32x32 RGB images
        else:
            if X.dim() > 2:
                X = X.view(n_samples, -1)

        weights = torch.ones(n_samples) / n_samples

        X = X.to(device)
        y = y.to(device)
        weights = weights.to(device)

        dataset = TensorDataset(X, y, weights)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for i in range(self.n_estimators):
            print(f"Training estimator {i + 1}/{self.n_estimators}...")
            if weak_learner == 'nn':
                model = SimpleNet(self.input_dim, self.num_classes).to(device)
            elif weak_learner == 'cnn':
                model = SimpleCNN(3, self.num_classes).to(device)
            elif weak_learner == 'linear':
                model = SimpleLinear(self.input_dim, self.num_classes).to(device)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            criterion = weighted_cross_entropy_loss

            # Train the weak learner
            model.train()
            start_time = time.time()
            for epoch in range(epochs):
                for X_batch, y_batch, w_batch in loader:

                    if device.type == 'cuda':
                        X_batch, y_batch, w_batch = X_batch.to(device), y_batch.to(device), w_batch.to(device)
                        torch.cuda.empty_cache()

                    optimizer.zero_grad()
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch, w_batch)
                    loss.backward()
                    optimizer.step()

            end_time = time.time()
            print(f"Estimator {i + 1} training time: {end_time - start_time:.2f} seconds")

            with torch.no_grad():
                preds = torch.argmax(model(X), dim=1)
                incorrect = (preds != y).float()
                err = torch.sum(weights * incorrect) / torch.sum(weights)
            
            if err.item() > 0.7 and weak_learner != 'linear':
                print(f"Estimator {i + 1} error too high ({err.item():.4f}), stopping training.")
                break
            
            err = max(err.item(), 1e-10)  # Avoid division by zero
            alpha = torch.log(torch.tensor((1 - err) / err,device=device)) + torch.log(torch.tensor(self.num_classes - 1, dtype=torch.float32, device=device))
            print(f"Estimator {i + 1} error: {err:.4f}, alpha: {alpha.item():.4f}")

            # Update model and weights
            weights = weights * torch.exp(alpha * incorrect)
            weights /= torch.sum(weights)  # Normalize weights
            self.models.append(model)
            self.alphas.append(alpha.item())

             # Calculate the accuracy of the model
            with torch.no_grad():
                self.eval()
                outputs = self.predict(X)
                accuracy = (outputs == y).float().mean().item()
                criterion = nn.CrossEntropyLoss()
                loss = criterion(self(X), y)
            print(f"Estimator {i + 1} train accuracy: {accuracy:.4f}, train loss: {loss.item():.4f}")

            if test_dataloader is not None:
                test_accuracy, test_loss = self.evaluate(test_dataloader, device)
                print(f"Estimator {i + 1} test accuracy: {test_accuracy:.4f}, test loss: {test_loss:.4f}")
            
            # Store training history
            self.training_history['train_loss'].append(loss.item())
            self.training_history['train_accuracy'].append(accuracy)
            self.training_history['test_loss'].append(test_loss)
            self.training_history['test_accuracy'].append(test_accuracy)
            self.training_history['error_rates'].append(err)
            self.training_history['alphas'].append(alpha.item())
        
        self.save_training_history()

    def save_training_history(self, save_path='training_history'):
        print("Saving training history...")
        os.makedirs(save_path, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"boosting_training_history_{timestamp}.csv"
        filepath = os.path.join(save_path, filename)

        try:
            n_records = len(self.training_history['train_accuracy'])

            if n_records == 0:
                print("No training history to save.")
                return
            
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = [
                    'train_accuracy', 'train_loss', 
                    'test_accuracy', 'test_loss', 'error_rates', 
                    'alphas'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for i in range(n_records):
                    row = {
                        'train_accuracy': f"{self.training_history['train_accuracy'][i]:.6f}",
                        'train_loss': f"{self.training_history['train_loss'][i]:.6f}",
                        'test_accuracy': f"{self.training_history['test_accuracy'][i]:.6f}",
                        'test_loss': f"{self.training_history['test_loss'][i]:.6f}",
                        'error_rates': f"{self.training_history['error_rates'][i]:.6f}",
                        'alphas': f"{self.training_history['alphas'][i]:.6f}",
                    }
                    writer.writerow(row)

                print(f"Training history saved to {filepath}")
                print(f"Total records saved: {n_records}")

        except Exception as e:
            print(f"Error saving training history: {e}")

    def predict(self, X):
        score = torch.zeros(X.shape[0], self.num_classes, device=X.device)

        for model, alpha in zip(self.models, self.alphas):
            logits = model(X)
            score += alpha * logits
        return torch.argmax(score, dim=1)
    
    def forward(self, X):
        score = torch.zeros(X.shape[0], self.num_classes, device=X.device)

        for model, alpha in zip(self.models, self.alphas):
            with torch.no_grad():
                logits = model(X)
                score += alpha * logits
        return score
    
    def evaluate(self, dataloader, device):
        self.eval()
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                loss = criterion(outputs, labels)
        loss_value = loss.item()
        
        accuracy = correct / total
        return accuracy, loss_value
