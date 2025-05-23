import torch.nn as nn
import matplotlib.pyplot as plt
import torch
from torch.optim import Optimizer
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import numpy as np

class Lion(Optimizer):
    def __init__(self, params, lr=1e-4, beta1=0.9, beta2=0.99, weight_decay=0.1):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                lr = group['lr']
                beta1 = group['beta1']
                beta2 = group['beta2']
                wd = group['weight_decay']
                state = self.state[p]
                if len(state) == 0:
                    state['momentum'] = torch.zeros_like(p)

                momentum = state['momentum']
                
                update = beta1 * momentum + (1 - beta1) * grad
                update = torch.sign(update)
                momentum.mul_(beta2).add_(grad, alpha=1 - beta2)
                if wd != 0:
                    update.add_(p, alpha=wd)
                p.add_(update, alpha=-lr)

        return loss


def process_data():
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    df = pd.read_csv("train.csv")
    df.drop(columns=["id"], inplace=True)

    num_features = ["person_age", "person_income", "person_emp_length", "loan_amnt", "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length"]
    cat_features = ["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]
    scaler = StandardScaler()
    df[num_features] = scaler.fit_transform(df[num_features])
    label_encoders = {}
    for col in cat_features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    X = df.drop(columns=["loan_status"])
    y = df["loan_status"].values
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    
    batch_size = 32
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, X_train.shape[1]

def train(model, train_loader, val_loader, num_epochs=10, lr=1e-4, beta1=0.9, beta2=0.99, weight_decay=0.1):
    criterion = nn.BCELoss()
    optimizer = Lion(model.parameters(), lr=lr, beta1=beta1, beta2=beta2, weight_decay=weight_decay) 
    train_losses, val_losses = [], []
    train_roc_auc, val_roc_auc = [], []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        y_train_true, y_train_pred = [], []
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            y_train_true.extend(y_batch.numpy())
            y_train_pred.extend(outputs.detach().numpy())
        
        train_losses.append(total_loss / len(train_loader))
        train_roc_auc.append(roc_auc_score(y_train_true, y_train_pred))
        
        model.eval()
        val_loss = 0
        y_val_true, y_val_pred = [], []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                
                y_val_true.extend(y_batch.numpy())
                y_val_pred.extend(outputs.numpy())
        
        val_losses.append(val_loss / len(val_loader))
        val_roc_auc.append(roc_auc_score(y_val_true, y_val_pred))
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Train ROC-AUC: {train_roc_auc[-1]:.4f}, Val ROC-AUC: {val_roc_auc[-1]:.4f}")
    return train_losses, val_losses, train_roc_auc, val_roc_auc

def show_graphs(train_losses, val_losses, train_roc_auc, val_roc_auc, num_epochs=10):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs+1), train_losses, label="Train Loss")
    plt.plot(range(1, num_epochs+1), val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curve")

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs+1), train_roc_auc, label="Train ROC-AUC")
    plt.plot(range(1, num_epochs+1), val_roc_auc, label="Validation ROC-AUC")
    plt.xlabel("Epochs")
    plt.ylabel("ROC-AUC")
    plt.legend()
    plt.title("ROC-AUC Curve")
    plt.show()

class ResidualBlockWithDropout(nn.Module):
    def __init__(self, hidden_size, dropout_p):
        super(ResidualBlockWithDropout, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.BatchNorm1d(hidden_size * 4),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_p)
        )
    
    def forward(self, x):
        return x + self.block(x)

class ModelWithDropout(nn.Module):
    def __init__(self, input_dim, dropout_p, hidden_size=128, num_blocks=3):
        super(ModelWithDropout, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.blocks = nn.ModuleList([ResidualBlockWithDropout(hidden_size, dropout_p) for _ in range(num_blocks)])
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        for block in self.blocks:
            x = block(x)
        x = self.fc2(x)
        return self.sigmoid(x)

train_loader, val_loader, input_dim = process_data()

num_epochs = 15
lr = 1e-4
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.99
model = ModelWithDropout(input_dim, 0.1)

train_losses, val_losses, train_roc_auc, val_roc_auc = train(model, train_loader, val_loader, num_epochs, lr=lr, weight_decay=weight_decay, beta1=beta1, beta2=beta2)
show_graphs(train_losses, val_losses, train_roc_auc, val_roc_auc, num_epochs)