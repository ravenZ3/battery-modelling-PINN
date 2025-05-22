import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.model import MLP as Encoder
from models.model import Predictor
from datasets.dataloader import MITdata
import numpy as np
import argparse
from sklearn.metrics import mean_squared_error

class ResBlock(nn.Module):
    def __init__(self, input_channel, output_channel, stride):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channel, output_channel, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm1d(output_channel),
            nn.ReLU(),
            nn.Conv1d(output_channel, output_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(output_channel)
        )
        self.skip_connection = nn.Sequential()
        if output_channel != input_channel:
            self.skip_connection = nn.Sequential(
                nn.Conv1d(input_channel, output_channel, kernel_size=1, stride=stride),
                nn.BatchNorm1d(output_channel)
            )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.skip_connection(x) + out
        out = self.relu(out)
        return out

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.encoder = Encoder(input_dim=17, output_dim=32, layers_num=3, hidden_dim=60, droupout=0.2)
        self.predictor = Predictor(input_dim=32)

    def forward(self, x):
        x = self.encoder(x)
        x = self.predictor(x)
        return x

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = ResBlock(input_channel=1, output_channel=8, stride=1)  # N,8,17
        self.layer2 = ResBlock(input_channel=8, output_channel=16, stride=2)  # N,16,9
        self.layer3 = ResBlock(input_channel=16, output_channel=24, stride=2)  # N,24,5
        self.layer4 = ResBlock(input_channel=24, output_channel=16, stride=1)  # N,16,5
        self.layer5 = ResBlock(input_channel=16, output_channel=8, stride=1)  # N,8,5
        self.layer6 = nn.Linear(8*5, 1)

    def forward(self, x):
        N, L = x.shape[0], x.shape[1]
        x = x.view(N, 1, L)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out.view(N, -1))
        return out.view(N, 1)

def count_parameters(model):
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {count} trainable parameters')
    return count

def train_model(model, train_loader, valid_loader, epochs=200, lr=1e-2, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x1, x2, y1, y2 in train_loader:
            x1, x2, y1, y2 = x1.to(device), x2.to(device), y1.to(device), y2.to(device)
            optimizer.zero_grad()
            pred = model(x1)  # Predict y1 from x1
            loss = criterion(pred, y1)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for x1, x2, y1, y2 in valid_loader:
                x1, x2, y1, y2 = x1.to(device), x2.to(device), y1.to(device), y2.to(device)
                pred = model(x1)
                loss = criterion(pred, y1)
                valid_loss += loss.item()
        
        print(f'Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader):.4f}, Valid Loss: {valid_loss/len(valid_loader):.4f}')
    
    return model

def evaluate_model(model, test_loader, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for x1, x2, y1, y2 in test_loader:
            x1, y1 = x1.to(device), y1.to(device)
            pred = model(x1)
            predictions.extend(pred.cpu().numpy().flatten())
            actuals.extend(y1.cpu().numpy().flatten())
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    return predictions, actuals, rmse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--normalization_method', type=str, default='min-max', help='min-max, z-score')
    parser.add_argument('--epochs', type=int, default=200, help='epochs')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--save_folder', type=str, default='results/comparison', help='save folder')
    parser.add_argument('--log_dir', type=str, default='comparison_log.txt', help='log file')  # Added log_dir
    return parser.parse_args()

def load_data(args):
    root = 'data/MIT_data'
    train_list, test_list = [], []
    for batch in ['2017-05-12', '2017-06-30', '2018-04-12']:
        batch_root = os.path.join(root, batch)
        files = os.listdir(batch_root)
        for f in files:
            id = int(f.split('-')[-1].split('.')[0])
            if id % 5 == 0:
                test_list.append(os.path.join(batch_root, f))
            else:
                train_list.append(os.path.join(batch_root, f))
    
    data = MITdata(root=root, args=args)
    trainloader = data.read_all(specific_path_list=train_list)
    testloader = data.read_all(specific_path_list=test_list)
    return trainloader['train_2'], trainloader['valid_2'], testloader['test_3']

if __name__ == '__main__':
    args = get_args()
    os.makedirs(args.save_folder, exist_ok=True)
    
    # Load data
    train_loader, valid_loader, test_loader = load_data(args)
    
    # Device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Train and evaluate MLP
    mlp = MLP().to(device)
    print("Training MLP...")
    count_parameters(mlp)
    mlp = train_model(mlp, train_loader, valid_loader, epochs=args.epochs, lr=args.lr, device=device)
    mlp_preds, mlp_actuals, mlp_rmse = evaluate_model(mlp, test_loader, device)
    print(f"MLP RMSE: {mlp_rmse:.4f}")
    
    # Train and evaluate CNN
    cnn = CNN().to(device)
    print("Training CNN...")
    count_parameters(cnn)
    cnn = train_model(cnn, train_loader, valid_loader, epochs=args.epochs, lr=args.lr, device=device)
    cnn_preds, cnn_actuals, cnn_rmse = evaluate_model(cnn, test_loader, device)
    print(f"CNN RMSE: {cnn_rmse:.4f}")
    
    # Load PINN results for comparison
    pinn_preds = np.load('results/MIT results/Experiment1/pred_label.npy').flatten()
    pinn_actuals = np.load('results/MIT results/Experiment1/true_label.npy').flatten()
    pinn_rmse = np.sqrt(mean_squared_error(pinn_actuals, pinn_preds))
    print(f"PINN RMSE: {pinn_rmse:.4f}")
    
    # Save comparison results
    np.save(os.path.join(args.save_folder, 'mlp_predictions.npy'), mlp_preds)
    np.save(os.path.join(args.save_folder, 'cnn_predictions.npy'), cnn_preds)
    with open(os.path.join(args.save_folder, 'rmse_comparison.txt'), 'w') as f:
        f.write(f"MLP RMSE: {mlp_rmse:.4f}\n")
        f.write(f"CNN RMSE: {cnn_rmse:.4f}\n")
        f.write(f"PINN RMSE: {pinn_rmse:.4f}\n")
