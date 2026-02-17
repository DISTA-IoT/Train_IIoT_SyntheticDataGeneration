import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.optim as optim
from modules import MLP
from sklearn.utils import resample
import numpy as np

feats = [
    'Durata','CabEnabled_M1', 'CabEnabled_M8', 'HMI_ACPntSts_T2', 'HMI_ACPntSts_T7', 'HMI_DCPntSts_T2',
    'HMI_DCPntSts_T7', 'HMI_Iline', 'HMI_Irsts_T2', 'HMI_Irsts_T7', 'HMI_VBatt_T2', 'HMI_VBatt_T4', 'HMI_VBatt_T5',
    'HMI_VBatt_T7', 'HMI_Vline', 'HMI_impSIL', 'LineVoltType', 'MDS_LedLimVel', 'MDS_StatoMarcia', '_GPS_LAT',
    '_GPS_LON', 'ldvvelimps', 'ldvveltreno', 'usB1BCilPres_M1', 'usB1BCilPres_M3', 'usB1BCilPres_M6', 'usB1BCilPres_M8',
    'usB1BCilPres_T2', 'usB1BCilPres_T4', 'usB1BCilPres_T5', 'usB1BCilPres_T7', 'usB2BCilPres_M1', 'usB2BCilPres_M3',
    'usB2BCilPres_M6', 'usB2BCilPres_M8', 'usB2BCilPres_T2', 'usB2BCilPres_T4', 'usB2BCilPres_T5', 'usB2BCilPres_T7',
    'usBpPres', 'usMpPres'
]

label = 'ERTMS_PiastraSts'

def resample_to_percentages(X, y, percentages):
    X0, X1, X2 = X[y == 0], X[y == 1], X[y == 2]
    
    p0, p1, p2 = percentages
    N_total = len(y)
    N0 = int(p0 * N_total)
    N1 = int(p1 * N_total)
    N2 = int(p2 * N_total)

    X0_r, y0_r = resample(X0, np.zeros(len(X0)), n_samples=N0, replace=True)
    X1_r, y1_r = resample(X1, np.ones(len(X1)), n_samples=N1, replace=True)
    X2_r, y2_r = resample(X2, np.full(len(X2), 2), n_samples=N2, replace=True)

    X_res = np.vstack([X0_r, X1_r, X2_r])
    y_res = np.hstack([y0_r, y1_r, y2_r])

    return X_res, y_res


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Training anomaly detector')
    
    parser.add_argument(
        '--epochs',
        type=int, 
        help='Number of training epochs',
        default=3)
    
    parser.add_argument(
        '--percentages',
        type=float,
        nargs=3,
        help='List of three percentages (normal, anomaly, fault)',
        default=[0.9, 0.08, 0.02]
    )

    args = parser.parse_args()

    epochs = args.epochs
    class_weights = args.percentages

    dataset = pd.read_csv('dataset.csv')
    print('intial dataset len', len(dataset), 'column num', len(dataset.columns))
    dataset_new = dataset.copy()
    dataset_new['Timestamp'] = pd.to_datetime(dataset_new['Timestamp'], errors='coerce')
    # timestamp will be essential
    dataset_new = dataset_new.dropna(subset=['Timestamp'])
    print("label counts", dataset_new[label].value_counts())
    # repeated signals:
    dataset_new = dataset_new.drop_duplicates(subset=['Descrizione', 'Timestamp'])
    dataset_new.set_index('Timestamp', inplace=True)
    dataset_new = dataset_new.dropna(axis=1, how='all')
    dataset_new = dataset_new.dropna()
    print('after first filtering dataset len', len(dataset_new), 'column num', len(dataset_new.columns))

    Y = dataset_new[label].astype(int).values
    X = dataset_new[feats].values

    anomaly_classifier = MLP(
        **{
            'input_dim': X.shape[1],
            'output_dim': 3,
            'layer_norm': True,
            'mode':'OF'})


    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_train_resampled, Y_train_resampled = resample_to_percentages(
            X_train, Y_train, class_weights
        )

    train_scaler = MinMaxScaler()
    X_train_scaled_resampled = train_scaler.fit_transform(X_train_resampled)
    X_test_scaled = train_scaler.transform(X_test)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(anomaly_classifier.parameters(), lr=0.001)

    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train_scaled_resampled).float(),
        torch.tensor(Y_train_resampled).long())
    test_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_test_scaled).float(),
        torch.tensor(Y_test).long())

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False)

    for epoch in range(epochs):
        anomaly_classifier.train()
        for x, y in train_dataloader:
            optimizer.zero_grad()
            output, _ = anomaly_classifier(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

        anomaly_classifier.eval()
        total_correct = 0
        with torch.no_grad():
            for x, y in test_dataloader:
                output, _ = anomaly_classifier(x)
                predicted = torch.argmax(output, dim=1)
                total_correct += (predicted == y).sum().item()
        accuracy = total_correct / len(test_dataset)
        print(f'Epoch {epoch+1}, Test Accuracy: {accuracy:.4f}')


    # final plot:
    all_X_scaled = train_scaler.transform(X)
    all_X_scaled = torch.Tensor(all_X_scaled).float()
    anomaly_classifier.eval()
    all_preds, manifold = anomaly_classifier(all_X_scaled)
    all_preds = torch.argmax(all_preds, dim=1)
    all_correct = (all_preds == Y).sum().item()
    all_accuracy = all_correct / len(Y)
    print("Accuracy finale: ", all_accuracy)
    manifold = manifold.numpy()

    plt.figure(figsize=(10, 8))
    plt.scatter(manifold[:, 0], manifold[:, 1], c=Y.astype(int), s=15, alpha=0.7)
    plt.title('Manifold Projection', fontsize=14)
    plt.xlabel('Manifold Component 1', fontsize=12)
    plt.ylabel('Manifold Component 2', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig('boundary_attacks/manifold.png', dpi=150, bbox_inches='tight')
    plt.show()

    with open('boundary_attacks/training_scaler.pkl', 'wb') as f:
        pickle.dump(train_scaler, f)
    torch.save(anomaly_classifier.state_dict(), 'boundary_attacks/anomaly_classifier.pt')

if __name__ == '__main__':
    main()

