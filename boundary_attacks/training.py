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

columns_to_generate = [
    'Durata','CabEnabled_M1', 'CabEnabled_M8', 'ERTMS_PiastraSts', 'HMI_ACPntSts_T2', 'HMI_ACPntSts_T7', 'HMI_DCPntSts_T2',
    'HMI_DCPntSts_T7', 'HMI_Iline', 'HMI_Irsts_T2', 'HMI_Irsts_T7', 'HMI_VBatt_T2', 'HMI_VBatt_T4', 'HMI_VBatt_T5',
    'HMI_VBatt_T7', 'HMI_Vline', 'HMI_impSIL', 'LineVoltType', 'MDS_LedLimVel', 'MDS_StatoMarcia', '_GPS_LAT',
    '_GPS_LON', 'ldvvelimps', 'ldvveltreno', 'usB1BCilPres_M1', 'usB1BCilPres_M3', 'usB1BCilPres_M6', 'usB1BCilPres_M8',
    'usB1BCilPres_T2', 'usB1BCilPres_T4', 'usB1BCilPres_T5', 'usB1BCilPres_T7', 'usB2BCilPres_M1', 'usB2BCilPres_M3',
    'usB2BCilPres_M6', 'usB2BCilPres_M8', 'usB2BCilPres_T2', 'usB2BCilPres_T4', 'usB2BCilPres_T5', 'usB2BCilPres_T7',
    'usBpPres', 'usMpPres'
]


columns_to_keep = [
    'Durata',
    'ldvveltreno',
    'HMI_Iline',
    'ldvvelimps',
    'MDS_LedLimVel',
    '_GPS_LAT',
    '_GPS_LON',
]


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Training anomaly detector')
    
    parser.add_argument(
        '--epochs',
        type=int, 
        help='Number of training epochs',
        default=10)
    
    args = parser.parse_args()

    epochs = args.epochs

    dataset_new = pd.read_csv('boundary_attacks/anomalies_ds.csv')
    dataset_new['Timestamp'] = pd.to_datetime(dataset_new['Timestamp'], errors='coerce')

    event_occurrences = dataset_new['Descrizione'].value_counts()
    eventi_totali = len(event_occurrences)

    print(f'nel nostro dataset abbiamo {eventi_totali} TIPI DI EVENTI diversi')
    record_totali = len(dataset_new)
    print(f'sparsi in un totale di {record_totali} record')

    record_anomalie = len(dataset_new[dataset_new['anomaly']])
    print(f'di cui {record_anomalie} corrispondono ad anomalie')

    X = dataset_new[columns_to_generate].values

    anomaly_classifier = MLP(
        **{
            'input_dim': X.shape[1],
            'output_dim': 2,
            'layer_norm': True,
            'mode':'OF'})


    Y = dataset_new['anomaly'].astype(int).values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    train_scaler = MinMaxScaler()
    X_train_scaled = train_scaler.fit_transform(X_train)
    X_test_scaled = train_scaler.transform(X_test)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(anomaly_classifier.parameters(), lr=0.001)

    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train_scaled).float(),
        torch.tensor(Y_train).long())
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


    all_X_scaled = train_scaler.transform(X)
    all_X_scaled = torch.Tensor(all_X_scaled).float()
    anomaly_classifier.eval()
    all_preds, manifold = anomaly_classifier(all_X_scaled)
    all_preds = (torch.sigmoid(output.squeeze()) > 0.5).long()
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

