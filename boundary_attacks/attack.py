from modules import MLP
import torch
import pickle
from hsja import HSJAWithTracking

columns_to_generate = [
    'Durata','CabEnabled_M1', 'CabEnabled_M8', 'ERTMS_PiastraSts', 'HMI_ACPntSts_T2', 'HMI_ACPntSts_T7', 'HMI_DCPntSts_T2',
    'HMI_DCPntSts_T7', 'HMI_Iline', 'HMI_Irsts_T2', 'HMI_Irsts_T7', 'HMI_VBatt_T2', 'HMI_VBatt_T4', 'HMI_VBatt_T5',
    'HMI_VBatt_T7', 'HMI_Vline', 'HMI_impSIL', 'LineVoltType', 'MDS_LedLimVel', 'MDS_StatoMarcia', '_GPS_LAT',
    '_GPS_LON', 'ldvvelimps', 'ldvveltreno', 'usB1BCilPres_M1', 'usB1BCilPres_M3', 'usB1BCilPres_M6', 'usB1BCilPres_M8',
    'usB1BCilPres_T2', 'usB1BCilPres_T4', 'usB1BCilPres_T5', 'usB1BCilPres_T7', 'usB2BCilPres_M1', 'usB2BCilPres_M3',
    'usB2BCilPres_M6', 'usB2BCilPres_M8', 'usB2BCilPres_T2', 'usB2BCilPres_T4', 'usB2BCilPres_T5', 'usB2BCilPres_T7',
    'usBpPres', 'usMpPres'
]

def read_data(train_path):
    import pandas as pd

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
    Y = dataset_new['anomaly'].astype(int).values

    return X, Y

def read_model(model_path):
    anomaly_classifier = MLP(
        **{
            'input_dim': X.shape[1],
            'output_dim': 2,
            'layer_norm': True,
            'mode':'OF'})

    anomaly_classifier.load_state_dict(torch.load(model_path))
    anomaly_classifier.eval()
    
    return anomaly_classifier


def attack(model, scaler_path, X, Y, n_iterations, max_queries):
    
    print("\n" + "="*70)
    print("ESECUZIONE ATTACCO HSJA")
    print("="*70)

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    x_start = X.mean(0)
    model.eval()
    hsja = HSJAWithTracking(model, X, scaler, max_queries=max_queries, verbose=True)
    boundary_points = hsja.attack(x_start, n_iterations=n_iterations)

    return boundary_points


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Attack boundary using HSJA')
    
    parser.add_argument(
        '--n_iterations',
        type=int, 
        help='Number of iterations for the attack',
        default=100)
    
    parser.add_argument(
        '--model_path', 
        type=str, 
        help='Path to the model',
        default='boundary_attacks/anomaly_classifier.pt')
    
    parser.add_argument(
        '--train_path', 
        type=str, 
        help='Path to the training data',
        default='boundary_attacks/anomalies_ds.csv')
    
    parser.add_argument(
        '--scaler_path', 
        type=str, 
        help='Path to the scaler',
        default='boundary_attacks/training_scaler.pkl')
    

    parser.add_argument(
        '--max_queries',
        type=int,
        help='Maximum number of queries to perform',
        default=500)

    args = parser.parse_args()

    n_iterations = args.n_iterations
    model_path = args.model_path
    train_path = args.train_path
    max_queries = args.max_queries
    scaler_path = args.scaler_path

    X, y = read_data(train_path)
    model = read_model(model_path)
    

    attack(model, scaler_path, X, y, n_iterations, max_queries)
    

    