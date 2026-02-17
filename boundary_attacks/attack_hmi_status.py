from modules import MLP
import torch
import pickle
from hsja import HSJAWithTracking
import pandas as pd

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


def read_data(train_path):

    dataset = pd.read_csv(train_path)
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

    return X, Y


def read_model(model_path):
    anomaly_classifier = MLP(
        **{
            'input_dim': X.shape[1],
            'output_dim': 3,
            'layer_norm': True,
            'mode':'OF'})

    anomaly_classifier.load_state_dict(torch.load(model_path))
    anomaly_classifier.eval()
    
    return anomaly_classifier


def attack(model, scaler_path, X, Y, n_iterations, max_queries, periodic_reprojection):
    
    print("\n" + "="*70)
    print("ESECUZIONE ATTACCO HSJA")
    print("="*70)

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    x_start = X.mean(0)
    model.eval()
    hsja = HSJAWithTracking(model, X, scaler, max_queries=max_queries, verbose=True)
    boundary_points = hsja.attack(x_start, n_iterations=n_iterations, periodic_reprojection=periodic_reprojection)

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
        default='dataset.csv')
    
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

    parser.add_argument(
        '--periodic_reprojection',
        type=int,
        help='Periodic reprojection',
        default=10
    )

    args = parser.parse_args()

    n_iterations = args.n_iterations
    model_path = args.model_path
    train_path = args.train_path
    max_queries = args.max_queries
    scaler_path = args.scaler_path
    periodic_reprojection = args.periodic_reprojection

    print("All arguments:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")

    X, y = read_data(train_path)
    model = read_model(model_path)
    

    attack(model, scaler_path, X, y, n_iterations, max_queries, periodic_reprojection)
    

    