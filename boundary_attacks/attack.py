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
    
    parser.add_argument('--max_queries', type=int, help='Maximum number of queries to perform')

    args = parser.parse_args()

    n_iterations = args.n_iterations
    model_path = args.model_path
    train_path = args.train_path
    max_queries = args.max_queries
