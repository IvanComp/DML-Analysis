import subprocess
import itertools
import argparse
import os

# Configuration
DATASETS = ['cifar10', 'stl10', 'mnist', 'oxfordpet']
MODELS = ['cnn', 'squeezenet', 'shufflenet', 'resnet']
NUM_CLIENTS = 2
DATA_DISTR = 0.5
DEFAULT_ROUNDS = 10
DEFAULT_EPOCHS = 1
DEFAULT_BASELINE = ['all']

def run_experiment(dataset, model, baseline, rounds, epochs, num_clients, data_distr):
    command = [
        "python3", "flower_baseline.py",
        "--dataset", dataset,
        "--model", model,
        "--baseline"
    ] + baseline + [
        "--rounds", str(rounds),
        "--epochs", str(epochs),
        "--num_clients", str(num_clients),
        "--data-distr", str(data_distr)
    ]
    
    print(f"\n" + "="*60)
    print(f"RUNNING: Dataset={dataset}, Model={model}, Baseline={baseline}")
    print(f"COMMAND: {' '.join(command)}")
    print("="*60 + "\n")
    
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running experiment for {dataset}/{model}: {e}")
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user. Exiting...")
        exit(0)

def main():
    parser = argparse.ArgumentParser(description='Automated Experiment Runner for Flower Baselines')
    parser.add_argument('--baseline', type=str, nargs='+', default=DEFAULT_BASELINE, help='Aggregation strategies to run')
    parser.add_argument('--rounds', type=int, default=DEFAULT_ROUNDS, help='Number of rounds per experiment')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS, help='Number of local epochs')
    parser.add_argument('--num_clients', type=int, default=NUM_CLIENTS, help='Number of clients')
    parser.add_argument('--data-distr', type=float, default=DATA_DISTR, help='Alpha for Dirichlet distribution')
    
    args = parser.parse_args()
    
    # Run all combinations
    combinations = list(itertools.product(DATASETS, MODELS))
    total = len(combinations)
    
    print(f"Starting {total} experiments...")
    
    for i, (dataset, model) in enumerate(combinations):
        print(f"\nExperiment {i+1} of {total}")
        run_experiment(
            dataset=dataset,
            model=model,
            baseline=args.baseline,
            rounds=args.rounds,
            epochs=args.epochs,
            num_clients=args.num_clients,
            data_distr=args.data_distr
        )

if __name__ == "__main__":
    main()
