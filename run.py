import subprocess

def get_classification_settings():
    """
    Function to ask the user for classification settings and return them.
    """
    print("Please provide the following details for the classification task:")
    
    # Prompt user for input
    account_name = input("Enter the account to use (e.g., PAS2903): ")
    conda_location = input("Enter the location of conda (e.g., path to conda): ")
    environment_name = input("Enter the name of conda environment: ")
    process_time = input("Enter the number of minutes as process time (15 or 30): ")
    if process_time == 30:
        process_time = 30
    else:
        process_time = 15
    dataset_name = input("Enter the name of dataset (CIFAR10 or CIFAR100): ")
    if dataset_name == "CIFAR100":
        dataset_name = "CIFAR100"
    else:
        dataset_name = "CIFAR10"
    model_name = input("Enter the model to use (ResNet18 or ResNet152): ")
    if model_name == "ResNet152":
        model_name = "ResNet152"
    else:
        model_name = "ResNet18"
    optimizer_name = input("Enter the optimizer to use (SGD or Adam): ")
    if optimizer_name == "Adam":
        optimizer_name = "Adam"
    else:
        optimizer_name = "SGD"
    learning_rate = float(input("Enter the learning rate (0.01 or 0.02): "))
    if learning_rate == 0.01:
        learning_rate = 0.01
    else:
        learning_rate = 0.02
    batch_size = int(input("Enter the batch size (512 or 1024): "))
    if batch_size == 512:
        batch_size = 512
    else:
        batch_size = 1024
    epoch = int(input("Enter the number of epoch (20 or 50): "))
    if epoch == 50:
        epoch = 50
    else:
        epoch = 20

    # Store all values in a dictionary
    settings = {
        "account_name": account_name,
        "conda_location": conda_location,
        "environment_name": environment_name,
        "process_time": process_time,
        "dataset_name": dataset_name,
        "model_name": model_name,
        "optimizer_name": optimizer_name,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epoch": epoch
    }
    create_sbatch_file(settings)

def create_sbatch_file(settings):

    sbatch_command = f"""#!/bin/bash

    source {settings['conda_location']}
    conda activate {settings['environment_name']}

    python cnn.py \
    --batch_size {settings['batch_size']} \
    --learning_rate {settings['learning_rate']} \
    --model_name {settings['model_name']} \
    --dataset_name {settings['dataset_name']} \
    --optimizer_name {settings['optimizer_name']} \
    --epochs {settings['epoch']}
    """

    # Write the sbatch script to the file
    with open("node.sbatch", "w") as sbatch_file:
        sbatch_file.write(sbatch_command)

    formatted_process_time = f"0:{settings['process_time']}:0"
    
    # Submit the sbatch job
    subprocess.run(["sbatch", "-N", "1", "--gpus-per-node=1", "-t", formatted_process_time, "-A", settings['account_name'], "node.sbatch"])
    
# Example usage
if __name__ == "__main__":
    get_classification_settings()
