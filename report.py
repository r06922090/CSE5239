import csv
import re
import os

def extract_info_from_slurm_file(slurm_file):
    """
    Extract relevant information (batch size, learning rate, model, dataset, optimizer,
    epochs, test loss, test accuracy, and test time for the last epoch or the last available data) 
    from the SLURM output file.
    """
    info = {}

    # Regular expressions for extracting information
    batch_size_pattern = r"Batch Size: (\d+)"
    learning_rate_pattern = r"Learning Rate: (\d+\.\d+)"
    model_pattern = r"Model: (\w+)"
    dataset_pattern = r"Dataset: (\w+)"
    optimizer_pattern = r"Optimizer: (\w+)"
    epochs_pattern = r"Epochs: (\d+)"
    test_loss_pattern = r"Test Loss: ([\d\.]+)"
    test_acc_pattern = r"Test Acc: ([\d\.]+)"
    test_time_pattern = r"Average Time per Image during Testing: ([\d\.]+)s"  # Capture test time per image in seconds

    # Read the SLURM output file
    with open(slurm_file, 'r') as f:
        content = f.read()

        # Extracting information using regex
        info["batch_size"] = re.search(batch_size_pattern, content).group(1) if re.search(batch_size_pattern, content) else None
        info["learning_rate"] = re.search(learning_rate_pattern, content).group(1) if re.search(learning_rate_pattern, content) else None
        info["model"] = re.search(model_pattern, content).group(1) if re.search(model_pattern, content) else None
        info["dataset"] = re.search(dataset_pattern, content).group(1) if re.search(dataset_pattern, content) else None
        info["optimizer"] = re.search(optimizer_pattern, content).group(1) if re.search(optimizer_pattern, content) else None
        info["epochs"] = re.search(epochs_pattern, content).group(1) if re.search(epochs_pattern, content) else None
        
        # Extracting test results for all epochs and checking for early stopping
        test_loss_matches = re.findall(test_loss_pattern, content)
        test_acc_matches = re.findall(test_acc_pattern, content)
        test_time_matches = re.findall(test_time_pattern, content)

        # Ensure there is test loss and accuracy data
        if test_loss_matches and test_acc_matches:
            info["test_loss"] = test_loss_matches[-1]  # Get the last test loss
            info["test_acc"] = test_acc_matches[-1]  # Get the last test accuracy
        
        # Extract test time for the last epoch (final epoch, or the last available data)
        last_epoch = int(info["epochs"]) if info["epochs"] else 0  # Default to 0 if not found
        info["test_time"] = None

        # Check for the last test time in the logs
        if test_time_matches:
            info["test_time"] = test_time_matches[-1]  # Get the last test time

    return info

def append_to_csv(csv_file, slurm_file):
    """
    Extract information from the SLURM output file and append it to a CSV file,
    checking for duplicate job IDs.
    """
    # Extract the relevant information from the SLURM file
    info = extract_info_from_slurm_file(slurm_file)
    
    # Check if the extracted information is valid
    if None in info.values():
        print("Warning: Some data was missing from the SLURM file: {}".format(slurm_file))
        return

    # Check if the CSV file exists
    file_exists = os.path.isfile(csv_file)

    # Read existing job IDs from the CSV to avoid duplicates
    existing_job_ids = set()
    if file_exists:
        with open(csv_file, mode='r', newline='') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header row
            existing_job_ids = {row[0] for row in reader}

    # Extract job ID from the SLURM file name (e.g., slurm-34479662.out)
    job_id = os.path.basename(slurm_file).split('.')[0]
    
    # Check if job ID is already in the CSV file
    if job_id in existing_job_ids:
        print("Duplicate Job ID detected: {}. Skipping.".format(job_id))
        return

    # Open the CSV file in append mode
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)

        # If the file does not exist, write the header
        if not file_exists:
            writer.writerow(["Job ID", "Batch Size", "Learning Rate", "Model", "Dataset", "Optimizer", "Epochs", "Test Loss", "Test Accuracy", "Test Time (Last Epoch)"])

        # Write the extracted data to the CSV
        writer.writerow([job_id, info["batch_size"], info["learning_rate"], info["model"], info["dataset"], 
                         info["optimizer"], info["epochs"], info["test_loss"], info["test_acc"], info["test_time"]])

    print("Data from {} has been successfully appended to {}".format(slurm_file, csv_file))

def process_all_slurm_files(directory, csv_file):
    """
    Process all SLURM output files in the given directory and append their data to a CSV file.
    """
    highest_acc = -1  # Initialize with a low value for comparison
    highest_acc_info = {}

    # List all files in the directory
    slurm_files = [f for f in os.listdir(directory) if f.startswith("slurm-") and f.endswith(".out")]
    
    for slurm_file in slurm_files:
        slurm_file_path = os.path.join(directory, slurm_file)
        
        # Extract info from SLURM file
        info = extract_info_from_slurm_file(slurm_file_path)
        
        # If test accuracy is higher than the current highest, update highest accuracy and info
        if info.get("test_acc") is not None and float(info["test_acc"]) > highest_acc:
            highest_acc = float(info["test_acc"])
            highest_acc_info = info
        
        append_to_csv(csv_file, slurm_file_path)

    # After processing all files, print the highest accuracy and corresponding information
    if highest_acc_info:
        print("\nHighest Accuracy: {:.2f}%".format(highest_acc))
        print("Batch Size: {}".format(highest_acc_info["batch_size"]))
        print("Learning Rate: {}".format(highest_acc_info["learning_rate"]))
        print("Model: {}".format(highest_acc_info["model"]))
        print("Dataset: {}".format(highest_acc_info["dataset"]))
        print("Optimizer: {}".format(highest_acc_info["optimizer"]))
        print("Epochs: {}".format(highest_acc_info["epochs"]))
        print("Test Loss: {}".format(highest_acc_info["test_loss"]))
        print("Test Time: {}".format(highest_acc_info["test_time"]))

    else:
        print("No valid accuracy data found.")

def main():
    # Directory containing SLURM output files
    directory = './'  # Change to the path where your SLURM files are located
    
    # Specify the CSV file to which data should be appended
    csv_file = 'record.csv'

    # Process all SLURM files in the directory
    process_all_slurm_files(directory, csv_file)

if __name__ == "__main__":
    main()