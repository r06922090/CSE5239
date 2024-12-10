import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import os
import argparse

def main(args):
    # Transform definitions
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Load dataset
    if args.dataset_name == "CIFAR100":
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    else:
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    # DataLoader setup
    batch_size = args.batch_size
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Model selection
    if args.model_name == "ResNet152":
        net = torchvision.models.resnet152(pretrained=True)
    else:
        net = torchvision.models.resnet18(pretrained=True)

    net = net.cuda()

    # Optimizer and loss function
    learning_rate = args.learning_rate
    if args.optimizer_name == "Adam":
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    else:
        optimizer = optim.SGD(net.parameters(), lr=learning_rate)

    criterion = nn.CrossEntropyLoss()
    
    print("Training Configuration:")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset_name}")
    print(f"Optimizer: {args.optimizer_name}")
    print(f"Epochs: {args.epochs}")

    patience = 5  # Number of epochs with no improvement to wait before stopping
    best_test_loss = float('inf')
    epochs_without_improvement = 0

    # Training and testing loop
    for epoch in range(args.epochs):
        # Training loop
        net.train()  # Set the model to training mode
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.cuda(), labels.cuda()  # Move data to GPU
            optimizer.zero_grad()  # Zero out the gradients
            outputs = net(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update the model parameters

            # Track the loss and accuracy
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        # Print training stats
        train_loss_avg = train_loss / (batch_idx+1)
        train_acc = 100. * correct / total
        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss_avg} - Train Acc: {train_acc}")

        # Testing loop
        net.eval()  # Set the model to evaluation mode
        test_loss = 0
        correct = 0
        total = 0
        total_time = 0
        with torch.no_grad():  # Disable gradient tracking for evaluation
            for batch_idx, (inputs, labels) in enumerate(testloader):
                start_time = time.time()  # Start time for this batch
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                end_time = time.time()  # End time for this batch
                batch_time = end_time - start_time
                total_time += batch_time

        # Calculate average test loss and accuracy
        test_loss_avg = test_loss / (batch_idx+1)
        test_acc = 100. * correct / total
        print(f"Test Loss: {test_loss_avg} - Test Acc: {test_acc}")
        
        # Calculate total time per image for the entire test phase
        avg_time_per_image = total_time / (len(testloader) * batch_size)
        print(f"Average Time per Image during Testing: {avg_time_per_image:.6f}s")

        # Check for early stopping
        if test_loss_avg < best_test_loss:
            best_test_loss = test_loss_avg
            epochs_without_improvement = 0  # Reset counter if we improve
        else:
            epochs_without_improvement += 1

        # If no improvement for 'patience' epochs, stop training
        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch+1} as validation loss hasn't improved.")
            break

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train a classification model on CIFAR')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, required=True, help='Learning rate for the optimizer')
    parser.add_argument('--model_name', type=str, required=True, choices=['ResNet18', 'ResNet152'], help='Model architecture')
    parser.add_argument('--dataset_name', type=str, required=True, choices=['CIFAR10', 'CIFAR100'], help='Dataset to use')
    parser.add_argument('--optimizer_name', type=str, required=True, choices=['SGD', 'Adam'], help='Optimizer to use')
    parser.add_argument('--epochs', type=int, required=True, help='Number of training epochs')

    # Parse the arguments
    args = parser.parse_args()

    # Call the main function
    main(args)