import warnings
warnings.filterwarnings("ignore")
import torch
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, Subset
from torch import nn
from torch import optim
import torchvision.datasets as datasets
import os
import random
import numpy as np
from models import CustomResNet
from dataset import RemappedDataset

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the data transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load the ImageNet dataset
    full_dataset = datasets.ImageFolder(os.path.join('./data/tiny-imagenet-200', 'train'), transform=transform)
    all_classes = full_dataset.classes
    selected_classes = random.sample(all_classes, 10)
    class_to_idx = full_dataset.class_to_idx
    selected_class_indices = [class_to_idx[cls] for cls in selected_classes]
    new_label_mapping = {original: new for new, original in enumerate(selected_class_indices)}

    selected_indices = [idx for idx, (_, class_idx) in enumerate(full_dataset.samples) if class_idx in selected_class_indices]
    subset_dataset = Subset(full_dataset, selected_indices)
    remapped_dataset = RemappedDataset(subset_dataset, new_label_mapping)  # Apply the label remapping

    # Create DataLoader for the remapped dataset
    train_loader = DataLoader(remapped_dataset, batch_size=32, shuffle=True, num_workers=4)

    # Initialize the ResNet50 model with pre-trained weights
    print('######################################')
    print('Load pretrained model:')
    pretrained_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2).to(device)
    model = CustomResNet(pretrained_model, num_classes=10, feature_size=32).to(device)
    

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    train = False
    if train:
        num_epochs = 20  # Define the number of epochs

        # Train the model
        print('######################################')
        print('Start tuning:')
        train_features = []  # List to store features
        train_logits = []  # List to store logits (for softmax scores)
        train_labels = []  # List to store labels
        
        for epoch in range(num_epochs):
            model.train()
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                features, logits = model(inputs)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                train_features.append(features.detach().cpu().numpy())  # Store features
                train_logits.append(logits.detach().cpu().numpy())  # Convert logits to softmax scores and store
                train_labels.append(labels.cpu().numpy())  # Store labels

            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')
        # Save the trained model
        print('######################################')
        print('Store tuned model:')
        torch.save(model.state_dict(), 'resnet50_imagenet.pth')

        # Concatenate all features, logits (softmax scores), and labels
        train_features_array = np.concatenate(train_features, axis=0)
        train_logits_array = np.concatenate(train_logits, axis=0)
        train_labels_array = np.concatenate(train_labels, axis=0)

        # Combine features, logits (softmax scores), and labels for CSV saving
        combined_array = np.hstack((train_features_array, train_logits_array, train_labels_array.reshape(-1, 1)))

        # Save to CSV file
        print('######################################')
        print('Saving training features, logits (softmax scores), and labels to CSV:')
        np.savetxt("train_features_logits_labels.csv", combined_array, delimiter=",", fmt='%f')
    else:
        model_state_path = 'resnet50_imagenet.pth'
        model_state = torch.load(model_state_path, map_location=device)
        model.load_state_dict(model_state)
        print('######################################')
        print('Load tuned model:')


    


    # Load the test ImageNet dataset
    test_dataset = datasets.ImageFolder(os.path.join('./data/tiny-imagenet-200', 'test'), transform=transform)
    test_indices = [idx for idx, (path, class_idx) in enumerate(test_dataset.samples) if test_dataset.classes[class_idx] in selected_classes]
    print(f"Number of matched test samples: {len(test_indices)}")
    test_subset = Subset(test_dataset, test_indices)
    print(f"New label mapping: {new_label_mapping}")
    remapped_test_dataset = RemappedDataset(test_subset, new_label_mapping)  
    test_loader = DataLoader(remapped_test_dataset, batch_size=32, shuffle=False, num_workers=4)


    # Assuming the continuation from the previous script
    print('######################################')
    print('Start testing:')
    # Set the model to evaluation mode
    model.eval()

    test_features = []  # List to store features
    test_logits = []  # List to store logits (for softmax scores)
    test_labels = []  # List to store labels

    # No need to track gradients here
    with torch.no_grad():
        for inputs, labels in test_loader:
            print("Testing loop entered.")
            inputs, labels = inputs.to(device), labels.to(device)
            features, logits = model(inputs)

            test_features.append(features.cpu().numpy())  # Store features
            test_logits.append(logits.cpu().numpy())  # Convert logits to softmax scores and store
            test_labels.append(labels.cpu().numpy())  # Store labels

    
    # Concatenate all features, logits (softmax scores), and labels
    test_features_array = np.concatenate(test_features, axis=0)
    test_logits_array = np.concatenate(test_logits, axis=0)
    test_labels_array = np.concatenate(test_labels, axis=0)
    
    # Optionally, calculate and print test accuracy
    correct_predictions = np.sum(np.argmax(test_logits_array, axis=1) == test_labels_array)
    total_samples = test_labels_array.shape[0]
    test_accuracy = correct_predictions / total_samples
    print(f'Test Accuracy: {test_accuracy:.4f}')


    # Combine features, logits (softmax scores), and labels for CSV saving
    combined_test_array = np.hstack((test_features_array, test_logits_array, test_labels_array.reshape(-1, 1)))

    # Save to CSV file
    print('######################################')
    print('Saving test features, logits (softmax scores), and labels to CSV:')
    np.savetxt("test_features_logits_labels.csv", combined_test_array, delimiter=",", fmt='%f')



if __name__ == '__main__':
    main()
