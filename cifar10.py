# cp train.csv /home/rivachen/OOD_learning_GP_models_in_R/data_32/ImageNet/
import warnings
warnings.filterwarnings("ignore")
import torch
from torchvision import models, transforms, datasets
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, Subset
from torch import nn
from torch import optim
import torchvision.datasets as datasets
import os
import numpy as np
from torch.utils.data import random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from models.densenet import *
from dataset import *
import torchvision
from tqdm import tqdm


torch.manual_seed(42)
np.random.seed(42)

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)


#############################s#####  Train   ###########################################################

    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4),
                                   transforms.ToTensor(), transforms.Normalize(mean, std)])
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    train_set = torchvision.datasets.CIFAR10('./Datasets/CIFAR-10',
                              train=True, transform=train_transform, download=True)
    test_set = torchvision.datasets.CIFAR10('./Datasets/CIFAR-10',
                             train=False, transform=test_transform, download=True)
    num_classes = 10
    num_channels = 3

    total_size = len(train_set)
    train_ratio = 0.8
    val_ratio = 0.2

    # Calculate sizes for each split
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)

    # Perform the split
    train_dataset, validation_dataset = random_split(train_set, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    validation_loader = DataLoader(validation_dataset, batch_size=256, shuffle=False, num_workers=4)

    n_channels = 3
    n_features=64
    model = DenseNet3(100, 10, 3, feature_size=n_features).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    lr=0.1
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001, nesterov=True)

    ckpt_dir = os.path.join('ckpt', f'cifar10-{n_features}')
    os.makedirs(ckpt_dir, exist_ok=True)

    train = True
    if train:
        num_epochs = 100  # Define the number of epochs
        # Train the model
        print('######################################')
        print('Start training:')
        
        for epoch in tqdm(range(num_epochs)):
            model.train()
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                # inputs = inputs.view(-1, 3, 32, 32)
                inputs = inputs.view(-1, 3, 32, 32)
                optimizer.zero_grad()
                features, logits = model(inputs)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
            train_loss = loss.item()

            model.eval()
            validation_loss = 0.0
            with torch.no_grad():
                for inputs, labels in validation_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    inputs = inputs.view(-1, 3, 32, 32)
                    features, logits = model(inputs)
                    loss = criterion(logits, labels)
                    validation_loss += loss.item()
            validation_loss /= len(validation_loader)
            

            # scheduler.step(validation_loss)

            print(f'Epoch {epoch+1}, Training Losss: {train_loss}, Validation Loss: {validation_loss}')

            # Update learning rate
            if epoch == 49:
                optimizer.param_groups[0]['lr'] *= lr * 0.1
            # elif epoch == 74:
            #     optimizer.param_groups[0]['lr'] *= lr * 0.01
            elif epoch == 89:
                optimizer.param_groups[0]['lr'] *= lr * 0.01

        # Save the trained model
        print('######################################')
        print('Store trained model:')
        torch.save(model.state_dict(), os.path.join(ckpt_dir, 'desnet_cifar10.pth'))
    else:
        print('######################################')
        print('Load model:')
        model_state_path = os.path.join(ckpt_dir, 'desnet_cifar10.pth')
        model_state = torch.load(model_state_path, map_location=device)
        model.load_state_dict(model_state)
        print('Model loaded successfully')

##################################  Train   ############################################################

    # test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=4)
    # Assuming the continuation from the previous script
    print('######################################')
    print('Re-saving training data:')
    # Set the model to evaluation mode
    model.eval()

    train_features = []  # List to store features
    train_logits = []  # List to store logits (for softmax scores)
    train_labels = []  # List to store labels

    # No need to track gradients here
    with torch.no_grad():
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.view(-1, 3, 32, 32)
            features, logits = model(inputs)

            train_features.append(features.cpu().numpy())  # Store features
            train_logits.append(logits.cpu().numpy())  # Convert logits to softmax scores and store
            train_labels.append(labels.cpu().numpy())  # Store labels

    
    # Concatenate all features, logits (softmax scores), and labels
    train_features_array = np.concatenate(train_features, axis=0)
    train_logits_array = np.concatenate(train_logits, axis=0)
    train_labels_array = np.concatenate(train_labels, axis=0)
    train_labels_array = train_labels_array.reshape(-1, 1)
    
    # Optionally, calculate and print test accuracy
    correct_predictions = np.sum(np.argmax(train_logits_array, axis=1) == train_labels_array.squeeze())
    A = np.argmax(train_logits_array, axis=1)
    # print(A.shape, correct_predictions)
    # print(test_labels_array.shape, test_logits_array.shape, test_labels_array.shape[0])
    total_samples = train_labels_array.shape[0]
    train_accuracy = correct_predictions / total_samples
    print(f'Train Accuracy: {train_accuracy:.4f}')

    n_train = 50000
    train_features_array = train_features_array[:n_train, :]
    train_logits_array = train_logits_array[:n_train, :]
    train_labels_array = train_labels_array[:n_train]

    # Generate column names
    feature_names = [f'feature_{i+1}' for i in range(n_features)]
    logit_names = [f'logit_{i+1}' for i in range(10)]
    label_name = ['label']
    column_names = feature_names + logit_names + label_name

    # Create a header string with column names
    header_string = ','.join(column_names)
    # Combine features, logits (softmax scores), and labels for CSV saving
    combined_train_array = np.hstack((train_features_array, train_logits_array, train_labels_array.reshape(-1, 1)))

    # Save to CSV file
    print('######################################')
    print('Saving train features, logits (softmax scores), and labels to CSV:')
    save_path = os.path.join(ckpt_dir, "train_features_logits_labels.csv")
    np.savetxt(save_path, combined_train_array, delimiter=",", fmt='%f', header=header_string, comments='')


##################################  Test   ############################################################

    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=4)
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
        for inputs, labels in tqdm(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.view(-1, 3, 32, 32)
            features, logits = model(inputs)

            test_features.append(features.cpu().numpy())  # Store features
            test_logits.append(logits.cpu().numpy())  # Convert logits to softmax scores and store
            test_labels.append(labels.cpu().numpy())  # Store labels

    
    # Concatenate all features, logits (softmax scores), and labels
    test_features_array = np.concatenate(test_features, axis=0)
    test_logits_array = np.concatenate(test_logits, axis=0)
    test_labels_array = np.concatenate(test_labels, axis=0)
    test_labels_array = test_labels_array.reshape(-1, 1)
    
    # Optionally, calculate and print test accuracy
    correct_predictions = np.sum(np.argmax(test_logits_array, axis=1) == test_labels_array.squeeze())
    A = np.argmax(test_logits_array, axis=1)
    # print(A.shape, correct_predictions)
    # print(test_labels_array.shape, test_logits_array.shape, test_labels_array.shape[0])
    total_samples = test_labels_array.shape[0]
    test_accuracy = correct_predictions / total_samples
    print(f'Test Accuracy: {test_accuracy:.4f}')

    n_test = 5000
    test_features_array = test_features_array[:n_test, :]
    test_logits_array = test_logits_array[:n_test, :]
    test_labels_array = test_labels_array[:n_test]

    # Generate column names
    feature_names = [f'feature_{i+1}' for i in range(n_features)]
    logit_names = [f'logit_{i+1}' for i in range(10)]
    label_name = ['label']
    column_names = feature_names + logit_names + label_name

    # Create a header string with column names
    header_string = ','.join(column_names)
    # Combine features, logits (softmax scores), and labels for CSV saving
    combined_test_array = np.hstack((test_features_array, test_logits_array, test_labels_array.reshape(-1, 1)))

    # Save to CSV file
    print('######################################')
    print('Saving test features, logits (softmax scores), and labels to CSV:')
    save_path = os.path.join(ckpt_dir, "test_features_logits_labels.csv")
    np.savetxt(save_path, combined_test_array, delimiter=",", fmt='%f', header=header_string, comments='')


##################################  OOD Datasets   ############################################################

# INaturalist
    # print('######################################')
    # print('Testing on INaturalist')
    # inaturalist_loader = INaturalistDataLoader(root_dir='./inaturalist/data', batch_size=32, download=True).get_data_loader()
    # model.eval()
    # ood_features = []  # List to store features
    # ood_logits = []  # List to store logits (for softmax scores)
    # ood_labels = []  # List to store labels
    # print('Start testing')
    # with torch.no_grad():
    #     batch_counter = 0
    #     for inputs, labels in inaturalist_loader:
    #         if batch_counter >= 300:  # Check if 200 batches have been processed
    #             break 
    #         inputs, labels = inputs.to(device), labels.to(device)
    #         inputs = inputs.view(-1, 3, 32, 32)
    #         features, logits = model(inputs)

    #         ood_features.append(features.cpu().numpy())  # Store features
    #         ood_logits.append(logits.cpu().numpy())  # Convert logits to softmax scores and store
    #         # ood_labels.append(labels.cpu().numpy())  # Store labels
    #         batch_counter += 1

    
    # # Concatenate all features, logits (softmax scores), and labels
    # ood_features_array = np.concatenate(ood_features, axis=0)[:2000, :]
    # ood_logits_array = np.concatenate(ood_logits, axis=0)[:2000, :]
    # ood_labels_array = np.full((2000, 1), 10)

    # combined_features = np.concatenate([test_features_array, ood_features_array], axis=0)
    # combined_logits = np.concatenate([test_logit_array, ood_logits_array], axis=0)
    # combined_labels = np.concatenate([test_labels_array, ood_labels_array], axis=0)

    # tsne = TSNE(n_components=2, random_state=42)
    # tsne_results = tsne.fit_transform(combined_features) 
    # combined_array = np.hstack((combined_features, combined_logits, combined_labels))

    # print('Getting tSNE features')
    # # Plotting
    # plt.figure(figsize=(10, 8))
    # scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=combined_labels_for_plot[:, 0], cmap='viridis', alpha=0.6)
    # plt.colorbar(scatter, label='Labels')
    # plt.title('t-SNE of Combined Test and OOD Data')
    # plt.xlabel('tSNE1')
    # plt.ylabel('tSNE2')

    # plt.savefig("INaturalist.png")

    


    # print('######################################')
    # print('Saving data for INaturalist to CSV:')

    # column_names = [f"feature_{i+1}" for i in range(32)] + [f"logit_{i+1}" for i in range(10)] + ["label"]
    # df = pd.DataFrame(combined_array, columns=column_names)
    # df['tSNE'] = tsne_results[:, 0]
    # df['tSNE'] = tsne_results[:, 1]
    # df['class'] = ['test']*2000 + ['OOD']*2000

    # df.to_csv("INaturalist_test.csv", index=False)
    
## SUN


    # print('######################################')
    # print('Testing on SUN')
    # sun397_loader = SUN397DataLoader(root_dir='./data/sun397', batch_size=32, download=False).get_data_loader()
    # model.eval()
    # ood_features = []  # List to store features
    # ood_logits = []  # List to store logits (for softmax scores)
    # ood_labels = []  # List to store labels
    # print('Start testing')
    # with torch.no_grad():
    #     batch_counter = 0
    #     for inputs, labels in sun397_loader:
    #         if batch_counter >= 100:  # Check if 200 batches have been processed
    #             break 
    #         inputs, labels = inputs.to(device), labels.to(device)
    #         inputs = inputs.view(-1, 3, 32, 32)
    #         features, logits = model(inputs)

    #         ood_features.append(features.cpu().numpy())  # Store features
    #         ood_logits.append(logits.cpu().numpy())  # Convert logits to softmax scores and store
    #         # ood_labels.append(labels.cpu().numpy())  # Store labels
    #         batch_counter += 1

    
    # # Concatenate all features, logits (softmax scores), and labels
    # ood_features_array = np.concatenate(ood_features, axis=0)[:2000, :]
    # ood_logits_array = np.concatenate(ood_logits, axis=0)[:2000, :]
    # ood_labels_array = np.full((2000, 1), 10)

    # combined_features = np.concatenate([test_features_array, ood_features_array], axis=0)
    # print(test_logits_array.shape, ood_logits_array.shape)
    # combined_logits = np.concatenate([test_logits_array, ood_logits_array], axis=0)
    # print(test_labels_array.shape, ood_labels_array.shape)
    # combined_labels = np.concatenate([test_labels_array, ood_labels_array], axis=0)
    # print(combined_features.shape, combined_logits.shape, combined_labels.shape)
    # combined_array = np.hstack((combined_features, combined_logits, combined_labels))

    # tsne = TSNE(n_components=2, random_state=42)
    # tsne_results = tsne.fit_transform(combined_features) 

    # print('Getting tSNE features')
    # # Plotting
    # plt.figure(figsize=(10, 8))
    # scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=combined_labels[:, 0], cmap='viridis', alpha=0.6)
    # plt.colorbar(scatter, label='Labels')
    # plt.title('t-SNE of Combined Test and OOD Data')
    # plt.xlabel('tSNE1')
    # plt.ylabel('tSNE2')

    # plt.savefig("SUN.png")


    # print('######################################')
    # print('Saving data for SUN to CSV:')

    # column_names = [f"feature_{i+1}" for i in range(32)] + [f"logit_{i+1}" for i in range(10)] + ["label", "tSNE1", "tSNE2", "class"]
    # class_labels = ['test'] * 2000 + ['OOD'] * 2000  # Adjust the numbers as needed

    # combined_data_with_tsne = np.hstack((combined_array, tsne_results, np.array(class_labels).reshape(-1, 1)))

    # with open("SUN_test.csv", "w") as file:
    #     file.write(",".join(column_names) + "\n")
        
    #     for row in combined_data_with_tsne:
    #         str_row = [str(item) for item in row]  # Convert all elements to strings
    #         file.write(",".join(str_row) + "\n")

## Places

    # places365_loader = Places365DataLoader(root_dir='./data/places365', batch_size=32, download=True).get_data_loader()



## DTD

    # print('######################################')
    # print('Testing on DTD')
    # dtd_loader = DTDDataLoader(root_dir='./data/dtd', batch_size=32, download=False).get_data_loader()
    # model.eval()
    # ood_features = []  # List to store features
    # ood_logits = []  # List to store logits (for softmax scores)
    # ood_labels = []  # List to store labels
    # print('Start testing')
    # with torch.no_grad():
    #     batch_counter = 0
    #     for inputs, labels in dtd_loader:
    #         if batch_counter >= 100:  # Check if 200 batches have been processed
    #             break 
    #         inputs, labels = inputs.to(device), labels.to(device)
    #         inputs = inputs.view(-1, 3, 32, 32)
    #         features, logits = model(inputs)

    #         ood_features.append(features.cpu().numpy())  # Store features
    #         ood_logits.append(logits.cpu().numpy())  # Convert logits to softmax scores and store
    #         # ood_labels.append(labels.cpu().numpy())  # Store labels
    #         batch_counter += 1

    
    # # Concatenate all features, logits (softmax scores), and labels
    # ood_features_array = np.concatenate(ood_features, axis=0)[:2000, :]
    # ood_logits_array = np.concatenate(ood_logits, axis=0)[:2000, :]
    # ood_labels_array = np.full((2000, 1), 10)

    # combined_features = np.concatenate([test_features_array, ood_features_array], axis=0)
    # # print(test_logits_array.shape, ood_logits_array.shape)
    # combined_logits = np.concatenate([test_logits_array, ood_logits_array], axis=0)
    # # print(test_labels_array.shape, ood_labels_array.shape)
    # combined_labels = np.concatenate([test_labels_array, ood_labels_array], axis=0)
    # # print(combined_features.shape, combined_logits.shape, combined_labels.shape)
    # combined_array = np.hstack((combined_features, combined_logits, combined_labels))

    # tsne = TSNE(n_components=2, random_state=42)
    # tsne_results = tsne.fit_transform(combined_features) 

    # print('Getting tSNE features')
    # # Plotting
    # plt.figure(figsize=(10, 8))
    # scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=combined_labels[:, 0], cmap='viridis', alpha=0.6)
    # plt.colorbar(scatter, label='Labels')
    # plt.title('t-SNE of Combined Test and OOD Data')
    # plt.xlabel('tSNE1')
    # plt.ylabel('tSNE2')

    # plt.savefig("DTD.png")


    # print('######################################')
    # print('Saving data for DTD to CSV:')

    # column_names = [f"feature_{i+1}" for i in range(32)] + [f"logit_{i+1}" for i in range(10)] + ["label", "tSNE1", "tSNE2", "class"]
    # class_labels = ['test'] * 2000 + ['OOD'] * 2000  # Adjust the numbers as needed

    # combined_data_with_tsne = np.hstack((combined_array, tsne_results, np.array(class_labels).reshape(-1, 1)))

    # with open("DTD_test.csv", "w") as file:
    #     file.write(",".join(column_names) + "\n")
        
    #     for row in combined_data_with_tsne:
    #         str_row = [str(item) for item in row]  # Convert all elements to strings
    #         file.write(",".join(str_row) + "\n")


    dset = 'SVHN'
    # dset='CIFAR10'
    # dset = 'FashionMNIST'
    # dset = 'ImageNet-c'

    if dset == 'SVHN':
        print('######################################')
        print('Testing on SVHN')
        loader = SVHNDataLoader(root_dir='./data/svhn', batch_size=512, download=True).get_data_loader()
    elif dset == 'CIFAR10':
        print('######################################')
        print('Testing on CIFAR10')
        normalizer = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                      std=[x/255.0 for x in [63.0, 62.1, 66.7]])
        transform = transforms.Compose([transforms.ToTensor(), normalizer])
        train_dataset = datasets.CIFAR10('./Datasets/CIFAR-10', train=True, download=True, transform=transform)
        loader = torch.utils.data.DataLoader(train_dataset, batch_size=512, shuffle=True)
    elif dset == 'FashionMNIST':
        print('######################################')
        print('Testing on FashionMNIST') 
        transform = transforms.Compose([ transforms.Resize((32, 32)), 
                                transforms.Grayscale(num_output_channels=3),
                                transforms.ToTensor()])
        tset = torchvision.datasets.FashionMNIST("./Datasets", download=True, train=True, transform=transform)
        # Get data loader
        loader = torch.utils.data.DataLoader(tset, shuffle=False, batch_size=512)
    elif dset == 'ImageNet-c':
        print('######################################')
        print('Testing on ImageNet-c') 
        transform = transforms.Compose([transforms.RandomCrop(32),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        test_datasets = datasets.ImageFolder(os.path.join('data/Imagenet'), transform=transform) 
        loader = torch.utils.data.DataLoader(test_datasets, batch_size=512, shuffle=True)
    
    # Evaluate features
    model.eval()
    ood_features = []  # List to store features
    ood_logits = []  # List to store logits (for softmax scores)
    ood_labels = []  # List to store labels
    print('Start testing')
    with torch.no_grad():
        batch_counter = 0
        for inputs, labels in tqdm(loader):
            if batch_counter >= 100:  # Check if 200 batches have been processed
                break 
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.view(-1, 3, 32, 32)
            features, logits = model(inputs)

            ood_features.append(features.cpu().numpy())  # Store features
            ood_logits.append(logits.cpu().numpy())  # Convert logits to softmax scores and store
            # ood_labels.append(labels.cpu().numpy())  # Store labels
            batch_counter += 1

    
    # Concatenate all features, logits (softmax scores), and labels
    # n_test = 5000
    ood_features_array = np.concatenate(ood_features, axis=0)[:n_test, :]
    ood_logits_array = np.concatenate(ood_logits, axis=0)[:n_test, :]
    ood_labels_array = np.full((n_test, 1), 10)

    combined_features = np.concatenate([test_features_array, ood_features_array], axis=0)
    combined_logits = np.concatenate([test_logits_array, ood_logits_array], axis=0)
    combined_labels = np.concatenate([test_labels_array, ood_labels_array], axis=0)
    combined_array = np.hstack((combined_features, combined_logits, combined_labels))

    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(combined_features) 

    print('Getting tSNE features')
    # Plotting
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=combined_labels[:, 0], cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Labels')
    plt.title('t-SNE of Combined Test and OOD Data')
    plt.xlabel('tSNE1')
    plt.ylabel('tSNE2')

    plt.savefig(os.path.join(ckpt_dir, f"{dset}.png"))


    print('######################################')
    print(f'Saving data for {dset} to CSV:')

    column_names = [f"feature_{i+1}" for i in range(n_features)] + [f"logit_{i+1}" for i in range(10)] + ["label", "tSNE1", "tSNE2", "class"]
    class_labels = ['test'] * n_test + ['OOD'] * n_test  # Adjust the numbers as needed

    combined_data_with_tsne = np.hstack((combined_array, tsne_results, np.array(class_labels).reshape(-1, 1)))

    with open(os.path.join(ckpt_dir, f"{dset}_test.csv"), "w") as file:
        file.write(",".join(column_names) + "\n")
        
        for row in combined_data_with_tsne:
            str_row = [str(item) for item in row]  # Convert all elements to strings
            file.write(",".join(str_row) + "\n")




if __name__ == '__main__':
    main()


    # sun397_loader = SUN397DataLoader(root_dir='./data/sun397', batch_size=32, download=True).get_data_loader()
    # places365_loader = Places365DataLoader(root_dir='./data/places365', batch_size=32, download=True).get_data_loader()
    # dtd_loader = DTDDataLoader(root_dir=root_dir, batch_size=32, download=True)
