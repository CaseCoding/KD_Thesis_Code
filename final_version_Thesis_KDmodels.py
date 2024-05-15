import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchmetrics import Accuracy, Precision, Recall, F1Score
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import datetime

# Check if GPU is available, and if not, use the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import pickle

# Function to load a pickle file
def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

# Replace 'file_path.pkl' with the actual path to your pickled files
image_data_path = './data/image_data.pkl'
one_hot_encoded_data_path = './data/one_hot_encoded_data.pkl'
test_image_data_path = './data/test_image_data.pkl'
test_one_hot_encoded_data_path = './data/test_one_hot_encoded_data.pkl'

# Unpickling the data
image_data = load_pickle(image_data_path)
one_hot_encoded_data = load_pickle(one_hot_encoded_data_path)
test_image_data = load_pickle(test_image_data_path)
test_one_hot_encoded_data = load_pickle(test_one_hot_encoded_data_path)


# We need to create a dataset of all our pickle files for the pytorch models to use. We compute some normalization methods for our models to perform better.
from torch.utils.data import Dataset, DataLoader

class SimpleDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)

        return image, label


calculated_mean = torch.tensor([0.5075, 0.5064, 0.5082]).mean().item()
calculated_std = torch.tensor([0.2556, 0.2558, 0.2541]).mean().item()


transforms_data = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[calculated_mean], std=[calculated_std])
])

# Assuming `train_images` and `train_labels` are your pre-loaded training data and labels
# And `test_images` and `test_labels` are your pre-loaded test data and labels
train_dataset = SimpleDataset(image_data, one_hot_encoded_data, transform=transforms_data)
test_dataset = SimpleDataset(test_image_data, test_one_hot_encoded_data, transform=transforms_data)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2) 

def calculate_mean_std(dataloader):
    # Variables to store total sum, square sum, and number of batches
    mean = 0.0
    squared_sum = 0.0
    n_samples = 0

    for data, _ in dataloader:
        data = data.transpose(0, 1).contiguous().view(3, -1)
        # Update total sum and square sum
        mean += data.mean(dim=1)
        squared_sum += data.pow(2).mean(dim=1)
        n_samples += 1

    # Final calculation
    mean /= n_samples
    std = (squared_sum / n_samples - mean.pow(2)).sqrt()

    return mean, std

mean, std = calculate_mean_std(train_loader)
print(f"Calculated Mean: {mean}")
print(f"Calculated Std: {std}")




for inputs, labels in train_loader:
    print("Original labels shape:", labels.shape)  # Print the shape of the labels
    print("Original labels sample:", labels[0])    # Print the first label entry

    # Convert one-hot to class indices if necessary
    converted_labels = labels.squeeze(1).max(dim=1)[1] if labels.dim() == 3 else labels
    print("Converted labels shape:", converted_labels.shape)
    print("Converted labels sample:", converted_labels[0])

    # Check the rest of your loop...
    break  # Remove or comment this out to check all batches, but be cautious with large datasets

# after setting up all the dataset things, we can initialize the Teacher and Student. These are edited specifically to work on grayscale images in a specific resolution. 
# if you want to use other sample data with this code, please edit the classes accordingly.

# Deeper neural network class to be used as teacher:
class DeepNN(nn.Module):
    def __init__(self, num_classes=6):
        super(DeepNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(4608, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Lightweight neural network class to be used as student:
class LightNN(nn.Module):
    def __init__(self, num_classes=6):
        super(LightNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(2304, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
##########
    
def train(model, train_loader, epochs, learning_rate, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            # Remove the middle dimension and convert one-hot encoded labels to class indices
            labels = labels.squeeze(1).max(dim=1)[1]
            
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")


#####

def test(model, test_loader, device):
    model.to(device)
    model.eval()

    # Initialize metrics
    accuracy_metric = Accuracy(num_classes=6, task='multiclass').to(device)
    precision_metric = Precision(num_classes=6, task='multiclass').to(device)
    recall_metric = Recall(num_classes=6, task='multiclass').to(device)
    f1_metric = F1Score(num_classes=6, task='multiclass').to(device)

    all_preds = []
    all_labels = []

    # Loop through batches in the test set
    with torch.no_grad():
        for batch_index, (inputs, labels) in enumerate(test_loader):
            # Convert labels from one-hot encoded to class indices
            labels = labels.squeeze(1).max(dim=1)[1]
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            # Update metrics
            accuracy_metric.update(predicted, labels)
            precision_metric.update(predicted, labels)
            recall_metric.update(predicted, labels)
            f1_metric.update(predicted, labels)
            print("Predictions:", predicted)
            print("True labels:", labels)

            all_preds.append(predicted.numpy())
            all_labels.append(labels.numpy())

    # Compute overall metrics after going through all batches
    overall_accuracy = accuracy_metric.compute()
    overall_precision = precision_metric.compute()
    overall_recall = recall_metric.compute()
    overall_f1_score = f1_metric.compute()

    # Print overall metrics
    print(f'Overall Test - '
          f'Accuracy: {overall_accuracy:.2f}, '
          f'Precision: {overall_precision:.3f}, '
          f'Recall: {overall_recall:.3f}, '
          f'F1: {overall_f1_score:.3f}')

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels) 

    cm = confusion_matrix(all_labels, all_preds)

    cm_df = pd.DataFrame(cm)
    # Saving the results to a csv with a recognizable name
    model_class_name = model.__class__.__name__
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_file_path = f'./results/confusion_matrix_{model_class_name}_{timestamp}.csv'
    cm_df.to_csv(csv_file_path, index=False)

# Return a dictionary of the overall metrics
    return {
        "accuracy": overall_accuracy.item(),
        "precision": overall_precision.item(),
        "recall": overall_recall.item(),
        "f1_score": overall_f1_score.item()
    }






#####

torch.manual_seed(42)
nn_deep = DeepNN(num_classes=6).to(device)
# Train the teacher using the best hyperparameters
train(nn_deep, train_loader, epochs=25, learning_rate=0.001, device=device)
results_deep = test(nn_deep, test_loader, device)

# Instantiate the lightweight network:
torch.manual_seed(42) 
nn_light = LightNN(num_classes=6).to(device)

torch.manual_seed(42)
new_nn_light = LightNN(num_classes=6).to(device)

# Print the norm of the first layer of the initial lightweight model
print("Norm of 1st layer of nn_light:", torch.norm(nn_light.features[0].weight).item())
# Print the norm of the first layer of the new lightweight model
print("Norm of 1st layer of new_nn_light:", torch.norm(new_nn_light.features[0].weight).item())

total_params_deep = "{:,}".format(sum(p.numel() for p in nn_deep.parameters()))
print(f"DeepNN parameters: {total_params_deep}")
total_params_light = "{:,}".format(sum(p.numel() for p in nn_light.parameters()))
print(f"LightNN parameters: {total_params_light}")

# Train the student using best hyperparameters
train(nn_light, train_loader, epochs=10, learning_rate=0.001, device=device)
results_light_ce = test(nn_light, test_loader, device)

# Print all metrics for the deep model
print("Deep Model Metrics:")
for metric, value in results_deep.items():
    print(f"{metric.capitalize()}: {value:.2f}%")

# Print all metrics for the lightweight model
print("Lightweight Model Metrics:")
for metric, value in results_light_ce.items():
    print(f"{metric.capitalize()}: {value:.2f}%")


##############  Knowledge distillation run

def train_knowledge_distillation(teacher, student, train_loader, epochs, learning_rate, T, soft_target_loss_weight, ce_loss_weight, device):
    ce_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)

    teacher.eval()  # Teacher set to evaluation mode
    student.train() # Student to train mode

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            # Remove the middle dimension and convert one-hot encoded labels to class indices
            labels = labels.squeeze(1).max(dim=1)[1]
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass with the teacher model - do not save gradients here as we do not change the teacher's weights
            with torch.no_grad():
                teacher_logits = teacher(inputs)

            # Forward pass with the student model
            student_logits = student(inputs)

            # Soften the student logits by applying softmax first and log() second
            soft_targets = nn.functional.softmax(teacher_logits / T, dim=-1)
            soft_prob = nn.functional.log_softmax(student_logits / T, dim=-1)

            # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
            soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (T**2)

            # Calculate the true label loss
            label_loss = ce_loss(student_logits, labels)

            # Weighted sum of the two losses
            loss = soft_target_loss_weight * soft_targets_loss + ce_loss_weight * label_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()



# Use the first knowledge distillation method, this is used for our studentOut model trained on outputs of the teacher.
# Initiliaze using best hyper parameters from previous testing
train_knowledge_distillation(teacher=nn_deep, student=new_nn_light, train_loader=train_loader, epochs=25, learning_rate=0.001, T=2, soft_target_loss_weight=0.25, ce_loss_weight=0.75, device=device)
test_accuracy_light_ce_and_kd = test(new_nn_light, test_loader, device)

# Compare the student test accuracy with and without the teacher, after distillation
print(f"Student accuracy with CE + KD: {test_accuracy_light_ce_and_kd['accuracy']:.2f}%")
print(f"Recall with CE + KD: {test_accuracy_light_ce_and_kd['recall']:.3f}")
print(f"F1 Score with CE + KD: {test_accuracy_light_ce_and_kd['f1_score']:.3f}")


### Cosine loss minimization run

class ModifiedDeepNNCosine(nn.Module):
    def __init__(self, num_classes=6):
        super(ModifiedDeepNNCosine, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, padding=1),  # Adjusted for 1 input channel
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Assuming an input image size of 48x48 for correct flattened size calculation
        self.classifier = nn.Sequential(
            nn.Linear(4608, 512),  # Adjusted linear layer size
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        flattened_conv_output = torch.flatten(x, 1)
        x = self.classifier(flattened_conv_output)
        flattened_conv_output_after_pooling = torch.nn.functional.avg_pool1d(flattened_conv_output, 2)
        return x, flattened_conv_output_after_pooling

class ModifiedLightNNCosine(nn.Module):
    def __init__(self, num_classes=6):
        super(ModifiedLightNNCosine, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # Assuming grayscale images
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Adjusted linear layer size to match 1024
        self.classifier = nn.Sequential(
            nn.Linear(2304, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        flattened_conv_output = torch.flatten(x, 1)
        x = self.classifier(flattened_conv_output)
        return x, flattened_conv_output

# We do not have to train the modified deep network from scratch of course, we just load its weights from the trained instance
modified_nn_deep = ModifiedDeepNNCosine(num_classes=6).to(device)
modified_nn_deep.load_state_dict(nn_deep.state_dict())

# Once again ensure the norm of the first layer is the same for both networks
print("Norm of 1st layer for deep_nn:", torch.norm(nn_deep.features[0].weight).item())
print("Norm of 1st layer for modified_deep_nn:", torch.norm(modified_nn_deep.features[0].weight).item())

# Initialize a modified lightweight network with the same seed as our other lightweight instances. This will be trained from scratch to examine the effectiveness of cosine loss minimization.
torch.manual_seed(42)
modified_nn_light = ModifiedLightNNCosine(num_classes=6).to(device)
print("Norm of 1st layer:", torch.norm(modified_nn_light.features[0].weight).item())

#######

# Adjust sample input for grayscale images
sample_input = torch.randn(128, 1, 48,48).to(device) # Batch size: 128, Filters: 1 (grayscale), Image size: 32x32

# Pass the input through the student
logits, hidden_representation = modified_nn_light(sample_input)

# Print the shapes of the tensors
print("Student logits shape:", logits.shape) # batch_size x total_classes
print("Student hidden representation shape:", hidden_representation.shape) # batch_size x hidden_representation_size

# Pass the input through the teacher
logits, hidden_representation = modified_nn_deep(sample_input)

# Print the shapes of the tensors
print("Teacher logits shape:", logits.shape) # batch_size x total_classes
print("Teacher hidden representation shape:", hidden_representation.shape) # batch_size x hidden_representation_size

######## 

def train_cosine_loss(teacher, student, train_loader, epochs, learning_rate, hidden_rep_loss_weight, ce_loss_weight, device):
    ce_loss = nn.CrossEntropyLoss()
    cosine_loss = nn.CosineEmbeddingLoss()
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)

    teacher.to(device)
    student.to(device)
    teacher.eval()  # Teacher set to evaluation mode
    student.train()  # Student to train mode

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            labels = labels.squeeze(1).max(dim=1)[1]

            optimizer.zero_grad()

            # Forward pass with the teacher model and keep only the hidden representation
            with torch.no_grad():
                _, teacher_hidden_representation = teacher(inputs)

            # Forward pass with the student model
            student_logits, student_hidden_representation = student(inputs)

            # Calculate the cosine loss. Target is a vector of ones. From the loss formula above we can see that is the case where loss minimization leads to cosine similarity increase.
            hidden_rep_loss = cosine_loss(student_hidden_representation, teacher_hidden_representation, target=torch.ones(inputs.size(0)).to(device))

            # Calculate the true label loss
            label_loss = ce_loss(student_logits, labels)

            # Weighted sum of the two losses
            loss = hidden_rep_loss_weight * hidden_rep_loss + ce_loss_weight * label_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")


        #### multiple output testing
def test_multiple_outputs(model, test_loader, device):
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            labels = labels.squeeze(1).max(dim=1)[1]

            outputs, _ = model(inputs) # Disregard the second tensor of the tuple
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")


    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm_df = pd.DataFrame(cm)


    # Generate a concise, unique identifier for the model - this is so we can recognize what model we are actually using
    model_class_name = model.__class__.__name__
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_file_path = f'./results/confusion_matrix_{model_class_name}_{timestamp}.csv'
    cm_df.to_csv(csv_file_path, index=False)

    # Return the accuracy and path to saved confusion matrix CSV

    return accuracy, csv_file_path

# Train and test the lightweight network with cross entropy loss utilizing best hyperparameters.
train_cosine_loss(teacher=modified_nn_deep, student=modified_nn_light, train_loader=train_loader, epochs=25, learning_rate=0.001, hidden_rep_loss_weight=0.25, ce_loss_weight=0.75, device=device)
test_accuracy_light_ce_and_cosine_loss, cm_csv_path = test_multiple_outputs(modified_nn_light, test_loader, device)

########## intermediate regressor run

# Pass the sample input only from the convolutional feature extractor
convolutional_fe_output_student = nn_light.features(sample_input)
convolutional_fe_output_teacher = nn_deep.features(sample_input)

# Print their shapes
print("Student's feature extractor output shape: ", convolutional_fe_output_student.shape)
print("Teacher's feature extractor output shape: ", convolutional_fe_output_teacher.shape)

class ModifiedDeepNNRegressor(nn.Module):
    def __init__(self, num_classes=6):
        super(ModifiedDeepNNRegressor, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, padding=1),  # Adjusted for 1 input channel
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Assuming an input image size of 48x48
        self.classifier = nn.Sequential(
            nn.Linear(4608, 512),  # Corrected to match output from the feature extractor
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        conv_feature_map = x
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x, conv_feature_map

class ModifiedLightNNRegressor(nn.Module):
    def __init__(self, num_classes=6):
        super(ModifiedLightNNRegressor, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # Adjusted for 1 input channel
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.regressor = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1)
        )        # Assuming an input image size of 48x48
        self.classifier = nn.Sequential(
            nn.Linear(2304, 256),  # Adjusted to match output from the feature extractor
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        regressor_output = self.regressor(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x, regressor_output

# Example of initializing and preparing the models
modified_deep_regressor = ModifiedDeepNNRegressor(num_classes=6).to(device)
modified_light_regressor = ModifiedLightNNRegressor(num_classes=6).to(device)

#### MSE LOSS CALCULATIONS


def train_mse_loss(teacher, student, train_loader, epochs, learning_rate, feature_map_weight, ce_loss_weight, device):
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)

    teacher.to(device)
    student.to(device)
    teacher.eval()  # Teacher set to evaluation mode
    student.train()  # Student to train mode

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            labels = labels.squeeze(1).max(dim=1)[1]

            optimizer.zero_grad()

            # Ignore teacher logits, get only the feature map
            with torch.no_grad():
                _, teacher_feature_map = teacher(inputs)

            # Forward pass with the student model
            student_logits, student_feature_map = student(inputs)

            # Calculate the MSE loss for the feature maps
            hidden_rep_loss = mse_loss(student_feature_map, teacher_feature_map)

            # Calculate the Cross-Entropy loss for the actual labels
            label_loss = ce_loss(student_logits, labels)

            # Weighted sum of the two losses
            loss = feature_map_weight * hidden_rep_loss + ce_loss_weight * label_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")


# Initialize a ModifiedLightNNRegressor
torch.manual_seed(42)
modified_nn_light_reg = ModifiedLightNNRegressor(num_classes=6).to(device)

# We do not have to train the modified deep network from scratch of course, we just load its weights from the trained instance
modified_nn_deep_reg = ModifiedDeepNNRegressor(num_classes=6).to(device)
modified_nn_deep_reg.load_state_dict(nn_deep.state_dict())

# Train and test once again using the best hyperparameters
train_mse_loss(teacher=modified_nn_deep_reg, student=modified_nn_light_reg, train_loader=train_loader, epochs=25, learning_rate=0.001, feature_map_weight=0.25, ce_loss_weight=0.75, device=device)
test_accuracy_light_ce_and_mse_loss, cm_csv_path = test_multiple_outputs(modified_nn_light_reg, test_loader, device)

print(f"Student accuracy with CE + CosineLoss: {test_accuracy_light_ce_and_cosine_loss:.2f}%")
print(f"Student accuracy with CE + RegressorMSE: {test_accuracy_light_ce_and_mse_loss:.2f}%")




