'''
yuanzhe
v1.0, 0312_2023, 0325_2023, 0328_2023: Implements LR with elasticnet and MIL training.
v1.1, 0331_2023
v1.2, 0402_2023
v2.0, 0406_2023: MIL_ElasticNet class updates the gradient and loss only for the specific instance within the same bag.
v2.1, 0412_2023: Change max pooling: (max(max, (1-min)))
v2.2, 0413_2023: If the max > (1- min), the max is used as the predicted probability for the bag; otherwise, the min is used.
v3.0, 0415_2023: Add one hidden layer to the class MIL_ElasticNet(nn.Module)
v3.1, 0416_2023: Add heatmap to double check hyperparameters space (alpha, l1_ratio)
v3.2, 0417_2023: Learning rate schedule
v3.3, 0419_2023: Gridsearch hyperparameters
v3.4, 0420_2023: Search hyperparameters from four space
v3.5, 0421_2023: CustomScaler retain original range and units, only subtracting the mean value of each feature
V3.6, 0501_2023: Add hidden layer
v3.7, 0504_2023: Threshold pooling
v3.8, 0510_2023: Average loss of selected instances from all bags in a minibatch
v3.9, 0513_2023: 4-level WPD 256 features
v3.10, 0516_2023: Fixed parallelizing MIL needs to consider the bags have different lengths and putting them into a batch is not natively supported by PyTorch
v3.11, 0519_2023: Check selected instances with fixed threshold pooling
v3.12, 0531_2023: Test the CNN feature alone to train the MIL.
v3.13, 0609_2023: Use the fine-tuned resnet34_conv4_x features alone
v3.14, 0620_2023: A clearer implementation ensures that perform stratified K-fold at the bag level, debug in CV

'''
import torch
import pickle
import logging
import os, argparse
import pandas as pd
import numpy as np
from numpy import interp
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam, SGD, RMSprop   # adam
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix, recall_score
import seaborn as sns
import matplotlib.pyplot as plt
import utils
from tqdm import tqdm
logging.basicConfig(filename='0505_log_file.log', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

random_seeds = [42, 65, 123, 152, 289, 
                363, 785, 987, 1024, 1525, 
                2125, 2666, 3535, 7551, 8962, 
                9361, 9898, 10256, 15221, 16656,
                26789, 29875, 32565, 33689, 36876,
                52135, 52136, 56399, 65895, 68787]
alphas = np.logspace(-6, 2, 7) 
l1_ratios = np.linspace(0.05, 0.95, 0.05)
# l1_ratios = np.linspace(0.05, 0.95, 0.1)
hidden_layer_sizes1 = np.arange(50, 200, 5) # 50-200
hidden_layer_sizes2 = np.arange(30, 120, 5) # 30-120
# learning_rates = [0.0001, 0.0002, 0.0003, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.01, 0.025, 0.05, 0.1]
learning_rates = [0.0001, 0.0002, 0.0003, 0.0005, 0.0007, 0.001, 0.002]
# Todo: cos_lr 
n_folds = 5
epochs = 100
tol = 1e-3
batch_size = 32
threshold_pooling = 0.90
# threshold_pooling = 0.85
# threshold_pooling = 0.80

def preprocess_data(data_folder, bag_labels_file):
    bag_labels = pd.read_csv(bag_labels_file, header=None, names=['bag_id', 'label'])
    X_instance = []
    y_instance = []
    bag_ids = []
    for filename in tqdm(os.listdir(data_folder)):
        if filename.endswith('.csv'):
            bag_id = filename.split('.')[0]
            if bag_id in bag_labels['bag_id'].values:
                df = pd.read_csv(os.path.join(data_folder, filename), header=None)
                bag_label = bag_labels[bag_labels['bag_id'] == bag_id]['label'].values[0]
                X_instance.extend(df.iloc[:, :].values)
                y_instance.extend([bag_label] * df.shape[0])
                bag_ids.extend([bag_id] * len(df))

    X_instance = np.array(X_instance)
    y_instance = np.array(y_instance)
    bag_ids = np.array(bag_ids)
    unique_bag_ids, bag_indices = np.unique(bag_ids, return_index=True)
    y_unique = y_instance[bag_indices]

    return X_instance, y_instance, bag_ids, unique_bag_ids, y_unique

def save_hyperparameters_to_txt(errors_matrix, filename='0505_hyperparameters_results.txt'):
    with open(filename, 'w') as f:
        for i, alpha in enumerate(alphas):
            for j, l1_ratio in enumerate(l1_ratios):
                for k, hidden_layer_size1 in enumerate(hidden_layer_sizes1):
                    for l, hidden_layer_size2 in enumerate(hidden_layer_sizes2):
                        for q, learning_rate in enumerate(learning_rates):
                            f.write(f"Alpha: {alpha}, L1_ratio: {l1_ratio}, Hidden_layer_size1: {hidden_layer_size1}, Hidden_layer_size2: {hidden_layer_size2}, Learning_rate: {learning_rate}, Error: {errors_matrix[i, j, k, l, q]}\n")
                 
def collate_fn(batch):
    X, Y = zip(*batch)
    max_bag_length = max([len(x) for x in X])
    X_padded = []
    for x in X:
        x_padded = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=0)
        x_padded = torch.nn.functional.pad(x_padded, pad=(0, 0, 0, max_bag_length - len(x_padded)))
        X_padded.append(x_padded)
    X_padded = torch.stack(X_padded)
    Y = torch.Tensor(Y).view(-1, 1)
    return X_padded, Y  

def cross_validate(alpha, l1_ratio, hidden_layer_size1, hidden_layer_size2, learning_rate, X_train, y_train,
                   bag_ids_train, device):
    data_df = pd.DataFrame(X_train)
    data_df['label'] = y_train
    data_df['bag_id'] = bag_ids_train

    # Get the unique bag ids and their corresponding labels         
    unique_bag_ids = data_df['bag_id'].unique()
    unique_bag_labels = data_df.groupby('bag_id')['label'].first()
    aucs = []

    skf = StratifiedKFold(n_splits=n_folds).split(unique_bag_ids, unique_bag_labels)

    # Perform stratified K-fold at the bag level, 0620
    for train_fold_indices, val_fold_indices in skf:
        train_bag_ids = unique_bag_ids[train_fold_indices]
        val_bag_ids = unique_bag_ids[val_fold_indices]
      
        # Find all tiles corresponding to the selected bag IDs
        train_tile_indices = data_df['bag_id'].isin(train_bag_ids)
        val_tile_indices = data_df['bag_id'].isin(val_bag_ids)

        train_data = data_df[train_tile_indices]
        val_data = data_df[val_tile_indices]
        X_train_fold, y_train_fold, bag_ids_train_fold = train_data.drop(['label', 'bag_id'], axis=1), train_data[
            'label'], train_data['bag_id']
        X_val_fold, y_val_fold, bag_ids_val_fold = val_data.drop(['label', 'bag_id'], axis=1), val_data['label'], \
                                                   val_data['bag_id']
        X_train_fold = X_train_fold.values
        X_val_fold = X_val_fold.values
        y_train_fold = y_train_fold.values
        y_val_fold = y_val_fold.values
        bag_ids_train_fold = bag_ids_train_fold.values
        bag_ids_val_fold = bag_ids_val_fold.values
        mil_clf = MIL_ElasticNet(input_size=X_train_fold.shape[1], hidden_layer_size1=hidden_layer_size1,
                                 hidden_layer_size2=hidden_layer_size2, alpha=alpha, l1_ratio=l1_ratio, device=device)
        mil_clf = mil_clf.to(device)
        mil_clf.fit(X_train_fold, y_train_fold, bag_ids_train_fold, learning_rate=learning_rate, max_iter=1000, tol=tol,
                    batch_size=batch_size)
        y_pred_prob_val = mil_clf.predict_proba(X_val_fold)
        y_val_fold = np.array(y_val_fold, dtype=int)
        y_pred_prob_val = np.array(y_pred_prob_val)
        fpr, tpr, _ = roc_curve(y_val_fold, y_pred_prob_val)
        auc_score = auc(fpr, tpr)
        aucs.append(auc_score)
    avg_auc = np.mean(aucs)
    return avg_auc, mil_clf

class MIL_Dataset(Dataset):
    def __init__(self, X, y, bag_ids):
        self.X = X
        self.y = y
        self.bag_ids = bag_ids

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.bag_ids[idx]

class MIL_ElasticNet(nn.Module):
    def __init__(self, input_size, hidden_layer_size1, hidden_layer_size2, alpha, l1_ratio, device):
        super(MIL_ElasticNet, self).__init__()
        self.device = device
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.hidden_layer1 = nn.Linear(input_size, hidden_layer_size1)
        self.batch_norm1 = nn.BatchNorm1d(hidden_layer_size1)
        self.hidden_layer2 = nn.Linear(hidden_layer_size1, hidden_layer_size2)
        self.batch_norm2 = nn.BatchNorm1d(hidden_layer_size2)
        self.output_layer = nn.Linear(hidden_layer_size2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.hidden_layer1(x)
        x = self.batch_norm1(x)
        x = self.sigmoid(x)
        x = self.hidden_layer2(x)
        x = self.batch_norm2(x)
        x = self.sigmoid(x)
        x = self.output_layer(x)
        x = self.sigmoid(x)
        return x  

    def fit(self, X, y, bag_ids, learning_rate, max_iter, tol, batch_size, threshold=threshold_pooling, writer=None):
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=self.alpha * (1 - self.l1_ratio))
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
        criterion = nn.BCELoss()
        self.train()

        X_np = np.array(X, dtype=np.float32)
        y_np = np.array(y, dtype=np.float32)
        label_encoder = LabelEncoder()
        bag_ids_np = label_encoder.fit_transform(bag_ids)

        dataset = data_utils.TensorDataset(torch.from_numpy(X_np), torch.from_numpy(y_np), torch.from_numpy(bag_ids_np))
        dataloader = data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Training 
        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0
            # Third threshold pooling, select instances based on prob
            for batch_id, (X_batch, y_batch, bag_ids_batch) in enumerate(dataloader):
                optimizer.zero_grad()
                batch_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                sum_instances = 0
                unique_bag_ids = torch.unique(bag_ids_batch)

                for bag_id in unique_bag_ids:
                    bag_indices = (bag_ids_batch == bag_id).nonzero(as_tuple=True)[0]
                    X_bag = X_batch[bag_indices]
                    y_bag = y_batch[bag_indices].to(self.device)
                    selected_indices = []
                    max_probs = []

                    for i, x_instance in enumerate(X_bag):
                        x_instance = x_instance.unsqueeze(0).to(self.device)
                        prob = self(x_instance).detach().cpu().numpy()
                        if y_bag[i] == 1:
                            max_probs.append(prob)
                        else:
                            max_probs.append(1 - prob)

# If the probability is above a threshold, append the instance index to selected_indices
                        if (y_bag[i] == 1 and prob > threshold) or (y_bag[i] == 0 and 1 - prob > threshold):
                            selected_indices.append(i)

# If no instances selected, select the instance with max(prob) or max(1-prob)
                    if not selected_indices:
                        if y_bag[i] == 1:
                            max_instance_index = np.argmax(max_probs)
                        else:
                            max_instance_index = np.argmax([1 - prob for prob in max_probs])
                        selected_indices.append(max_instance_index)          

                    selected_instances = X_bag[selected_indices].to(self.device)
                    y_preds = self(selected_instances)
                    loss = criterion(y_preds, y_bag[selected_indices])
                    batch_loss = batch_loss + loss.item() * len(selected_indices)
                    sum_instances += len(selected_indices)
                    
                batch_loss /= sum_instances
                # Backpropagation
                batch_loss.backward()
                optimizer.step()

                total_loss += batch_loss.item()
                num_batches += 1      
            avg_loss = total_loss / num_batches
            scheduler.step(avg_loss)
            logging.info(f'loss: {avg_loss}')
            if avg_loss < tol:
                break
            
        # with open('num_instances.txt', 'w') as f:
        #    for key, value in num_instances_dict.items():
        #       f.write(f'Epoch: {key[0]}, Batch: {key[1]}, Bag: {key[2]}, Num instances: {value}\n')
                    
    def predict_proba(self, X):
        self.eval()
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            y_pred = self(X)
        return y_pred.cpu().numpy()

    def predict_classes(self, X, threshold=0.5):
        y_pred = self.predict_proba(X)
        return (y_pred > threshold).astype(int)

def main():
    parser = argparse.ArgumentParser(description='Process dataset arguments.')
    parser.add_argument('--dataset', type=str, help='Dataset/feature to be used')

    try:
        logging.debug('Starting the main function')
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        args = parser.parse_args()
        if args.dataset == 'TCGA_7wavelet_246':
            data_folder = '/a/bear.cs.fiu.edu./disk/bear-c/users/rxm1351/yz/0403_lr_246/dataset_tcga_1047/0408_tcga_246/'
            bag_labels_file = '/a/bear.cs.fiu.edu./disk/bear-c/users/rxm1351/yz/0403_lr_246/dataset_tcga_1047/TCGA.csv'
        elif args.dataset == 'TCGA_cnn_512':
            data_folder = '/a/bear.cs.fiu.edu./disk/bear-c/users/rxm1351/yz/dataset_tcga_1047/0531_resnet/'
            bag_labels_file = '/a/bear.cs.fiu.edu./disk/bear-c/users/rxm1351/yz/dataset_tcga_1047/TCGA.csv'
        elif args.dataset == 'TCGA_cnn_256':
            data_folder = '/a/bear.cs.fiu.edu./disk/bear-c/users/rxm1351/yz/dataset_tcga_1047/0610_FT_resnet/'
            bag_labels_file = '/a/bear.cs.fiu.edu./disk/bear-c/users/rxm1351/yz/dataset_tcga_1047/TCGA.csv'
        else:
            raise ValueError(f"Invalid dataset: {args.dataset}")

        X_instance, y_instance, bag_ids, unique_bag_ids, y_unique = preprocess_data(data_folder, bag_labels_file)
        X_unique, _, y_unique, _ = train_test_split(np.unique(bag_ids),
                                                    y_instance[np.unique(bag_ids, return_index=True)[1]], test_size=0.2,
                                                    stratify=y_instance[np.unique(bag_ids, return_index=True)[1]],
                                                    random_state=random_seed)
        tprs = []
        aucs = []
        test_aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        errors_matrix = np.zeros((len(alphas), len(l1_ratios), len(hidden_layer_sizes1), len(hidden_layer_sizes2), len(learning_rates))) 
      
        for random_seed in random_seeds:
            # Split into training and test sets, 836 train
            train_indices, test_indices = train_test_split(np.arange(len(X_unique)), test_size=0.2, random_state=random_seed, stratify=y_unique)
            bag_ids_train = X_unique[train_indices]
            bag_ids_test = X_unique[test_indices]
            train_indices = np.where(np.isin(bag_ids, bag_ids_train))[0]
            test_indices = np.where(np.isin(bag_ids, bag_ids_test))[0]
            X_train, X_test = X_instance[train_indices], X_instance[test_indices]
            y_train, y_test = y_instance[train_indices], y_instance[test_indices]
            y_train = y_train.astype(int)
            bag_ids_train, bag_ids_test = bag_ids[train_indices], bag_ids[test_indices]       
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)                    
            best_auc = 0
            best_model = None
            best_alpha = None
            best_l1_ratio = None
            best_hidden_layer_size1 = None
            best_hidden_layer_size2 = None
            best_learning_rate = None
            best_seed = None
          
            for l, learning_rate in enumerate(learning_rates):
                for i, alpha in enumerate(alphas):
                    for j, l1_ratio in enumerate(l1_ratios):
                        for k, hidden_layer_size1 in enumerate(hidden_layer_sizes1):
                            for q, hidden_layer_size2 in enumerate(hidden_layer_sizes2):
                                avg_auc, mil_clf = cross_validate(alpha, l1_ratio, hidden_layer_size1, hidden_layer_size2, learning_rate, X_train, y_train, bag_ids_train, device, batch_size)
                                errors_matrix[i, j, k, l, q] = 1 - avg_auc

                                if avg_auc > best_auc:
                                    best_auc = avg_auc
                                    best_model = mil_clf
                                    best_alpha = alpha
                                    best_l1_ratio = l1_ratio
                                    best_hidden_layer_size1 = hidden_layer_size1
                                    best_hidden_layer_size2 = hidden_layer_size2
                                    best_learning_rate = learning_rate
                                    best_seed = random_seed
        
        save_hyperparameters_to_txt(errors_matrix, filename='hyperparameters_results.txt')
        epochs =  100
        writer = SummaryWriter()
        for epoch in range(epochs):
                best_model.fit(X_train, y_train, bag_ids_train, best_learning_rate, max_iter, tol, batch_size=batch_size, writer=writer)
                logging.info(f'Epoch {epoch + 1}/{epochs}')             
        writer.close()
             
        y_true_bag = []
        y_pred_bag = []
        y_pred_prob_test = best_model.predict_proba(X_test)
        for bag_id in np.unique(bag_ids_test):
            y_pred_bag_prob = y_pred_prob_test[bag_ids_test == bag_id]
            y_pred_bag.append(np.mean(y_pred_bag_prob))
            y_true_bag.append(y_test[bag_ids_test == bag_id][0])
        
        y_true_bag = np.array(y_true_bag, dtype=int)
        y_pred_bag = np.array(y_pred_bag)
        fpr, tpr, _ = roc_curve(y_true_bag, y_pred_bag)
        auc_score = auc(fpr, tpr)
       
        with open('output_file.txt', 'w') as output_file:
            output_file.write(f"Test set AUC for seed {random_seed}: {auc_score}\n")
        test_aucs.append(auc_score)                
            
        with open('output_Best AUC.txt', 'w') as f:
            print(f"Best AUC: {best_auc}, Best alpha: {best_alpha}, Best l1_ratio: {best_l1_ratio}, Best hidden_layer_size1: {best_hidden_layer_size1}, Best hidden_layer_size2: {best_hidden_layer_size2}, Best seed: {best_seed}", file=f)
            
        with open('best_model.pkl', 'wb') as f:
            pickle.dump(best_model, f)
            
        with open("X_test_shapes.txt", "w") as f:
            print(f"X_test {bag_id} has shape {X_test.shape}", file=f)
         
        bag_ids_test_unique = np.unique(bag_ids_test)
        y_pred_bag_test = []
        y_true_bag_test = []
        y_pred_prob_test = best_model.predict_proba(X_test)
           
        for bag_id in bag_ids_test_unique:
            bag_indices = [i for i, x in enumerate(bag_ids_test) if x == bag_id]
            max_prob = y_pred_prob_test[bag_indices].max()
            min_prob = y_pred_prob_test[bag_indices].min()
            if max_prob > (1 - min_prob):
                y_pred_bag_test.append(max_prob)
            else:
                y_pred_bag_test.append(min_prob)  
            y_true_bag_test.append(y_test[bag_indices[0]])
        
        y_true_bag_test = np.array(y_true_bag_test, dtype=int)
        y_pred_bag_test = np.array(y_pred_bag_test)
       
        fpr, tpr, _ = roc_curve(y_true_bag_test, y_pred_bag_test)
        test_auc_score = auc(fpr, tpr)
        aucs.append(test_auc_score)
        tprs.append(interp(mean_fpr, fpr, tpr))
        mean_tpr = np.mean(tprs, axis=0)
        std_tpr = np.std(tprs, axis=0)
      
        # plt.figure(figsize=(8, 6))
        # plt.plot(mean_fpr, mean_tpr, label=f"CNN features threshold (0.85)_pooling_MIL_ElasticNet (AUC = {np.mean(aucs):.3f} $\pm$ {np.std(aucs):.3f})", lw=2)
        # plt.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, alpha=0.3)
        # plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('Fine_tuned_conv4_x_(third Res block)_features_MIL_ElasticNet')
        # plt.legend(loc="lower right")
        # plt.legend(loc="lower right")
        # plt.savefig('Fine_tuned_conv4_x_(third Res block)_features_MIL_ElasticNet.pdf')
        # plt.show()
        
        logging.debug('Finishing the main function')
    except Exception as e:
        logging.error(f'An error occurred in the main function: {str(e)}', exc_info=True)

if __name__ == "__main__":
    main()


