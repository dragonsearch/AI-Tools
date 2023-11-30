import torch
import matplotlib.pyplot as plt
import numpy as np
import pickle


def save_ckpt(model, optimizer, PATH, params={}):
    """
    Args:
        model (torch.nn.Module): Model to be saved
        optimizer (torch.optim.Optimizer): Optimizer to be saved
        PATH (str): Path where the model will be saved
        params (dict): Dictionary with the parameters to be saved     
    """
    #Join 2 dicts using | (python 3.9) (for python 3.5+ {**x, **y})
    save_dict = {'model_state_dict':model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }# | params
    torch.save(save_dict, PATH)
    print(f'Saved model (ckpt) into: {PATH}')

def load_ckpt(model,optimizer,PATH):
    """
    Args:
        model (torch.nn.Module): Model to be loaded
        optimizer (torch.optim.Optimizer): Optimizer to be loaded
        mode (str): Mode of the model (train or eval)
        PATH (str): Path where the model is saved
    """
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f'Loaded model (ckpt) from: {PATH}')
    return checkpoint

def count_and_print_least_common_classes(arr,idx2class):
    # Calculate counts of each class
    unique_classes, counts = np.unique(arr, return_counts=True)
    # Sort the classes based on counts in ascending order
    sorted_indices = np.argsort(counts)
    # Get the least 10 common classes
    least_common_classes = unique_classes[sorted_indices[:30]]
    least_counts = counts[sorted_indices[:30]]
    # Apply the idx function to change the class names
    least_common_classes = np.array(list(map(lambda x: idx2class[x], least_common_classes)))
    # Print the least 10 common classes
    print("Least 10 common classes after applying idx function:")
    i = 0
    for class_name in least_common_classes:
        print("Class: ", class_name, " Count: ", least_counts[i])
        i+=1
    return least_common_classes, least_counts

def wrong_class(y_pred, labels, samples):
    wrong_predictions = []
    predicted_indices = torch.argmax(y_pred, dim=1)
    comparison = predicted_indices != labels
    different_elements = torch.nonzero(comparison).squeeze()
    if different_elements.shape[0] != 0:
        print("found wrong elements")
        for element in different_elements:
            wrong_predictions.append({'samples': samples[element], 'label': labels[element], 'prediction': y_pred[element]})


def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    print(f"Saved {name} to disk")

def plot_loss_curves(loss_res, metrics_res):
    """Plots training curves of loss and metrics dictionaries.
    """
    loss = loss_res["train"]
    test_loss = loss_res["val"]

    accuracy = metrics_res["accuracy"]["train"]
    val_accuracy = metrics_res["accuracy"]["train"]

    epochs = range(len(loss_res["train"]))

    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="val_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, val_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()

def set_seeds(seed=42):
    """Sets random sets for torch operations.

    Args:
        seed (int, optional): Random seed to set. Defaults to 42.
    """
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)

def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)