import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd

def count_model_params(model):
    """ Counting the number of learnable parameters in a nn.Module """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params

def smooth(f, K=5):
    """ Smoothing a function using a low-pass filter (mean) of size K """
    kernel = np.ones(K) / K
    f = np.concatenate([f[:int(K//2)], f, f[int(-K//2):]])  # to account for boundaries
    smooth_f = np.convolve(f, kernel, mode="same")
    smooth_f = smooth_f[K//2: -K//2]  # removing boundary-fixes
    return smooth_f


def plot_values_linear_n_log(values_list, mode: str="Loss"):
    """ Plotting loss/accuracy values for linear and log scale + smoothing """
    plt.style.use('seaborn')
    fig, ax = plt.subplots(1,2)
    fig.set_size_inches(18,5)

    smooth_acc = smooth(values_list, 31)
    ax[0].plot(values_list, c="blue", label=f"Training {mode}", linewidth=3, alpha=0.5)
    ax[0].plot(smooth_acc, c="red", label=f"Smoothed {mode}", linewidth=3)
    ax[0].legend(loc="best")
    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel(f"{mode}")
    ax[0].set_title("Training Progress (linearscale)")

    ax[1].plot(values_list, c="blue", label=f"Training {mode}", linewidth=3, alpha=0.5)
    ax[1].plot(smooth_acc, c="red", label=f"Smoothed {mode}", linewidth=3)
    ax[1].legend(loc="best")
    ax[1].set_xlabel("Iteration")
    ax[1].set_ylabel(f"{mode}")
    ax[1].set_yscale("log")
    ax[1].set_title("Training Progress (logscale)")

    plt.show()


def plot_2_confusion_m(y_true_1, y_pred_1, y_true_2, y_pred_2, title1, title2, classes):
    cf_matrix_1 = confusion_matrix(y_true_1, y_pred_1)
    df_cm_1 = pd.DataFrame(cf_matrix_1/np.sum(cf_matrix_1) *10, index = [i for i in classes],
                    columns = [i for i in classes])
    cf_matrix_2 = confusion_matrix(y_true_2, y_pred_2)
    df_cm_2 = pd.DataFrame(cf_matrix_2/np.sum(cf_matrix_2) *10, index = [i for i in classes],
                    columns = [i for i in classes])

    # Here we create a figure instance, and two subplots
    fig = plt.figure(figsize=(26, 17))

    ax1 = fig.add_subplot(221)
    ax1.set_title(title1)
    ax2 = fig.add_subplot(222)
    ax2.set_title(title2)
    # We use ax parameter to tell seaborn which subplot to use for this plot
    sn.heatmap(data=df_cm_1, annot=True, ax=ax1)
    sn.heatmap(data=df_cm_2, annot=True, ax=ax2)
    plt.show()

def plot_train_n_eval_metric(train_list, eval_list, mode: str="Loss"):
    assert len(train_list) == len(eval_list)
    x = np.arange(0, len(train_list))
    # plot training and validation loss/accuracy 
    plt.plot(x, train_list, label = f"Training {mode}")
    plt.plot(x, eval_list, label = f"Evaluation {mode}")

    plt.legend()
    plt.show()


def plot_conv_kernels(model, layer_num, kernel_num, title):
    """ Plotting the kernels of a convolutional layer """
    # retrieve weights from the second hidden layer
    filters, biases = model.network[layer_num].cpu().weight.data, model.network[0].bias.data
    # normalize filter values to 0-1 so we can visualize them
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)
    # plot first few filters
    n_filters, ix = kernel_num, 1
    fig= plt.figure(figsize=(6,16))
    fig.suptitle(title)

    # plt.figure(figsize=(6,16))
    
    for i in range(n_filters):
        # get the filter
        f = filters[i, :, :, :]
        # plot each channel separately
        for j in range(3):
            # specify subplot and turn of axis
            ax = plt.subplot(n_filters, 3, ix, )
            ax.set_xticks([])
            
            ax.set_yticks([])
            # plot filter channel in grayscale
            plt.imshow(f[:, :, j], cmap='gray')
            ix += 1
    # show the figure
    plt.show()
    # move the model back to the GPU
    model.network[layer_num].cuda()

def show_n_samples_from_batch(sample_imgs, sample_labels, n_samples=8):
    """ Plotting n_samples from a batch of images """
    fig, ax = plt.subplots(1,n_samples)
    fig.set_size_inches(3 * n_samples, 3)

    ids = np.random.randint(low=0, high=len(sample_imgs), size=n_samples)

    for i, n in enumerate(ids):
        img = sample_imgs[n]

        # because of normalization of dataset images have strange range, let's fix it with normalization to range [0, 1] for adequate visualization
        img = img.clone().detach()
        img += np.abs(img.min())
        img /= img.max()

        label = sample_labels[n]
        ax[i].imshow(img.permute(1,2,0))
        ax[i].set_title(f"#{n}  Label: {label}")
        ax[i].axis("off")
    plt.show()