from tqdm import tqdm
import torch



def evaluate_model(model, criterion, eval_loader, device):
    dataset_size = len(eval_loader.dataset)

    n_correct = 0

    y_pred = [] # list of predictionsof the model
    y_true = [] # list of true labels

    eval_loss_list = []

    with torch.no_grad():
        progress_bar = tqdm(enumerate(eval_loader), total=len(eval_loader))
        for i, (imgs, labels) in progress_bar: 
            #everything needs to be on the same device
            imgs = imgs.to(device)
            labels = labels.to(device)
            y_true.extend(labels.cpu().numpy()) # Save Truth
        
            # forward pass
            # flattened_imgs = imgs.flatten(start_dim=1)
            preds = model(imgs)

            pred_labels = torch.argmax(preds, dim=-1)
            loss = criterion(preds, labels)
            eval_loss_list.append(loss.item())

            y_pred.extend(pred_labels.cpu().numpy()) # Save Prediction
            cur_correct = len(torch.where(pred_labels == labels)[0])
            n_correct = n_correct + cur_correct

    accuracy = n_correct / dataset_size * 100
    print(f"Test accuracy: {round(accuracy,2)}%")
    return accuracy, y_true, y_pred, eval_loss_list