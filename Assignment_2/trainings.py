from tqdm import tqdm
import evaluations
import numpy as np
import torch


def train(model, criterion, train_loader, eval_loader, optimizer, epoch_num, device, regularization=None, scheduler=None):
    loss_list = []
    acc_list = []
    loss_epoch_list = []
    acc_epoch_list = []
    loss_epoch_list_eval = []
    acc_epoch_list_eval = []

    # use this scheduler or similar
    # scheduler = ReduceLROnPlateau(optimizer, 'min', patience = 5)

    for epoch in range(epoch_num):
        progress_bar = tqdm(train_loader, total=len(train_loader))
        model.train()
        for i, (imgs, labels) in enumerate(progress_bar):
            # using GPU
            imgs, labels = imgs.to(device), labels.to(device)
    
            # forward pass
            preds = model(imgs)
    
            # computing error
            loss = criterion(preds, labels)
            loss_list.append(loss.item())

            if(regularization == "L1"):
                l1_lambda = 0.0001
                l1_norm = sum(torch.norm(param, 1) for param in model.parameters())

                loss = loss + l1_lambda * l1_norm
            # elif(regularization == "L2"):
            #     # "from scratch" implementation of L2 regularization, but pytorch has it built in as weight_decay parameter in the optimizer
            #     l2_lambda = 0.001
            #     l2_norm = sum(p.pow(2.0).sum()
            #                     for p in model.parameters())
 
            #     loss = loss + l2_lambda * l2_norm

            # computing accuracy
            train_acc = (preds.argmax(dim=1) == labels).float().mean()
            acc_list.append(train_acc.item())

            # removing accumulated gradients
            optimizer.zero_grad()
    
            # backprogating error to compute gradients
            loss.backward()
    
            # updating arameters
            optimizer.step()
    
            if(i % 100 == 0 or i == len(train_loader) - 1):
                progress_bar.set_description(f"Epoch {epoch + 1} Iter {i + 1}: loss {loss.item():.5f}. ")
        loss_epoch_list.append(loss_list[-1])
        acc_epoch_list.append(acc_list[-1])

        # evaluating on validation set
        model.eval()
        temp_accuracy, temp_y_true, temp_y_pred, temp_eval_loss_list = evaluations.evaluate_model(model, criterion, eval_loader, device)
        loss_epoch_list_eval.append(temp_eval_loss_list[-1])
        acc_epoch_list_eval.append(temp_accuracy)

        if(scheduler != None):
            scheduler.step(eval_loader / len(eval_loader))

    return loss_list, acc_list, loss_epoch_list, acc_epoch_list, loss_epoch_list_eval, acc_epoch_list_eval