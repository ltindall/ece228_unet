import numpy as np
import torch
from torch.autograd import Variable
from matplotlib import pyplot as plt
import os



def f1_score(y_true, y_pred, threshold):


    y_pred = (y_pred >= threshold).astype(np.uint8)
    
    true_positive = np.sum(y_pred * y_true)

    total_positive = np.sum(y_pred)
    total_true = np.sum(y_true)
    
    precision = true_positive/total_positive if total_positive != 0 else 0
    recall = true_positive/total_true if total_true != 0 else 0
        
    f1 = (2*precision*recall)/(precision+recall) if (precision+recall) != 0 else 0
    
    return f1, precision, recall


def eval_net(GPU, model, EMDataLoader, criterion): 
    model.eval()
    
    avg_loss = 0
    avg_f1_score = 0
    avg_precision = 0
    avg_recall = 0
    
    dataset_size = 0
    
    for i, data in enumerate(EMDataLoader):
        
        imgs = data[0]
        lbls = data[1]
        
        
        for (img,lbl) in zip(imgs,lbls): 
        
            # convert to pytorch cuda variable 
            x = Variable(torch.FloatTensor(img)).detach()
            target = Variable(torch.FloatTensor(lbl)).detach()
            if GPU: 
                x = x.cuda()
                target = target.cuda()

            x = torch.unsqueeze(x,0)
            target = torch.unsqueeze(target, 0)

            # get output and loss 
            output = model(x)
            loss = criterion(output, target)

            avg_loss += loss.data[0] 

            f1, precision, recall = f1_score(target.data.cpu().numpy(), output.data.cpu().numpy(), 0.1)

            avg_f1_score += f1
            avg_precision += precision
            avg_recall += recall
            
            dataset_size += 1


    avg_loss /= dataset_size
    avg_f1_score /= dataset_size
    avg_precision /= dataset_size
    avg_recall /= dataset_size

    model.train()
    
    return avg_loss, avg_f1_score, avg_precision, avg_recall
    
    
def load_pretrained_models(model, output_dir):
    
    
    if os.path.isdir(output_dir):
        model_list = [f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))]

        model_list.sort()
        
        if len(model_list) > 0: 
            model.load_state_dict(torch.load('%s/%s' % (output_dir, model_list[-1])))
            print("loaded model %s" % model_list[-1])
        else: 
            print("No model loaded. Fresh start.")
           

    return model




def training(GPU, TrainDataLoader, ValDataLoader, ValDataset, model, optimizer, criterion, epochs, batch_size, output_dir, warm_start=True):
    
    
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
        
    num_batches = len(TrainDataLoader)
    
    tr_loss = []
    tr_f1 = []
    tr_prec = []
    tr_rec = []
    
    val_loss = []
    val_f1 = []
    val_prec = []
    val_rec = []
    
    if warm_start:
         model = load_pretrained_models(model, output_dir)
            
    model.train()
    
    for epoch in range(epochs): 
        
        avg_loss = 0
        for i, data in enumerate(TrainDataLoader):
            
            imgs = data[0]
            lbls = data[1]
            
            # convert to pytorch cuda variable 
            x = Variable(torch.FloatTensor(imgs))
            target = Variable(torch.FloatTensor(lbls))
            if GPU: 
                x = x.cuda()
                target = target.cuda()

                
            # get output and loss 
            output = model(x)
            loss = criterion(output, target)
            
            # backpropagate 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            

        
        # do one random prediction 
        i = np.random.randint(0,len(ValDataset))   
        
        val_img,val_target = ValDataset.__getitem__(i)
        
        
        
        #val_img = val[i]
        val_img = np.expand_dims(val_img, axis=0)
        val_img = Variable(torch.FloatTensor(val_img))
        if GPU: 
            val_img = val_img.cuda()
        prediction = model(val_img)
        prediction = prediction.data.cpu().numpy()
        prediction = prediction.squeeze()
        
        val_img = val_img.data.cpu().numpy()
        v_img = np.moveaxis(val_img,0,-1) if val_img.shape[0] == 3 else np.squeeze(val_img)
        v_img = 127*(v_img+1)
        
        # plot predication and real label 
        f, (ax1,ax2,ax3) = plt.subplots(1,3, sharey=True,figsize=(15,5), dpi=80)
        if v_img.shape[-1] == 3: 
            
            ax1.imshow(v_img.astype(np.uint8))
        else: 
            ax1.imshow(v_img, cmap="gray")
        ax1.set_title("Image")
        ax2.imshow(prediction, cmap="gray")
        ax2.set_title("Prediction")
        ax3.imshow(np.squeeze(val_target), cmap="gray")
        ax3.set_title("Real label")
        plt.suptitle("Epoch: %d" %(epoch))
        plt.show()
        
        #if epoch % 10 == 0: 
        train_avg_loss, train_avg_f1_score, train_avg_precision, train_avg_recall = eval_net(GPU, model, TrainDataLoader, criterion)
        val_avg_loss, val_avg_f1_score, val_avg_precision, val_avg_recall = eval_net(GPU, model, ValDataLoader, criterion)

        tr_loss.append(train_avg_loss)
        tr_f1.append(train_avg_f1_score)
        tr_prec.append(train_avg_precision)
        tr_rec.append(train_avg_recall)

        val_loss.append(val_avg_loss)
        val_f1.append(val_avg_f1_score)
        val_prec.append(val_avg_precision)
        val_rec.append(val_avg_recall)
        

        f, (ax1,ax2,ax3) = plt.subplots(1,3, figsize=(15,5), dpi=80)
        ax1.plot(range(epoch+1),tr_loss, label="train loss")
        ax1.plot(range(epoch+1),val_loss, label="val loss")
        ax1.set_title("BCE Loss")
        ax1.legend(loc="upper left")
        ax2.plot(range(epoch+1),tr_f1, label="train F1")
        ax2.plot(range(epoch+1),val_f1,label="val F1")
        ax2.set_title("F1 Score")
        ax2.legend(loc="lower right")
        ax3.plot(range(epoch+1),tr_prec,label="train precision")
        ax3.plot(range(epoch+1),tr_rec,label="train recall")
        ax3.plot(range(epoch+1),val_prec,label="val precision")
        ax3.plot(range(epoch+1),val_rec,label="val recall")
        ax3.set_title("Precision and Recall")
        ax3.legend(loc="lower right")
        ax1.set(xlabel='epochs')
        ax2.set(xlabel='epochs')
        ax3.set(xlabel='epochs')

        ax2.tick_params(axis='both', which='both', labelsize=7)
        ax3.tick_params(axis='both', which='both', labelsize=7)

        plt.suptitle("Epoch: %d" %(epoch))
        plt.show()

        print("epoch [%d/%d] " % (epoch,epochs))
        print("train loss = %.4f, train f1 score = %.4f \ntrain precision = %.4f, train recall = %.4f \n" % (train_avg_loss,train_avg_f1_score, train_avg_precision,train_avg_recall))
        print("val loss = %.4f, val f1 score = %.4f \nval precision = %.4f, val recall = %.4f \n\n" % (val_avg_loss,val_avg_f1_score,val_avg_precision,val_avg_recall))
        
        if epoch % 100 == 0: 
            torch.save(model.state_dict(), '%s/model_epoch%d.pth' % (output_dir, epoch))

        
        
