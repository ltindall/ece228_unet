import numpy as np
import torch
from torch.autograd import Variable
from matplotlib import pyplot as plt



def f1_score(y_true, y_pred, threshold):


    y_pred = (y_pred >= threshold).astype(np.uint8)
    true_positive = np.sum(y_pred * y_true)

    total_positive = np.sum(y_pred)
    total_true = np.sum(y_true)
    
    precision = true_positive/total_positive if total_positive != 0 else 0
    recall = true_positive/total_true if total_true != 0 else 0
        
    f1 = (2*precision*recall)/(precision+recall) if (precision+recall) != 0 else 0
    
    return f1, precision, recall


def eval_net(GPU, model, inputs, targets, criterion): 
    #model.eval()
    
    avg_loss = 0
    avg_f1_score = 0
    avg_precision = 0
    avg_recall = 0
    
    for (img,lbl) in zip(inputs,targets): 
            
        # get batch of images and labels 
        #imgs = inputs[i*batch_size:(i+1)*batch_size]
        #lbls = targets[i*batch_size:(i+1)*batch_size]

        
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

        f1, precision, recall = f1_score(lbl, output.data.cpu().numpy(), 0.1)
        
        avg_f1_score += f1
        avg_precision += precision
        avg_recall += recall


    avg_loss /= len(targets)
    avg_f1_score /= len(targets)
    avg_precision /= len(targets)
    avg_recall /= len(targets)

    #model.train()
    
    return avg_loss, avg_f1_score, avg_precision, avg_recall
    


def training(GPU, model, inputs, targets,val,val_target, optimizer, criterion, epochs, batch_size):
    model.train()
    
    num_batches = int(len(inputs) / batch_size)
    
    tr_loss = []
    tr_f1 = []
    tr_prec = []
    tr_rec = []
    
    val_loss = []
    val_f1 = []
    val_prec = []
    val_rec = []
    
    for epoch in range(epochs): 
        
        avg_loss = 0
        for i in range(num_batches):
            
            if i*batch_size >= len(inputs):
                break
                
            # get batch of images and labels 
            imgs = inputs[i*batch_size:(i+1)*batch_size]
            lbls = targets[i*batch_size:(i+1)*batch_size]
            
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
            
            
            avg_loss += loss.data[0]
            #print("loss = ",loss.data[0])
            

        avg_loss /= num_batches
        #print('epoch: ' + str(epoch) + ', train loss: ' + str(avg_loss))
        
        
        # do one random prediction 
        i = np.random.randint(0,val.shape[0])        
        val_img = val[i]
        val_img = np.expand_dims(val_img, axis=0)
        val_img = Variable(torch.FloatTensor(val_img))
        if GPU: 
            val_img = val_img.cuda()
        prediction = model(val_img)
        prediction = prediction.data.cpu().numpy()
        prediction = prediction.squeeze()
        
        
        v_img = np.moveaxis(val[i],0,-1) if val[i].shape[0] == 3 else np.squeeze(val[i])
        v_img = 127*(v_img+1)
        
        # plot predication and real label 
        f, (ax1,ax2,ax3) = plt.subplots(1,3, sharey=True,figsize=(15,5), dpi=80)
        if v_img.shape[-1] == 3: 
            ax1.imshow(v_img.astype(np.uint8))
        else: 
            ax1.imshow(v_img.astype(np.uint8),cmap="gray")
        ax1.set_title("Image")
        ax2.imshow(prediction, cmap="gray")
        ax2.set_title("Prediction")
        ax3.imshow(np.squeeze(val_target[i]), cmap="gray")
        ax3.set_title("Real label")
        plt.suptitle("Epoch: %d" %(epoch))
        plt.show()
        
        #if epoch % 10 == 0: 
        train_avg_loss, train_avg_f1_score, train_avg_precision, train_avg_recall = eval_net(GPU, model, inputs, targets, criterion)
        val_avg_loss, val_avg_f1_score, val_avg_precision, val_avg_recall = eval_net(GPU, model, val, val_target, criterion)

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
        
        
