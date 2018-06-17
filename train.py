import numpy as np
import torch
from torch.autograd import Variable
from matplotlib import pyplot as plt
import os
from scipy import ndimage



## based off of code from https://www.kaggle.com/glenslade/alternative-metrics-kernel?scriptVersionId=2617366
def mean_avg_prec(y_true, y_pred, threshold): 
    
    #print("max = ",np.max(y_pred))
    #print("min = ",np.min(y_pred))
    #y_pred = 127*(y_pred+1)
    y_pred = (y_pred >= 0.2).astype(np.uint8)
    #print(y_pred)
    # Compute mask and number of objects
    #print("max y_pred = ",np.max(y_pred))
    #print("min y_pred = ",np.min(y_pred))
    mask_pred, num_pred = ndimage.label(y_pred)
    mask_true, num_true = ndimage.label(y_true)
    
    y_pred_arr = np.zeros((num_pred, np.squeeze(y_pred).shape[0], np.squeeze(y_pred).shape[1]))
    y_true_arr = np.zeros((num_true, np.squeeze(y_true).shape[0], np.squeeze(y_true).shape[1]))
    
    for i in range(1,num_pred): 
        y_pred_arr[i] = np.where(mask_pred==i, 1, 0)
    
    for i in range(1,num_true): 
        y_true_arr[i] = np.where(mask_true==i, 1,0)
        
    #for mask in y_true_arr: 
    #    plt.imshow(mask, cmap="gray")
    #    plt.show()
    
    #plt.imshow(np.squeeze(mask_pred), cmap="gray")
    #plt.show()
    
    #plt.imshow(np.squeeze(y_pred), cmap="gray")
    #plt.show()
    
    #plt.imshow(np.squeeze(y_true),cmap="gray")
    #plt.show()
    
    
    
    #print("Number of true objects: %d" % num_true)
    #print("Number of predicted objects: %d" % num_pred)

    
    # Compute iou score for each prediction
    iou = []
    for pr in range(num_pred):
        bol = 0  # best overlap
        bun = 1e-9  # corresponding best union
        for tr in range(num_true):
            olap = y_pred_arr[pr] * y_true_arr[tr]  # Intersection points
            osz = np.sum(olap)  # Add the intersection points to see size of overlap
            if osz > bol:  # Choose the match with the biggest overlap
                bol = osz
                bun = np.sum(np.maximum(y_pred_arr[pr], y_true_arr[tr]))  # Union formed with sum of maxima
        iou.append(bol / bun)

    # Loop over IoU thresholds
    p = 0
    #print("Thresh\tTP\tFP\tFN\tPrec.")
    

    for t in np.arange(0.5, 1.0, 0.05):
        matches = iou > t
        tp = np.count_nonzero(matches)  # True positives
        fp = num_pred - tp  # False positives
        fn = num_true - tp  # False negatives
        if tp+fp+fn == 0:  
            p += 1
        else:
            p += tp / (tp + fp + fn)
            

        
        #print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, tp / (tp + fp + fn)))


    
    mean_avg_p = p/10
    #print("AP\t-\t-\t-\t{:1.3f}".format(mean_avg_p))
    
    return mean_avg_p
    

def f1_score(y_true, y_pred, threshold):


    y_pred = (y_pred >= threshold).astype(np.uint8)
    y_pred_neg = y_pred==0
    
    true_positive = np.sum(y_pred * y_true)
    y_false = y_true==0
    true_negative = np.sum(y_pred_neg * y_false)

    total_positive = np.sum(y_pred)
    total_true = np.sum(y_true)
    total_false = np.sum(y_false)

    precision = true_positive/total_positive if total_positive != 0 else 0
    recall = true_positive/total_true if total_true != 0 else 0
    specificity = true_negative/total_false if total_false != 0 else 0
        
    f1 = (2*precision*recall)/(precision+recall) if (precision+recall) != 0 else 0
    
    return f1, precision, recall, specificity


def eval_net(GPU, model, EMDataLoader, criterion, val_img, threshold): 
    #model.eval()
    
    avg_loss = 0
    avg_f1_score = 0
    avg_precision = 0
    avg_recall = 0
    avg_specificity = 0
    #mean_average_precision = 0
    
    dataset_size = 0
    
    for i, data in enumerate(EMDataLoader):
        
        imgs = data[0]
        lbls = data[1]
        
        
        
        for (img,lbl) in zip(imgs,lbls): 
            lbl[lbl!=0] = 1
        
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
            
            
            #plt.imshow(np.squeeze(output.data.cpu().numpy()), cmap="gray")
            #plt.title("test %d" % dataset_size)
            #plt.show()
            
            #pred = model(val_img)
            #pred = pred.data.cpu().numpy()
            #pred = pred.squeeze()
            
            #print(pred)
            #plt.imshow(pred, cmap="gray")
            #plt.show()
            
            
            f1, precision, recall, specificity = f1_score(target.data.cpu().numpy(), output.data.cpu().numpy(), threshold)
          
                
            
            #mean_average_precision += mean_avg_prec(target.data.cpu().numpy(), output.data.cpu().numpy(), 0.1)

            avg_f1_score += f1
            avg_precision += precision
            avg_recall += recall
            avg_specificity += specificity
            
            dataset_size += 1


    avg_loss /= dataset_size
    avg_f1_score /= dataset_size
    avg_precision /= dataset_size
    avg_recall /= dataset_size
    avg_specificity /= dataset_size
    #mean_average_precision /= dataset_size

    #model.train()
    
    return avg_loss, avg_f1_score, avg_precision, avg_recall, avg_specificity

def get_map(GPU, model, EMDataLoader): 
    #model.eval()
    
    
    mean_average_precision = 0
    
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
            
            mean_average_precision += mean_avg_prec(target.data.cpu().numpy(), output.data.cpu().numpy(), 0.1)
            
            dataset_size += 1


    mean_average_precision /= dataset_size

    #model.train()
    
    return mean_average_precision
    
    
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
    tr_map = []
    
    val_loss = []
    val_f1 = []
    val_prec = []
    val_rec = []
    val_map = []
    
    if warm_start:
         model = load_pretrained_models(model, output_dir)
            
    model.train()
    
    for epoch in range(epochs): 
        
        avg_loss = 0
        for i, data in enumerate(TrainDataLoader):
            
            imgs = data[0]
            lbls = data[1]
            lbls[lbls!=0] = 1
            
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
        
        #print(prediction)
        
        v_img = np.squeeze(val_img.data.cpu().numpy())
        #print(v_img.shape)
        v_img = np.moveaxis(v_img,0,-1) if v_img.shape[0] == 3 else np.squeeze(v_img)
        v_img = 127*(v_img+1)
        
        #print(v_img.shape)
        
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
        train_avg_loss, train_avg_f1_score, train_avg_precision, train_avg_recall, train_avg_specificity = eval_net(GPU, model, TrainDataLoader, criterion, val_img, 0.1)
        val_avg_loss, val_avg_f1_score, val_avg_precision, val_avg_recall, val_avg_specificity = eval_net(GPU, model, ValDataLoader, criterion, val_img, 0.1)

        
        
        
        tr_loss.append(train_avg_loss)
        tr_f1.append(train_avg_f1_score)
        tr_prec.append(train_avg_precision)
        tr_rec.append(train_avg_recall)
        #tr_map.append(train_mean_avg_p)

        val_loss.append(val_avg_loss)
        val_f1.append(val_avg_f1_score)
        val_prec.append(val_avg_precision)
        val_rec.append(val_avg_recall)
        #val_map.append(val_mean_avg_p)
        

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
        
        #ax4.plot(range(epoch+1), tr_map, label="train MAP")
        #ax4.plot(range(epoch+1), val_map, label="val MAP")
        #ax4.set_title("Mean Average Precision")
        #ax4.legend(loc="upper left")
        
        ax1.set(xlabel='epochs')
        ax2.set(xlabel='epochs')
        ax3.set(xlabel='epochs')
        #ax4.set(xlabel='epochs')
        
        ax2.tick_params(axis='both', which='both', labelsize=7)
        ax3.tick_params(axis='both', which='both', labelsize=7)

        plt.suptitle("Epoch: %d" %(epoch))
        plt.show()

        print("epoch [%d/%d] " % (epoch,epochs))
        print("train loss = %.4f, train f1 score = %.4f \ntrain precision = %.4f, train recall = %.4f \n" % (train_avg_loss,train_avg_f1_score, train_avg_precision,train_avg_recall))
        print("val loss = %.4f, val f1 score = %.4f \nval precision = %.4f, val recall = %.4f \n\n" % (val_avg_loss,val_avg_f1_score,val_avg_precision,val_avg_recall))
        
        if epoch % 20 == 0 and epoch != 0: 
            
            train_mean_avg_p = get_map(GPU, model, TrainDataLoader)
            val_mean_avg_p = get_map(GPU, model, ValDataLoader)
            
            tr_map.append(train_mean_avg_p)
            val_map.append(val_mean_avg_p)

            
            recs = []
            precs = []
            specs = []
            for j in np.arange(0.1, 1.0, 0.1): 
                _a,_b, p, r,s = eval_net(GPU, model, TrainDataLoader, criterion, val_img, j)
                precs.append(p)
                recs.append(r)
                specs.append(1-s)
                
            val_recs = []
            val_precs = []
            val_specs = []
            for j in np.arange(0.1, 1.0, 0.1): 
                _a,_b, p, r,s = eval_net(GPU, model, ValDataLoader, criterion, val_img, j)
                val_precs.append(p)
                val_recs.append(r)
                val_specs.append(1-s)


            
            f, (ax1,ax2,ax3) = plt.subplots(1,3, figsize=(15,5), dpi=80)
            ax1.plot(range(int(epoch/20)), tr_map, label="train MAP")
            ax1.plot(range(int(epoch/20)), val_map, label="val MAP")
            ax1.set_title('Mean Average Precision')
            ax1.set(xlabel='epoch * 20')
            ax1.set(ylabel='Mean Average Precision')
            ax1.legend(loc="upper left")

            ax2.plot(recs, precs, label="train curve")
            ax2.plot(val_recs, val_precs, label="val curve")
            ax2.set_title("Precision-Recall Curve")
            ax2.set(xlabel='Recall')
            ax2.set(ylabel='Precision')
            ax2.legend(loc="lower left")
            

            ax3.plot(specs, recs, label="train curve")
            ax3.plot(val_specs, val_recs, label="val curve")
            ax3.set_title("ROC Curve")
            ax3.set(xlabel='1-Specificity')
            ax3.set(ylabel='Sensitivity')
            ax3.legend(loc="lower right")

            plt.suptitle("Epoch: %d" %(epoch))
            plt.show()
            
            
            
            
            torch.save(model.state_dict(), '%s/model_epoch%d.pth' % (output_dir, epoch))

        
        
