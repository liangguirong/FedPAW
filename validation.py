import numpy as np
import torch
from torch.nn import functional as F
from eval.utils.metrics import compute_metrics_test
import torch.nn as nn

def epochVal_metrics_test(model, dataLoader, thresh):
    training = model.training
    model.eval()

    gt = torch.FloatTensor().cuda()
    pred = torch.FloatTensor().cuda()
    losses = []
    gt_study = {}
    pred_study = {}
    studies = []
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        for i, (study, index, image,image2,_,_,labels_id) in enumerate(dataLoader):
            image, label = image.cuda(), labels_id.cuda()
            output = model(image)
            output = F.softmax(output, dim=1)
            _, max_idx = torch.max(label, dim=-1)
            loss = loss_fn(output, max_idx.long())
            losses.append(loss.item())

            for i in range(len(study)):
                if study[i] in pred_study:
                    assert torch.equal(gt_study[study[i]], label[i])
                    pred_study[study[i]] = torch.max(pred_study[study[i]], output[i])
                else:
                    gt_study[study[i]] = label[i]
                    pred_study[study[i]] = output[i]
                    studies.append(study[i])

          
        
        for study in studies:
            gt = torch.cat((gt, gt_study[study].view(1, -1)), 0)
            pred = torch.cat((pred, pred_study[study].view(1, -1)), 0)
       
        AUROCs, Accus, Senss, Specs, pre, F1 = compute_metrics_test(gt, pred,  thresh=thresh, competition=True)
    
    model.train(training)
    loss = np.array(losses).mean()

    return AUROCs, Accus, Senss, Specs, pre, F1, loss
