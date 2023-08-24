from torch import optim
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from utils.AverageMeter import *
import torch.nn.functional as F
from scipy import stats
from torch.autograd import Variable
device = torch.device('cuda:0')

class Client:
    def __init__(self, args, num_classes, con, lambda_a, lambda_i,train_loader,train_loader_u):
        """
       Client init method

       Parameters:
           args: List containing arguments to configure the training

       Returns:
           None
       """
        self.num_classes = num_classes
        self.client_id = None
        self.local_model = self.build_models()
        self.train_loader = None
        self.train_loader_u = None
        self.val_loader = None
        self.steps = 0
        self.con = con
        self.args = args
        self.lambda_a = lambda_a
        self.lambda_i = lambda_i
        self.lambda_j = args.lambda_j
        self.mu = 1e-2
        self.criterion = nn.CrossEntropyLoss()
        self.kl_divergence = nn.KLDivLoss()
        self.mse = torch.nn.MSELoss()
        self.peer = self.build_models()
        self.minw = args.minw
        if self.num_classes == 7:
            self.ema_p = 0.99
        else:
            self.ema_p = 0.999
        self.train_loader = train_loader
        self.train_loader_u = train_loader_u
        self.sum_n = len(self.train_loader_u.dataset)+len(self.train_loader.dataset)
    def build_models(self):
        """
          create an efficientnet pretrained model
        """
        model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=self.num_classes)
        num_ftrs = model._fc.in_features
        model._fc = nn.Linear(num_ftrs, self.num_classes)
        model = model.cuda()
        return model
    def set(self, client_id, global_model,alpha_global,beta_global):
        """
         initials client
       """
        self.client_id = client_id
        self.local_model.load_state_dict(global_model)
        self.alpha = alpha_global
        self.beta = beta_global
    def un_supervised_loss(self,x, x_u1, x_aug, peers_wghts,curr_round):
        """
             unsupervised loss
             Parameters:
                 x: input batch
                 x_aug: augment input batch
                 peers_wghts: similar peers
             Returns:
                 loss
            """
        loss_u = 0
        if not self.peers_found:
            with torch.no_grad():
                y_pred = self.local_model(x)
                prob = torch.softmax(y_pred.detach(), dim=-1)
            self.update_beta(prob)
            mask = self.calculate_mask(prob)
            conf = [i for i in range(len(prob)) if mask[i] >= self.minw]
            if len(conf) > 0:
                x_conf = x_aug[conf]
                y_hard = self.local_model(x_conf)
                mx_prob, mx_ind = torch.max(prob, axis=1)
                mx_ind = mx_ind[conf]
                log_pred = F.log_softmax(y_hard, dim=-1)
                masked_loss = F.nll_loss(log_pred, mx_ind.long(), reduction='none') * mask[conf].float()
                loss_u += masked_loss.mean() * self.lambda_a
            return loss_u, len(conf), 0, 0
        else:
            with torch.no_grad():
                y_pred = self.local_model(x)
                ye_pred = self.local_model(x_u1)
                prob = torch.softmax(y_pred.detach(), dim=-1)
            peer_preds = []
            peere_preds = []
            for wght in peers_wghts:
                self.peer.load_state_dict(wght)
                self.peer.eval()
                with torch.no_grad():
                    peer_pred = self.peer(x)
                    peere_pred = self.peer(x_u1)
                    peer_preds.append(peer_pred)
                    peere_preds.append(peere_pred)
            sum_prob = self.stabilization_loss(y_pred, ye_pred, peer_preds, peere_preds, curr_round)
            self.update_beta(prob)
            sum_prob = torch.softmax(sum_prob, dim=-1)
            mask = self.calculate_mask(sum_prob)
            mx_prob, mx_ind = torch.max(sum_prob, axis=1)
            conf = [i for i in range(len(sum_prob)) if mask[i] >= self.minw]
            if len(conf) > 0:
                x_conf = x_aug[conf]
                y_hard = self.local_model(x_conf)
                mx_ind = mx_ind[conf]
                log_pred = F.log_softmax(y_hard, dim=-1)
                masked_loss = F.nll_loss(log_pred, mx_ind.long(), reduction='none') * mask[conf].float()
                loss_u += masked_loss.mean() * self.lambda_a
            return loss_u, len(conf), 0, 0
   
    def draw(self):
        ab_pairs = []
        for i in range(self.args.num_classes):
            if len(self.sum_x[i]) == 0:
                self.min_x.append([])
                self.max_x.append([])
                ab_pairs.append((self.alpha[i], self.beta[i]))
            else:
                self.sum_x[i] = sorted(self.sum_x[i], key=lambda x: x)
                self.min_x.append(self.sum_x[i][0])
                self.max_x.append(self.sum_x[i][-1])
                ab_pairs.append((self.alpha[i], self.beta[i]))
    def train(self,client_id, curr_round, curr_lr, peers_wghts, fed_prox, global_model):
        """
         trains client locally

         Parameters:
             client_id:
             curr_round:
             steps:
             curr_lr:
             peers_wghts:
             fed_prox:
             global_model

         Returns:
             client state
       """
        self.local_model.train()
        if peers_wghts != None:
            self.peers_found = True
        else:
            self.peers_found = False

        optimizer = optim.Adam(self.local_model.parameters(), lr=curr_lr)
        criterion = nn.CrossEntropyLoss()
        train_loss = AverageMeter()
        train_acc = AverageMeter()
        trainloader_l_iter = enumerate(self.train_loader)
        trainloader_u_iter = enumerate(self.train_loader_u)
        self.steps = len(self.train_loader_u)
        if curr_round < 1:
            self.last_p_model = (torch.ones(self.args.num_classes) * self.steps).cuda()
        conf_cont = []
        ce_lsses = []
        mse_lsses = []
        self.sum_x = []
        self.min_x = []
        self.max_x = []
        self.sum_p_model = (torch.zeros(self.args.num_classes)).cuda()
        for i in range(self.args.num_classes):
            self.sum_x.append([])
        for i in range(self.steps):
            # Check if the label loader has a batch available
            try:
                _, sample_batched = next(trainloader_l_iter)
            except:
                # Curr loader doesn't have data, then reload data
                del trainloader_l_iter
                trainloader_l_iter = enumerate(self.train_loader)
                _, sample_batched = next(trainloader_l_iter)

            try:
                _, sample_batched_u = next(trainloader_u_iter)
            except:
                del trainloader_u_iter
                trainloader_u_iter = enumerate(self.train_loader_u)
                _, sample_batched_u = next(trainloader_u_iter)

            x = sample_batched[0].type(torch.cuda.FloatTensor)
            y = sample_batched[1].type(torch.cuda.LongTensor)
            x_u = sample_batched_u[0].type(torch.cuda.FloatTensor)
            x_u_aug = sample_batched_u[2].type(torch.cuda.FloatTensor)
            x_u1 = sample_batched_u[3].type(torch.cuda.FloatTensor)
            n = x.size(0)

            with torch.set_grad_enabled(True):
                output = self.local_model(x)
                loss = criterion(output, y)
                loss_u, conf, ce_lss, mse_lss = self.un_supervised_loss(x_u, x_u1, x_u_aug, peers_wghts, curr_round)
                loss += loss_u
                conf_cont.append(conf)
                ce_lsses.append(ce_lss)
                mse_lsses.append(mse_lss)

                #########################we implement FedProx Here###########################
                # referring to https://github.com/IBM/FedMA/blob/4b586a5a22002dc955d025b890bc632daa3c01c7/main.py#L819
                if fed_prox == 'True' and i > 0 and curr_round > 0:
                    w_diff = torch.tensor(0., device=device)
                    for w, w_t in zip(global_model.parameters(), self.local_model.parameters()):
                        w_diff += torch.pow(torch.norm(w - w_t), 2)
                    loss += self.mu / 2. * w_diff
                #############################################################################

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                prediction = output.cpu().max(1, keepdim=True)[1]
                train_acc.update(prediction.cpu().eq(y.cpu().view_as(prediction.cpu())).sum().item() / n)
                train_loss.update(loss.item())
        self.draw()
        self.last_p_model = self.sum_p_model
        print('rnd:{}, clnt{}: trlss:{}, tracc:{}'.format(curr_round, client_id, round(train_loss.avg, 4),
                                                          round(train_acc.avg, 4)))
        return (self.sum_n, self.local_model.state_dict(), train_loss, train_acc,self.alpha,self.beta,self.min_x,self.max_x)


