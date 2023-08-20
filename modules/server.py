from scipy.stats import truncnorm
from modules.client import *
from scipy.spatial import KDTree
from dataloaders import dataset
from torch.utils.data import DataLoader
import copy
import logging
from torchvision import transforms
import os
import sys
import time
from validation import epochVal_metrics_test
from scipy import stats
import numpy as np
from dataloaders.randaug import RandAugment
import pandas as pd
device = torch.device('cuda:0')
class Server:
    def __init__(self, args):
        """
        Server init method

        Parameters:
            args: List containing arguments to configure the training

        Returns:
            None
        """

        self.args = args
        self.global_model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=self.args.num_classes)
        num_ftrs = self.global_model._fc.in_features
        self.global_model._fc = nn.Linear(num_ftrs, self.args.num_classes)
        self.global_model = self.global_model.cuda()
        self.train_loaders = []
        self.val_loaders = []
        self.test_loaders = []
        self.server_loader = None
        self.betaMatrix = np.zeros((self.args.num_clients, self.args.num_clients))
        'create a dumy image used to measure the similarity between the clients'
        mu, std, lower, upper = 0, 1, 0, 255
        self.dumyImg = torch.from_numpy(
            (truncnorm((lower - mu) / std, (upper - mu) / std, loc=mu, scale=std).rvs((1, 3, 224, 224))) / 255).type(
            torch.cuda.FloatTensor)
    def build_clients(self,args):
        """
        build_clients: create client class and peer from a pretrained efficientnet

        Parameters:
            None

        Returns:
            None
        """
        self.trainer_locals = []
        clients = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        for client_id in clients:
            self.client = Client(args,self.num_classes, self.con, self.lambda_a, self.lambda_i,self.train_loaders[client_id],self.train_loaders_u[client_id])
            self.trainer_locals.append(self.client)
        self.peer = self.build_models()
        self.peer.eval()
    def configure(self):
        """
        configure server class from user argument

        Parameters:
            None

        Returns:
            None
        """
        self.steps = self.args.steps
        self.num_clients = self.args.num_clients
        self.num_classes = self.args.num_classes
        self.connected_clients = self.args.connected_clients
        self.batch_size = self.args.batch_size
        self.num_rounds = self.args.num_rounds
        self.curr_lr = self.args.curr_lr
        self.lambda_a = self.args.lambda_a
        self.lambda_i = self.args.lambda_i
        self.con = self.args.con
        self.clients_state = self.args.clients_state
        self.num_peers = self.args.num_peers
        self.trained_clients = []
        self.vid_to_cid = {}
        self.client_pred = {}
        self.clnts_bst_acc = [0] * self.args.num_clients
        self.client_peers = {
            'client0': [],
            'client1': [],
            'client2': [],
            'client3': [],
            'client4': [],
            'client5': [],
            'client6': [],
            'client7': [],
            'client8': [],
            'client9': []
        }
        num_features = 622
        self.method = self.args.method
        self.is_normalized = self.args.is_normalized
        self.include_acc = self.args.include_acc
        if self.include_acc == True:
            num_features += 2
        self.save_check = self.args.save_check
        self.is_PA = self.args.is_PA
        self.include_C8 = self.args.include_C8
        self.fed_prox = self.args.fed_prox
        self.name = self.method + '_c8' + str(self.include_C8) + '_avg' + str(self.is_PA) + '_prox' + str(self.fed_prox)

    def build_models(self):
        """
        create an efficientnet pretrained model

        Parameters:
            None

        Returns:
            None
        """
        model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=self.num_classes)
        num_ftrs = model._fc.in_features
        model._fc = nn.Linear(num_ftrs, self.num_classes)
        model = model.cuda()
        return model
    def set_configuration(self, lambda_a=0.5, con=0.6, curr_lr=0.00005, num_rounds=200):
        """
        modify some server configurations

        Parameters:
            lambda_a: unlabeled coefficient of type float
            con: confidence treashold of type float
            curr_lr: learning rate of type float
            num_rounds: number of round of type int

        Returns:
            None
        """
        self.num_rounds = num_rounds
        self.curr_lr = curr_lr
        self.lambda_a = lambda_a
        self.con = con
    def load_client_weights(self, client_id):
        """
       Loads client weights

       Parameters:
           client_id: client id of type int

       Returns:
           client
       """
        checkpoint = torch.load(os.path.join(self.clients_state, 'Client{}.pt'.format(client_id)))
        return checkpoint['model_state_dict']
    def build_client_similarity(self, updates):
        """
        build similarities based on the clients predictions

        Parameters:
            updates: tuple contains clients

        Returns:
            KDTree clients similarities in KDTree format
        """
        for client_id, _, model in updates:
            self.peer.load_state_dict(model)
            self.peer.eval()
            with torch.no_grad():
                self.client_pred[client_id] = np.squeeze(self.peer(self.dumyImg).cpu().numpy())
        self.vid_to_cid = list(self.client_pred.keys())
        self.vectors = list(self.client_pred.values())
        self.tree = KDTree(self.vectors)
    def bray_curtis_distance(self,x, y):
        """
        计算布雷柯蒂斯距离
        :param x: 第一条曲线，一个二维numpy数组
        :param y: 第二条曲线，一个二维numpy数组
        :return: 两条曲线之间的布雷柯蒂斯距离
        """
        x = np.asarray(x)
        y = np.asarray(y)
        numerator = np.abs(x - y).sum()
        denominator = np.abs(x + y).sum()
        distance = numerator / denominator
        similarity = 1 / (distance + 1)
        return similarity
    def similar_matrix(self,min_x,max_x,alpha,beta):
        for h in range(self.args.num_clients):
            for i in range(self.args.num_clients):
                sum_similarity = 0
                for j in range(self.args.num_classes):
                    if min_x[h][j] == [] and min_x[i][j] == []:
                        similarity = 1
                    else:
                        if min_x[h][j] != []:
                            x1_min = min_x[h][j]
                            x1_max = max_x[h][j]
                        else:
                            x1_min = min_x[i][j]
                            x1_max = max_x[i][j]
                        if min_x[i][j] != []:
                            x2_min = min_x[i][j]
                            x2_max = max_x[i][j]
                        else:
                            x2_min = min_x[h][j]
                            x2_max = max_x[h][j]
                        a1, b1 = alpha[h][j], beta[h][j]
                        a2, b2 = alpha[i][j], beta[i][j]
                        if a1 <= 1 or b1 <= 1:
                            max_id = torch.tensor(0).cuda()
                        else:
                            max_id = (a1 - 1) / (a1 + b1 - 2)
                        max_pdf = stats.beta(a1, b1).pdf(max_id)
                        x1 = torch.clamp(torch.tensor(np.linspace(x1_min.cpu(), x1_max.cpu(), 100)),
                                         max=max_id.cpu())
                        y1 = stats.beta(a1, b1).pdf(x1) / max_pdf
                        if a2 <= 1 or b2 <= 1:
                            max_id = torch.tensor(0).cuda()
                        else:
                            max_id = (a2 - 1) / (a2 + b2 - 2)
                        max_pdf = stats.beta(a2, b2).pdf(max_id)
                        x2 = torch.clamp(torch.tensor(np.linspace(x2_min.cpu(), x2_max.cpu(), 100)),
                                         max=max_id.cpu())
                        y2 = stats.beta(a2, b2).pdf(x2) / max_pdf
                        similarity = self.bray_curtis_distance(np.column_stack((x1, y1)), np.column_stack((x2, y2)))
                    sum_similarity = sum_similarity + similarity
                self.betaMatrix[h][i] = sum_similarity / self.args.num_classes
    def get_similar_clients(self, client_id, n=1, cid2=-1, client_gate=0):
        clients_sim = self.betaMatrix[client_id]
        arg_sort = reversed(
            np.argsort(clients_sim))  # from smaller to larger value-->the bigger value at the end
        clients_idx = []
        cnt = 0
        for i in arg_sort:
            if i != client_id and i != cid2 and clients_sim[i] >= client_gate:
                clients_idx.append(i)
                cnt += 1
                if cnt == n:
                    break
        if len(clients_idx) == 0:
            return None
        return clients_idx
    def get_peers(self, client_id, curr_round):
        """
        find top T similar peers (T = args.num_peers)

        Parameters:
            client_id: client id of type int
            curr_round: current round of type int

        Returns:
            weights of top T similar peers when is_PA = False
            or the average weights of top T similar peers when is_PA = True
        """
        weights = []
        if self.include_C8:
            cd2 = -1
        else:
            cd2 = 8
        if client_id in self.trained_clients and self.method != 'Random':
            # Get peers based on the best validation accuracy
            if self.method == 'FedPAW':
                sims = self.get_similar_clients(client_id, self.num_peers)
                for pid in sims:
                    self.client_peers['client' + str(client_id)].append(pid)
                    w = self.load_client_weights(pid)
                    if self.is_PA:
                        weights.append((pid, len(self.train_loaders_u[pid].dataset)+len(self.train_loaders[pid].dataset), copy.deepcopy(w)))
                    else:
                        weights.append(copy.deepcopy(w))
        # Random peers
        else:
            return None
        if self.is_PA:
            return [self.average(weights)]
        else:
            return weights[:self.num_peers]

    def get_best_acc_client(self, client_id, n, cid2=-1):
        """
       Get indexes for n peers best validation accuracy on the server validation data

       Parameters:
           client_id: client id of type int
           n: number of returned indexes of type int
           cid2: excluded from the search (i.e. when it is already included)

       Returns:
           list of n peers indexes
       """
        arg_sort = reversed(np.argsort(self.clnts_bst_acc))
        clients_idx = []
        cnt = 0
        if self.include_C8:
            c8 = -1
        else:
            c8 = 8
        for i in arg_sort:
            if i != client_id and i != cid2 and i != c8:
                clients_idx.append(i)
                cnt += 1
                if n == 1:
                    return i
                if cnt == n:
                    break
        return clients_idx
    def aggregate(self, w_locals):
        """
        FedAvg method based on the samples in each client

        Parameters:
            w_locals: List containing <sample_numbers, local  parameters> pairs of all local clients

        Returns:
            averaged_params: aggregated model
        """
        training_num = 0
        for idx in range(len(w_locals)):
            (_, sample_num, averaged_params) = w_locals[idx]
            training_num += sample_num
        (_, sample_num, averaged_params) = w_locals[0]
        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                _, local_sample_number, local_model_params = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w
        return averaged_params
    def aggregateab(self, w_locals):
        """
        FedAvg method based on the samples in each client

        Parameters:
            w_locals: List containing <sample_numbers, local  parameters> pairs of all local clients

        Returns:
            averaged_params: aggregated model
        """
        training_num = 0
        for idx in range(len(w_locals)):
            (_, sample_num, averaged_params) = w_locals[idx]
            training_num += sample_num

        (_, sample_num, averaged_params) = w_locals[0]
        for k in range(len(averaged_params)):
            for i in range(0, len(w_locals)):
                _, local_sample_number, local_model_params = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w
        return averaged_params

    def average(self, w_locals):
        """
        Averages top T peers weights

        Parameters:
            w_locals: List containing <sample_numbers, local  parameters> pairs of all local clients

        Returns:
            averaged_params: top T similar peers averaged model
        """
        training_num = 0
        for idx in range(len(w_locals)):
            (_, sample_num, averaged_params) = w_locals[idx]
            training_num += sample_num

        (_, sample_num, averaged_params) = w_locals[0]
        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                _, local_sample_number, local_model_params = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w
        return averaged_params
    def split(self,dataset, num_users):
        server_num = int(0.2 * len(dataset))
        num_items = int((len(dataset) - server_num) / num_users)
        dict_users, all_idxs = {}, [i for i in range(len(dataset))]
        dict_server = set(np.random.choice(all_idxs, server_num, replace=False))
        all_idxs = list(set(all_idxs) - dict_server)
        for i in range(num_users):
            dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
            all_idxs = list(set(all_idxs) - dict_users[i])
        dict_users[num_users - 1] = set(list(dict_users[num_users - 1]) + all_idxs)
        return dict_users, dict_server
    def dirichlet_split_noniid(self,train_labels, alpha, n_clients):
        '''
        参数为alpha的Dirichlet分布将数据索引划分为n_clients个子集
        '''
        n_classes = train_labels.max() + 1
        label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)
        # (K, N)的类别标签分布矩阵X，记录每个client占有每个类别的多少

        class_idcs = [np.argwhere(train_labels == y).flatten()
                      for y in range(n_classes)]
        # 记录每个K个类别对应的样本下标

        client_idcs = [[] for _ in range(n_clients)]
        # 记录N个client分别对应样本集合的索引
        for c, fracs in zip(class_idcs, label_distribution):
            # np.split按照比例将类别为k的样本划分为了N个子集
            # for i, idcs 为遍历第i个client对应样本集合的索引
            for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
                client_idcs[i] += [idcs]
        client_idcs = [np.concatenate(idcs) for idcs in client_idcs]
        return client_idcs

    def get_client_dataset(self,train_dataset, X, alpha=0.5):  # 0.5

        train_labels = train_dataset.dataset.labels[train_dataset.indices]
        train_labels = np.array(train_labels)
        # train_labels = np.array(train_dataset.labels)
        client_idcs = self.dirichlet_split_noniid(train_labels, alpha, X)
        client_datasets = []
        for client_indices in client_idcs:
            # client_subset = torch.utils.data.Subset(train_dataset, client_indices)  # 等价？
            client_subset = [train_dataset.indices[i] for i in client_indices]
            client_datasets.append(client_subset)
        return client_datasets
    def prepare_data(self,args):
        self.train_loaders = []
        self.train_loaders_u = []
        self.val_loaders = []
        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
        validation_dataset = dataset.CheXpertDataset(root_dir=args.root_path,
                                               csv_file=args.csv_file_test,
                                               transform=transforms.Compose([
                                                   transforms.Resize((224, 224)),
                                                   transforms.ToTensor(),
                                                   normalize,
                                               ]))
        self.server_loader = DataLoader(dataset=validation_dataset, batch_size=args.batch_size,
                                     shuffle=False, num_workers=args.num_workers, pin_memory=True)
        train_dataset = dataset.CheXpertDataset(root_dir=args.root_path,
                                                csv_file=args.csv_file_train,
                                                transform=transforms.Compose([
                                                    transforms.Resize((224, 224)),
                                                    transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    normalize,
                                                ]),strong_transform=transforms.Compose([
            transforms.Resize((224, 224)),RandAugment(),transforms.ToTensor(), normalize,
        ]))
        sup_train_dataset, unsup_train_dataset = torch.utils.data.random_split(train_dataset,
                                                                               [int(len(train_dataset) * 0.2),
                                                                                len(train_dataset) - int(
                                                                                    len(train_dataset) * 0.2)])
        l_dict_users = self.get_client_dataset(sup_train_dataset, args.num_clients, alpha=args.alpha)
        u_dict_users = self.get_client_dataset(unsup_train_dataset, args.num_clients, alpha=args.alpha)
        for clnt in range(self.num_clients):
            train_ds = dataset.DatasetSplit(train_dataset, l_dict_users[clnt])
            train_dsu = dataset.DatasetSplit(train_dataset, u_dict_users[clnt])
            val_ds = validation_dataset
            train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle='True', num_workers=args.num_workers,
                                      pin_memory=True)
            train_loader_u = DataLoader(train_dsu, batch_size=self.batch_size, shuffle='True', num_workers=args.num_workers,
                                       pin_memory=True)
            val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle='False', num_workers=args.num_workers, pin_memory=True)
            self.train_loaders.append(train_loader)
            self.train_loaders_u.append(train_loader_u)
            self.val_loaders.append(val_loader)

    def append_client(self, client_id):
        """
          Appends current client to the trained clients list

          Parameters:
              client_id: client id of type int

          Returns:
              None
        """
        if client_id not in self.trained_clients:
            self.trained_clients.append(client_id)
    def test(self,args, save_mode_path=None,net=None,val=False):
        if net is not None:
            model = net.cuda()
        else:
            checkpoint_path = save_mode_path
            checkpoint = torch.load(checkpoint_path)
            model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=args.num_classes)
            num_ftrs = model._fc.in_features
            model._fc = nn.Linear(num_ftrs, args.num_classes)
            model = model.cuda()
            model.load_state_dict(checkpoint['state_dict'])
        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
        if val==False:
            csv_file_dataset = args.csv_file_test
        else:
            csv_file_dataset = args.csv_file_val
        val_test_dataset = dataset.CheXpertDataset(root_dir=args.root_path,
                                               csv_file=csv_file_dataset,
                                               transform=transforms.Compose([
                                                   transforms.Resize((224, 224)),
                                                   transforms.ToTensor(),
                                                   normalize,
                                               ]))
        test_dataloader = DataLoader(dataset=val_test_dataset, batch_size=args.batch_size,
                                     shuffle=False, num_workers=args.num_workers, pin_memory=True)
        AUROCs, Accus, Senss, Specs, Preci, F1, loss = epochVal_metrics_test(model, test_dataloader, thresh=0.4)
        AUROC_avg = np.array(AUROCs).mean()
        Accus_avg = np.array(Accus).mean()
        Senss_avg = np.array(Senss).mean()
        Specs_avg = np.array(Specs).mean()
        Preci_avg = np.array(Preci).mean()
        F1_avg = np.array(F1).mean()
        return AUROC_avg, Accus_avg, Senss_avg, Specs_avg, F1_avg, loss
    def save_client_weights(self, client_id, weights):
        """
       Saves client weights

       Parameters:
           client_id: client id of type int
           weights: client weights

       Returns:
           None
       """
        torch.save({
            'client': client_id, 'model_state_dict': weights
        }, os.path.join(self.clients_state, 'Client{}.pt'.format(client_id)))

    def run_fed(self,args):
        logging.basicConfig(filename="log.txt", level=logging.INFO,
                            format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        logging.info(str(args))
        start_time = time.time()
        test_metrics = {
            'test_loss': [],
            'test_auc': [],
            'test_acc': [],
            'test_sen': [],
            'test_spe': [],
            'test_f1': [],
        }
        alpha_global = (torch.ones(self.args.num_classes)).cuda()
        beta_global = (torch.ones(self.args.num_classes)).cuda()
        for curr_round in range(self.num_rounds):
            clnts_updates, sum_alpha, sum_beta, sum_min_x, sum_max_x = [], [], [], [], []
            clnts_alphas = []
            clnts_betas = []
            clients = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            print('<----Training---->')
            print('training clients (round:{}, connected:{})'.format(curr_round, clients))
            for client_id in clients:
                # set initial weights from global model
                w_global = self.global_model.state_dict()
                # initials client
                self.trainer_locals[client_id].set(client_id, w_global, alpha_global, beta_global)
                # get top T similar peers for the client
                peers = self.get_peers(client_id, curr_round)
                # train client locally
                sz, wgt, lss, acc, alpha, beta, min_x, max_x = self.trainer_locals[client_id].train(client_id, curr_round, self.curr_lr,
                                                                                 peers, self.fed_prox,
                                                                                 self.global_model)
                sum_alpha.append(copy.deepcopy(alpha))
                sum_beta.append(copy.deepcopy(beta))
                sum_min_x.append(copy.deepcopy(min_x))
                sum_max_x.append(copy.deepcopy(max_x))
                # save client weights
                self.save_client_weights(client_id, copy.deepcopy(wgt))
                clnts_updates.append((client_id, sz, copy.deepcopy(wgt)))
                clnts_alphas.append((client_id, 0.5, copy.deepcopy(alpha)))
                clnts_betas.append((client_id, 0.5, copy.deepcopy(beta)))

            # update trained clients list
            for client_id in clients:
                self.append_client(client_id)
            # models aggregation
            w_global = self.aggregate(clnts_updates)
            alpha_global = self.aggregateab(clnts_alphas)
            beta_global = self.aggregateab(clnts_betas)
            # update the global model weights for the next round of the training
            self.global_model.load_state_dict(w_global)
            if 'FedPAW' in self.method:
                self.similar_matrix(sum_min_x, sum_max_x, sum_alpha, sum_beta)
            if curr_round % 10 == 0 or curr_round == self.num_rounds - 1:
                save_mode_path = os.path.join(self.clients_state, 'epoch_' + str(curr_round) + '.pth')
                torch.save({
                    'state_dict': w_global,
                }
                    , save_mode_path
                )
                AUROC_avg, Accus_avg, Senss_avg, Specs_avg, F1_avg, loss = self.test(args, save_mode_path)
                logging.info("\nTEST Student: Epoch: {}".format(curr_round))
                logging.info(
                    "\nTEST AUROC: {:.4f}, Accus: {:.4f}, Senss: {:.4f}, Specs: {:.4f}, F1: {:.4f}"
                    .format(AUROC_avg, Accus_avg, Senss_avg, Specs_avg, F1_avg))
                test_metrics['test_auc'].append(AUROC_avg)
                test_metrics['test_acc'].append(Accus_avg)
                test_metrics['test_sen'].append(Senss_avg)
                test_metrics['test_spe'].append(Specs_avg)
                test_metrics['test_f1'].append(F1_avg)
                test_metrics['test_loss'].append(loss)
                logging.info(
                    "\nMeanTEST AUROC: {:.4f}+{:.4f}, Accus: {:.4f}+{:.4f}, Senss: {:.4f}+{:.4f}, Specs: {:.4f}+{:.4f},F1: {:.4f}+{:.4f}"
                    .format(np.mean(test_metrics['test_auc']), np.std(test_metrics['test_auc']),
                            np.mean(test_metrics['test_acc']),
                            np.std(test_metrics['test_acc']),
                            np.mean(test_metrics['test_sen']), np.std(test_metrics['test_sen']),
                            np.mean(test_metrics['test_spe']),
                            np.std(test_metrics['test_spe']),
                            np.mean(test_metrics['test_f1']), np.std(test_metrics['test_f1'])))
        logging.info('server done. ({}s)'.format(time.time() - start_time))
        metrics_pd = pd.DataFrame.from_dict(test_metrics)
        metrics_pd.to_csv(os.path.join(self.clients_state, "test_metrics.csv"))

