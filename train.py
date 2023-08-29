import argparse
from modules.server import *
import torch.backends.cudnn as cudnn
import numpy as np
import random
import os
device = torch.device('cuda:0')
parser = argparse.ArgumentParser(description='FedPAW')
parser.add_argument('--seed', type=int,  default=1, help='random seed')
parser.add_argument('--clients_state', default='model', type=str, metavar='PATH',
                    help='path to save the clients state temporarly')
parser.add_argument('--num_rounds', type=int, default=100)
parser.add_argument('--steps', type=int, default=200, help='number of iteration')
parser.add_argument('--curr_lr', type=float, default=2e-4)
parser.add_argument('--batch_size', type=int, default=12)
parser.add_argument('--num_classes', type=int, default=7)
parser.add_argument('--con', type=float, default=0.6, help='confidence treashold')
parser.add_argument('--lambda_a', type=float, default=0.5, help='unlabeled coefficient')
parser.add_argument('--lambda_i', type=float, default=0.01, help='consistency coefficient')
parser.add_argument('--lambda_j', type=float, default=1, help='consistency coefficient')
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--num_clients', type=int, default=10, help='num of clients')
parser.add_argument('--connected_clients', type=int, default=10, help='connected clients')
parser.add_argument('--num_peers', type=int, default=2, help='number of peers used in peer learning')
parser.add_argument('--method', default='FedPAW', type=str,
                    help='current options FedPAW')
parser.add_argument('--is_normalized', default='True', help='normalize the features on the similairty matrix')
parser.add_argument('--include_acc', default=False,type=bool,
                    help='include clients accuarcy in the similarity calculation for FedPerl')
parser.add_argument('--save_check', default=True, type=bool,help='save check points')
parser.add_argument('--calculate_val', default=False,type=bool,
                    help='calculate validation accuracy for clients after each round')
parser.add_argument('--is_PA', default=False, type=bool,help='apply peer anonymization')
parser.add_argument('--include_C8', default=True, type=bool,help='include client 8 in the training')
parser.add_argument('--fed_prox', default=False, type=bool,help='apply fedprox')
parser.add_argument('--root_path', type=str, default='D:/Deeplearning/FedIRM-main/data/HAM10000_images_part_1/', help='dataset root dir')
parser.add_argument('--csv_file_train', type=str, default='data/skin_split/train.csv', help='training set csv file')
parser.add_argument('--csv_file_val', type=str, default='data/skin_split/validation.csv', help='validation set csv file')
parser.add_argument('--csv_file_test', type=str, default='data/skin_split/test.csv', help='testing set csv file')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument("--num_workers", default=0, type=int)
parser.add_argument('--ema', type=float,  default=0.999, help='ema_decay')
parser.add_argument('--minw', type=float, default=1e-3)
if __name__ == '__main__':
    args = parser.parse_args()
    os.environ['TORCH_HOME'] = 'pytorch/models'
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    server = Server(args)
    server.configure()
    server.prepare_data(args)
    server.build_clients(args)
    server.run_fed(args)
