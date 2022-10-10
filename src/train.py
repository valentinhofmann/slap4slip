import argparse
import pickle
import random

import numpy as np
import torch

from helpers import get_best
from model import SparseConceptGAE


def main():

    # Read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs.')
    parser.add_argument('--mode', type=str, help='Mode of training.')
    parser.add_argument('--cuda', type=int, default=0, help='Device to train.')
    parser.add_argument('--year', type=int, default=2013, help='Year for which to train model.')
    parser.add_argument('--random_seed', type=int, help='Random seed.')
    parser.add_argument('--lr', type=float, default=1e-04, help='Learning rate.')
    parser.add_argument('--lambda_r', type=float, default=1e-02, help='Regularization constant.')
    parser.add_argument('--window_size', type=int, default=0, help='Window size used for embedding training.')
    parser.add_argument('--theta_c', type=int, default=0, help='Sparsity threshold.')
    parser.add_argument('--linear', default=False, action='store_true', help='Use linear encoder.')
    args = parser.parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    mode = args.mode
    year = args.year

    # Define path to files and filename
    filepath_data = ''
    filepath_model = ''
    filename = '{}_{}'.format(mode, year)
    if mode == 'embs' or mode == 'joint':
        filename += '_{:02d}'.format(args.window_size)
    if args.theta_c:
        filename += '_{:03d}'.format(args.theta_c)
    if args.linear:
        filename += '_l'

    # Load data
    if mode == 'base' or mode == 'counts':
        with open(filepath_data + 'splits/data_counts_{}.p'.format(year), 'rb') as f:
            data = pickle.load(f)
    elif mode == 'embs':
        with open(filepath_data + 'splits/data_embs_{}_{:02d}.p'.format(year, args.window_size), 'rb') as f:
            data = pickle.load(f)
    elif mode == 'joint':
        with open(filepath_data + 'splits/data_counts_{}.p'.format(year), 'rb') as f:
            data_counts = pickle.load(f)
        with open(filepath_data + 'splits/data_embs_{}_{:02d}.p'.format(year, args.window_size), 'rb') as f:
            data_embs = pickle.load(f)

    # Define hyperparameters
    if mode == 'base':
        input_dim = data.x.size(0)
    else:
        input_dim = 1000
    hidden_dim = 100
    output_dim = 10

    # Initialize model
    model = SparseConceptGAE(
        mode,
        input_dim,
        hidden_dim,
        output_dim,
        args.lr,
        args.lambda_r,
        args.linear
    )

    # Define device
    device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() else 'cpu')

    # Move model to device
    model = model.to(device)

    # Prepare data and move to device
    if mode == 'base':
        x = torch.diag(torch.ones(data.x.size(0))).to(device)
        train_pos_edge_index = data.train_pos_edge_index.to(device)
        dev_pos_edge_index = data.val_pos_edge_index
        dev_neg_edge_index = data.val_neg_edge_index
        test_pos_edge_index = data.test_pos_edge_index
        test_neg_edge_index = data.test_neg_edge_index
    elif mode == 'joint':
        x = torch.cat((data_counts.x, data_embs.x), dim=1).to(device)
        train_pos_edge_index = data_counts.train_pos_edge_index.to(device)
        dev_pos_edge_index = data_counts.val_pos_edge_index
        dev_neg_edge_index = data_counts.val_neg_edge_index
        test_pos_edge_index = data_counts.test_pos_edge_index
        test_neg_edge_index = data_counts.test_neg_edge_index
    else:
        x = data.x.to(device)
        train_pos_edge_index = data.train_pos_edge_index.to(device)
        dev_pos_edge_index = data.val_pos_edge_index
        dev_neg_edge_index = data.val_neg_edge_index
        test_pos_edge_index = data.test_pos_edge_index
        test_neg_edge_index = data.test_neg_edge_index

    print('Mode {}, year {}, learning rate {:.0e}, lambda {:.0e}, sparsity {:03d}...'.format(
        mode, year, args.lr, args.lambda_r, args.theta_c))

    best_auc, _, _ = get_best(filepath_model + 'results/{}.txt'.format(filename), args.theta_c)
    print('Best AUC so far: {}'.format(best_auc))

    for epoch in range(1, args.epochs + 1):
        model.train(x, train_pos_edge_index)
        if epoch % 10 == 0:

            # Measure model performance
            auc_dev, ap_dev, y, pred = model.test(x, train_pos_edge_index, dev_pos_edge_index, dev_neg_edge_index)
            auc_test, ap_test, y, pred = model.test(x, train_pos_edge_index, test_pos_edge_index, test_neg_edge_index)

            # Measure model sparsity
            if args.linear:
                sparse_weight = model.model.gae.encoder.linear1.weight
                pruned = len([c for c in sparse_weight.T if torch.all(c == 0)])
            else:
                sparse_weight = model.model.gae.encoder.conv1.weight
                pruned = len([r for r in sparse_weight if torch.all(r == 0)])

            with open(filepath_model + 'results/{}.txt'.format(filename), 'a+') as f:
                f.write('{:.0e}\t{:.0e}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{}\n'.format(
                    args.lr, args.lambda_r, auc_dev, ap_dev, auc_test, ap_test, pruned))
            if best_auc is None or auc_dev > best_auc:
                if pruned >= args.theta_c:
                    best_auc = auc_dev
                    torch.save(model.state_dict(), filepath_model + 'trained/{}.torch'.format(filename))
                    with open(filepath_model + 'preds/{}.p'.format(filename), 'wb') as f:
                        pickle.dump([y, pred], f)


if __name__ == '__main__':
    main()
