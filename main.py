from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataset import EMGDataset
import torch.optim as optim
import numpy as np
import autoencoder
import argparse
import random
import torch
import os

parser = argparse.ArgumentParser(description='Force-Aware Interface via Electromyography for Natural VR/AR Interaction')
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--num-channels', type=int, default=8, help='Number of EMG channels')
parser.add_argument('--num-forces', type=int, default=5, help='Number of forces')
parser.add_argument('--num-force-levels', type=int, default=2, help='Number of force levels')
parser.add_argument('--num-frames', type=int, default=32, help='Number of frames')
parser.add_argument('--num-frequencies', type=int, default=64, help='Number of STFT frequencies')
parser.add_argument('--window-length', type=int, default=256, help='Window length for STFT')
parser.add_argument('--hop-length', type=int, default=32, help='Hop length for STFT')
parser.add_argument('--hop-length-test', type=int, default=8, help='Hop length for evaluation')
parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
parser.add_argument('--num-epochs', type=int, default=30, help='Number of training epochs')
parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay factor')
FLAGS = parser.parse_args()


def emg_dataloader(args):
    kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}
    dataset = EMGDataset(args.dataset_path)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=True, **kwargs)


def train(model, dataloader, optimizer, args):
    model.train()
    correct = 0
    all = 0
    losses_classification = []
    losses_regression = []
    for emg, force, force_class in dataloader:
        if args.cuda:
            emg, force, force_class = emg.cuda(), force.cuda(), force_class.cuda()
        logits = model(emg)
        # Classification
        loss_classification = F.cross_entropy(logits, force_class)
        losses_classification.append(loss_classification.item())
        correct += logits.max(1)[1].eq(force_class).all(1).sum().item()
        all += force_class.shape[0] * force_class.shape[2]
        # Regression
        logits = logits.transpose(1, 3)
        probs = F.softmax(logits, 3)
        weights = torch.from_numpy(np.array([5], dtype=np.float32))
        if args.cuda:
            weights = weights.cuda()
        loss_regression = F.mse_loss((F.relu(probs[..., 1] - 0.5) * weights).transpose(1, 2), force)
        losses_regression.append(loss_regression.item())
        # Optimization
        optimizer.zero_grad()
        (loss_classification + 4 * loss_regression).backward()
        optimizer.step()
    loss_classification = np.mean(losses_classification)
    loss_regression = np.mean(losses_regression)
    accuracy = 100.0 * float(correct) / all
    return loss_classification, loss_regression, accuracy


def evaluate(model, args):
    file_ids = list(range(11))
    session_ids = list(range(3, 28, 3))
    model.eval()
    correct = np.zeros((len(session_ids), len(file_ids)), dtype=np.float32)
    correct_framewise = np.zeros(args.hop_length_test, dtype=np.float32)
    all = np.zeros_like(correct)
    all_framewise = np.zeros_like(correct_framewise)
    losses_classification = []
    losses_regression = []
    force_pred_session_all = []
    force_gt_session_all = []
    with torch.no_grad():
        for i, sid in enumerate(session_ids):
            force_pred_session = []
            force_gt_session = []
            for j, fid in enumerate(file_ids):
                emg = torch.from_numpy(np.load(os.path.join(args.data_path, "Session{:d}".format(sid), "emg_test_{:d}.npy".format(fid))))
                force = torch.from_numpy(np.load(os.path.join(args.data_path, "Session{:d}".format(sid), "force_test_{:d}.npy".format(fid))))
                force_class = torch.from_numpy(np.load(os.path.join(args.data_path, "Session{:d}".format(sid), "force_class_test_{:d}.npy".format(fid))))
                if args.cuda:
                    emg, force, force_class = emg.cuda(), force.cuda(), force_class.cuda()
                logits = model(emg)
                logits = logits[..., -args.hop_length_test:]
                force = force[..., -args.hop_length_test:]
                force_class = force_class[..., -args.hop_length_test:]
                # Classification
                loss_classification = F.cross_entropy(logits, force_class)
                losses_classification.append(loss_classification.item())
                results = logits.max(1)[1].eq(force_class).all(1)
                correct[i, j] = results.sum().item()
                all[i, j] = force_class.shape[0] * force_class.shape[2]
                correct_framewise += results.sum(0).cpu().numpy()
                all_framewise += force_class.shape[0]
                # Regression
                logits = logits.transpose(1, 3)
                probs = F.softmax(logits, 3)
                weights = torch.from_numpy(np.array([5], dtype=np.float32))
                if args.cuda:
                    weights = weights.cuda()
                force_pred = (F.relu(probs[..., 1] - 0.5) * weights).transpose(1, 2)
                loss_regression = F.mse_loss(force_pred, force)
                losses_regression.append(loss_regression.item())
                force_pred = force_pred.cpu().numpy().transpose(0, 2, 1).reshape(-1, args.num_forces)
                force = force.cpu().numpy().transpose(0, 2, 1).reshape(-1, args.num_forces)
                force_pred_session.append(force_pred[:1840])
                force_gt_session.append(force[:1840])
            force_pred_session_all.append(np.stack(force_pred_session, axis=0))
            force_gt_session_all.append(np.stack(force_gt_session, axis=0))
    force_pred_session_all = np.stack(force_pred_session_all, axis=0)
    force_gt_session_all = np.stack(force_gt_session_all, axis=0)
    NRMSE = 100.0 * np.sqrt(((force_pred_session_all - force_gt_session_all)**2).mean()) / 2.5
    NRMSE_filewise = 100.0 * np.sqrt(((force_pred_session_all - force_gt_session_all)**2).mean((0, 2, 3))) / 2.5
    R2 = 100.0 * (1.0 - ((force_pred_session_all - force_gt_session_all)**2).sum() / ((force_gt_session_all - force_gt_session_all.mean())**2).sum())
    R2_filewise = 100.0 * (1.0 - ((force_pred_session_all - force_gt_session_all)**2).sum((0, 2, 3)) / ((force_gt_session_all - force_gt_session_all.mean())**2).sum((0, 2, 3)))
    loss_classification = np.mean(losses_classification)
    loss_regression = np.mean(losses_regression)
    accuracy = 100.0 * correct.mean() / all.mean()
    accuracy_filewise = 100.0 * correct.mean(0) / all.mean(0)
    accuracy_framewise = 100.0 * correct_framewise / all_framewise
    return loss_classification, loss_regression, accuracy, accuracy_filewise, accuracy_framewise, NRMSE, NRMSE_filewise, R2, R2_filewise


def main(args):
    args.cuda = torch.cuda.is_available()
    # Random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # Folders
    args.model_path = os.path.join(os.getcwd(), 'Checkpoints')
    args.data_path = os.path.join(os.getcwd(), 'Data')
    args.dataset_path = os.path.join(os.getcwd(), 'Dataset')
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    if not os.path.exists(args.data_path) or not os.path.exists(args.dataset_path):
        raise Exception("Dataset not found!")
    # Model
    model = autoencoder.EMGNet(args)
    if args.cuda:
        model.cuda()
    trainloader = emg_dataloader(args)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Training
    for epoch in range(1, args.num_epochs + 1):
        loss_cls_test, loss_reg_test, acc_test, acc_filewise_test, acc_framewise_test, NRMSE_test, NRMSE_filewise_test, R2_test, R2_filewise_test = evaluate(model, args)
        loss_cls_train, loss_reg_train, acc_train = train(model, trainloader, optimizer, args)
        print("Epoch {:d} | Train Cls loss: {:.4f} | Train Reg Loss: {:.4f} | Test Cls loss: {:.4f} | Test Reg Loss: {:.4f}| Train accuracy(%): {:.2f} | Test accuracy(%): {:.2f} | NRMSE(%): {:.2f} | R2(%): {:.2f}".format(
            epoch, loss_cls_train, loss_reg_train, loss_cls_test, loss_reg_test, acc_train, acc_test, NRMSE_test, R2_test))
        print("Test action-wise accuracy(%): {}".format(np.array2string(acc_filewise_test, precision=2, separator=', ')))
        print("Test action-wise NRMSE(%): {}".format(np.array2string(NRMSE_filewise_test, precision=2, separator=', ')))
        print("Test action-wise R2(%): {}".format(np.array2string(R2_filewise_test, precision=2, separator=', ')))
        # Checkpoint
        info = {'epoch': epoch, 'state_dict': model.state_dict()}
        filename = "checkpoint-all-model{:d}.pth".format(epoch)
        filepath = os.path.join(args.model_path, filename)
        torch.save(info, filepath)


if __name__ == '__main__':
    main(FLAGS)
