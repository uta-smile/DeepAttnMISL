
"""
Model training Code for "Whole Slide Images Based Cancer Survival Prediction using Attention Guided Deep
Multiple Instance Learning Networks"

Jiawen Yao, Xinliang Zhu et al. Medical Image Analysis, Available online 19 July 2020, 101789
https://doi.org/10.1016/j.media.2020.101789

"""
import gc
import torch
import numpy as np
from MIL_dataloader import MIL_dataloader
from tqdm import tqdm
from utils.surv_utils import cox_log_rank, CIndex_lifeline
from DeepAttnMISL_model import DeepAttnMIL_Surv
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import os
from utils.Early_Stopping import EarlyStopping
from sklearn.model_selection import KFold
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

parser = argparse.ArgumentParser(description='DeepAttnMISL')
parser.add_argument('--cluster_num', type=int, default=10, help='cluster number')
parser.add_argument('--feat_path', type=str, default='/home/jiawen/Code/Survival_Pytorch/data/MedIA/VGG/NLST/each_patient/kmeans/10/',
                    help='deep features and cluster label of each patient (e.g. npz files)')
# csv file stored as patient id, img_path, patient-level survival label
parser.add_argument('--img_label_path', type=str, default='./NLST/NLST_all_patch_expandedlabels.csv')
parser.add_argument('--batch_size', type=int, default=1, help='has to be 1')
parser.add_argument('--nepochs', type=int, default=100, help='The maxium number of epochs to train')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate (default: 1e-4)')



def _neg_partial_log(prediction, T, E):
    """
    calculate cox loss, Pytorch implementation by Huang, https://github.com/huangzhii/SALMON
    :param X: variables
    :param T: Time
    :param E: Status
    :return: neg log of the likelihood
    """

    current_batch_len = len(prediction)
    # print(current_batch_len)
    R_matrix_train = np.zeros([current_batch_len, current_batch_len], dtype=int)
    for i in range(current_batch_len):
        for j in range(current_batch_len):
            R_matrix_train[i, j] = T[j] >= T[i]

    train_R = torch.FloatTensor(R_matrix_train)
    train_R = train_R.cuda()
    train_ystatus = torch.FloatTensor(E).cuda()

    theta = prediction.reshape(-1)
    exp_theta = torch.exp(theta)

    loss_nn = - torch.mean((theta - torch.log(torch.sum(exp_theta * train_R, dim=1))) * train_ystatus)

    return loss_nn




def prediction(model, queryloader, testing=False):

    model.eval()

    lbl_pred_all = None
    status_all = []
    survtime_all = []
    iter = 0

    tbar = tqdm(queryloader, desc='\r')
    with torch.no_grad():
        for i_batch, sampled_batch in enumerate(tbar):

            X, survtime, lbl, cls_num, mask = sampled_batch['feat'], sampled_batch['time'], sampled_batch['status'], sampled_batch['cluster_num'], sampled_batch['mask']

            graph = [X[i].cuda() for i in range(cluster_num)]
            lbl = lbl.cuda()

            time = survtime.data.cpu().numpy()
            status = lbl.data.cpu().numpy()

            time = np.squeeze(time)
            status = np.squeeze(status)

            survtime_all.append(time/30.0)
            status_all.append(status)

        # ===================forward=====================


            lbl_pred = model(graph, mask.cuda())

            if iter == 0:
                lbl_pred_all = lbl_pred
                survtime_torch = survtime
                lbl_torch = lbl
            else:
                lbl_pred_all = torch.cat([lbl_pred_all, lbl_pred])
                lbl_torch = torch.cat([lbl_torch, lbl])
                survtime_torch = torch.cat([survtime_torch, survtime])

            iter += 1


    survtime_all = np.asarray(survtime_all)
    status_all = np.asarray(status_all)

    loss_surv = _neg_partial_log(lbl_pred_all, survtime_all, status_all)

    l1_reg = None
    for W in model.parameters():
            if l1_reg is None:
                l1_reg = torch.abs(W).sum()
            else:
                l1_reg = l1_reg + torch.abs(W).sum()  # torch.abs(W).sum() is equivalent to W.norm(1)

    loss = loss_surv + 1e-5 * l1_reg
    print("\nval_loss_nn: %.4f, L1: %.4f" % (loss_surv, 1e-5 * l1_reg))

    pvalue_pred = cox_log_rank(lbl_pred_all.data, lbl_torch, survtime_torch)
    c_index = CIndex_lifeline(lbl_pred_all.data, lbl_torch, survtime_torch)


    if not testing:
        print('\n[val]\t loss (nn):{:.4f}'.format(loss.data.item()),
                      'c_index: {:.4f}, p-value: {:.3e}'.format(c_index, pvalue_pred))
    else:
        print('\n[testing]\t loss (nn):{:.4f}'.format(loss.data.item()),
              'c_index: {:.4f}, p-value: {:.3e}'.format(c_index, pvalue_pred))

    return loss.data.item(), c_index


def train_epoch(epoch, model, optimizer, trainloader,  measure=1, verbose=1):
    model.train()

    lbl_pred_all = None
    lbl_pred_each = None

    survtime_all = []
    status_all = []

    iter = 0
    gc.collect()
    loss_nn_all = []

    tbar = tqdm(trainloader, desc='\r')

    for i_batch, sampled_batch in enumerate(tbar):

        X, survtime, lbl, mask = sampled_batch['feat'], sampled_batch['time'], sampled_batch['status'], sampled_batch['mask']


        graph = [X[i].cuda() for i in range(cluster_num)]
        lbl = lbl.cuda()
        masked_cls = mask.cuda()

        # ===================forward=====================
        lbl_pred = model(graph, masked_cls)  # prediction

        time = survtime.data.cpu().numpy()
        status = lbl.data.cpu().numpy()

        time = np.squeeze(time)
        status = np.squeeze(status)
        pred_score = lbl_pred.detach().cpu().numpy()  # [Batch, 1]

        survtime_all.append(time/30.0)  # if time are days
        status_all.append(status)

        if i_batch == 0:
            lbl_pred_all = lbl_pred
            survtime_torch = survtime
            lbl_torch = lbl

        if iter == 0:
            lbl_pred_each = lbl_pred

        else:
            lbl_pred_all = torch.cat([lbl_pred_all, lbl_pred])
            lbl_pred_each = torch.cat([lbl_pred_each, lbl_pred])

            lbl_torch = torch.cat([lbl_torch, lbl])
            survtime_torch = torch.cat([survtime_torch, survtime])


        iter += 1

        if iter % 16 == 0 or i_batch == len(trainloader)-1:
            # Update the loss when collect 16 data samples

            survtime_all = np.asarray(survtime_all)
            status_all = np.asarray(status_all)

            # print(survtime_all)

            if np.max(status_all) == 0:
                print("encounter no death in a batch, skip")
                lbl_pred_each = None
                survtime_all = []
                status_all = []
                iter = 0
                continue

            optimizer.zero_grad()  # zero the gradient buffer

            loss_surv = _neg_partial_log(lbl_pred_each, survtime_all, status_all)


            l1_reg = None
            for W in model.parameters():
                if l1_reg is None:
                    l1_reg = torch.abs(W).sum()
            else:
                l1_reg = l1_reg + torch.abs(W).sum()  # torch.abs(W).sum() is equivalent to W.norm(1)

            loss = loss_surv + 1e-5 * l1_reg
    # ===================backward====================
            loss.backward()
            optimizer.step()

            torch.cuda.empty_cache()
            lbl_pred_each = None
            survtime_all = []
            status_all = []
            loss_nn_all.append(loss.data.item())
            iter = 0

            gc.collect()

    if measure:
        pvalue_pred = cox_log_rank(lbl_pred_all.data, lbl_torch, survtime_torch)
        c_index = CIndex_lifeline(lbl_pred_all.data, lbl_torch, survtime_torch)

        if verbose > 0:
            print("\nEpoch: {}, loss_nn: {}".format(epoch, np.mean(loss_nn_all)))
            print('\n[Training]\t loss (nn):{:.4f}'.format(np.mean(loss_nn_all)),
                  'c_index: {:.4f}, p-value: {:.3e}'.format(c_index, pvalue_pred))


def train(train_path, test_path, model_save_path, num_epochs, lr, cluster_num = 10):


    model = DeepAttnMIL_Surv(cluster_num=cluster_num).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = 5e-4)

    Data = MIL_dataloader(data_path=train_path, cluster_num = cluster_num, train=True)
    trainloader, valloader = Data.get_loader()

    TestData = MIL_dataloader(test_path, cluster_num=cluster_num, train=False)

    testloader = TestData.get_loader()

    # initialize the early_stopping object
    early_stopping = EarlyStopping(model_path=model_save_path,
                                   patience=15, verbose=True)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)

    save_epoch = range(10, 100, 5)
    val_ci_list = []
    val_losses = []

    for epoch in range(num_epochs):

        train_epoch(epoch, model, optimizer, trainloader)
        valid_loss, val_ci = prediction(model, valloader)
        scheduler.step(valid_loss)
        val_losses.append(valid_loss)

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

        if epoch in save_epoch:
            val_ci_list.append(val_ci)
            print('saving epoch in {}, vali loss: {}, val ci:{}'.format(epoch, valid_loss, val_ci))
            torch.save(model.state_dict(), model_save_path.replace('.pth', '_epoch_{}.pth'.format(epoch)))


    model_test = DeepAttnMIL_Surv(cluster_num = cluster_num).cuda()  # set to get features or risks

    # Use the final saved model to test this time
    model_test.load_state_dict(torch.load(model_save_path))

    _, c_index = prediction(model_test, testloader, testing=True)

    return c_index


if __name__ == '__main__':


    # To run the code, should prepare extracted features and then perform clustering on them
    # You can organize the data in your most convenient way. I saved each patient in a npz file
    # It contains patient patch path, clustering label and the patient level survival label

    args = parser.parse_args()

    img_label_path = args.img_label_path
    batch_size = args.batch_size
    num_epochs = args.nepochs

    cluster_num = args.cluster_num
    feat_path = args.feat_path

    lr = args.lr

    all_paths = pd.read_csv(img_label_path)

    # expand_label = pd.read_csv(all_paths)
    surv = all_paths['surv']
    status = all_paths['status'].tolist()
    pid = all_paths['pid'].tolist()

    uniq_pid = np.unique(pid)  # unique patients id
    uniq_st = []
    # print("number of patients: ", uniq_pid)

    for each_pid in uniq_pid:
        temp = pid.index(each_pid)
        uniq_st.append(status[temp])

    testci = []
    index_num = 1

    pid_ind = range(len(uniq_st))

    kf = KFold(n_splits=5, random_state=666, shuffle=True)
    fold = 0
    for train_index, test_index in kf.split(pid_ind):

        print("Now training fold:{}".format(fold))


        test_pid = [uniq_pid[i] for i in test_index]
        print('testing pid', len(test_pid))

        train_val_npz = [str(uniq_pid[i])+'.npz' for i in train_index]
        test_npz = [str(uniq_pid[i])+'.npz' for i in test_index]


        train_val_patients_pca = [os.path.join(feat_path , each_path) for each_path in train_val_npz]
        test_patients_pca = [os.path.join(feat_path, each_path) for each_path in test_npz]

        print('training pid', len(train_val_patients_pca))
        print('testing pid', len(test_pid))


        model_save_path = './saved_model/NLST_model_fold_{}_c_{}.pth'.format(fold, cluster_num)
        test_ci = train(train_val_patients_pca, test_patients_pca, model_save_path,
                                    num_epochs=num_epochs, lr=lr, cluster_num=cluster_num)

        testci.append(test_ci)

        fold += 1


    print(testci)
    print(np.mean(testci))

