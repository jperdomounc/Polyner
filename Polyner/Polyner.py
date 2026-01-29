import model
import torch
import numpy as np
import dataset as dataset
import SimpleITK as sitk
import tinycudann as tcnn
from tqdm import tqdm
from torch.utils import data
from scipy import io as scio
from torch.optim import lr_scheduler
from skimage.morphology import erosion, square

import os, csv
import matplotlib
matplotlib.use("Agg")          # 无显示环境（集群）下绘图
import matplotlib.pyplot as plt

def train(img_id, config):

    # data's path and paramters
    # -----------------------
    in_path = config["file"]["in_dir"]
    out_path = config["file"]["out_dir"]
    model_path = config["file"]["model_dir"]
    proj_path = '{}/proj.nii'.format(in_path)
    proj_pos_path_u = '{}/fanSensorPosition_fanangle.nii'.format(in_path)
    proj_pos_path_v = '{}/fanSensorPosition_coneangle.nii'.format(in_path)
    mask_path = '{}/mask.nii'.format(in_path)
    h, w, d, SOD = config["file"]["h"], config["file"]["w"], config["file"]["d"], config["file"]["SOD"]
    voxel_size = config["file"]["voxel_size"]
    num_angle, _, _ = sitk.GetArrayFromImage(sitk.ReadImage(proj_path)).shape

    # training hyper-parameters
    # -----------------------
    lr = config["train"]["lr"]
    gpu = config["train"]["gpu"]
    epoch = config["train"]["epoch"]
    save_epoch = config["train"]["save_epoch"]
    lr_decay_epoch = config["train"]["lr_decay_epoch"]
    lr_decay_coefficient = config["train"]["lr_decay_coefficient"]
    batch_size = config["train"]["batch_size"]
    num_sample_ray = config["train"]["num_sample_ray"]
    lamb = config["train"]["lambda"]

    device = torch.device('cuda:{}'.format(str(gpu) if torch.cuda.is_available() else 'cpu'))

    # mask
    # -----------------------
    # mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
    # mask = np.rot90(np.pad(mask, ((int(SOD - (mask.shape[0] / 2)), int(SOD - (mask.shape[0] / 2))-1),
    #                               (int(SOD - (mask.shape[1] / 2)), int(SOD - (mask.shape[1] / 2))-1),
    #                               (int(SOD - (mask.shape[2] / 2)), int(SOD - (mask.shape[2] / 2))-1)))).copy()
    # mask = torch.tensor(mask).float().unsqueeze(0).unsqueeze(0).to(device)
    # mask = torch.where(mask == 1, 0., 1.)

    mask = torch.zeros(1, 1, h, w, d).float().to(device)

    # energy spectrum
    # -----------------------
    spectrum = scio.loadmat('./{}/DECBCTSpectrum110KVP.mat'.format(in_path))['DECBCTSpectrum110KVP_bin']

    e_1, e_n = 1, 6
    spectrum = spectrum[e_1-1:e_n, 0] # 0->LE, 1->HE
    spectrum = spectrum / np.sum(spectrum)
    e_level = len(spectrum)
    spectrum = torch.tensor(spectrum, dtype=torch.float).view(1, 1, -1).to(device)

    # model
    # -----------------------
    dc_loss = torch.nn.L1Loss().to(device)
    ase_loss = model.Attenuation_Smootion_Over_Energies_Loss(lamb=lamb, mask=mask).to(device)

    network = tcnn.NetworkWithInputEncoding(n_input_dims=3, n_output_dims=e_level,
                                            encoding_config=config["encoding"], network_config=config["network"]).to(device)
    optimizer = torch.optim.Adam(params=network.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay_epoch, gamma=lr_decay_coefficient)

    # data loader
    # -----------------------
    train_loader = data.DataLoader(
        dataset=dataset.TrainData(proj_path=proj_path, proj_pos_path_u=proj_pos_path_u, proj_pos_path_v=proj_pos_path_v, SOD=SOD,
                                  num_sample_ray=num_sample_ray, num_angle=num_angle, voxel_size=voxel_size),
                                  batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(
        dataset=dataset.TestData(h=(2 * SOD) + 1, w=(2 * SOD) + 1, d=(2 * SOD) + 1), batch_size=1, shuffle=False)
    # optimization & reconstruction
    # -----------------------

    os.makedirs(out_path, exist_ok=True)
    epoch_loss_hist = []  # the loss of each epoch

    loop_tqdm = tqdm(range(epoch), leave=False)
    for e in loop_tqdm:
        network.train()
        loss_log = 0
        for i, (ray, proj) in enumerate(train_loader):
            ray = ray.to(device).float().view(-1, 3)   # (batch_size*num_sample_ray*2*SOD, 3)
            proj = proj.to(device).float()  # (batch_size, num_sample_ray)
            # (batch_size*num_sample_ray*2*SOD, e_level)
            intensity_pre = network(ray).view(-1, num_sample_ray, 2 * SOD, e_level).float()
            # forward model
            proj_pre = torch.exp(-voxel_size *
                                 torch.sum(intensity_pre, dim=2).squeeze(-1))  # (batch_size, num_sample_ray, e_level)
            proj_pre = -torch.log(torch.sum(proj_pre * spectrum, dim=-1).squeeze(-1))  # (batch_size, num_sample_ray)
            # calculate loss
            loss = dc_loss(proj_pre, proj.to(proj_pre.dtype)) + ase_loss(intensity=intensity_pre, ray=ray)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_log = loss_log + loss.item()
        
        # —— epoch 结束 —— 记录平均 loss
        avg_loss = loss_log / max(1, len(train_loader))
        epoch_loss_hist.append(avg_loss)
        
        scheduler.step()
        loop_tqdm.set_description("Image #{}".format(img_id))
        loop_tqdm.set_postfix(lr=scheduler.get_last_lr()[0], loss=loss_log / len(train_loader))

        # model save & reconstruction
        if (e + 1) % save_epoch == 0:
            img_all = []
            kx, ky, kz = int(((2 * SOD) - h)/2), int(((2 * SOD) - w)/2), int(((2 * SOD) - d)/2)
            with torch.no_grad():
                torch.save(network.state_dict(), '{}/model_{}.pkl'.format(model_path, img_id))
                for i, (xyz) in enumerate(test_loader):
                    xyz = xyz.float().view(-1, 3)  # (h*w*d, 3) - keep on CPU

                    # Batched inference to avoid OOM
                    chunk_size = 100000  # adjust based on your GPU memory
                    num_points = xyz.shape[0]
                    energy_idx = int(np.mean(np.arange(0, e_level)))

                    img_pre_list = []
                    for start_idx in range(0, num_points, chunk_size):
                        end_idx = min(start_idx + chunk_size, num_points)
                        xyz_chunk = xyz[start_idx:end_idx].to(device)  # only move chunk to GPU
                        chunk_out = network(xyz_chunk)[:, energy_idx]
                        img_pre_list.append(chunk_out.cpu())

                    # 拼接分块结果并reshape
                    img_pre = torch.cat(img_pre_list, dim=0).view((2 * SOD) + 1, (2 * SOD) + 1, (2 * SOD) + 1)
                    img_pre = img_pre.float().numpy()[kx:kx + h, ky:ky + w, kz:kz + d]
                    # img_pre = np.flip(img_pre, axis=1)
                    img_pre = np.transpose(img_pre, (2, 0, 1))

                sitk.WriteImage(sitk.GetImageFromArray(img_pre), '{}/polyner_RANDO.nii'.format(out_path))

            # —— 记录 loss 到 CSV（避免变量名 w）——
            csv_path = f'{out_path}/loss_log.csv'
            with open(csv_path, 'w', newline='') as fcsv:
                csv_writer = csv.writer(fcsv)
                csv_writer.writerow(['epoch', 'avg_loss'])
                for ep_idx, loss_val in enumerate(epoch_loss_hist, 1):
                    csv_writer.writerow([ep_idx, loss_val])

            # —— 画 loss 曲线 —— 
            plt.figure()
            xs = np.arange(1, len(epoch_loss_hist) + 1)
            plt.semilogy(xs, epoch_loss_hist, linewidth=2)
            plt.xlabel('Epoch'); plt.ylabel('Average training loss')
            plt.title('Loss Curve'); plt.grid(True); plt.tight_layout()
            plt.savefig(f'{out_path}/loss_curve.png', dpi=200)
            plt.close()