# ----------------------------------------------#
# Pro    : cbct
# File   : dataset.py
# Date   : 2023/2/22
# Author : Qing Wu
# Email  : wuqing@shanghaitech.edu.cn
# ----------------------------------------------#
import model
import torch
import numpy as np
import dataset
import time
import SimpleITK as sitk
import tinycudann as tcnn
from tqdm import tqdm
from torch.utils import data
from scipy import io as scio
from torch.optim import lr_scheduler
from skimage.morphology import erosion, square

def train(img_id, config):

    # data's path and paramters
    # -----------------------
    in_path = config["file"]["in_dir"]
    out_path = config["file"]["out_dir"]
    model_path = config["file"]["model_dir"]
    proj_path = '{}/ma_sinogram_{}.nii'.format(in_path, img_id)
    proj_pos_path = '{}/fanSensorPos.nii'.format(in_path)
    mask_path = '{}/mask_{}.nii'.format(in_path, img_id)
    h, w, SOD = config["file"]["h"], config["file"]["w"], config["file"]["SOD"]
    d = 1  # Set to 1 for 2D fan beam
    SDD = config["file"]["SDD"]
    detector_h, detector_w = config["file"]["detector_h"], config["file"]["detector_w"]
    cone_angle = config["file"]["cone_angle"]
    n_projections = config["file"]["n_projections"]
    voxel_size = config["file"]["voxel_size"]
    num_angle, _ = sitk.GetArrayFromImage(sitk.ReadImage(proj_path)).shape

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

    # mask - extend to 3D
    # -----------------------
    mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
    if len(mask.shape) == 2:
        # Extend 2D mask to 3D
        mask = np.repeat(mask[:, :, np.newaxis], d, axis=2)
    mask = np.pad(mask, ((int(SOD - (mask.shape[0] / 2)), int(SOD - (mask.shape[0] / 2))-1),
                         (int(SOD - (mask.shape[1] / 2)), int(SOD - (mask.shape[1] / 2))-1),
                         (int(SOD - (mask.shape[2] / 2)), int(SOD - (mask.shape[2] / 2))-1))).copy()
    mask = torch.tensor(mask).float().unsqueeze(0).unsqueeze(0).to(device)
    mask = torch.where(mask == 1, 0., 1.)

    # energy spectrum
    # -----------------------
    spectrum = scio.loadmat('./{}/GE14Spectrum120KVP.mat'.format(in_path))['GE14Spectrum120KVP']

    e_1, e_n = 20, 120
    spectrum = spectrum[e_1-1:e_n, 1]
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
        dataset=dataset.TrainData3D(proj_path=proj_path, proj_pos_path=proj_pos_path, SOD=SOD, SDD=SDD,
                                    detector_h=detector_h, detector_w=detector_w, cone_angle=cone_angle,
                                    num_sample_ray=num_sample_ray, num_angle=num_angle, voxel_size=voxel_size),
                                    batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(
        dataset=dataset.TestData3D(h=(2 * SOD) + 1, w=(2 * SOD) + 1, d=(2 * SOD) + 1), batch_size=1, shuffle=False)

    # optimization & reconstruction
    # -----------------------
    loop_tqdm = tqdm(range(epoch), leave=False)
    epoch_start_time = time.time()
    for e in loop_tqdm:
        network.train()
        loss_log = 0
        for i, (ray, proj) in enumerate(train_loader):
            ray = ray.to(device).float().view(-1, 3)   # (batch_size*num_sample_ray*ray_length, 3)
            proj = proj.to(device).float()  # (batch_size, num_sample_ray)
            # (batch_size*num_sample_ray*ray_length, e_level)
            ray_length = ray.shape[0] // (ray.shape[0] // (num_sample_ray * batch_size))
            intensity_pre = network(ray).view(-1, num_sample_ray, ray_length // num_sample_ray, e_level).float()
            # forward model - 3D cone beam projection
            proj_pre = torch.exp(-voxel_size *
                                 torch.sum(intensity_pre, dim=2))  # (batch_size, num_sample_ray, e_level)
            proj_pre = -torch.log(torch.sum(proj_pre * spectrum, dim=-1))  # (batch_size, num_sample_ray)
            # calculate loss
            loss = dc_loss(proj_pre, proj.to(proj_pre.dtype)) + ase_loss(intensity=intensity_pre, ray=ray)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_log = loss_log + loss.item()
        scheduler.step()
        loop_tqdm.set_description("Image #{}".format(img_id))
        loop_tqdm.set_postfix(lr=scheduler.get_last_lr()[0], loss=loss_log / len(train_loader))

        # model save & reconstruction
        if (e + 1) % save_epoch == 0:
            img_all = []
            kx, ky, kz = int(1 + ((2 * SOD) - h)/2), int(((2 * SOD) - w)/2), int(((2 * SOD) - d)/2)
            final_loss = loss_log / len(train_loader)
            # Calculate iterations per second
            elapsed_time = time.time() - epoch_start_time
            iterations_per_sec = (e + 1) / elapsed_time if elapsed_time > 0 else 0
            with torch.no_grad():
                torch.save(network.state_dict(), '{}/model_{}.pkl'.format(model_path, img_id))
                for i, (xyz) in enumerate(test_loader):
                    xyz = xyz.to(device).float().view(-1, 3)  # (h*w*d, 3)
                    img_pre = network(xyz)[:, int(np.mean(np.arange(0, e_level)))].view((2 * SOD) + 1, (2 * SOD) + 1, (2 * SOD) + 1)
                    img_pre = img_pre.float().cpu().detach().numpy()[kx:kx + h, ky:ky + w, kz:kz + d]
                    img_pre = np.flip(img_pre, axis=1)

                sitk.WriteImage(sitk.GetImageFromArray(img_pre), '{}/polyner_{}_{:.5f}_{:.2f}it_s.nii'.format(out_path, img_id, final_loss, iterations_per_sec))