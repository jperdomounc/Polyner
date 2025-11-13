# ----------------------------------------------#
# Pro    : cbct
# File   : Polyner_3d.py
# Date   : 2025
# Author : Adapted for 3D Cone-Beam CT
# ----------------------------------------------#
import model
import torch
import numpy as np
import dataset_3d
import time
import SimpleITK as sitk
import tinycudann as tcnn
from tqdm import tqdm
from torch.utils import data
from scipy import io as scio
from torch.optim import lr_scheduler
from skimage.morphology import erosion, cube


def train(img_id, config):

    # data's path and parameters
    # -----------------------
    in_path = config["file"]["in_dir"]
    out_path = config["file"]["out_dir"]
    model_path = config["file"]["model_dir"]
    proj_path = '{}/ma_projection_{}.nii'.format(in_path, img_id)
    proj_u_pos_path = '{}/detectorUPos.nii'.format(in_path)
    proj_v_pos_path = '{}/detectorVPos.nii'.format(in_path)
    mask_path = '{}/mask_{}.nii'.format(in_path, img_id)

    # 3D volume parameters
    h, w, d = config["file"]["h"], config["file"]["w"], config["file"]["d"]
    SOD = config["file"]["SOD"]
    SDD = config["file"]["SDD"]
    num_samples = config["file"]["num_samples"]
    voxel_size = config["file"]["voxel_size"]

    # Read projection data to get num_angles
    proj_data = sitk.GetArrayFromImage(sitk.ReadImage(proj_path))
    num_angle = proj_data.shape[0]  # (num_angle, num_det_v, num_det_u)

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

    # 3D mask
    # looks fine v
    # -----------------------
    mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
    # Pad mask to match reconstruction volume size
    # Assumes mask is centered in the volume
    pad_h = int(SOD - (mask.shape[0] / 2))
    pad_w = int(SOD - (mask.shape[1] / 2))
    pad_d = int(SOD - (mask.shape[2] / 2))

    mask = np.pad(mask, ((pad_h, pad_h-1 if pad_h > 0 else 0),
                         (pad_w, pad_w-1 if pad_w > 0 else 0),
                         (pad_d, pad_d-1 if pad_d > 0 else 0)))

    # Rotate for correct orientation (may need adjustment based on your data)
    mask = np.rot90(mask, k=1, axes=(0, 1)).copy()

    mask = torch.tensor(mask).float().unsqueeze(0).unsqueeze(0).to(device)
    mask = torch.where(mask == 1, 0., 1.)  # Invert mask (0 for metal, 1 for tissue)

    # energy spectrum
    # -----------------------

    spectrum = scio.loadmat('./{}/DECBCTSpectrum110KVP.mat'.format(in_path))['DECBCTSpectrum110KVP']

    e_1, e_n = 20, 110
    spectrum = spectrum[e_1-1:e_n, 0] # 0->LE, 1->HE
    spectrum = spectrum / np.sum(spectrum)
    e_level = len(spectrum)
    spectrum = torch.tensor(spectrum, dtype=torch.float).view(1, 1, -1).to(device)

    # 3D model - KEY CHANGE: n_input_dims=3 for xyz coordinates
    # -----------------------
    dc_loss = torch.nn.L1Loss().to(device)
    ase_loss = model.Attenuation_Smootion_Over_Energies_Loss(lamb=lamb, mask=mask).to(device)

    # CRITICAL: Change n_input_dims from 2 to 3 for 3D coordinates
    network = tcnn.NetworkWithInputEncoding(
        n_input_dims=3,  # Changed from 2 to 3 for xyz
        n_output_dims=e_level,
        encoding_config=config["encoding"],
        network_config=config["network"]
    ).to(device)

    optimizer = torch.optim.Adam(params=network.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay_epoch, gamma=lr_decay_coefficient)

    # 3D data loader
    # -----------------------
    train_loader = data.DataLoader(
        dataset=dataset_3d.TrainData3D(
            proj_path=proj_path,
            proj_u_pos_path=proj_u_pos_path,
            proj_v_pos_path=proj_v_pos_path,
            SOD=SOD,
            SDD=SDD,
            num_samples=num_samples,
            num_sample_ray=num_sample_ray,
            num_angle=num_angle,
            voxel_size=voxel_size
        ),
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = data.DataLoader(
        dataset=dataset_3d.TestData3D(h=h, w=w, d=d),
        batch_size=1,
        shuffle=False
    )

    # optimization & reconstruction
    # -----------------------
    loop_tqdm = tqdm(range(epoch), leave=False)
    epoch_start_time = time.time()

    for e in loop_tqdm:
        network.train()
        loss_log = 0

        for i, (ray, proj) in enumerate(train_loader):
            # ray: (batch_size, num_sample_ray, num_samples, 3)
            # proj: (batch_size, num_sample_ray)
            ray = ray.to(device).float().view(-1, 3)  # (batch_size*num_sample_ray*num_samples, 3)
            proj = proj.to(device).float()  # (batch_size, num_sample_ray)

            # Network inference: input 3D coordinates, output attenuation at each energy
            # Output: (batch_size*num_sample_ray*num_samples, e_level)
            intensity_pre = network(ray).view(-1, num_sample_ray, num_samples, e_level).float()

            # Forward model: Beer's Law with polyenergetic spectrum
            # Line integral: sum along ray (dim=2)
            proj_pre = torch.exp(-voxel_size *
                                torch.sum(intensity_pre, dim=2).squeeze(-1))  # (batch_size, num_sample_ray, e_level)

            # Spectrum weighting and log projection
            proj_pre = -torch.log(torch.sum(proj_pre * spectrum, dim=-1).squeeze(-1))  # (batch_size, num_sample_ray)

            # Calculate loss
            loss = dc_loss(proj_pre, proj.to(proj_pre.dtype)) + ase_loss(intensity=intensity_pre, ray=ray)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_log = loss_log + loss.item()

        scheduler.step()
        loop_tqdm.set_description("3D Image #{}".format(img_id))
        loop_tqdm.set_postfix(lr=scheduler.get_last_lr()[0], loss=loss_log / len(train_loader))

        # Model save & 3D reconstruction
        if (e + 1) % save_epoch == 0:
            final_loss = loss_log / len(train_loader)

            # Calculate iterations per second
            elapsed_time = time.time() - epoch_start_time
            iterations_per_sec = (e + 1) / elapsed_time if elapsed_time > 0 else 0

            with torch.no_grad():
                torch.save(network.state_dict(), '{}/model_3d_{}.pkl'.format(model_path, img_id))

                # Reconstruct 3D volume
                for xyz in test_loader:
                    xyz = xyz.to(device).float().view(-1, 3)  # (h*w*d, 3)

                    # Reconstruct at middle energy level
                    img_pre = network(xyz)[:, int(np.mean(np.arange(0, e_level)))]
                    img_pre = img_pre.view(h, w, d)
                    img_pre = img_pre.float().cpu().detach().numpy()

                    # Flip for correct orientation (may need adjustment)
                    img_pre = np.flip(img_pre, axis=1)

                    # Save 3D volume
                    sitk.WriteImage(
                        sitk.GetImageFromArray(img_pre),
                        '{}/polyner_3d_{}_{:.5f}_{:.2f}it_s.nii'.format(out_path, img_id, final_loss, iterations_per_sec)
                    )

    print("3D Cone-Beam CT reconstruction complete!")
    return network
