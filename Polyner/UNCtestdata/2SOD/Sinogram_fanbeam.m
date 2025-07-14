dimensions = [216, 216, 76]; % 1*1*1 mm, DEMSCBCT
%%
filename = "..\RANDO_MAR_Ca_110kVp_VMI_mu_75_keV.bin";

fid = fopen(filename, 'r');
if fid == -1
    error('Could not open file');
end
img = fread(fid, prod(dimensions), 'single');
fclose(fid);
img = reshape(img, dimensions);
figure;
imagesc(img(:,:,42));
colormap(gray);
colorbar;

%%
% slice = single(img(:,:,42));
slice = img(:,:,42);
%%
% % proj_geom = astra_create_proj_geom('fanflat', det_width, det_count, angles, source_origin, origin_det)
% %
% % Create a 2D flat fan beam geometry.  See the API for more information.
% % det_width: distance between two adjacent detectors
% % det_count: number of detectors in a single projection
% % angles: projection angles in radians, should be between -pi/4 and 7pi/4
% % source_origin: distance between the source and the center of rotation
% % origin_det: distance between the center of rotation and the detector array
% % proj_geom: MATLAB struct containing all information of the geometry
%%
angles = linspace2(-pi/4, 7*pi/4, 360);
voxel_size = 1; % mm
pixel_size = 1; % mm
det_columns = 520/pixel_size;
SOD = 410/voxel_size; % mm
ODD = 410/voxel_size; % mm
proj_geom = astra_create_proj_geom('fanflat', pixel_size, det_columns, angles, SOD, ODD);

x_len = size(slice, 2)/voxel_size; % mm
y_len = size(slice, 1)/voxel_size; % mm 

vol_geom = astra_create_vol_geom(x_len, y_len);

[sinogram_id, sinogram] = astra_create_sino_gpu(slice, proj_geom, vol_geom);

figure(1); imshow(slice, []);
figure(2); imshow(sinogram, []);

% Create a data object for the reconstruction
rec_id = astra_mex_data2d('create', '-vol', vol_geom);

% Set up the parameters for a reconstruction algorithm using the GPU
cfg = astra_struct('SIRT_CUDA');
cfg.ReconstructionDataId = rec_id;
cfg.ProjectionDataId = sinogram_id;

% Available algorithms:
% SIRT_CUDA, SART_CUDA, EM_CUDA, FBP_CUDA (see the FBP sample)

% Create the algorithm object from the configuration structure
alg_id = astra_mex_algorithm('create', cfg);

% Run 150 iterations of the algorithm
astra_mex_algorithm('iterate', alg_id, 150);

% Get the result
rec = astra_mex_data2d('get', rec_id);
figure(3); imshow(rec, []);

%% 
out_dir = '..\2SOD';   
if ~exist(out_dir,'dir'); mkdir(out_dir); end

%%
slice_file = fullfile(out_dir,'slice42_2SOD.bin');
fid = fopen(slice_file,'w'); fwrite(fid, slice, 'single'); fclose(fid);

%% 
sino_for_save = sinogram';                              
sino_file = fullfile(out_dir,'sino42_2SOD.bin');
fid = fopen(sino_file,'w'); fwrite(fid, sino_for_save,'single'); fclose(fid);

%%
rec_file = fullfile(out_dir,'rec42_2SOD.bin');
fid = fopen(rec_file,'w'); fwrite(fid, rec, 'single'); fclose(fid);

%% 
rmse = sqrt(mean((single(rec(:)) - single(slice(:))).^2));
mae  = mean(abs(single(rec(:)) - single(slice(:))));
fprintf('RMSE = %.4g   |   MAE = %.4g\n', rmse, mae);

diff_img = rec - slice;       

%% 
figure('Name','Slice | Sinogram | Recon | Difference','Position',[100 100 1200 600]);

subplot(2,2,1);
imagesc(slice, [0 1]); axis image off; colormap gray;
title('Original Slice (cm^{-1})');

subplot(2,2,2);
imagesc(sino_for_save); axis xy; colormap gray;   % xy 保证行→上‑下,列→左‑右
xlabel('Detector index'); ylabel('Projection angle idx');
title('Sinogram (angles × detectors)');

subplot(2,2,3);
imagesc(rec, [0 1]); axis image off; colormap gray;
title('Reconstruction (cm^{-1})');

subplot(2,2,4);
imagesc(diff_img); axis image off; colormap gray;
title(sprintf('Difference (RMSE %.3g)', rmse));
colorbar;
%%

% Clean up. Note that GPU memory is tied up in the algorithm object,
% and main RAM in the data objects.
astra_mex_algorithm('delete', alg_id);
astra_mex_data2d('delete', rec_id);
astra_mex_data2d('delete', sinogram_id);
