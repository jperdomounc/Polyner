%% Generate a binary metal mask from a SIRT reconstruction (NIfTI)
% This script loads a SIRT reconstruction NIfTI file, identifies metal
% regions using intensity thresholding and morphological operations, and
% saves a binary metal mask as a new NIfTI file.
%
% Requirements:
%   MATLAB R2017b+ for niftiread, niftiinfo, and niftiwrite
%   MATLAB R2018b+ when show_plots=true (xline and sgtitle)
%   Image Processing Toolbox for morphological operations
%   Statistics and Machine Learning Toolbox for prctile
%
% Usage:
%   Update the parameters in the User parameters section, then run the script.
%
% Original author note: Generated with Claude
% Original date: 2026-02-18

clear; clc; close all;

%% User parameters

% Resolve defaults relative to this repository. The example input was created
% on an earlier experiment branch and may not be present in a fresh checkout.
script_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(script_dir);
input_nifti = fullfile(repo_root, 'Polyner', ...
    'rec_RANDO_Zr_SIRT_90_04mm.nii');
output_mask = fullfile(repo_root, 'Polyner', 'input', 'metal_mask.nii');
compress_output = true;

% Thresholding method: 'manual', 'percentile', or 'otsu'.
threshold_method = 'percentile';

% Manual threshold value (used only if threshold_method = 'manual').
% Typical metal HU values may exceed 3000. Adjust for the actual SIRT range;
% this value is not generally appropriate for linear attenuation coefficients.
manual_threshold = 3000;

% Percentile threshold (used only if threshold_method = 'percentile').
% For example, 99.5 treats the top 0.5% of voxel intensities as metal.
percentile_value = 99.5;

% Morphological operations.
do_fill_holes = true;
do_remove_noise = true;
min_voxel_count = 20;
do_close = true;
close_radius = 1;
do_dilate = true;
dilate_radius = 2;

% Visualization.
show_plots = true;
slice_axis = 3;  % 1 = sagittal, 2 = coronal, 3 = axial

%% Load data

if ~isfile(input_nifti)
    error(['Input file not found: %s\nUpdate input_nifti in the User ' ...
        'parameters section.'], input_nifti);
end

fprintf('Loading NIfTI file: %s\n', input_nifti);
nii = niftiread(input_nifti);
info = niftiinfo(input_nifti);
nii = double(nii);

fprintf('  Volume size: [%d x %d x %d]\n', ...
    size(nii, 1), size(nii, 2), size(nii, 3));
fprintf('  Voxel size:  [%.2f x %.2f x %.2f] mm\n', ...
    info.PixelDimensions(1), info.PixelDimensions(2), ...
    info.PixelDimensions(3));
fprintf('  Intensity range: [%.2f, %.2f]\n', min(nii(:)), max(nii(:)));

%% Compute threshold

switch lower(threshold_method)
    case 'manual'
        threshold = manual_threshold;
        fprintf('Using manual threshold: %.2f\n', threshold);

    case 'percentile'
        threshold = prctile(nii(:), percentile_value);
        fprintf('Using percentile (%.1f%%) threshold: %.2f\n', ...
            percentile_value, threshold);

    case 'otsu'
        % Apply Otsu to the upper intensity tail to separate metal from
        % bone/tissue.
        upper_mask = nii(:) > prctile(nii(:), 90);
        upper_vals = nii(upper_mask);
        if isempty(upper_vals)
            error('No voxels found above the 90th percentile. Check the data.');
        end
        threshold = multithresh(upper_vals, 1);
        fprintf('Using Otsu threshold (on top 10%%): %.2f\n', threshold);

    otherwise
        error(['Unknown threshold method: %s. Use ''manual'', ' ...
            '''percentile'', or ''otsu''.'], threshold_method);
end

%% Create binary mask

fprintf('Creating binary metal mask...\n');
metal_mask = nii > threshold;
fprintf('  Initial metal voxels: %d (%.4f%% of volume)\n', ...
    sum(metal_mask(:)), 100 * sum(metal_mask(:)) / numel(metal_mask));

%% Morphological cleanup

if do_fill_holes
    fprintf('  Filling holes (slice by slice)...\n');
    for s = 1:size(metal_mask, slice_axis)
        switch slice_axis
            case 1
                metal_mask(s, :, :) = imfill( ...
                    squeeze(metal_mask(s, :, :)), 'holes');
            case 2
                metal_mask(:, s, :) = imfill( ...
                    squeeze(metal_mask(:, s, :)), 'holes');
            case 3
                metal_mask(:, :, s) = imfill( ...
                    squeeze(metal_mask(:, :, s)), 'holes');
            otherwise
                error('slice_axis must be 1, 2, or 3.');
        end
    end
end

if do_remove_noise
    fprintf('  Removing components smaller than %d voxels...\n', ...
        min_voxel_count);
    components = bwconncomp(metal_mask, 26);
    component_sizes = cellfun(@numel, components.PixelIdxList);
    small_components = find(component_sizes < min_voxel_count);
    for k = 1:length(small_components)
        metal_mask(components.PixelIdxList{small_components(k)}) = false;
    end
    fprintf('    Removed %d small components\n', length(small_components));
end

if do_close
    fprintf('  Applying morphological closing (radius=%d)...\n', close_radius);
    close_element = strel('sphere', close_radius);
    metal_mask = imclose(metal_mask, close_element);
end

metal_mask_tight = metal_mask;

if do_dilate
    fprintf('  Dilating mask (radius=%d)...\n', dilate_radius);
    dilate_element = strel('sphere', dilate_radius);
    metal_mask = imdilate(metal_mask, dilate_element);
end

fprintf('  Final metal voxels: %d (%.4f%% of volume)\n', ...
    sum(metal_mask(:)), 100 * sum(metal_mask(:)) / numel(metal_mask));

%% Save mask

fprintf('Saving metal mask to: %s\n', output_mask);
info_mask = info;
info_mask.Datatype = 'uint8';
info_mask.BitsPerPixel = 8;
niftiwrite(uint8(metal_mask), output_mask, info_mask, ...
    'Compressed', compress_output);

if do_dilate
    [output_dir, output_name, output_ext] = fileparts(output_mask);
    tight_mask_name = fullfile(output_dir, ...
        [output_name '_tight' output_ext]);
    fprintf('Saving tight (non-dilated) mask to: %s\n', tight_mask_name);
    niftiwrite(uint8(metal_mask_tight), tight_mask_name, info_mask, ...
        'Compressed', compress_output);
end

fprintf('Done!\n\n');

%% Visualization

if show_plots
    figure('Name', 'Intensity Histogram', 'Position', [100 500 800 400]);
    histogram(nii(:), 1000, 'FaceColor', [0.3 0.5 0.8], ...
        'EdgeColor', 'none');
    hold on;
    xline(threshold, 'r--', 'LineWidth', 2, ...
        'Label', sprintf('Threshold = %.1f', threshold));
    xlabel('Voxel Intensity');
    ylabel('Count');
    title('SIRT Reconstruction Intensity Histogram');
    set(gca, 'YScale', 'log');
    grid on;

    switch slice_axis
        case 1
            metal_slices = find(squeeze(any(any(metal_mask, 2), 3)));
        case 2
            metal_slices = find(squeeze(any(any(metal_mask, 1), 3)));
        case 3
            metal_slices = find(squeeze(any(any(metal_mask, 1), 2)));
    end

    if isempty(metal_slices)
        warning(['No metal was detected. Adjust the threshold method or ' ...
            'threshold value.']);
    else
        num_show = min(9, length(metal_slices));
        slice_step = max(1, floor(length(metal_slices) / num_show));
        show_indices = metal_slices(1:slice_step:end);
        show_indices = show_indices(1:min(9, length(show_indices)));

        figure('Name', 'Metal Mask Overlay', ...
            'Position', [100 50 1200 800]);
        num_columns = ceil(sqrt(length(show_indices)));
        num_rows = ceil(length(show_indices) / num_columns);

        for i = 1:length(show_indices)
            s = show_indices(i);
            subplot(num_rows, num_columns, i);

            switch slice_axis
                case 1
                    image_slice = squeeze(nii(s, :, :));
                    mask_slice = squeeze(metal_mask(s, :, :));
                case 2
                    image_slice = squeeze(nii(:, s, :));
                    mask_slice = squeeze(metal_mask(:, s, :));
                case 3
                    image_slice = squeeze(nii(:, :, s));
                    mask_slice = squeeze(metal_mask(:, :, s));
            end

            display_range = [prctile(nii(:), 1), prctile(nii(:), 99)];
            image_normalized = mat2gray(image_slice, display_range);
            image_rgb = repmat(image_normalized, [1 1 3]);

            overlay = image_rgb;
            overlay(:, :, 1) = max(overlay(:, :, 1), ...
                double(mask_slice) * 0.7);
            overlay(:, :, 2) = overlay(:, :, 2) .* ...
                (1 - double(mask_slice) * 0.5);
            overlay(:, :, 3) = overlay(:, :, 3) .* ...
                (1 - double(mask_slice) * 0.5);

            imshow(permute(overlay, [2 1 3]));
            title(sprintf('Slice %d', s));
        end
        sgtitle('Metal Mask Overlay (red = metal)', 'FontSize', 14);
    end

    if sum(metal_mask(:)) < 5e6
        figure('Name', '3D Metal Mask', 'Position', [500 100 600 500]);
        mask_smooth = smooth3(double(metal_mask), 'gaussian', 3);
        surface_patch = patch(isosurface(mask_smooth, 0.5));
        surface_patch.FaceColor = [0.8 0.2 0.2];
        surface_patch.EdgeColor = 'none';
        surface_patch.FaceAlpha = 0.8;

        lighting gouraud;
        camlight('headlight');
        camlight('left');
        axis equal tight;
        view(3);
        rotate3d on;
        title('3D Metal Mask Rendering');
        xlabel('X'); ylabel('Y'); zlabel('Z');
        grid on;
    end
end

%% Summary

fprintf('\n=== SUMMARY ===\n');
fprintf('Input file:         %s\n', input_nifti);
fprintf('Threshold method:   %s\n', threshold_method);
fprintf('Threshold value:    %.2f\n', threshold);
fprintf('Metal voxels:       %d\n', sum(metal_mask(:)));
fprintf('Volume percentage:  %.4f%%\n', ...
    100 * sum(metal_mask(:)) / numel(metal_mask));
fprintf('Output mask:        %s\n', output_mask);
if do_dilate
    fprintf('Output tight mask:  %s\n', tight_mask_name);
end
fprintf('===================\n');
