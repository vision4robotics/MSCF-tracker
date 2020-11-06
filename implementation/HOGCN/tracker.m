
function results = tracker(params)
%% Initialization
% Get sequence info
learning_rate = params.learning_rate;
lambda_1 = params.admm_lambda_1;
lambda_2 = params.admm_lambda_2;
phi      = params.admm_phi;
delta    = params.delta;
nu       = params.nu;
eta      = params.eta;
sz_ratio = params.sz_ratio;
update_interval = 1;
half_sub_cen    = 6;
cen = 2;
tau = 0; 


[seq, im] = get_sequence_info(params.seq);
params = rmfield(params, 'seq');
if isempty(im)
    seq.rect_position = [];
    [~, results] = get_sequence_results(seq);
    return;
end
% Init position
pos = seq.init_pos(:)';
% context position
target_sz = seq.init_sz(:)';
params.init_sz = target_sz;

% Feature settings
features = params.t_features;

% Set default parameters
params = init_default_params(params);

% Global feature parameters
if isfield(params, 't_global')
    global_fparams = params.t_global;
else
    global_fparams = [];
end

global_fparams.use_gpu = params.use_gpu;
global_fparams.gpu_id = params.gpu_id;

% Define data types
params.data_type = zeros(1, 'single');
params.data_type_complex = complex(params.data_type);

global_fparams.data_type = params.data_type;

init_target_sz = target_sz;

% Check if color image
if size(im,3) == 3
    if all(all(im(:,:,1) == im(:,:,2)))
        is_color_image = false;
    else
        is_color_image = true;
    end
else
    is_color_image = false;
end

if size(im,3) > 1 && is_color_image == false
    im = im(:,:,1);
end

% Check if mexResize is available and show warning otherwise.
params.use_mexResize = true;
global_fparams.use_mexResize = true;
try
    [~] = mexResize(ones(5,5,3,'uint8'), [3 3], 'auto');
catch err
    params.use_mexResize = false;
    global_fparams.use_mexResize = false;
end

% Calculate search area and initial scale factor
search_area = prod(init_target_sz * params.search_area_scale);
if search_area > params.image_sample_size
    currentScaleFactor = sqrt(search_area / params.image_sample_size);
elseif search_area < params.image_sample_size
    currentScaleFactor = sqrt(search_area / params.image_sample_size);
else
    currentScaleFactor = 1.0;
end

% target size at the initial scale
base_target_sz = target_sz / currentScaleFactor;

% window size, taking padding into account
switch params.search_area_shape
    case 'proportional'
        img_sample_sz = floor(base_target_sz * params.search_area_scale);     % proportional area, same aspect ratio as the target
    case 'square'
        img_sample_sz = repmat(sqrt(prod(base_target_sz * params.search_area_scale)), 1, 2); % square area, ignores the target aspect ratio
    case 'fix_padding'
        img_sample_sz = base_target_sz + sqrt(prod(base_target_sz * params.search_area_scale) + (base_target_sz(1) - base_target_sz(2))/4) - sum(base_target_sz)/2; % const padding
    case 'custom'
        img_sample_sz = [base_target_sz(1)*2 base_target_sz(2)*2];
end

[features, global_fparams, feature_info] = init_features(features, global_fparams, is_color_image, img_sample_sz, 'exact');

% Set feature info
img_support_sz = feature_info.img_support_sz;
feature_sz = unique(feature_info.data_sz, 'rows', 'stable');
feature_cell_sz = unique(feature_info.min_cell_size, 'rows', 'stable');
num_feature_blocks = size(feature_sz, 1);

small_filter_sz{1} = floor(base_target_sz/(feature_cell_sz(1,1)));

% Get feature specific parameters
feature_extract_info = get_feature_extract_info(features);

% Size of the extracted feature maps
feature_sz_cell = mat2cell(feature_sz, ones(1,num_feature_blocks), 2);
filter_sz = feature_sz;
filter_sz_cell = permute(mat2cell(filter_sz, ones(1,num_feature_blocks), 2), [2 3 1]);

filter_sz_cell_ours{1} = filter_sz_cell{1}; 

% The size of the label function DFT. Equal to the maximum filter size
[output_sz_hand, k1] = max(filter_sz, [], 1);

output_sz = output_sz_hand;

k1 = k1(1);
% Get the remaining block indices
block_inds = 1:num_feature_blocks;
block_inds(k1) = [];



% Initialize dynamic hybrid label
% central_target_space = eta * prod(init_target_sz); 

% update dynamic hybrid label
    central_target      = base_target_sz/sz_ratio;

% Construct the Gaussian label function
    yf = cell(numel(num_feature_blocks), 1);
    for i = 1:num_feature_blocks
        sz = filter_sz_cell{i};
        output_sigma = sqrt(prod(floor(base_target_sz/feature_cell_sz(i)))) * params.output_sigma_factor;
        rg1           = circshift(-floor((sz(1)-1)/2):ceil((sz(1)-1)/2), [0 -floor((sz(1)-1)/2)]);
        cg1           = circshift(-floor((sz(2)-1)/2):ceil((sz(2)-1)/2), [0 -floor((sz(2)-1)/2)]);
        [rs1, cs1]    = ndgrid(rg1,cg1);   
        y1            = exp(-0.5 * (((rs1.^2 + cs1.^2) / output_sigma^2)));
        yf            = fft2(y1); 
    end
    
% Construct 1st direction square label
    for i = 1:num_feature_blocks
        central_target(1) = max(central_target(1),5);
        central_target(2) = max(central_target(2),5);
        rg2 = ones(1,sz(1));
        cg2 = ones(1,sz(2));    
        [rs2,cs2] = ndgrid(rg2,cg2);
        rs2(:,ceil((central_target(1)-1)/2):sz(1)-floor((central_target(1)-1)/2)) = 0;
        rs2(ceil((central_target(2)-1)/2):sz(1)-floor((central_target(2)-1)/2),:) = 0;
        cs2(:,ceil((central_target(1)-1)/2):sz(1)-floor((central_target(1)-1)/2)) = 0;
        cs2(ceil((central_target(2)-1)/2):sz(1)-floor((central_target(2)-1)/2),:) = 0;
        y2 = ((rs2+cs2)/2);
    end
    
%Constrcut 2nd direction square label
    for i = 1:num_feature_blocks
        rg3 = ones(1,sz(1));
        cg3 = ones(1,sz(2));    
        [cs3,rs3] = ndgrid(rg3,cg3);
        rs3(:,ceil((central_target(2)-1)/2):sz(2)-floor((central_target(2)-1)/2)) = 0;
        rs3(ceil((central_target(1)-1)/2):sz(1)-floor((central_target(1)-1)/2),:) = 0;
        cs3(:,ceil((central_target(2)-1)/2):sz(2)-floor((central_target(2)-1)/2)) = 0;
        cs3(ceil((central_target(1)-1)/2):sz(1)-floor((central_target(1)-1)/2),:) = 0;
        y3 = ((rs3+cs3)/2);
    end
    
%Constrcut the overlap label
    for i = 1:num_feature_blocks
        rg4 = ones(1,sz(1));
        cg4 = ones(1,sz(2));    
        [cs4,rs4] = ndgrid(rg4,cg4);
        min_sz = min(central_target(1),central_target(2));
        rs4(:,ceil((min_sz-1)/2):sz(2)-floor((min_sz-1)/2)) = 0;
        rs4(ceil((min_sz-1)/2):sz(1)-floor((min_sz-1)/2),:) = 0;
        cs4(:,ceil((min_sz-1)/2):sz(2)-floor((min_sz-1)/2)) = 0;
        cs4(ceil((min_sz-1)/2):sz(1)-floor((min_sz-1)/2),:) = 0;    
        y4 = ((rs4+cs4)/2);
    end
    
% Constrcut the hybrid label
    y_2 = ((y2./y2(1,1))+(y3./y3(1,1))-(y4./(y4(1,1))));
    y_2_f = fft2(y_2);
    
% Construct the distance matrix
    sub_cen = 2*half_sub_cen + 1;
    dis = zeros(sub_cen);
    for i=1:sub_cen
        for j=1:sub_cen
            dist_2   = sqrt((i-(half_sub_cen+1))^2 + (j-(half_sub_cen+1))^2);
            dis(i,j) = nu/(1+delta*exp(dist_2));
        end
    end
% Compute the cosine windows
cos_window = cellfun(@(sz) hann(sz(1))*hann(sz(2))', feature_sz_cell, 'uniformoutput', false);


% Pre-computes the grid that is used for socre optimization
ky = circshift(-floor((filter_sz_cell{1}(1) - 1)/2) : ceil((filter_sz_cell{1}(1) - 1)/2), [1, -floor((filter_sz_cell{1}(1) - 1)/2)]);
kx = circshift(-floor((filter_sz_cell{1}(2) - 1)/2) : ceil((filter_sz_cell{1}(2) - 1)/2), [1, -floor((filter_sz_cell{1}(2) - 1)/2)])';
newton_iterations = params.newton_iterations;

% Use the translation filter to estimate the scale
% lfl: parameters for scale estimation
scale_sigma = sqrt(params.num_scales) * params.scale_sigma_factor;
ss = (1:params.num_scales) - ceil(params.num_scales/2);
ys = exp(-0.5 * (ss.^2) / scale_sigma^2);
ysf = single(fft(ys));
if mod(params.num_scales,2) == 0
    scale_window = single(hann(params.num_scales+1));
    scale_window = scale_window(2:end);
else
    scale_window = single(hann(params.num_scales));
end
ss = 1:params.num_scales;
scaleFactors = params.scale_step.^(ceil(params.num_scales/2) - ss);
if params.scale_model_factor^2 * prod(params.init_sz) > params.scale_model_max_area
    params.scale_model_factor = sqrt(params.scale_model_max_area/prod(params.init_sz));
end

if prod(params.init_sz) > params.scale_model_max_area
    params.scale_model_factor = sqrt(params.scale_model_max_area/prod(params.init_sz));
end
scale_model_sz = floor(params.init_sz * params.scale_model_factor);

% set maximum and minimum scales
min_scale_factor = params.scale_step ^ ceil(log(max(5 ./ img_support_sz)) / log(params.scale_step));
max_scale_factor = params.scale_step ^ floor(log(min([size(im,1) size(im,2)] ./ base_target_sz)) / log(params.scale_step));

seq.time = 0;

% Define the learning variables
cf_f = cell(num_feature_blocks, 1);

% Allocate
scores_fs_feat = cell(1,1,3);

%% Main loop here
while true
    % Read image
    if seq.frame > 0
        [seq, im] = get_sequence_frame(seq);
        if isempty(im)
            break;
        end
        if size(im,3) > 1 && is_color_image == false
            im = im(:,:,1);
        end
    else
        seq.frame = 1;
    end

    tic();
    
    % Target localization step
    
    % Do not estimate translation and scaling on the first frame, since we 
    % just want to initialize the tracker there
    if seq.frame > 1
        old_pos = inf(size(pos));
        iter = 1;
        
        %translation search
        while iter <= params.refinement_iterations && any(old_pos ~= pos)
            % Extract features at multiple resolutions
            sample_pos = round(pos);
            xt = extract_features(im, sample_pos, currentScaleFactor, features, global_fparams, feature_extract_info);
            % Do windowing of features
            xtw = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xt, cos_window, 'uniformoutput', false);
            % Compute the fourier series
            xtf = cellfun(@fft2, xtw, 'uniformoutput', false);
            
            % Compute convolution for each feature block in the Fourier domain
            % and the sum over all blocks.
            scores_fs_feat{k1} = gather(sum(bsxfun(@times, conj(cf_f{k1}), xtf{k1}), 3));
            scores_fs_sum = scores_fs_feat{k1};
            for k = block_inds
                scores_fs_feat{k} = gather(sum(bsxfun(@times, conj(cf_f{k}), xtf{k}), 3));
                scores_fs_feat{k} = resizeDFT2(scores_fs_feat{k}, output_sz);
                scores_fs_sum = scores_fs_sum +  scores_fs_feat{k};
            end
             
            % Also sum over all feature blocks.
            % Gives the fourier coefficients of the convolution response.
            scores_fs = permute(gather(scores_fs_sum), [1 2 4 3]);
            
            responsef_padded = resizeDFT2(scores_fs, output_sz);
            response = ifft2(responsef_padded, 'symmetric');
            [disp_row, disp_col, sind] = resp_newton(response, responsef_padded, newton_iterations, ky, kx, output_sz);

            % Compute the translation vector in pixel-coordinates and round
            % to the closest integer pixel.
            translation_vec = [disp_row, disp_col] .* (img_support_sz./output_sz) * currentScaleFactor;            
            old_pos = pos;
            pos = sample_pos + translation_vec;
            
            if params.clamp_position
                pos = max([1 1], min([size(im,1) size(im,2)], pos));
            end
            
            % lfl: SCALE SPACE SEARCH
            xs = get_scale_sample(im, pos, base_target_sz, currentScaleFactor * scaleFactors, scale_window, scale_model_sz);
            xsf = fft(xs,[],2);
            scale_response = real(ifft(sum(sf_num .* xsf, 1) ./ (sf_den + params.scale_lambda) ));            
            % find the maximum scale response
            recovered_scale = find(scale_response == max(scale_response(:)), 1);
            % update the scale
            currentScaleFactor = currentScaleFactor * scaleFactors(recovered_scale);
            if currentScaleFactor < min_scale_factor
                currentScaleFactor = min_scale_factor;
            elseif currentScaleFactor > max_scale_factor
                currentScaleFactor = max_scale_factor;
            end  
            % Evaluate the effect of side peak 
                % current frame
                map_shift = fftshift(response);
                map_sz = size(map_shift);
                [r_max,cen_pos] = max(map_shift(:));
                [rpos,cpos] = ind2sub(size(map_shift),cen_pos);
                 
                if rpos+half_sub_cen+1>map_sz(1)
                    rpos=map_sz(1)-half_sub_cen-1;
                elseif rpos-half_sub_cen-1<0
                    rpos=half_sub_cen+1;
                end
                if cpos+half_sub_cen+1>map_sz(2)
                    cpos=map_sz(2)-half_sub_cen-1;
                elseif cpos-half_sub_cen-1<0
                    cpos=half_sub_cen+1;
                end
  
                sub_cen = map_shift(rpos-half_sub_cen:rpos+half_sub_cen,cpos-half_sub_cen:cpos+half_sub_cen); 
                sub_cen(half_sub_cen : half_sub_cen+cen,half_sub_cen : half_sub_cen+cen) = 0;
                find_local_max = imregionalmax(sub_cen);
                
                local_max = sub_cen.*find_local_max.*dis;
                
                [rsd_max,~] = max(local_max(:));   
                tau = rsd_max/r_max;
                
           iter = iter + 1;
        end
    end
    
    if seq.frame == 1
        omega_f = yf ;
    else 
        omega_f = yf + eta*(1-tau)* y_2_f;
    end
    
    %% Model update step
    % extract image region for training sample
    sample_pos = round(pos);
    xl = extract_features(im, sample_pos, currentScaleFactor, features, global_fparams, feature_extract_info);
    % do windowing of features
    xlw = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xl, cos_window, 'uniformoutput', false);
    % compute the fourier series
    xlf = cellfun(@fft2, xlw, 'uniformoutput', false);
    % train the CF model for each feature

    for k = 1: 1
         
            if (seq.frame == 1)
                model_xf = xlf{k};

                %initialize rp_f
                rp_f = yf;
                r_f  = yf;             
            else
                model_xf = ((1 - learning_rate) * model_xf) + (learning_rate * xlf{k});
                rp_f = r_f;
            end
            
        if (seq.frame==1||mod(seq.frame,update_interval)==0)    
            
            g_f      = single(zeros(size(xlf{k})));
            h_f      = g_f;
            l_f      = g_f;
            cf_f{1}  = g_f;
            xi_f     = single(zeros(size(yf)));
            mu       = params.admm_mu;
            gamma    = params.admm_gamma;
            betha    = 10;
            mumax    = 10000;
            gammamax = 10000;
            i      = 1;
            ii     = 1;
            iii    = 1;
            
            T = prod(filter_sz_cell_ours{k});
            S_xx = sum(conj(model_xf) .* model_xf, 3);
            % ADMM solving process
            while (i <= params.admm_iterations )
                %   solve for G- please refer to the paper for more details
                B = S_xx + (T * mu) ;
                S_lx = sum(conj(model_xf) .* l_f, 3);
                S_hx = sum(conj(model_xf) .* h_f, 3);
                g_f = (((1/(T*mu)) * bsxfun(@times, omega_f, model_xf)) - ((1/mu) * l_f) + h_f) - ...
                    bsxfun(@rdivide,(((1/(T*mu)) * bsxfun(@times, model_xf, (S_xx .* omega_f))) - ((1/mu) * bsxfun(@times, model_xf, S_lx)) + (bsxfun(@times, model_xf, S_hx))), B);

                %   solve for H
                h = (T/((mu*T)+ lambda_1))* ifft2((mu*g_f) + l_f);
                [sx,sy,h] = get_subwindow_no_window(h, floor(filter_sz_cell_ours{k}/2) , small_filter_sz{k});
                t = zeros(filter_sz_cell_ours{k}(1), filter_sz_cell_ours{k}(2), size(h,3));
                t(sx,sy,:) = h;
                h_f = fft2(t);

                %   update L
                l_f = l_f + (mu * (g_f - h_f));
                cf_f{k} = g_f;
                %   update mu- betha = 10.
                mu = min(betha * mu, mumax);
                i = i+1;
            end

            while(ii <= params.admm_iterations)
            %solve for s
                S_gx = sum(bsxfun(@times, conj(h_f), model_xf), 3);
                s_f  = 1/(1+gamma*T)*(S_gx + gamma*T*r_f + T*xi_f);                    
            %solve for r
                %r_f   = 1/(lambda_2*(1+Psi^2)+phi*(1-Psi^2)+gamma*T)*(lambda_2*(1+Psi^2)*(yf + eta*(1-Psi)* y_2_f) + phi*(1-Psi^2)*rp_f + gamma*T*s_f - T*xi_f);  
                r_f   = 1/(lambda_2+phi+gamma*T)*(lambda_2*omega_f + phi*rp_f + gamma*T*s_f - T*xi_f);   
                %r_f   = 1/(lambda_2+gamma*T)*(lambda_2*(yf + eta*y_2_f) + gamma*T*s_f - T*xi_f);
            %update Xi
                xi_f  = xi_f + gamma*(s_f - r_f);                   
            %    update gamma- betha =10.
                gamma = min(betha * gamma, gammamax);
                ii    = ii+1; 
            end
            
            while (iii <= params.admm_iterations )
                %   solve for G- please refer to the paper for more details
                B = S_xx + (T * mu) ;
                S_lx = sum(conj(model_xf) .* l_f, 3);
                S_hx = sum(conj(model_xf) .* h_f, 3);
                g_f = (((1/(T*mu)) * bsxfun(@times, r_f, model_xf)) - ((1/mu) * l_f) + h_f) - ...
                    bsxfun(@rdivide,(((1/(T*mu)) * bsxfun(@times, model_xf, (S_xx .* r_f))) - ((1/mu) * bsxfun(@times, model_xf, S_lx)) + (bsxfun(@times, model_xf, S_hx))), B);
                %   solve for H
                h = (T/((mu*T)+ lambda_1))* ifft2((mu*g_f) + l_f);
                [sx,sy,h] = get_subwindow_no_window(h, floor(filter_sz_cell_ours{k}/2) , small_filter_sz{k});
                t = zeros(filter_sz_cell_ours{k}(1), filter_sz_cell_ours{k}(2), size(h,3));
                t(sx,sy,:) = h;
                h_f = fft2(t);
                %   update L
                l_f = l_f + (mu * (g_f - h_f));
                cf_f{k} = g_f;
                %   update mu- betha = 10.
                mu = min(betha * mu, mumax);
                iii = iii+1;
             end
        end
    end
            scores_fs_feat{k1} = gather(sum(bsxfun(@times, conj(cf_f{k1}), xlf{k1}), 3));
            scores_fs_sum = scores_fs_feat{k1};
            for k = block_inds
                scores_fs_feat{k} = gather(sum(bsxfun(@times, conj(cf_f{k}), xlf{k}), 3));
                scores_fs_feat{k} = resizeDFT2(scores_fs_feat{k}, output_sz);
                scores_fs_sum = scores_fs_sum +  scores_fs_feat{k};
            end
             
    
    %% Upadate Scale
    xs = get_scale_sample(im, pos, base_target_sz, currentScaleFactor * scaleFactors, scale_window, scale_model_sz);
    xsf = fft(xs,[],2);
    new_sf_num = bsxfun(@times, ysf, conj(xsf));
    new_sf_den = sum(xsf .* conj(xsf), 1);
    
    if seq.frame == 1
        sf_den = new_sf_den;
        sf_num = new_sf_num;
    else
        sf_den = (1 - params.learning_rate_scale) * sf_den + params.learning_rate_scale * new_sf_den;
        sf_num = (1 - params.learning_rate_scale) * sf_num + params.learning_rate_scale * new_sf_num;
    end
    % Update the target size (only used for computing output box)
    target_sz = base_target_sz * currentScaleFactor;
    
    %fprintf('target_sz: %f, %f \n', target_sz(1), target_sz(2));
    
    %save position and calculate FPS
    tracking_result.center_pos = double(pos);
    tracking_result.target_size = double(target_sz);
    seq = report_tracking_result(seq, tracking_result);
    
    seq.time = seq.time + toc();
    
    %% Visualization
    if params.visualization
        figure(462);
        rect_position_vis = [pos([2,1]) - (target_sz([2,1]) - 1)/2, target_sz([2,1])];
        im_to_show = double(im)/255;
        if size(im_to_show,3) == 1
            im_to_show = repmat(im_to_show, [1 1 3]);
        end
        
        imagesc(im_to_show);
        hold on;
        rectangle('Position',rect_position_vis, 'EdgeColor','g', 'LineWidth',2);
        text(10, 10, [int2str(seq.frame) '/'  int2str(size(seq.image_files, 1))], 'color', [0 1 1]);
        hold off;
        axis off;axis image;set(gca, 'Units', 'normalized', 'Position', [0 0 1 1])
                    
        drawnow
    end
end

[~, results] = get_sequence_results(seq);

disp(['fps: ' num2str(results.fps)])

