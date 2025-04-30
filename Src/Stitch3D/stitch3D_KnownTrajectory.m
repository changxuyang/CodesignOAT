function scanOut = stitch3D_KnownTrajectory(scanIn, trajectory, dataName, isPlotting)
%% Defaults
if nargin < 3
    isPlotting = false;
end


%% Data size

f = size( scanIn, 4 );


%% Memory allocation

est_shifts = zeros( f-1, 3 );       % frame to frame shifts
% trajectory = zeros(  f , 3 );       % trajectory

coordinates_frame_0 = [0 0 0];      % coordinates during the stitching process (stitched frame)
coordinates_frame_c = [0 0 0];      % coordinates during the stitching process (new frame (current))

scanOut = ( scanIn(:,:,:,1) );        % stitched scan


%% Stitching process

if isPlotting
    fig1 = figure();
    set(fig1, 'Position', [50, 50, 900, 600]);
end

% pr = progmeter(f-1);
for idf = 2:f
    idf;
    % extract frames
    scan_p = ( scanIn(:,:,:,idf-1) ); % previous scan
    scan_c = ( scanIn(:,:,:,idf+0) ); % current scan
    
    % preprocess frames (remove noise and artefacts)
    scan_p_proc = masking(scan_p, 'histogram', 0.001);
    scan_c_proc = masking(scan_c, 'histogram', 0.001);
    
    % estimate spatial shifts
%     [ shifts, fourier_matrix ] = fourier_shift( scan_p_proc, scan_c_proc, 25, 1.5 );
    [ shifts, fourier_matrix ] = fourier_shift( scan_p_proc, scan_c_proc, 20, 1.5 );
    row_shift = shifts(1); col_shift = shifts(2); sli_shift = shifts(3);
    
    % update trajectory with selected shifts ( T_n = T_{n-1} - t_n )
    % est_shifts(idf-1,:) = [ row_shift, col_shift, sli_shift ];
    % trajectory(idf-0,:) = trajectory(idf-1,:) - est_shifts(idf-1,:);
    coordinates_frame_c = round(trajectory(idf,:));
    
    % merge new frame with stitched frame
    [ scanOut, coordinates_frame_0 ] = mergeframes3D( scanOut, scan_c, coordinates_frame_0, coordinates_frame_c );
    %% 2024.3.11添加
    global MIPxy_0 % 2D images
    global MIPyz_0
    global MIPzx_0
    %%
    % extract new MIPs and rotate MIPs of scan_p and scan_c for visualization
    [MIPxy_0,MIPyz_0,MIPzx_0] = getMIPs(scanOut, '3D');
    [MIPxy_p,MIPyz_p,MIPzx_p] = getMIPs(scan_p, '3D');
    [MIPxy_c,MIPyz_c,MIPzx_c] = getMIPs(scan_c, '3D');
    
    % fourier_matrix = imgaussfilt3(fourier_matrix,1.0);
    [fourier_matrix1, fourier_matrix2, fourier_matrix3] = getMIPs(fourier_matrix, '3D');
    
    if isPlotting
        % plot merged scan
        subplot(3,4,1)
        imagesc(MIPxy_0); axis equal; axis off
        title('Merged Frame')
        subplot(3,4,5)
        imagesc(MIPyz_0'); axis equal; axis off
        subplot(3,4,9)
        imagesc(MIPzx_0); axis equal; axis off
        % plot previous frame
        subplot(3,4,2)
        imagesc(MIPxy_p); axis equal; axis off
        title('Previous Frame')
        subplot(3,4,6)
        imagesc(MIPyz_p'); axis equal; axis off
        subplot(3,4,10)
        imagesc(MIPzx_p); axis equal; axis off
        % plot new scan that was just merged
        subplot(3,4,3)
        imagesc(MIPxy_c); axis equal; axis off
        title('New Frame')
        subplot(3,4,7)
        imagesc(MIPyz_c'); axis equal; axis off
        subplot(3,4,11)
        imagesc(MIPzx_c); axis equal; axis off
        % plot fourier matrices
        subplot(3,4,4)
        imagesc(fourier_matrix1); axis equal; axis off
        title('Fourier Tensor')
        subplot(3,4,8)
        imagesc(fourier_matrix2); axis equal; axis off
        subplot(3,4,12)
        imagesc(fourier_matrix3); axis equal; axis off
        pause(0.001);
    end
   
%     pr.progIt;
end

%% plot merged frame and trajectories

[n0,m0] = size( MIPxy_0 );
[nc,mc] = size( MIPxy_p );

if isPlotting
    figure()
    hold on
    box on
    plot3(trajectory(:,1), trajectory(:,2), trajectory(:,3))
    view([45 45])
    axis equal
    xlabel('x axis')
    ylabel('y axis')
    zlabel('z axis')
    title('3D Trajectories')
    savefig('3D Trajectories');
    
    figure()
    hold on
    box on
    MIPxy_flip = flipud(MIPxy_0);
    imagesc(MIPxy_flip)
    axis equal
    axis off
    colormap bone
    title([dataName '-MIPz'],'Interpreter', 'none');
    % savefig(['MIPz_' dataName]);
    % imwrite(MIPxy_flip, bone, ['MIPz_' dataName, '.jpg']);
     
    % MIPx
    figure()
    hold on
    box on
    MIPyz_flip = flipud(MIPyz_0);
    imagesc(MIPyz_flip)
    axis equal
    axis off
    colormap bone
    title([dataName '-MIPx'],'Interpreter', 'none');
    % savefig(['MIPx_' dataName]);
    % imwrite(MIPyz_flip, bone, ['MIPx_' dataName, '.jpg']);
    
    figure()
    hold on
    box on
    MIPzx_flip = flipud(MIPzx_0);
    imagesc(MIPzx_flip)
    axis equal
    axis off
    colormap bone
    title([dataName '-MIPy'],'Interpreter', 'none');
    % savefig(['MIPy_' dataName]);
    % imwrite(MIPzx_flip, bone, ['MIPy_' dataName, '.jpg']);
end


end

function [ mask ] = circmask( volume, distance )

%%

[n,m,k] = size(volume);

row_vec = (1:n)';
col_vec = (1:m);
sli_vec = zeros(1,1,k); sli_vec(1:end) = (1:k);

tensor_row = repmat(row_vec, [1,m,k]);
tensor_col = repmat(col_vec, [n,1,k]);
tensor_sli = repmat(sli_vec, [n,m,1]);

mask = sqrt((tensor_row-n/2).^2 + (tensor_col-m/2).^2 + (tensor_sli-k/2).^2);

mask(find(mask>distance)) = 0;

end

function [ shifts, tau_fftshift ] = fourier_shift( frame1, frame2, maxshift, sigma_filter )

if (nargin==2)
    maxshift = 25;
    sigma_filter = 1.5;
elseif (nargin==3)
    sigma_filter = 1.5;
end

% check for apporpriate dimensions of input data
d1 = ndims(frame1);
d2 = ndims(frame2);

if ( ~( d1==3 || d1==2 ) || ~( d2==3 || d2==2 ) || ~isequal(d1,d2) )
    shifts = 0;
    tau_fftshift = 0;
    return;
end

%%

if (d1==2 && d2==2)
    
    % extract numbers of voxels and adapt in uneven
    [n1,m1] = size(frame1);
    [n2,m2] = size(frame2);

    n = max(n1,n2);
    m = max(m1,m2);

    if (mod(n,2)~=0); n=n+1; end
    if (mod(m,2)~=0); m=m+1; end


    % 3D fourier transform of images
    FRAME1 = fft2(frame1,n,m);
    FRAME2 = fft2(frame2,n,m);

    % computation of tau in frequency and spatial domain
    TAU = FRAME1.*conj(FRAME2)./abs(FRAME1.*conj(FRAME2));
    tau = ifft2(TAU);

    % process tau for optimised shift estimation
    tau_fftshift = fftshift(tau);
    tau_fftshift = tau_fftshift.*circmask(tau_fftshift,maxshift);
    tau_fftshift = imgaussfilt(tau_fftshift,sigma_filter);

    % find dirac position
    [ value_new, index_new ] = max(tau_fftshift(:));
    [ row_shift,col_shift ] = ind2sub(size(tau_fftshift),index_new);


    % transform tensor indices into shifts
    row_shift = row_shift-n/2;
    col_shift = col_shift-m/2;

    % '+1' since peak at (1,1) means shift of (0,0)
    row_shift = -row_shift+1;
    col_shift = -col_shift+1;

    shifts = [ row_shift, col_shift ];
    
    
elseif (d1==3 && d2==3)

    % extract numbers of voxels and adapt in uneven
    [n1,m1,k1] = size(frame1);
    [n2,m2,k2] = size(frame2);

    n = max(n1,n2);
    m = max(m1,m2);
    k = max(k1,k2);

    if (mod(n,2)~=0); n=n+1; end
    if (mod(m,2)~=0); m=m+1; end
    if (mod(k,2)~=0); k=k+1; end


    % 3D fourier transform of images
    FRAME1 = fftn(frame1,[n,m,k]);
    FRAME2 = fftn(frame2,[n,m,k]);

    % computation of tau in frequency and spatial domain
    TAU = FRAME1.*conj(FRAME2)./abs(FRAME1.*conj(FRAME2));
    tau = ifftn(TAU);

    % process tau for optimised shift estimation
    tau_fftshift = fftshift(tau);
    tau_fftshift = tau_fftshift.*circmask(tau_fftshift,maxshift);
    tau_fftshift = imgaussfilt3(tau_fftshift,sigma_filter);

    % find dirac position
    [ value_new, index_new ] = max(tau_fftshift(:));
    [ row_shift,col_shift,sli_shift ] = ind2sub(size(tau_fftshift),index_new);


    % transform tensor indices into shifts
    row_shift = row_shift-n/2;
    col_shift = col_shift-m/2;
    sli_shift = sli_shift-k/2;

    % '+1' since peak at (1,1,1) means shift of (0,0,0)
    row_shift = -row_shift+1;
    col_shift = -col_shift+1;
    sli_shift = -sli_shift+1;

    shifts = [ row_shift, col_shift, sli_shift ];

else
    
    shifts = 0;
    tau_fftshift = 0;
    
end


end

function [ MIPxy, MIPyz, MIPzx ] = getMIPs( Dataset,format )

if strcmp(format,'3D')
    MIPxy = squeeze(max(Dataset,[],3));
    MIPyz = squeeze(max(Dataset,[],1));
    MIPzx = squeeze(max(Dataset,[],2));
elseif strcmp(format,'4D')
    MIPxy = squeeze(max(Dataset,[],3));
    MIPyz = squeeze(max(Dataset,[],1));
    MIPzx = squeeze(max(Dataset,[],2));
end


end

function [ image_out ] = masking( image, mode, threshold )

if strcmp(mode, 'absolute')

    mask = double( scale_image(image) > threshold );
    image_out = mask .* image;
    
elseif strcmp(mode, 'histogram')
    
    values = scale_image(image(:));
    n = length(values);
    [vals, ind] = sort(values,'descend');
    
    new_threshold = vals(ceil(threshold*n));
    
    mask = double( scale_image(image) > new_threshold );
    image_out = mask .* image;
    % image_out = mask;

elseif strcmp(mode, 'histmask')
    
    values = scale_image(image(:));
    n = length(values);
    [vals, ind] = sort(values,'descend');
    
    new_threshold = vals(ceil(threshold*n));
    
    image_out = double( scale_image(image) > new_threshold );
    
else
    
    image_out = double( scale_image(image) > threshold );
    
end

end
function [ scan_merged, coordinates_scan ] = mergeframes3D( scan1, scan2, coordinates_scan1, coordinates_scan2, mode )

if (nargin==4)
    mode = 'max';
end


% extract image sizes of image1 and image2
[nof_rows_1,nof_columns_1,nof_slices_1] = size(scan1);
[nof_rows_2,nof_columns_2,nof_slices_2] = size(scan2);

% compute coordinates of new images
coordinates_scan(1) = min([coordinates_scan1(1), coordinates_scan2(1)]);
coordinates_scan(2) = min([coordinates_scan1(2), coordinates_scan2(2)]);
coordinates_scan(3) = min([coordinates_scan1(3), coordinates_scan2(3)]);

% get number of pixels for merged image
nof_rows_scan    = max([coordinates_scan1(1)+nof_rows_1, coordinates_scan2(1)+nof_rows_2])       - min([coordinates_scan1(1), coordinates_scan2(1)]);
nof_columns_scan = max([coordinates_scan1(2)+nof_columns_1, coordinates_scan2(2)+nof_columns_2]) - min([coordinates_scan1(2), coordinates_scan2(2)]);
nof_slices_scan  = max([coordinates_scan1(3)+nof_slices_1, coordinates_scan2(3)+nof_slices_2])   - min([coordinates_scan1(3), coordinates_scan2(3)]);

% memory allocation for merged image
scan = zeros(nof_rows_scan, nof_columns_scan, nof_slices_scan, 2);

row_range1 = (coordinates_scan1(1) - coordinates_scan(1)) + (1:nof_rows_1);
row_range2 = (coordinates_scan2(1) - coordinates_scan(1)) + (1:nof_rows_2);

column_range1 = (coordinates_scan1(2) - coordinates_scan(2)) + (1:nof_columns_1);
column_range2 = (coordinates_scan2(2) - coordinates_scan(2)) + (1:nof_columns_2);

slice_range1 = (coordinates_scan1(3) - coordinates_scan(3)) + (1:nof_slices_1);
slice_range2 = (coordinates_scan2(3) - coordinates_scan(3)) + (1:nof_slices_2);


% put image1 and image2 according to shifts in plane1 and plane2 of the
% merger image
scan(row_range1, column_range1, slice_range1, 1) = scan1;
scan(row_range2, column_range2, slice_range2, 2) = scan2;


if (strcmp(mode,'max'))
        % select brighter pixel of both images
        scan_merged = squeeze(max(scan, [], 4));
elseif (strcmp(mode,'sum'))
        % select brighter pixel of both images
        scan_merged = (scan(:,:,:,1) + scan(:,:,:,2));
elseif (strcmp(mode,'mean'))
        % select brighter pixel of both images
        normalization = ones(nof_rows_scan,nof_columns_scan,nof_slices_scan) + double(scan(:,:,:,2)>0);
        scan_merged = (scan(:,:,:,1) + scan(:,:,:,2))./normalization;
else
        % select brighter pixel of both images
        scan_merged = squeeze(max(scan, [], 4));
end

pause(0.001);

end

function [ scaled_image ] = scale_image( image )
maximum = max( image(:) );
minimum = min( image(:) );

scaled_image = 1/(maximum - minimum)*( image - minimum );

end
