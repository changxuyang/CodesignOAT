clc;clear; close all;
% load( '..\..\Outputs\Results_Exp3DHemispherical_32elements_RtHand_thmFinger_FT\Matresults.mat' );
% load('./Trajectory_RtHand_thmFinger_FT_Seq1.mat')

load( '..\..\Outputs\Results_Exp3DHemispherical_32elements_RtHand_mdFinger_BK\Matresults.mat' );
load('./Trajectory_RtHand_mdFinger_BK_Seq2.mat')

scanOut = stitch3D_KnownTrajectory(results(:,:,:,1:2:end), trajectory, 'OutImage', 0);

MIP1 = mat2gray((max(scanOut,[],3)));MIP2 = squeeze(mat2gray((max(scanOut,[],2))));MIP3 = squeeze(mat2gray((max(scanOut,[],1))));
figure;imshow(MIP1,[0.5,1]);colormap('bone');title('XY-MIP image');
