

function assigned_states = executa_segmentacao(arquivo);
% close all;
% clear all;

load("ambiente.mat")

path = strcat("../Banco_A/", arquivo);
audio = audioread(path);
assigned_states = runSpringerSegmentationAlgorithm(audio, 2000, B_matrix, pi_vector, total_obs_distribution, false);

end