function [assigned_states] = executa_segmentacao();

close all;
clear all;

load("ambiente.mat")
audio = audioread("../Banco_A/a0200.wav");
[assigned_states] = runSpringerSegmentationAlgorithm(audio, 2000, B_matrix, pi_vector, total_obs_distribution, true);

end