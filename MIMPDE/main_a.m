clear, clc, close all;
% Random seed
rng(42);
% Number of k in K-nearest neighbor
opts.k = 5; 
% Common parameter settings 
opts.N  = 100;     % number of solutions
opts.T  = 100;    % maximum number of iterations
runNum =20;       % Number of independent runs
% Parameters 
opts.ng = 25;           % Strategy evaluation period 
opts.p_rate = 0.1;      % Percentage for pbest/pbad 
opts.delta_sharing = 0.04; %  Leader-follower learning rate 

% Algorithm list
Algs = ["mimpde"]; 
AlgsNum = length(Algs);

% Load dataset
Problem=["Colon","lung","Lymphoma","GLIOMA","TOX_171","Prostate_GE","Leukemia","ALLAML","nci9","arcene","CLL-SUB-111","SMK-CAN-187","GLI_85"];
DataPath='C:\Users\lenvov\Desktop\Code\DataSet\';

plotdata = zeros(AlgsNum,runNum,opts.T);    % Record the best solution of each algorithm per run per generation 
result = zeros(AlgsNum,opts.T);             % Record the mean of best solutions per generation after multiple runs of each algorithm
SFNum = zeros(AlgsNum,runNum);              % Number of features selected by each algorithm per run
Time = zeros(AlgsNum,runNum);               % Runtime of each algorithm per run
Accuracy = zeros(AlgsNum,runNum);
Gbest = zeros(AlgsNum,runNum);

for p = 1 : length(Problem) %length(Problem)
    p_name = Problem(p);
    dataname=strcat(DataPath,p_name,'.mat');
    load(dataname); 
    fprintf(">>>>>>>>>>load data: <%s>\n",p_name);
    
    [~, total_feats] = size(feat); % Get total number of features
    ho = 0.3;
    if total_feats >= 2000 % For high-dimensional data >=2000 
    K = 0.3; 
    end
    fprintf("Total Feats: %d, Using K=%.2f\n", total_feats, K);
    
    for i = 1 : AlgsNum
        fprintf("============== %s ==============\n",Algs(i));
        for n = 1 : runNum

              % Divide data into training and validation sets
            CV = cvpartition(label,'KFold',5);
            opts.Model = CV;
            [MIFeat, selected_mi_values] = MI_TOOL(feat,label,K); 
    
            % Store MI values in opts structure for passing to jfs and gMI_MPODE
            opts.mi_values = selected_mi_values;
            FS = jfs(Algs(i),MIFeat,label,opts);
             
            sf_idx = FS.sf;             % Define index of selected features
            plotdata(i,n,:) = FS.c;       % Record the best solution per generation for convergence graph FS.c is the best solution
            Time(i,n) = FS.t;
            SFNum(i,n) = FS.nf;
            Gbest(i,n) = FS.gb;
            Accuracy(i,n)  = jknn(MIFeat(:,sf_idx),label,opts); % Accuracy
        end
        fprintf('\n %s: \n Mean of Accuracy: %f\n Std of Accuracy: %f\n',...
                Algs(i), mean(Accuracy(i,:)), std(Accuracy(i,:)) );
        fprintf(' Mean of Gbest: %f\n Std of Gbest: %f\n',...
                   mean(Gbest(i,:)), std(Gbest(i,:)) );
        fprintf(" Number of Selected Feature: %f \n",mean(SFNum(i,:)));    % Average number of selected features across multiple runs
        fprintf(" Mean of Times: %f\n",mean(Time(i,:)) );                  % Average time
        
   end
% Data saving


if  AlgsNum == 6 && Algs(1) == "mimpde"
    savepath = 'C:\Users\lenvov\Desktop\Code\SaveData\Algorithms';
    plotpath = 'C:\Users\lenvov\Desktop\Code\SaveData\PlotData';
   
else
    savepath = 'D:\StudyData\MIMPDE\Data result\Com AlgorithmsA\';
    plotpath = 'D:\StudyData\MIMPDE\Data result\PlotData\com algorithmsA\';
   
end
    filename1 = strcat(savepath,p_name,'.mat');
    save(filename1,'Accuracy','SFNum','Time','Gbest');
    filename2 = strcat(plotpath,p_name,'.mat');
    save(filename2,'plotdata');
    fprintf("\n %s data saved successfully!\n",p_name);
end
