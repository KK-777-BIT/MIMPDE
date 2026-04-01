clc;
close all;
clear;
problem=["Colon","TOX_171","Leukemia","ALLAML","GLI_85","Prostate_GE","arcene","GLIOMA","lymphoma","nci9","CLL-SUB-111","lung","SMK-CAN-187"];
% Import source data files
OriginDataPath='C:\Users\lenvov\Desktop\datasets\OriginalData\';
DataPath='C:\Users\lenvov\Desktop\datasets\';
num = 9;

%% Process data
for i = 34:35  
    p_name=problem(i); % Get the name of the current dataset in the loop.
    fprintf("Reading data: %s\n",p_name);
    % Source data file path
     datapath=strcat(OriginDataPath,p_name,'.mat');  %TOX-171, Colon,Leukemia
    % Load data
    data = load(datapath);
    % Save mat file path
    savepath=strcat(DataPath,p_name,'.mat');
    
    if  i == 2 || i==8
        label = data(:,1);   %label in the first column of the file
        feat = data(:,2:end);    
    elseif i==1 || i==3 || i==4 || i==6 || i==9 || i==11 || i==13  
        label = data(:,end); %label in the last column of the file
        feat = data(:,1:end-1);
    elseif i==12           
        datapath=strcat(OriginDataPath,p_name,'.xlsx'); 
        feat = zeros(126,310);
        for n = 2:127
            S1 = "A" + n;
            S2 = "KX" + n;
            S = S1 + ":" + S2;
            data1(n,:) = xlsread(datapath,'Data',S);
            feat(n-1,:) = data1(n,:);
            
        end
            data2 = xlsread(datapath,'Binary response','A2:A127');
            label = data2;
            
    elseif i==5            
        feat = zeros(195,22);
        label = data(:,17);
        for i = 1:195
            for j = 1:23               
                if j ~=17
                  feat(i,j) = data(i,j);
                end
                
            end
        end
        feat(:,17)=[];
    else                             % Convert X,Y to feat,label
        feat=data.X;  %Extract feature data from the X field of the data structure and assign it to the variable feat.
        label=data.Y; %Extract label data from the Y field of the data structure and assign it to the variable label
    end
    save(savepath,'label','feat');
    fprintf("Data saved successfully!\n");
end