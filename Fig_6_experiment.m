close all
clear all
clc

count = 0;
t = 20;
% ratio of noise
ratio = 5;
step = t*2*ratio; % waitbar
wait=waitbar(0,'Beginning, the 1 st run for pinSVOR');

% number of class
r=3;

% number of data in one class
m = 300;
n = m/r;

for i = 1:t
    % load Synthetic Data training data and testing data
    %train data
    SIGMA = [30 30];
    for j=1:r
        mu(j,:) = [0 30*(j-1)];
        train_temp = mvnrnd(mu(j,:),SIGMA,n);
        if j ~= 1
            traindata = [traindata; train_temp];
            trainlabel = [trainlabel; j*ones(size(train_temp,1),1)];
        else
            traindata = train_temp;
            trainlabel = j*ones(size(train_temp,1),1);
        end
    end
    
    %test data
    for j = 1:r
        test_temp = mvnrnd(mu(j,:),SIGMA,200);
        if j ~= 1
            testdata = [testdata; test_temp];
            testlabel = [testlabel; j*ones(size(test_temp,1),1)];
        else
            testdata = test_temp;
            testlabel = j*ones(size(test_temp,1),1);
        end
    end
    
    for k = 1:ratio
        % ratio of noise
        p = 0.1*(k-1);
        
        % add noise
        class = 2;
        train_temp = traindata(trainlabel==class,:);
        d = size(train_temp,2); %dimension
        t_std = std(train_temp);
        index = randperm(n);
        num = fix(p*n);
        noise = 2*t_std.*rand(num,d);
        train_temp_n = train_temp(index(1:num),:)+noise;
        train_temp(index(1:num),:) = [];
        train_temp_n = [train_temp;train_temp_n];
        traindata(trainlabel==class,:) = train_temp_n;

        train.patterns = traindata;
        train.targets = trainlabel;
        test.patterns = testdata;
        test.targets = testlabel;

    
        str=['Process:  ',num2str(fix((count/step)*100)),'% , the '...
            ,num2str(i),' th run for pinSVOR, r = 0.',num2str(k-1)];
        waitbar(count/step,wait,str);
        %pinsvor
        bestpinSvor = pinsvor_ParamOptimization(train,test);
        mae(k,1,i) = MAE.calculateMetric(bestpinSvor.predictedTest,test.targets);
        mze(k,1,i) = MZE.calculateMetric(bestpinSvor.predictedTest,test.targets);
        %waitbar
        count = count +1;
        str=['Process:  ',num2str(fix((count/step)*100)),'% , the '...
            ,num2str(i),' th run for SVOR-EXC, r = 0.',num2str(k-1)];
        waitbar(count/step,wait,str);

        %svorexc 2
        bestSVOREX = svorex_ParamOptimization(train,test);
        mae(k,2,i) = MAE.calculateMetric(bestSVOREX.predictedTest,test.targets);
        mze(k,2,i) = MZE.calculateMetric(bestSVOREX.predictedTest,test.targets);
        %waitbar
        count = count +1;
        str=['Process:  ',num2str(fix((count/step)*100)),'% , the '...
            ,num2str(i),' th run for SVOR-IMC, r = 0.',num2str(k-1)];
        waitbar(count/step,wait,str);
    end
end
close(wait);

mean_mae = mean(mae,3);
mean_mze = mean(mze,3);


fprintf('\n %d classes of data in dataset', r);
fprintf('\n %d of data in dataset (m=%d)', n*r,n*r);
fprintf('\n')
for k = 1:ratio
    fprintf('\n')
    fprintf('\n 0.%d of data is courrpted with noise', k-1);
    fprintf('\n %d of data per class in dataset', n);
    fprintf('\n')
    fprintf('\n')

    % Report mae mze
    fprintf('Performance for pinSVOR\n');
    fprintf('Average MAE Test %f\n',mean_mae(k,1));
    fprintf('Average MZE Test %f \n',mean_mze(k,1));
    fprintf('\n');

    fprintf('Performance for SVOR-EXC\n');
    fprintf('Average MAE Test %f\n',mean_mae(k,2));
    fprintf('Average MZE Test %f \n',mean_mze(k,2));
    fprintf('\n');
end

%end of code
%biu pia
load splat
sound(y,Fs)

%% SVOREX
% Create an Algorithm object
function [bestSVOREX]= svorex_ParamOptimization(train,test)
    algorithmObj = SVORLin();
    % Clear parameter struct
    clear param;
    bestMAE=100;
    for C=10.^(-3:1:3)
       param = struct('C',C);
       SVOREX = algorithmObj.fitpredict(train,test,param);
       mae = MAE.calculateMetric(test.targets,SVOREX.predictedTest);
       if mae <= bestMAE
           bestMAE = mae;
           bestParam = param;
       end

    end

    % Fit the model and predict with test data
    bestSVOREX = algorithmObj.fitpredict(train,test,bestParam);
end

%% pinSVOR
function [bestpinSvor]=pinsvor_ParamOptimization(train,test)
    warning('off');
    % Create an Algorithm object
    algorithmObj = pinSVOR();

    % Clear parameter struct
    clear param;
    bestMAE=100;
    for C=10.^(-3:1:3)
        for tau= 0.1:0.1:0.9
            param = struct('C',C,'C2',C,'k',1,'tau',tau,'kernel','linear');
            try
               pinSvor = algorithmObj.fitpredict(train,test,param);
               mae = MAE.calculateMetric(test.targets,pinSvor.predictedTest);
               if mae <= bestMAE
                   bestMAE = mae;
                   bestParam = param;
               end
            end
        end
    end

    % Fit the model and predict with test data
    bestpinSvor = algorithmObj.fitpredict(train,test,bestParam);
    warning('on');
end
