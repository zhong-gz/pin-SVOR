clc
clear all
close all
% profile on
%number of data in one class
n = 100;

%number of class
r=4;
color = linspecer(r);
    
% noise data
% train data
for j=1:r
    mu(j,:) = [40*(j-1) 0]; %center of each class
end
SIGMA = [20 20]; %variance of each class


% train data
for j = 1:r
    train_temp = mvnrnd(mu(j,:),SIGMA,n);

    if j ~= 1
        traindata = [traindata; train_temp];
        trainlabel = [trainlabel; j*ones(size(train_temp,1),1)];
    else
        traindata = train_temp;
        trainlabel = j*ones(size(train_temp,1),1);
    end

end

% test data
for j = 1:r
    test_temp = mvnrnd(mu(j,:),SIGMA,100);
    if j ~= 1
        testdata = [testdata; test_temp];
        testlabel = [testlabel; j*ones(size(test_temp,1),1)];
    else
        testdata = test_temp;
        testlabel = j*ones(size(test_temp,1),1);
    end
end

train.patterns = traindata;
train.targets = trainlabel;
test.patterns = testdata;
test.targets = testlabel;

%parameter
C = 10;
C2 = C;
kernel = 'linear';
k = 0.01;
tau = 0.1;

algorithmObj = pinSVOR();
param = struct('C',C,'tau',tau,'kernel',kernel,'k',k); 
    
model = algorithmObj.fitpredict(train,test,param);
MAE = MAE.calculateMetric(model.predictedTest,test.targets);
fprintf('MAE Test %f \n',MAE);

h1 = figure(1);
visulization(r,train.targets,train.patterns,color,model,algorithmObj);
set(gcf,'Position',[100 300 600 400]);

%% plot
function [h] = ...
    visulization(r,targets,patterns,color,info,algorithmObj)
    for i = 1:r
        h = scatter(patterns(targets==i,1),patterns(targets==i,2),...
            50,color(i,:),'fill');
        hold on;
    end

    d = 1;
    t=1;
    [x1Grid,x2Grid] = ...
    meshgrid(min(patterns(:,1))-20*t:d:max(patterns(:,1)+20*t),...
        min(patterns(:,2))-20*t:d:max(patterns(:,2))+20*t);
    xGrid = [x1Grid(:),x2Grid(:)];
    scores = algorithmObj.predict(xGrid);
    
    %decision plane
    contour(x1Grid,x2Grid,reshape(scores,size(x1Grid)),...
        info.model.thresholds,...
        'LineWidth', 6);
    hold on

    %positive plane
    contour(x1Grid,x2Grid,reshape(scores,size(x1Grid)),...
        info.model.thresholds+1,':',...
        'LineWidth', 3,'LineColor',[1 0.6 0.6]);
    hold on

    %negative plane
    contour(x1Grid,x2Grid,reshape(scores,size(x1Grid)),...
        info.model.thresholds-1,':',...
        'LineWidth', 3,'LineColor',[0.6 0.6 1]);
    hold off
end