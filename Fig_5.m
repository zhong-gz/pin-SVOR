clc
clear all
close all

%number of data in one class
n = 100;

%number of class
r=3;
color = linspecer(r);
%train data
for i=1:r
    mu(i,:) = [30*(i-1) 0];
end
SIGMA = [5 5];
traindata = 0;

%ratio of noise
p = 0.2;

addpath('Algorithms\SVOREX\');
run("Algorithms\SVOREX\make.m");

for i = 1:r
    train_temp = mvnrnd(mu(i,:),SIGMA,n);
    
    %noise free
    if i ~= 1
        traindata_n = [traindata_n; train_temp];
        trainlabel_n = [trainlabel; i*ones(size(train_temp,1),1)];
    else
        traindata_n = train_temp;
        trainlabel_n = i*ones(size(train_temp,1),1);
    end
    
    if i ==ceil(r/2) 
        d = size(train_temp,2); %dimension
        t_std = std(train_temp);
        index = randperm(n);
        noise = 5*t_std(1).*rand(p*n,1);
        noise = [noise ones(size(noise,1),1)*2];
        train_temp_n = train_temp(index(1:p*n),:)+noise;
        train_temp(index(1:p*n),:) = [];

        train_temp_n = [train_temp;train_temp_n];
    else
        train_temp_n = train_temp;
    end

    if i ~= 1
        traindata = [traindata; train_temp_n];
        trainlabel = [trainlabel; i*ones(size(train_temp_n,1),1)];
    else
        traindata = train_temp_n;
        trainlabel = i*ones(size(train_temp_n,1),1);
    end
end

%test data
for i = 1:r
    test_temp = mvnrnd(mu(i,:),SIGMA,n*1.5);
    if i ~= 1
        testdata = [testdata; test_temp];
        testlabel = [testlabel; i*ones(size(test_temp,1),1)];
    else
        testdata = test_temp;
        testlabel = i*ones(size(test_temp,1),1);
    end
end

train.patterns = traindata;
train.targets = trainlabel;
test.patterns = testdata;
test.targets = testlabel;

% noise free
train_n.patterns = traindata_n;
train_n.targets = trainlabel_n;

% noise free fig a
algorithmObj = SVORLin();
clear param;
param = struct('C',10);
Svor = algorithmObj.fitpredict(train_n,test,param);
h1 = figure(1);
visulization(r,train_n.targets,train_n.patterns,color,Svor,algorithmObj);
set(gcf,'Position',[100 300 600 400]);

% svor-exc fig b
algorithmObj = SVORLin();
clear param;
param = struct('C',10);
pinSvor = algorithmObj.fitpredict(train,test,param);
h2 = figure(2);
visulization(r,train.targets,train.patterns,color,pinSvor,algorithmObj);
set(gcf,'Position',[500 300 600 400]);

% tau=0.2 fig c
algorithmObj = pinSVOR();
clear param;
param = struct('C',10,'tau',0.2,'kernel','linear');
pinSvor = algorithmObj.fitpredict(train,test,param);
h3 = figure(3);
visulization(r,train.targets,train.patterns,color,pinSvor,algorithmObj);
set(gcf,'Position',[900 300 600 400]);

% tau=0.5 fig d
algorithmObj = pinSVOR();
clear param;
param = struct('C',10,'tau',0.5,'kernel','linear');
pinSvor = algorithmObj.fitpredict(train,test,param);
h4 = figure(4);
visulization(r,train.targets,train.patterns,color,pinSvor,algorithmObj);
set(gcf,'Position',[1300 300 600 400]);

%% plot
function [h] = visulization(r,targets,patterns,color,info,algorithmObj)

    for i = 1:r
        h = scatter(patterns(targets==i,1),patterns(targets==i,2),...
            30,color(i,:),'fill');
        hold on;
    end

    d = 0.5;
    [x1Grid,x2Grid] = meshgrid(min(patterns(:,1))-20:d:max(patterns(:,1)+20),...
        min(patterns(:,2))-2:d:max(patterns(:,2))+2);
    xGrid = [x1Grid(:),x2Grid(:)];
    scores = algorithmObj.predict(xGrid);
    %decision plane
    contour(x1Grid,x2Grid,reshape(scores,size(x1Grid)),...
        info.model.thresholds,'LineWidth', 4);
    hold on

    %positive plane
    for i = 1:size(info.model.thresholds,2)
        contour(x1Grid,x2Grid,reshape(scores,size(x1Grid)),...
            [info.model.thresholds(i)+1 info.model.thresholds(i)+1],':',...
            'LineWidth', 3,'LineColor',[1 0.4 0.4]);
        hold on
    end
    %negative plane
    for i = 1:size(info.model.thresholds,2)
        contour(x1Grid,x2Grid,reshape(scores,size(x1Grid)),...
            [info.model.thresholds(i)-1 info.model.thresholds(i)-1],':',...
            'LineWidth', 3,'LineColor',[0.4 0.4 1]);
        hold on
    end

    fontsize = 23;
    y= max(patterns(:,2)+1.5);
%     text(-20,y,'$b_{0}-1$','Interpreter','latex','FontSize',fontsize);
%     text(-10,y,'$b_{0}$','Interpreter','latex','FontSize',fontsize);
%     text(0,y,'$b_0+1$','Interpreter','latex','FontSize',fontsize);
%     text(5,y,'$b_1-1$','Interpreter','latex','FontSize',fontsize);
%     text(15,y,'$b_1$','Interpreter','latex','FontSize',fontsize);
%     text(20,y,'$b_1+1$','Interpreter','latex','FontSize',fontsize);
%     text(35,y,'$b_2-1$','Interpreter','latex','FontSize',fontsize);
%     text(40,y,'$b_2$','Interpreter','latex','FontSize',fontsize);
%     text(45,y,'$b_2+1$','Interpreter','latex','FontSize',fontsize);
%     text(65,y,'$b_3-1$','Interpreter','latex','FontSize',fontsize);
%     text(70,y,'$b_3$','Interpreter','latex','FontSize',fontsize);
%     text(75,y,'$b_3+1$','Interpreter','latex','FontSize',fontsize);
    
%     text(-5,max(patterns(:,2)+3),'Class 1','FontSize',fontsize);
%     text(30-5,max(patterns(:,2)+3),'Class 2','FontSize',fontsize);
%     text(60-5,max(patterns(:,2)+3),'Class 3','FontSize',fontsize);
    
    axis([-inf inf,min(patterns(:,2))-1 max(patterns(:,2)+1)]); 
%     set(gca,'FontSize',fontsize,'Fontname', 'Times New Roman');
%     set(gca,'Visible','off');
    box on
    set(gca, 'linewidth', 1.7) 
    hold off
end