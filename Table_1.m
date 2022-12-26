close all
clear all
clc

count = 0;
t = 100; %independent run
step = t; %waitbar
wait=waitbar(0,'beginning');

for i = 1:t
    %% load Synthetic Data training data and testing data
    %number of data in one class
    n = 50;

    %number of class
    r=3;

    %train data
    for j=1:r
        mu(j,:) = [0 30*(j-1)];
    end
    SIGMA = [30 30];
    traindata = 0;

    for j = 1:r
        train_temp = mvnrnd(mu(j,:),SIGMA,n);
        %noise free
        if j ~= 1
            traindata = [traindata; train_temp];
            trainlabel = [trainlabel; j*ones(size(train_temp,1),1)];
        else
            traindata = train_temp;
            trainlabel = j*ones(size(train_temp,1),1);
        end

    end

    train.patterns = traindata;
    train.targets = trainlabel;
    test.patterns = traindata;
    test.targets = trainlabel;
    
    %pin-SVOR tau=0.1
    algorithmObj = pinSVOR();
    clear param;
    param = struct('C',10,'tau',0.1,'kernel','linear');
    pinsvor = algorithmObj.fitpredict(train,test,param);
    wtau1(i,:) = pinsvor.model.w;
    btau1(i,:) = pinsvor.model.thresholds;

    
    %pin-SVOR tau=0.2
    algorithmObj = pinSVOR();
    clear param;
    param = struct('C',10,'tau',0.2,'kernel','linear');
    pinsvor = algorithmObj.fitpredict(train,test,param);
    wtau2(i,:) = pinsvor.model.w;
    btau2(i,:) = pinsvor.model.thresholds;

    
    %pin-SVOR tau=0.3
    algorithmObj = pinSVOR();
    clear param;
    param = struct('C',10,'tau',0.3,'kernel','linear');
    pinsvor = algorithmObj.fitpredict(train,test,param);
    wtau3(i,:) = pinsvor.model.w;
    btau3(i,:) = pinsvor.model.thresholds;

    
    %pin-SVOR tau=0.4
    algorithmObj = pinSVOR();
    clear param;
    param = struct('C',10,'tau',0.4,'kernel','linear');
    pinsvor = algorithmObj.fitpredict(train,test,param);
    wtau4(i,:) = pinsvor.model.w;
    btau4(i,:) = pinsvor.model.thresholds;

    %waitbar
    count = count +1;
    str=['Process:  ',num2str((count/step)*100),'%'];
    waitbar(count/step,wait,str);
end

close(wait);

s_wtau1 = wtau1(:,1)./wtau1(:,2);
s_wtau2 = wtau2(:,1)./wtau2(:,2);
s_wtau3 = wtau3(:,1)./wtau3(:,2);
s_wtau4 = wtau4(:,1)./wtau4(:,2);

m_wtau1 = mean(s_wtau1);
m_wtau2 = mean(s_wtau2);
m_wtau3 = mean(s_wtau3);
m_wtau4 = mean(s_wtau4);
   
s_wtau1 = std(s_wtau1);
s_wtau2 = std(s_wtau2);
s_wtau3 = std(s_wtau3);
s_wtau4 = std(s_wtau4);

s_btau1 = btau1./wtau1(:,2);
s_btau2 = btau2./wtau2(:,2);
s_btau3 = btau3./wtau3(:,2);
s_btau4 = btau4./wtau4(:,2);


m_btau1 = mean(s_btau1);
m_btau2 = mean(s_btau2);
m_btau3 = mean(s_btau3);
m_btau4 = mean(s_btau4);

s_btau1 = std(s_btau1);
s_btau2 = std(s_btau2);
s_btau3 = std(s_btau3);
s_btau4 = std(s_btau4);  

fprintf('\n %d classes of data in dataset', r);
fprintf('\n %d of data in dataset (m=%d)', n*r,n*r);
fprintf('\n %d of data per class in dataset', n);
fprintf('\n')
fprintf('\n')
fprintf('*** pinSVOR model training finished ***\n')
fprintf('tau = 0.1 , w = %.4f$pm$%.4f\n',...
    m_wtau1,s_wtau1)
fprintf('b1 = %.4f$pm$%.4f\n',...
    m_btau1(1),s_btau1(1))
fprintf('b2 = %.4f$pm$%.4f\n',...
    m_btau1(2),s_btau1(2))
fprintf('\n')

fprintf('tau = 0.2 , w = %.4f$pm$%.4f\n',...
    m_wtau2,s_wtau2)
fprintf('b1 = %.4f$pm$%.4f\n',...
    m_btau2(1),s_btau2(1))
fprintf('b2 = %.4f$pm$%.4f\n',...
    m_btau2(2),s_btau2(2))
fprintf('\n')

fprintf('tau = 0.3 , w = %.4f$pm$%.4f \n',...
    m_wtau3,s_wtau3)
fprintf('b1 = %.4f$pm$%.4f\n',...
    m_btau3(1),s_btau3(1))
fprintf('b2 = %.4f$pm$%.4f\n',...
    m_btau3(2),s_btau3(2))
fprintf('\n')

fprintf('tau = 0.4 , w = %.4f$pm$%.4f\n',...
    m_wtau4,s_wtau4)
fprintf('b1 = %.4f$pm$%.4f\n',...
    m_btau4(1),s_btau4(1))
fprintf('b2 = %.4f$pm$%.4f\n',...
    m_btau4(2),s_btau4(2))
fprintf('\n')
    
    
%end of code
%biu pia
load splat
sound(y,Fs)
    

