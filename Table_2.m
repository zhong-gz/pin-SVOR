close all
clear all
clc

count = 0;
t = 20; %independent run
ratio = 4;
step = t*5*ratio; %waitbar
wait=waitbar(0,'beginning , the 1 st run for pinSVOR (tau = 0.1)');

%number of class
r=3;

%number of data
m = 1500; %total number of data
n = m/r; %number of data in one class

% parameter
c = 10;
c2 = c;

for i = 1:t
    %% load Synthetic Data training data and testing data
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
    
    for k = 1:ratio
        % ratio of noise
        p = 0.1*k;
        
        % add noise
        class = 2;
        train_temp = traindata(trainlabel==class,:);
        d = size(train_temp,2); %dimension
        t_std = std(train_temp);
        index = randperm(n);
        num = fix(p*n);
        noise = 2*t_std.*rand(num,d);
        train_temp_n = train_temp(index(1:p*n),:)+noise;
        train_temp(index(1:p*n),:) = [];
        train_temp_n = [train_temp;train_temp_n];
        traindata(trainlabel==class,:) = train_temp_n;

        train.patterns = traindata;
        train.targets = trainlabel;
        test.patterns = traindata;
        test.targets = trainlabel;

        str=['Process:  ',num2str(fix((count/step)*100)),'% , the '...
            ,num2str(i),' th run for pin-SVOR (tau = 0.1), r = 0.',num2str(k)];
        waitbar(count/step,wait,str);

        %pin tau=0.1
        % Create an Algorithm object
        algorithmObj = pinSVOR();
        clear param;
        param = struct('C',c,'C2',c2,'tau',0.1,'kernel','linear');
        pinsvor = algorithmObj.fitpredict(train,test,param);
        wtau1(k,i,:) = pinsvor.model.w;
        btau1(k,i,:) = pinsvor.model.thresholds;
        %waitbar
        count = count +1;
        str=['Process:  ',num2str(fix((count/step)*100)),'% , the '...
            ,num2str(i),' th run for pin-SVOR (tau = 0.2), r = 0.',num2str(k)];
        waitbar(count/step,wait,str);

        %pin tau=0.2
        % Create an Algorithm object
        algorithmObj = pinSVOR();
        clear param;
        param = struct('C',c,'C2',c2,'tau',0.2,'kernel','linear');
        pinsvor = algorithmObj.fitpredict(train,test,param);
        wtau2(k,i,:) = pinsvor.model.w;
        btau2(k,i,:) = pinsvor.model.thresholds;
        %waitbar
        count = count +1;
        str=['Process:  ',num2str(fix((count/step)*100)),'% , the '...
            ,num2str(i),' th run for pin-SVOR (tau = 0.3), r = 0.',num2str(k)];
        waitbar(count/step,wait,str);

        %pin tau=0.3
        % Create an Algorithm object
        algorithmObj = pinSVOR();
        clear param;
        param = struct('C',c,'C2',c2,'tau',0.3,'kernel','linear');
        pinsvor = algorithmObj.fitpredict(train,test,param);
        wtau3(k,i,:) = pinsvor.model.w;
        btau3(k,i,:) = pinsvor.model.thresholds;
        %waitbar
        count = count +1;
        str=['Process:  ',num2str(fix((count/step)*100)),'% , the '...
            ,num2str(i),' th run for pin-SVOR (tau = 0.4), r = 0.',num2str(k)];
        waitbar(count/step,wait,str);

        %pin tau=0.4
        % Create an Algorithm object
        algorithmObj = pinSVOR();
        clear param;
        param = struct('C',c,'C2',c2,'tau',0.4,'kernel','linear');
        pinsvor = algorithmObj.fitpredict(train,test,param);
        wtau4(k,i,:) = pinsvor.model.w;
        btau4(k,i,:) = pinsvor.model.thresholds;

        %waitbar
        count = count +1;
        str=['Process:  ',num2str(fix((count/step)*100)),'% , the '...
            ,num2str(i),' th run for SVOR-EXC, r = 0.',num2str(k)];
        waitbar(count/step,wait,str);
        %svorexc
        algorithmObj = SVORLin();
        clear param;
        param = struct('C',c);
        svorexc = algorithmObj.fitpredict(train,test,param);
        wexc(k,i,:) = svorexc.model.w;
        bexc(k,i,:) = svorexc.model.thresholds;
    end
end
    close(wait);

 for k = 1:4
    s_wtau1 = wtau1(k,:,1)./wtau1(k,:,2);
    s_wtau2 = wtau2(k,:,1)./wtau2(k,:,2);
    s_wtau3 = wtau3(k,:,1)./wtau3(k,:,2);
    s_wtau4 = wtau4(k,:,1)./wtau4(k,:,2);
    
    m_wtau1 = mean(s_wtau1);
    m_wtau2 = mean(s_wtau2);
    m_wtau3 = mean(s_wtau3);
    m_wtau4 = mean(s_wtau4);
       
    s_wtau1 = std(s_wtau1);
    s_wtau2 = std(s_wtau2);
    s_wtau3 = std(s_wtau3);
    s_wtau4 = std(s_wtau4);
    
    s_btau1 = btau1./wtau1(k,:,2);
    s_btau2 = btau2./wtau2(k,:,2);
    s_btau3 = btau3./wtau3(k,:,2);
    s_btau4 = btau4./wtau4(k,:,2);
    
    m_btau1 = mean(s_btau1);
    m_btau2 = mean(s_btau2);
    m_btau3 = mean(s_btau3);
    m_btau4 = mean(s_btau4);
    
    s_btau1 = std(s_btau1);
    s_btau2 = std(s_btau2);
    s_btau3 = std(s_btau3);
    s_btau4 = std(s_btau4);

    s_wexc = wexc(k,:,1)./wexc(k,:,2);
    m_wexc = mean(s_wexc);
    s_wexc = std(s_wexc);
    s_bexc = bexc./wexc(k,:,2);
    m_bexc = mean(s_bexc);
    s_bexc = std(s_bexc);
    
    fprintf('\n %d classes of data in dataset', r);
    fprintf('\n %d of data in dataset (m=%d)', n*r,n*r);
    fprintf('\n 0.%d of data is courrpted with noise', k);
    fprintf('\n %d of data per class in dataset', n);
    fprintf('\n')
    fprintf('\n')
    fprintf('*** pinSVOR model training finished ***\n')
    fprintf('tau = 0.1 , \nw = %.4f$pm$%.4f\n',...
        m_wtau1,s_wtau1)
    fprintf('b1 = %.4f$pm$%.4f\n',...
        m_btau1(1),s_btau1(1))
    fprintf('b2 = %.4f$pm$%.4f\n',...
        m_btau1(2),s_btau1(2))
    fprintf('\n')
    
    fprintf('tau = 0.2 , \nw = %.4f$pm$%.4f\n',...
        m_wtau2,s_wtau2)
    fprintf('b1 = %.4f$pm$%.4f\n',...
        m_btau2(1),s_btau2(1))
    fprintf('b2 = %.4f$pm$%.4f\n',...
        m_btau2(2),s_btau2(2))
    fprintf('\n')
    
    fprintf('tau = 0.3 , \nw = %.4f$pm$%.4f \n',...
        m_wtau3,s_wtau3)
    fprintf('b1 = %.4f$pm$%.4f\n',...
        m_btau3(1),s_btau3(1))
    fprintf('b2 = %.4f$pm$%.4f\n',...
        m_btau3(2),s_btau3(2))
    fprintf('\n')
    
    fprintf('tau = 0.4 , \nw = %.4f$pm$%.4f\n',...
        m_wtau4,s_wtau4)
    fprintf('b1 = %.4f$pm$%.4f\n',...
        m_btau4(1),s_btau4(1))
    fprintf('b2 = %.4f$pm$%.4f\n',...
        m_btau4(2),s_btau4(2))
    fprintf('\n')

    fprintf('*** SVOR-EXC model training finished ***\n')
    fprintf('w = %.4f$pm$%.4f\n',...
        m_wexc,s_wexc)
    fprintf('b1 = %.4f$pm$%.4f\n',...
        m_bexc(1),s_bexc(1))
    fprintf('b2 = %.4f$pm$%.4f\n',...
        m_bexc(2),s_bexc(2))
    fprintf('\n')
    
 end
    
    %end of code
    %biu pia
    load splat
    sound(y,Fs)