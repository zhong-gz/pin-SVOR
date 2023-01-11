clc
clear all
close all

dataset = 'newthyroid';
str = [dataset '\matlab'];
[train,test,data,label]=load_data(str);

[m,n] = size(data);
r = max(label);
k=5; %kfold
t=20; %independent run
ratio = 5;
step=k*t*ratio*2; %waitbar

kernel = 'linear'; %or 'rbf'

count=0;
wait=waitbar(0,'Beginning, the 1 st run for the 1 st fold pinSVM1VA');
for j=1:t %t independent run
    indices = crossvalind('Kfold',m,k);
    for i = 1:k      
        test_indic = (indices == i);
        train_indic = ~test_indic;
        train.patterns = data(train_indic,:);%train data and train label
        train.targets = label(train_indic,:);
        test.patterns = data(test_indic,:);%test data and test label
        test.targets = label(test_indic,:);
        traindata_k = train.patterns;

        for noise = 1:ratio
            p = 0.1*(noise-1);
            %add noise to train data
            for class = 1:r
                train_temp = traindata_k(train.targets==class,:);
                m_temp = size(train_temp,1);
                t_std = std(train_temp);
                index = randperm(m_temp);
                noise_number = ceil(p*m_temp);
                noisedata = 4*t_std.*(2*rand(noise_number,n)-1);
                train_temp_n = train_temp(index(1:noise_number),:)+noisedata;
                train_temp(index(1:noise_number),:) = [];
                train_temp_n = [train_temp;train_temp_n];

                if class ~= 1
                    traindata = [traindata; train_temp_n];
                    trainlabel = [trainlabel; class*ones(size(train_temp_n,1),1)];
                else
                    traindata = train_temp_n;
                    trainlabel = class*ones(size(train_temp_n,1),1);
                end
            end

            train.patterns = traindata;
            train.targets = trainlabel;

            %waitbar
            str=['Process:  ',num2str(fix((count/step)*100)),...
                '% , the ',num2str(j),' th run for the '...
                ,num2str(i),'th fold pinSVM, r = 0.',num2str(noise-1)];
            waitbar(count/step,wait,str);
    
            %pinsvm 1
            algorithmObj = pinSVM1VA();
            clear param;
            param = struct('C',10,'tau',0.2,'kernel',kernel,'k',0.1);
            model = algorithmObj.fitpredict(train,test,param);
            trainTime(noise,1,(j-1)*k+i) = model.trainTime;
            testTime(noise,1,(j-1)*k+i) = model.testTime;
            %waitbar
            count = count +1;
            str=['Process:  ',num2str(fix((count/step)*100)),...
                '% , the ',num2str(j),' th run for the '...
                ,num2str(i),'th fold pinSVOR, r = 0.',num2str(noise-1)];
            waitbar(count/step,wait,str);
    
            %pinsvor 2
            algorithmObj = pinSVOR();
            clear param;
            param = struct('C',10,'tau',0.2,'kernel',kernel,'k',0.1); %pinsvor pinsvm
            model = algorithmObj.fitpredict(train,test,param);
            trainTime(noise,2,(j-1)*k+i) = model.trainTime;
            testTime(noise,2,(j-1)*k+i) = model.testTime;
            %waitbar
            count = count +1;
        end
    end     
end

close(wait);

mean_time = mean(trainTime,3);
mean_test_time = mean(testTime,3);

fprintf('\n Dataset is %s', dataset);
fprintf('\n %d of data in dataset', m);
fprintf('\n %d of classed in dataset', m);
fprintf('\n')
fprintf('\n')

for noise = 1:ratio
    fprintf('\n 0.%d of data is courrpted with noise', noise-1);
    fprintf('\n')

    % Report train and test time
    fprintf('pinSVOR\n');
    fprintf('Average Train Time %f\n',mean_time(noise,2));
    fprintf('Average Test Time %f\n',mean_test_time(noise,2));
    fprintf('\n');
    
    fprintf('pinSVM\n');
    fprintf('Average Train Time %f\n',mean_time(noise,1));
    fprintf('Average Test Time %f\n',mean_test_time(noise,1));
    fprintf('\n');
end

%end of code
%biu pia
load splat
sound(y,Fs)

%% load data
function [train,test,sort_data,label]= load_data(path)
    oldFolder = cd("ordinal-regression dataset");
    cd(path);
    getfilename=ls('train*.*');
    filename = cellstr(getfilename);
    train_num = length(filename);
    train_stock(train_num) = struct('Name',filename(train_num),...
        'Data',textread(filename{train_num}));
    for ii=1:train_num-1
        train_stock(ii) = struct('Name',filename(ii),'Data',textread(filename{ii}));
    end

    getfilename=ls('test*.*');
    filename = cellstr(getfilename);
    test_num = length(filename);
    test_stock(test_num) = struct('Name',filename(test_num),...
        'Data',textread(filename{test_num}));
    for ii=1:test_num-1
        test_stock(ii) = struct('Name',filename(ii),'Data',textread(filename{ii}));
    end

    train.patterns = train_stock(1).Data(:,1:end-1);

    train.targets = train_stock(1).Data(:,end);

    test.patterns = test_stock(1).Data(:,1:end-1);

    test.targets = test_stock(1).Data(:,end);

    patterns = [train.patterns;test.patterns ];
    targets =[train.targets;test.targets];
    data=[patterns targets];
    r = max(targets);
    sort_data = data(data(:,end)==1,:);
    for i = 2:r
        data_temp = data(data(:,end)==i,:);
        sort_data = [sort_data;data_temp];
    end
    label = sort_data(:,end);
    sort_data = sort_data(:,1:end-1);

    cd (oldFolder);
end