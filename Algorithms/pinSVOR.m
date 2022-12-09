classdef pinSVOR < Algorithm
    %pinSVOR Support Vector for Ordinal Regression with Pinball Loss

    %   pinSVOR methods:
    %      fitpredict               - runs the corresponding algorithm,
    %                                   fitting the model and testing it in a dataset.
    %      fit                        - Fits a model from training data
    %      predict                    - Performs label prediction

    
    properties
        description = 'Support Vector for Ordinal Regression with Pinball Loss';        
        parameters = struct('C', 0.1, 'C2',10,'k', 0.1,'kernel','linear','tau',0.5);
    end
    properties (Access = private)
        algorithmMexPath = fullfile(fileparts(which('Algorithm.m')),'pinSVORnoLP');
    end
    
    methods
        function obj = pinSVOR(varargin)
            %pinSVOR constructs an object of the class pinSVOR and sets its default
            %   characteristics
            %   OBJ = pinSVOR builds pinSVOR
            obj.parseArgs(varargin);
        end
        
        function [projectedTrain, predictedTrain] = ...
                privfit(obj,train,parameters)
            %PRIVFIT trains the model for the pinSVOR method with TRAIN data and
            %vector of parameters PARAMETERS. 
%             if isempty(strfind(path,obj.algorithmMexPath))
%                 addpath(obj.algorithmMexPath);
%             end
            parameters.C2 = parameters.C;
            [beta, beta_star, delta, delta_star,lambda, b ,projectedTrain,projected] = ...
                obj.pinsvor(train.patterns, train.targets,parameters.C,parameters.tau,parameters.kernel,parameters.k,parameters.C2);
            predictedTrain = obj.assignLabels(projectedTrain, b);
            if strcmpi(parameters.kernel,'linear')
                model.w = ((1-parameters.tau)*(beta_star-beta)+parameters.tau*(delta-delta_star))'*train.patterns;
            end
            model.projected = projected;
            model.beta = beta;
            model.beta_star = beta_star;
            model.delta = delta;
            model.delta = beta_star;
            model.delta_star = delta_star;
            model.lambda=lambda;
            model.thresholds = b;
            model.parameters = parameters;
            model.train = train.patterns;
            obj.model = model;
            projectedTrain = projectedTrain';
            if ~isempty(strfind(path,obj.algorithmMexPath))
                rmpath(obj.algorithmMexPath);
            end
        end
        
        function [projected, predicted] = privpredict(obj, test)
            %PREDICT predicts labels of TEST patterns labels. The object needs to be fitted to the data first.
            kernelMatrix = computeKernelMatrix(obj.model.train',test',obj.model.parameters.kernel,obj.model.parameters.k);
            projected = obj.model.projected*kernelMatrix;
            
            predicted = obj.assignLabels(projected, obj.model.thresholds);
            projected = projected';
        end
    end
 
    methods (Static = true)
        function [beta, beta_star, delta, delta_star, lambda, b ,projectedTrain,projected] =...
                pinsvor(traindata,label,C,tau,kernel,s,C2)
            r = max(label);
            K = computeKernelMatrix(traindata',traindata',kernel,s);
            for j = 1:r
                n(j)= sum(label==j);
            end
            n_max = max(n);
            
            %% H
            H = [-(1-tau)*(1-tau)*K         (1-tau)*(1-tau)*K           tau*(1-tau)*K               -tau*(1-tau)*K                zeros(size(traindata,1),r);
                 (1-tau)*(1-tau)*K          -(1-tau)*(1-tau)*K          -tau*(1-tau)*K              tau*(1-tau)*K                 zeros(size(traindata,1),r);
                 tau*(1-tau)*K              -tau*(1-tau)*K              -tau*tau*K                  tau*tau*K                     zeros(size(traindata,1),r);
                 -tau*(1-tau)*K             tau*(1-tau)*K               tau*tau*K                   -tau*tau*K                    zeros(size(traindata,1),r);
                 zeros(r,size(traindata,1)) zeros(r,size(traindata,1))  zeros(r,size(traindata,1))  zeros(r,size(traindata,1))  zeros(r,r)];
            H = -1*H;

            %% f
            f = [(1-tau)*ones(size(traindata,1),1);
                 (1-tau)*ones(size(traindata,1),1);
                 zeros(size(traindata,1),1);
                 zeros(size(traindata,1),1);
                 zeros(r,1)];
            f = -1*f;

            %% Aeq
            %beta
            beta = zeros(r+1,size(traindata,1));
            all = 0;
            for j = 1:r-1
                beta(j,1+all:n(j)+all) = -(1-tau);
                all = all+n(j);
            end
            beta(r,size(K,1)-n(r)+1:size(K,1)) = 1-tau;

            %beta_star
            beta_star = zeros(r+1,size(traindata,1));
            all = n(1);
            for j = 1:r-1
                beta_star(j,1+all:n(j+1)+all) = 1-tau;
                all = all+n(j+1);
            end
            beta_star(r+1,1:n(1)) = 1-tau;
            
            %delta
            delta = zeros(r+1,size(traindata,1));
            all = 0;
            for j = 1:r-1
                delta(j,1+all:n(j)+all) = tau/2;
                all = all+n(j);
            end
            delta(r,size(K,1)-n(r)+1:size(K,1)) = -1*(tau/2);
            
            all = n(1);
            for j = 1:r-1
                delta(j,1+all:n(j+1)+all) = tau/2;
                all = all+n(j+1);
            end
            delta(r+1,1:n(1)) = tau/2;
            
            %delta_star
            delta_star = zeros(r+1,size(traindata,1));
            all = 0;
            for j = 1:r-1
                delta_star(j,1+all:n(j)+all) = -tau/2;
                all = all+n(j);
            end
            delta_star(r,size(K,1)-n(r)+1:size(K,1)) = (tau/2);
            
            all = n(1);
            for j = 1:r-1
                delta_star(j,1+all:n(j+1)+all) = -tau/2;
                all = all+n(j+1);
            end
            delta_star(r+1,1:n(1)) = -1*tau/2;
            
            %lamda
            lamda = zeros(r+1,r);
            for j = 1:r-1
                lamda(j,j) = -1;
            end
            lamda(r,r) = 1;
            
            for j = 1:r-1
                lamda(j,j+1) = 1;
            end
            lamda(r+1,1)=1;
            Aeq = [beta beta_star delta delta_star lamda];
            
            %% beq
            beq1 = zeros(r-1,1);
            beq2 = C2*ones(2,1);
            beq = [beq1;beq2];
            
            %% lb
            lb = zeros(4*size(traindata,1)+r,1);
            
            %% ub
            ub1 = C*ones(4*size(traindata,1),1);
            ub2 = inf(r,1);
            ub = [ub1 ; ub2];
            
            %% qp
            warning('off')
            threshold = 1e-8;
            opts = optimoptions('quadprog','display','off');
            x = quadprog(H,f,[],[],Aeq,beq,lb,ub,[],opts);
%             x(x<threshold)=0;
            
            %% alpha alpha_star lambda
            beta = x(1:size(traindata,1));
            beta_star = x(size(traindata,1)+1:2*(size(traindata,1)));
            delta = x(2*(size(traindata,1))+1:3*(size(traindata,1)));
            delta_star = x(3*(size(traindata,1))+1:4*(size(traindata,1)));
            lambda = x(4*(size(traindata,1))+1:4*(size(traindata,1))+r);
            alpha = tau*(C-beta-delta);
            alpha_star = tau*(C-beta_star-delta_star);
            
            %% projectedTrain
            projectedTrain = sum((1-tau)*(beta_star-beta)+tau*(delta-delta_star).*K);
            projected = ((1-tau)*(beta_star-beta)+tau*(delta-delta_star))';

            %% b
            all = 0;
            for j = 1:r-1
                clear SvIndices
                clear SvIndices_star
                beta_temp = beta(1+all:n(j)+all)+all;
                beta_startemp = beta_star(n(j)+all+1:n(j)+n(j+1)+all)+n(j)+all;

                SvIndices= find( beta_temp > threshold...
                    & beta_temp < C -threshold )';
                SvIndices_star= find( beta_startemp > threshold...
                    & beta_startemp < C -threshold)';

                if ~isempty(SvIndices) && ~isempty(SvIndices_star)
                    bj_temp = mean(((1-tau)*(beta_star-beta)+tau*(delta-delta_star))'*K(:,SvIndices))+1;
                    bj_star_temp = mean(((1-tau)*(beta_star-beta)+tau*(delta-delta_star))'*K(:,SvIndices_star))-1;
                    b_temp(j) = 0.5*(bj_temp+bj_star_temp);
                end
                
                if ~isempty(SvIndices) && isempty(SvIndices_star)
                    bj_temp = mean(((1-tau)*(beta_star-beta)+tau*(delta-delta_star))'*K(:,SvIndices))+1;
                    b_temp(j) = bj_temp;
                end
                
                if isempty(SvIndices) && ~isempty(SvIndices_star)
                    bj_star_temp = mean(((1-tau)*(beta_star-beta)+tau*(delta-delta_star))'*K(:,SvIndices_star))-1;
                    b_temp(j) = bj_star_temp;
                end

                if isempty(SvIndices) && isempty(SvIndices_star)
                    bj_temp = mean(((1-tau)*(beta_star-beta)+tau*(delta-delta_star))'*K(:,1+all:n(j)+all))+1;
                    bj_star_temp = mean(((1-tau)*(beta_star-beta)+tau*(delta-delta_star))'*K(:,n(j)+all+1:n(j)+n(j+1)+all))-1;
                    b_temp(j) = 0.5*(bj_temp+bj_star_temp);
                end
                all = all+n(j);
            end

            beta_temp = beta(sum(n)-n(r)+1:end);
            beta_startemp = beta_star(1:n(1));
            SvIndices= find( beta_temp > threshold...
                    & beta_temp < C -threshold )'+sum(n)-n(r);
            SvIndices_star= find( beta_startemp > threshold...
                & beta_startemp < C -threshold)';

            if isempty(SvIndices_star)
                bj_star_temp = min(((1-tau)*(beta_star-beta)+tau*(delta-delta_star))'*K(:,1:n(1)))-1;
                b(1) = bj_star_temp;
            else
                bj_star_temp = mean(((1-tau)*(beta_star-beta)+tau*(delta-delta_star))'*K(:,SvIndices_star))-1;
                b(1) = bj_star_temp;
            end

            if isempty(SvIndices)
                bj_temp = max(((1-tau)*(beta_star-beta)+tau*(delta-delta_star))'*K(:,sum(n)-n(r):end))+1;
                b(r+1) = bj_temp;
            else
                bj_temp = mean(((1-tau)*(beta_star-beta)+tau*(delta-delta_star))'*K(:,SvIndices))+1;
                b(r+1) = bj_temp;
            end

            b(2:r) = b_temp(1:end);
        end
        
        function predicted = assignLabels(projected, thresholds)

            numClasses = size(thresholds,2)-1;
            project2 = repmat(projected, numClasses,1);
            b_j = thresholds(2:end);
            b_j_1 = thresholds(1:end-1);
            project2 = abs(project2 - ...
                ((b_j+b_j_1)/2)'*ones(1,size(project2,2)));
            wx=project2;

            [~,predicted]=min(wx,[],1);

%             b = thresholds(2:end-1);
%             numClasses = size(b,2)+1;
%             %TEST assign the labels from projections and thresholds
%             project2 = repmat(projected, numClasses-1,1);
%             project2 = project2 - b'*ones(1,size(project2,2));
%             % Asignation of the class
%             % f(x) = max {Wx-bk<0} or Wx - b_(K-1) > 0
%             wx=project2;
%             % The procedure for that is the following:
%             % We assign the values > 0 to NaN
%             wx(wx(:,:)>0)=NaN;
% 
%             % Then, we choose the biggest one.
%             [maximum,predicted]=max(wx,[],1);
% 
%             % If a max is equal to NaN is because Wx-bk for all k is >0, so this
%             % pattern belongs to the last class.
%             predicted(isnan(maximum(:,:)))=numClasses;
             
            predicted = predicted';
        end
    end 
end

