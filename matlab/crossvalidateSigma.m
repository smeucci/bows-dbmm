function bestSigma = crossvalidateSigma(train, test, VC, minValue, maxValue, step)

    bestSigma = 0;
    bestAccuracy = 0;
    
    fprintf('\nSoft assignment sigma crossvalidation\n');
    
    for sigma = minValue:step:maxValue
        for i=1:length(train)  
          dmat = eucliddist(train(i).sift, VC);
          gaussian_kernel = gaussianKernel(dmat, sigma);
          unc = bsxfun(@rdivide, gaussian_kernel, sum(gaussian_kernel, 2));
          h = sum(unc, 1)/size(gaussian_kernel, 1);
          h = h./norm(h, 1);
          train(i).bof = h;
          clear h;
        end

        for i=1:length(test)    
          dmat = eucliddist(test(i).sift, VC);
          gaussian_kernel = gaussianKernel(dmat, sigma);
          unc = bsxfun(@rdivide, gaussian_kernel, sum(gaussian_kernel, 2));
          h = sum(unc, 1)/size(gaussian_kernel, 1);
          h = h./norm(h, 1);
          test(i).bof = h;
          clear h;
        end
        
        bof_train=(cat(1,train.bof));
        bof_test=(cat(1,test.bof));
        labels_train=cat(1,train.class);
        labels_test=cat(1,test.class);
        
        %%Classification
            Ktrain=zeros(size(bof_train,1),size(bof_train,1));
            for i=1:size(bof_train,1)
                for j=1:size(bof_train,1)

                    Ktrain(i,j) = sum(min(bof_train(i,:), bof_train(j,:)));
                end
            end

            Ktest=zeros(size(bof_test,1),size(bof_train,1));
            for i=1:size(bof_test,1)
                for j=1:size(bof_train,1)

                    Ktest(i,j) = sum(min(bof_test(i,:), bof_train(j,:)));
                end
            end

            % cross-validation only the first time
            if sigma == minValue
                C_vals=log2space(3,10,5);
                for i=1:length(C_vals);
                    opt_string=['-t 4  -v 5 -c ' num2str(C_vals(i))];
                    xval_acc(i)=svmtrain(labels_train,[(1:size(Ktrain,1))' Ktrain],opt_string);
                end
                [v,ind]=max(xval_acc);
            end

            model=svmtrain(labels_train,[(1:size(Ktrain,1))' Ktrain],['-t 4 -c ' num2str(C_vals(ind))] );
            fprintf('Sigma: %i\n', sigma);
            [~, acc, ~]=svmpredict(labels_test,[(1:size(Ktest,1))' Ktest],model);
            
            if acc(1) > bestAccuracy
               bestAccuracy = acc(1);
               bestSigma = sigma;
            end
    end

end
