function s = crossvalidateSigma(train, test, VC)

    for sigma = 50:20:500
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
        
        bof_train=double(cat(1,train.bof));
        bof_test=double(cat(1,test.bof));
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
            if sigma == 50
                C_vals=log2space(3,10,5);
                for i=1:length(C_vals);
                    opt_string=['-t 4  -v 5 -c ' num2str(C_vals(i))];
                    xval_acc(i)=svmtrain(labels_train,[(1:size(Ktrain,1))' Ktrain],opt_string);
                end
                [v,ind]=max(xval_acc);
            end

            % train the model and test
            model=svmtrain(labels_train,[(1:size(Ktrain,1))' Ktrain],['-t 4 -c ' num2str(C_vals(ind))] );
            % we supply the missing scalar product (actually the values of non-support vectors could be left as zeros.... consider this if the kernel is computationally inefficient.
            fprintf('Sigma: %i\n', sigma);
            [precomp_ik_svm_lab,conf]=svmpredict(labels_test,[(1:size(Ktest,1))' Ktest],model);


        
    end




end

