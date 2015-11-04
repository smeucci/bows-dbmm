function dmatk = kNearestNeighbours(dmat, K)
% Return matrix of the first k-nearest neighbours for each row 
% of a distance matrix
    
    % Sort distance matrix by each row
    dmat_sort = sort(dmat, 2);
    
    % Take k-th value of the sorted dmat, meaning the distance with the
    % k-th nearest codeword.
    dk = dmat_sort(:,K);

    % Threshold
    res = double(bsxfun(@lt, dmat, dk));
    res(find(res==0)) = Inf;
    
    dmatk = res.*dmat;
    
end

