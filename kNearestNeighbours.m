function dmatk = kNearestNeighbours(dmat, K)
% Return matrix of the first k-nearest neighbours for each row 
% of a distance matrix
    
    % Sort distance matrix by each row
    dmatk = sort(dmat, 2);
    
    if K < (size(dmatk, 2) - 1)
        % Save only K-nearest neighbours, set others to 0
        dmatk(:, K+1:end) = 0;
    end

end

