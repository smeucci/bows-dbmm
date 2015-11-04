function k = gaussianKernel(x, sigma)
    %Compute gaussian kernel with sigma
    k = 1/(sqrt(2*pi) * sigma) * exp(-0.5 * (x.^2)/sigma^2);
end

