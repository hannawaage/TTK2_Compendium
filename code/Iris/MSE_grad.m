function grad = MSE_grad(X,T,W, discriminant)
    grad = zeros(size(W));
    for k = 1:size(X,2)
        xk = X(:,k);
        tk = T(:,k);
        gk = discriminant(X(:,k),W);
        grad = grad + ((gk-tk) .* gk .* (1-gk))*xk.';        
    end
end

