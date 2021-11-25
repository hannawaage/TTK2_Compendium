function [W,n] = gradient_descent(gradient, W0, alpha, maxiter)
    W = W0;
    iterate = true;
    n = 1;
    while iterate && n < maxiter
       grad = gradient(W);
       W = W - alpha * grad;
       if mod(n,1000) == 0
           display(n)
       end
       n = n+1;
       iterate = norm(grad) > 1e-4;
    end
end

