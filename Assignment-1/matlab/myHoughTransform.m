function [H, rhoScale, thetaScale] = myHoughTransform(Im, threshold, rhoRes, thetaRes)
    Im(Im < threshold) = 0;
    [y,x] = find(Im);
    
    theta = (0:thetaRes:2*pi-thetaRes);
    
    % Preallocate memory for rho, improve algo efficiency
    rho = zeros(length(x), length(theta));
    
    for i = 1:size(x,1)
        rho(i,:) = x(i)*cos(theta)+y(i)*sin(theta);
    end
    rho = round(rho);
    
    % negative invalid
    rho(rho<0) = 0;
    
    rho_maximum = sqrt(size(Im,1).^2+size(Im,2).^2);
    h = zeros(ceil(rho_maximum/rhoRes),size(theta,2));
    
    for i = 1:size(rho,2)
        rho_out = rho(:,i);
        rho_out = round(rho_out/rhoRes);
        for j = 1:size(rho_out,1)
            idx = rho_out(j,1);
            if idx ~= 0
                h(idx,i) = h(idx,i)+1;
            end
        end
    end
    H = h;
    thetaScale = theta;
    rhoScale = (rhoRes:rhoRes:rho_maximum);

end