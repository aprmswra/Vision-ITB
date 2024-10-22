function [rhos, thetas] = myHoughLines(H, nLines)
    % Apply NMS
    neighborhoodSize = [15, 15]; 
    H_max = imdilate(H, strel('rectangle', neighborhoodSize));
    H_suppressed = H .* (H == H_max);
    
    % Find the coordinates of the nLines highest peaks
    [~, sortedind] = sort(H_suppressed(:), 'descend');
    [rhos, thetas] = ind2sub(size(H), sortedind(1:nLines));
end