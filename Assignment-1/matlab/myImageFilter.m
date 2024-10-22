function [img1] = myImageFilter(img0, h)
    % Determine padding size
    pad = (size(h,1)-1)/2;
    filt = size(h,1);
    
    % Pad the image
    pad_img = padarray(img0, [pad pad], 'replicate', 'both'); 
    
    % Preallocate the output image
    img1 = zeros(size(img0));
    
    % Vectorized convolution operation
    for i = 1:size(img0,1)
        for j = 1:size(img0,2)
            img1(i,j) = sum(sum(pad_img(i:i+filt-1, j:j+filt-1) .* h));
        end
    end
end