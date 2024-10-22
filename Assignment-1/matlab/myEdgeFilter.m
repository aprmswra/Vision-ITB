function [img1] = myEdgeFilter(img0, sigma)
    % Smoothing (Gaussian filter)
    h = 2 * ceil(3 * sigma) + 1;
    gaus = fspecial('gaussian', [h h], sigma);
    im = myImageFilter(img0, gaus);

    % Compute gradients (Sobel filters)
    sobelX = fspecial('sobel');
    sobelY = sobelX';
    imgx = myImageFilter(im, sobelX);
    imgy = myImageFilter(im, sobelY);

    % Calculate gradient magnitude and direction
    G_mag = sqrt(imgx.^2 + imgy.^2);
    Grad_dir = atan2d(imgy, imgx);
    Grad_dir(Grad_dir < 0) = Grad_dir(Grad_dir < 0) + 180;

    % Map gradient direction to one of four orientations
    Grad_dir_bins = [0, 45, 90, 135];
    [~, Grad_dir_mapped] = min(abs(Grad_dir - reshape(Grad_dir_bins, [1, 1, numel(Grad_dir_bins)])), [], 3);
    Grad_dir = Grad_dir_bins(Grad_dir_mapped);

    % Non-maximum suppression
    new_img = zeros(size(img0));
    for i = 2:size(im,1) - 1
        for j = 2:size(im,2) - 1
            magnitude_current = G_mag(i,j);
            direc_current = Grad_dir(i,j);
            switch direc_current
                case 0
                    magnitude_n1 = G_mag(i, j-1);
                    magnitude_n2 = G_mag(i, j+1);
                case 45
                    magnitude_n1 = G_mag(i-1, j+1);
                    magnitude_n2 = G_mag(i+1, j-1);
                case 90
                    magnitude_n1 = G_mag(i-1, j);
                    magnitude_n2 = G_mag(i+1, j);
                case 135
                    magnitude_n1 = G_mag(i-1, j-1);
                    magnitude_n2 = G_mag(i+1, j+1);
            end

            % Set the pixel value to 0 if it's not the maximum in its neighborhood
            if magnitude_current ~= max([magnitude_current, magnitude_n1, magnitude_n2])
                new_img(i,j) = 0;
            else
                new_img(i,j) = magnitude_current;
            end
        end
    end
    img1 = new_img;
end