%directory
datadir     = "./mydata";    
resultsdir  = "./myresults";

%parameters
sigma     = 2;
threshold = 0.03;
rhoRes    = 2;
thetaRes  = pi/90;
nLines    = 50;

imgdata_jpeg = dir(sprintf('%s/*.jpeg', datadir));
imgdata_jpg = dir(sprintf('%s/*.jpg', datadir));
imgdata = [imgdata_jpeg; imgdata_jpg];

for i = 1:numel(imgdata)
    %read and adjust images%
    [path, imgname, dummy] = fileparts(imgdata(i).name);
    img = imread(sprintf('%s/%s', datadir, imgdata(i).name));
    
    if (ndims(img) == 3)
        img = rgb2gray(img);
    end
    
    img = double(img) / 255;
    
    % step-by-step houghlines detection function calls%  
    [Im] = myEdgeFilter(img, sigma);   
    [H,rhoScale,thetaScale] = myHoughTransform(Im, threshold, rhoRes, thetaRes);
    [rhos, thetas] = myHoughLines(H, nLines);
    lines = houghlines(Im>threshold, 180*(thetaScale/pi), rhoScale, [rhos, thetas],'FillGap',5,'MinLength',10);
    
    %output
    fname = sprintf('%s/%s_01_edge.png', resultsdir, imgname);
    imwrite(sqrt(Im/max(Im(:))), fname);
    fname = sprintf('%s/%s_02_threshold.png', resultsdir, imgname);
    imwrite(Im > threshold, fname);
    fname = sprintf('%s/%s_03_hough.png', resultsdir, imgname);
    imwrite(H/max(H(:)), fname);
    fname = sprintf('%s/%s_04_lines.png', resultsdir, imgname);

    img_w = img;
    for j=1:numel(lines)
       img_w = drawLine(img_w, lines(j).point1, lines(j).point2); 
    end     
    imwrite(img_w, fname);
end
