function [images,labels] = read_images( directory , class )

%Read images
imageFiles = dir( strcat(directory,'/*.jpg') );

% Read all images into 'images' array and resize 
RESIZE = [64 64];

for i=1:size(imageFiles)
    
    images{i}=(imread( strcat(directory,'/',imageFiles(i).name) ));
    
%     
%     %pad non square images
%     BoxSize = max(size(images{i}));
%     simg = images{i};
%     % add black padding to ensure min dimension is 64
%     if (size(simg,1)<BoxSize)
%         diffr = ceil((BoxSize-size(simg,1))/2.0);
%         simg = [zeros(diffr,size(simg,2)); simg; zeros(diffr,size(simg,2))];
%     end
% 
%     if (size(simg,2)<BoxSize)
%         diffc = ceil((BoxSize-size(simg,2))/2.0);
%         simg = [zeros(size(simg,1),diffc), simg, zeros(size(simg,1),diffc)];
%     end
%     
    images{i}=imresize(images{i}, RESIZE);
    %imshow(images{i});
end

labels = zeros(size(images,2),2);
if (class == 'p')
    labels(:,1) = 1;
else
    labels(:,2) = 1;
end

end

