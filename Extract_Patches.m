% extract training image patches

directory = 'images';
imageFiles = dir( strcat(directory,'/*.jpg') );


% training samples

for i=1:size(imageFiles)
    imageNo = str2num( strtok( imageFiles(i).name ,'.jpg') );
    
    if (imageNo < 600)
        images{imageNo+1}=imread( strcat(directory,'/',imageFiles(i).name) );
    end
end

load('bboxes.mat','boxes');


posInd = 1;
negInd = 1;

% write positive (containing autorickshaw) images

for i=1:size(images,2)
    i
    img = images{i};
    bboxes = boxes{i};
    %imshow(img);
    %hold on ;
    for j=1:size(bboxes)
        
        bbox = abs(bboxes(j,:));
        %rectangle('Position',bbox,'EdgeColor',[1 0 0], 'LineWidth',2);
        
        
        SIZE = mean(bbox(3:4));
        
        if (bbox(4)>=bbox(3))
            diff = (bbox(4)-bbox(3))/2.0;
            bbox(1) = bbox(1)-diff;
            bbox(3) = bbox(3)+diff*2;
        else
            diff = (bbox(3)-bbox(4))/2.0;
            bbox(2) = bbox(2)-diff;
            bbox(4) = bbox(4)+diff*2;
        end
        
        bbox(1) = min( max(0, bbox(1)), size(img,2));
        bbox(2) = min( max(0, bbox(2)), size(img,1));
        bbox(3) = min(size(img,2)-bbox(1), bbox(3));
        bbox(4) = min(size(img,1)-bbox(2), bbox(4));
        
        %imshow(imcrop(img,bbox));
        
        imwrite( imresize( imcrop(img,bbox), [64 64] ), strcat( 'data\positive\' ,num2str(posInd),'.jpg') );
        posInd = posInd+1;
        
        for rr=1:4
            Rbox = bbox;
            Rbox(1:2) =  bbox(1:2)+randi([-ceil(0.1*SIZE), ceil(0.1*SIZE)] ,1,2);
            Rbox(1) = min( max(0, Rbox(1)), size(img,2));
            Rbox(2) = min( max(0, Rbox(2)), size(img,1));
            Rbox(3) = min(size(img,2)-Rbox(1), Rbox(3));
            Rbox(4) = min(size(img,1)-Rbox(2), Rbox(4));
            
            
%             size(imcrop(img,Rbox))
%             Rbox
%             size(img)
            imwrite( imresize( imcrop(img,Rbox), [64 64] ), strcat( 'data\positive\' ,num2str(posInd),'.jpg') );
            posInd = posInd+1;
        end
        
        
    end
    %hold off;
    
end






% write negative (not containing autorickshaw) images

for i=1:size(images,2)
    i
    img = images{i};
    bboxes = boxes{i};
    %imshow(img);
    %hold on ;
    for j=1:size(bboxes)
        
        bbox = abs(bboxes(j,:));
        %rectangle('Position',bbox,'EdgeColor',[1 0 0], 'LineWidth',2);
        
        
        % make bounding box square
        SIZE = mean(bbox(3:4));
        
        if (bbox(4)>=bbox(3))
            diff = (bbox(4)-bbox(3))/2.0;
            bbox(1) = bbox(1)-diff;
            bbox(3) = bbox(3)+diff*2;
        else
            diff = (bbox(3)-bbox(4))/2.0;
            bbox(2) = bbox(2)-diff;
            bbox(4) = bbox(4)+diff*2;
        end
        
        bbox(1) = min( max(0, bbox(1)), size(img,2));
        bbox(2) = min( max(0, bbox(2)), size(img,1));
        bbox(3) = min(size(img,2)-bbox(1), bbox(3));
        bbox(4) = min(size(img,1)-bbox(2), bbox(4));
        
        %imshow(imcrop(img,bbox));
        
        % do random shifts to get new positions in image
        for rr=1: (ceil(size(img,1)/32))
            Rbox = bbox;
            randomShift = randi([-ceil(2*SIZE), ceil(2*SIZE)] ,1,2);
            if (randomShift(1)<0)
                randomShift(1) = randomShift(1) - (0.7*SIZE);
            else
                randomShift(1) = randomShift(1) + (0.7*SIZE);
            end
            
            if (randomShift(2)<0)
                randomShift(2) = randomShift(2) - (0.7*SIZE);
            else
                randomShift(2) = randomShift(2) + (0.7*SIZE);
            end
            
            Rbox(3:4) =  Rbox(3:4).*(0.1+rand(1)*3);
            
            
            Rbox(1:2) =  bbox(1:2)+randomShift;
            Rbox(1) = min( max(0, Rbox(1)), size(img,2));
            Rbox(2) = min( max(0, Rbox(2)), size(img,1));
            Rbox(3) = min(size(img,2)-Rbox(1), Rbox(3));
            Rbox(4) = min(size(img,1)-Rbox(2), Rbox(4));
            
            boxRatio = Rbox(3)/Rbox(4);
            if ~(boxRatio<1.5 && boxRatio > 0.7)
                continue;
            end 
               
            
%             size(imcrop(img,Rbox))
%             Rbox
%             size(img)
            imwrite( imresize( imcrop(img,Rbox), [64 64] ), strcat( 'data\negative\' ,num2str(negInd),'.jpg') );
            negInd = negInd+1;
            
             %imshow(imcrop(img,Rbox));
        end
        
        
    end
    %hold off;
    
end




