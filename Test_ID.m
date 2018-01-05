
directory = 'images';
imageFiles = dir( strcat(directory,'/*.jpg') );

% training samples

for i=1:size(imageFiles)
    imageNo = str2num( strtok( imageFiles(i).name ,'.jpg') );
    
    if (imageNo >= 600)
        images{imageNo+1}=imread( strcat(directory,'/',imageFiles(i).name) );
    end
end

load('bboxes.mat','boxes');

myNet = load('myNet.mat');
myNet = myNet.net;


FinalIOU =0 ;
TestImageCount =0 ;
midAns = double.empty;

PredictedBoxes = {};

for testImgNo=601:size(images,2)

    %disp(datestr(clock,0));
    disp(strcat(num2str(testImgNo-1),'.jpg'));
    Oimage = images{testImgNo};
    image = (Oimage);
    MAXIMAGESIZE = 512;
    SCALE = MAXIMAGESIZE/max(size(image));
    image = imresize(image, SCALE);

    BoxSizeX = 64;
    BoxSizeY = 64;
    NSTEPS = 10;

    RES = zeros(MAXIMAGESIZE,MAXIMAGESIZE,NSTEPS);
    ROIS = double.empty;
    stepNo = 0;
    startSCALE = 0.1;

    ROI_Threshold = 0.8;

    SCALES = linspace(startSCALE,0.5,NSTEPS);

    for stepNo=1:NSTEPS
        scale = SCALES(stepNo);
        simg = imresize(image, scale);

        %disp(scale);

        if (max(size(simg))<min(BoxSizeX,BoxSizeY))
            scale=scale*( min(BoxSizeX,BoxSizeY) )/max(size(simg));
            simg = imresize(simg, ( min(BoxSizeX,BoxSizeY) )/max(size(simg)) );

        end

        diffr=0; diffc=0;
        % add black padding to ensure min dimension is 64
        if (size(simg,1)<BoxSizeY)
            diffr = ceil((BoxSizeY-size(simg,1))/2.0);
            simg = [zeros(diffr,size(simg,2),3); simg; zeros(diffr,size(simg,2),3)];
        end

        if (size(simg,2)<BoxSizeX)
            diffc = ceil((BoxSizeX-size(simg,2))/2.0);
            simg = [zeros(size(simg,1),diffc,3), simg, zeros(size(simg,1),diffc,3)];
        end
        %size(simg)
        %imshow(simg);

        myBox = [1,1,BoxSizeY,BoxSizeX];

        WindowStep=16;

        for r = 1:WindowStep:(size(simg,1)-BoxSizeY+1)

            for c = 1:WindowStep:(size(simg,2)-BoxSizeX+1)
                %fprintf('%f, %f\n',r,c);

                P = imcrop(simg, [ c,r,BoxSizeY-1,BoxSizeX-1 ]);
                %imshow( P );
                %disp(size(P))


                HOG = extractHOGFeatures(imresize(P,[64 64]));

                output = myNet(HOG');

                RES(r,c,stepNo) = output(1);
                scaledROI = [ max(0,r-diffr),max(0,c-diffc),min(BoxSizeY,size(Oimage,2)),min(BoxSizeX,size(Oimage,1)) ].*(1/(scale*SCALE));


                if (output(1) >= ROI_Threshold)
                    ROIS = [ROIS; [scaledROI, output(1)] ];
                end
            end
        end


    end

    %disp(datestr(clock,0));
    if (size(ROIS,1) ==0)
        ROIS = [0,0,0,0,0];
    end
    ROIS = [ROIS, ones(size(ROIS,1),1)];
    [srt,idx] = sort(ROIS(:,3),1,'descend');
    ROIS = ROIS(idx,:);


    for i=1:size(ROIS,1)
    %     break;
        if (ROIS(i,6) == 0)
            continue;
        end

        for j=i+1:size(ROIS,1)
            oA = rectint(ROIS(i,1:4), ROIS(j,1:4) );
            oI = oA/ (ROIS(i,3)*ROIS(i,4));
            oJ = oA/ (ROIS(j,3)*ROIS(j,4));

            if (oI>0.8 && oJ>0.8)
               ROIS(i,6) =  ROIS(i,6)+ROIS(j,6);
               ROIS(j,6) = 0;
               ROIS(i,1:4) = mean( [ROIS(i,1:4); ROIS(i,1:4)] );
            end
        end
    end

    [srt,idx] = sort(ROIS(:,5),1,'descend');
    ROIS = ROIS(idx,:);

    for i=1:size(ROIS,1)
    %     break;
        if (ROIS(i,6) == 0)
            continue;
        end

        for j=i+1:size(ROIS,1)
            oA = rectint(ROIS(i,1:4), ROIS(j,1:4) );
            oI = oA/ (ROIS(i,3)*ROIS(i,4));
            oJ = oA/ (ROIS(j,3)*ROIS(j,4));

            if (oJ>0.6 && ( ROIS(i,6)>ROIS(j,6) || ROIS(i,5)>ROIS(j,5) ))
               ROIS(i,6) =  ROIS(i,6)+ROIS(j,6);
               ROIS(j,6) = 0;

            end
        end
    end

    [srt,idx] = sort(ROIS(:,6),1,'descend');
    ROIS = ROIS(idx,:);

    for i=1:size(ROIS,1)
        %break;
        if (ROIS(i,6) == 0)
            continue;
        end

        for j=i+1:size(ROIS,1)
            oA = rectint(ROIS(i,1:4), ROIS(j,1:4) );
            oI = oA/ (ROIS(i,3)*ROIS(i,4));
            oJ = oA/ (ROIS(j,3)*ROIS(j,4));

            if (oJ>0.3 && ( ROIS(i,6)>ROIS(j,6)) )
               ROIS(i,6) =  ROIS(i,6)+ROIS(j,6);
               ROIS(j,6) = 0;
            end

            if (oI>0.9 && ( ROIS(i,6)>2*ROIS(j,6) || ROIS(i,5)>ROIS(j,5) ) )
               ROIS(i,6) =  ROIS(i,6)+ROIS(j,6);
               ROIS(j,6) = 0;
            end
        end
    end


    % [srt,idx] = sort(ROIS(:,5),1,'descend');
    % ROIS = ROIS(idx,:);
    % 

    pboxes = double.empty;

    %imshow(Oimage);
    %hold on;
    for i=1:min(10,size(ROIS,1))

        if (ROIS(i,6)<1)
            continue;
        end

        myRect = [ROIS(i,2), ROIS(i,1), ROIS(i,4), ROIS(i,3)];
        
        if (myRect(3)*myRect(4)==0)
            continue;
        end
        
        pboxes = [pboxes; myRect];

  %     myColor = hsv2rgb( [ROIS(i,5),0.8,0.9] );
        myColor = [1 0 0];
        
      %rectangle('Position',myRect,'EdgeColor',myColor, 'LineWidth',2);

    end
    
    bboxes = boxes{testImgNo};
    IOUSUM = 0;
    Nboxes = 0;
    for bb=1:size(bboxes,1)
       
         if (min(bboxes(bb,:))<0)
             continue;
         end
                  
        %rectangle('Position',bboxes(bb,:),'EdgeColor',[0 1 0], 'LineWidth',2);
        maxIOU = 0;
        for pb = 1:size(pboxes,1)
            oA = rectint(pboxes(pb,:), bboxes(bb,:) );
            oI = oA/ ( pboxes(pb,3)*pboxes(pb,4) +  bboxes(bb,3)*bboxes(bb,4) - oA);
            maxIOU = max(maxIOU , oI);
            
        end
        IOUSUM = IOUSUM + maxIOU;
        Nboxes =Nboxes+1; 
    end
    
    
    IOU = IOUSUM/max(1,Nboxes);
    midAns = [midAns;IOU];
    FinalIOU = FinalIOU + IOU;
    TestImageCount = TestImageCount + 1;
    fprintf('IOU : %f\n',IOU);
    %hold off;
    
    disp('Bounding Boxes:')
    disp(pboxes);
    PredictedBoxes{testImgNo} = pboxes;
end

FinalAccuracy = FinalIOU / max(1,TestImageCount);
fprintf('Average IOU : %f\n',FinalAccuracy);
disp('All predicted bounding boxes stored as predBoxes.mat');
save('predBoxes.mat','PredictedBoxes');

% 
% RESIZE = [64 64];
% image=imresize(image, RESIZE);
% hog = extractHOGFeatures(image);
% 
% output = myNet(hog')




