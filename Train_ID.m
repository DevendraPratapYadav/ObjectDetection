Extract_Patches

disp('Positive and Negative patches extracted');

TRAIN_DIR = 'data\'; 
TEST_DIR = 'data\';

[trip,trlp] = read_images(strcat(TRAIN_DIR,'positive\'), 'p');
[trin,trln] = read_images(strcat(TRAIN_DIR,'negative\'), 'n');
tri = [trip, trin]; trl = [trlp;trln];

% [teip,telp] = read_images(strcat(TEST_DIR,'positive\'), 'p');
% [tein,teln] = read_images(strcat(TEST_DIR,'negative\'), 'n');
% tei = [teip, tein]; tel = [telp;teln];

tei = tri; tel = trl;

hog = extractHOGFeatures(tri{1});
trf = zeros(size(tri,2),size(hog,2));
tef = zeros(size(tei,2),size(hog,2));
 
for i=1:size(tri,2) 
    hog = extractHOGFeatures(tri{i});
    trf(i,:) = hog;
end

for i=1:size(tei,2) 
    hog = extractHOGFeatures(tei{i});
    tef(i,:) = hog;
end

hiddenLayerSize = 100;
net =  patternnet([hiddenLayerSize hiddenLayerSize/2],'traincgp');
[net, tr] = train(net, trf', trl');
save ('myNet.mat','net');
% load net;


disp('Neural Network training complete. NN save to myNet.mat');


% 
% % Test the Network
% y = net(tef');
% ltest = tel';
% e = gsubtract(ltest,y);
% tind = vec2ind(ltest);
% yind = vec2ind(y);
% Compare = [tind;yind];
% percentErrors = sum(tind ~= yind)/numel(tind);
% % Show accuracy
% Accuracy=100-percentErrors*100;
% Accuracy
