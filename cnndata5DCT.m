%%% 1-19 should be changed depending on how you want to load the data %%%

% Extract folder information for image names
folderinfo = struct2cell(dir('XML_Files\'));

% Choose image(s)
k = 8;

% Obtain image name
pic = erase(folderinfo{1,k},'.xml');

img=imread(sprintf('TIF_Files%s%s.tif','\',pic));
xDoc = xmlread(sprintf('XML_Files%s%s.xml','\',pic));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Regions=xDoc.getElementsByTagName('Region'); 
% get a list of all the region tags

% Initialzing xy variable
xy = cell(1,Regions.getLength);

for regioni = 0:Regions.getLength-1
    Region=Regions.item(regioni);  % for each region tag

    %get a list of all the vertexes (which are in order)
    verticies=Region.getElementsByTagName('Vertex'); 
    xy{regioni+1}=zeros(verticies.getLength-1,2); %allocate space for them
    for vertexi = 0:verticies.getLength-1 %iterate through all verticies
        %get the x value of that vertex
        x=str2double(verticies.item(vertexi).getAttribute('X')); 
        %get the y value of that vertex
        y=str2double(verticies.item(vertexi).getAttribute('Y')); 
        xy{regioni+1}(vertexi+1,:)=[x,y]; % finally save them into the array
    end
end

% Manually setting parameters so previously used code works
lb = 32;
lp = 1000;
sampsize = 3000;

% Obtaining binary masks of ground truth and boundary, as well as centroids
bmask = zeros(lp,lp,'logical');
bndmask = zeros(lp,lp,'logical');
boundlist = [];
bnd = zeros(1,1);
centr = zeros(length(xy),2);
for i = 1:length(xy)
    coords = xy{i};
    centr(i,:) = [mean(coords(:,1)),mean(coords(:,2))];
    coords(coords <= 1) = 1;
    coords(coords >= lp) = lp;
    boundround = zeros(length(coords),4);
    boundround(:,1) = sub2ind([lp,lp],floor(coords(:,1)),floor(coords(:,2)));
    boundround(:,2) = sub2ind([lp,lp],floor(coords(:,1)),ceil(coords(:,2)));
    boundround(:,3) = sub2ind([lp,lp],ceil(coords(:,1)),floor(coords(:,2)));
    boundround(:,4) = sub2ind([lp,lp],ceil(coords(:,1)),ceil(coords(:,2)));
    boundlist = union(boundlist,unique(boundround));
    bmask = bmask + poly2mask(coords(:,1),coords(:,2),lp,lp);
end
bndmask(boundlist) = 1;
bndmask = transpose(bndmask);
bmask = logical(bmask);

% Combining the two masks to make a ternary mask
tmask = bmask + 2*bndmask;
tmask(tmask == 3) = 2;

% Implementing criteria for buffered borders
% 3 = outer border, 4 = inner border
pmask = tmask;
for i = 1:lp
    for j = 1:lp
        tempMat = tmask(max([1,i-2]):min([lp,i+2]),max([1,j-2]):min([lp,j+2]));
        if (tmask(i,j) == 0 && sum(sum(tempMat == 2)) > 0)
            pmask(i,j) = 3;
        end
        if (tmask(i,j) == 1 && sum(sum(tempMat == 2)) > 0)
            pmask(i,j) = 4;
        end
    end
end

% Defining dimensions for the patches
if mod(lb,2) == 1
    lefttrim = (lb-1)/2;
    righttrim = lefttrim;
end
if mod(lb,2) == 0
    lefttrim = lb/2 - 1;
    righttrim = lefttrim + 1;
end

% Making a new buffered ternary mask (for boundary and exterior pixels)
tmask2 = tmask;
for i = (1+lefttrim):(lp-righttrim)
    for j = (1+lefttrim):(lp-righttrim)
        blocktemp = tmask((i-lefttrim):(i+righttrim),(j-lefttrim):(j+righttrim));
        if tmask(i,j) ~= 2 && length(blocktemp(blocktemp == tmask(i,j)))/(size(blocktemp,1)*size(blocktemp,2)) < .8
            tmask2(i,j) = 3;
        end
    end
end

% Marking the points within a distance of 4 from any centroid
tmask2(tmask2 == 1) = 3;
pts = zeros(lp,lp,2);
[xpts,ypts] = meshgrid(1:lp,1:lp);
pts(:,:,1) = xpts;
pts(:,:,2) = ypts;
pts = reshape(pts,lp^2,2);
closepts = rangesearch(pts,centr,4);
for i = 1:length(closepts)
    tmask2(closepts{i}) = 1;
end

tmask2([1:lefttrim,(lp+1-righttrim):lp],:) = 3;
tmask2(:,[1:lefttrim,(lp+1-righttrim):lp]) = 3;

% Obtaining buffered pmask
pmask2 = pmask;
pmask2(tmask2 == 3) = 5;
pmask2(pmask == 3) = 3;
pmask2(pmask == 4) = 4;

pmask2([1:lefttrim,(lp+1-righttrim):lp],:) = 5;
pmask2(:,[1:lefttrim,(lp+1-righttrim):lp]) = 5;

pmask2size = zeros(5,1);
for i = 0:4
    pmask2size(i+1) = sum(sum(pmask2 == i));
end

% Obtaining DCT matrix
imgDCT = zeros(lp,lp,18);
for i = 2:(lp-2)
    for j = 2:(lp-2)
        for y = 1:3
            tempDCT = dct2(img((i-1):(i+2),(j-1):(j+2),y));
            imgDCT(i,j,6*y-5) = tempDCT(1,1);
            imgDCT(i,j,6*y-4) = tempDCT(1,2);
            imgDCT(i,j,6*y-3) = tempDCT(1,3);
            imgDCT(i,j,6*y-2) = tempDCT(2,1);
            imgDCT(i,j,6*y-1) = tempDCT(2,2);
            imgDCT(i,j,6*y) = tempDCT(3,1);
        end
    end
end

% Obtaining the patch samples
PatchSamp = zeros(lb,lb,18,length(unique(pmask))*sampsize);
subcent = zeros(sampsize,2,length(unique(pmask)));
patchesi = zeros(lb,lb,18,sampsize);
lincent = zeros(sampsize,length(unique(pmask)));

for i = 0:4
    pop = find(pmask2 == i);
    [subcent(:,1,i+1),subcent(:,2,i+1)] = ind2sub([lp,lp],pop(randperm(length(pop),sampsize)));
    lincent(:,i+1) = sub2ind([lp,lp],subcent(:,1,i+1),subcent(:,2,i+1));
    for j = 1:sampsize
       patchesi(:,:,:,j) = imgDCT((subcent(j,1,i+1)-lefttrim):(subcent(j,1,i+1)+righttrim),(subcent(j,2,i+1)-lefttrim):(subcent(j,2,i+1)+righttrim),:);
    end
    PatchSamp(:,:,:,sampsize*i+(1:sampsize)) = patchesi;
end

PatchSamp = single(PatchSamp);

% Creating the structure (using VGG syntax and variable names)
% id: unique patch number, data: uint8 patches
% label: 1 = background, 2 = object, 3 = boundary, 4 = ob, 5 = ib
% set: 1 = training set, 2 = testing/validation set
% imdb.images = struct('patch',img,'id',1:(3*sampsize),'data',PatchSamp,'centers',subcent,'label',[ones(1,sampsize),2*ones(1,sampsize),3*ones(1,sampsize)],'set',[ones(1,2*sampsize/3),2*ones(1,sampsize/3),ones(1,2*sampsize/3),2*ones(1,sampsize/3),ones(1,2*sampsize/3),2*ones(1,sampsize/3)]);
imdb.images = struct('patch',single(img),'id',1:(length(unique(pmask))*sampsize),'data',PatchSamp,'centers',subcent,'label',[ones(1,sampsize),2*ones(1,sampsize),3*ones(1,sampsize),4*ones(1,sampsize),5*ones(1,sampsize)],'set',[ones(1,2*sampsize/3),2*ones(1,sampsize/3),ones(1,2*sampsize/3),2*ones(1,sampsize/3),ones(1,2*sampsize/3),2*ones(1,sampsize/3),ones(1,2*sampsize/3),2*ones(1,sampsize/3),ones(1,2*sampsize/3),2*ones(1,sampsize/3)]);
imdb.meta = struct('classes','eibEI');
% If you want to verify that this structure is accurate, run the code below
%{

MaskSamp = zeros(lb,lb,25);
CompSamp = zeros(lb,lb,3,25,'uint8');
for i = 1:5
    for j = 1:5
    tempcent = imdb.images.centers((size(imdb.images.centers,1)/5)*j,:,i);
        MaskSamp(:,:,5*i+j-5) = pmask((tempcent(1)-lefttrim):(tempcent(1)+righttrim),(tempcent(2)-lefttrim):(tempcent(2)+righttrim));
        ImgSamp(:,:,:,5*i+j-5) = uint8(imdb.images.patch((tempcent(1)-lefttrim):(tempcent(1)+righttrim),(tempcent(2)-lefttrim):(tempcent(2)+righttrim),:));
    end
end

figure
for i = 1:25
    subplot(5,5,i)
    imagesc(ImgSamp(:,:,:,i))
end

figure
for i = 1:25
    subplot(5,5,i)
    imagesc(MaskSamp(:,:,i))
    caxis([0 4])
end
%}