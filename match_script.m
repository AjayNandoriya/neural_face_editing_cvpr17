flist = dir('C:\Users\qcri\Documents\Ajay\Face\dataset\img_align_celeba\*.jpg');
flist = dir('C:\Users\qcri\Documents\Ajay\Face\SOA\cvpr17\ss*.PNG');
for k=length(flist):-1:1
    fname = fullfile(flist(k).folder,flist(k).name);
    if(exist([fname(1:end-3) 'mat'],'file'))
        continue;
    end
    img = imread(fname);
    ptsDistorted = detectSURFFeatures(rgb2gray(img));
    [features, validPts] = extractFeatures(rgb2gray(img), ptsDistorted);
    save([fname(1:end-3) 'mat'],'features','validPts');
end

%%
clc
ref = load('C:\Users\qcri\Documents\Ajay\Face\SOA\cvpr17\ss10.mat');
flist = dir('C:\Users\qcri\Documents\Ajay\Face\dataset\img_align_celeba\*.mat');
parfor k=1:length(flist)
source = load(fullfile(flist(k).folder,flist(k).name));
[indexPairs,score] = matchFeatures(ref.features, source.features);
if(length(indexPairs)>3)
    fprintf('match=%s\n',flist(k).name);
end
end
return
