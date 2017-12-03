addpath('..\3DMM_edges-master');

flist = dir('C:\Users\qcri\Documents\Ajay\Face\dataset\img_align_celeba\*.jpg');

datapath= 'tmp';
mkdir(datapath);
datafile_id = fopen(fullfile(datapath,'datafile.txt'),'w');
for k=1:100%length(flist)
    
    img = im2double(imread(fullfile(flist(k).folder,flist(k).name)));
    try
        [normal,mask,~,L] = img2facedata(img);
    catch
        continue;
    end
    
    
    img_fname = fullfile(datapath,flist(k).name);
    normal_fname = fullfile(datapath,[flist(k).name(1:end-4) '_normal.png']);
    mask_fname = fullfile(datapath,[flist(k).name(1:end-4) '_mask.png']);   
    imwrite(img,img_fname);
    imwrite((normal+1)/2,normal_fname);
    imwrite(double(mask(:,:,1)),mask_fname);
    cnn_str = sprintf('%s %s %s %s',flist(k).name,flist(k).name,[flist(k).name(1:end-4) '_mask.png'],[flist(k).name(1:end-4) '_normal.png']);
    for lid = 1:length(L)
        cnn_str = sprintf('%s %f',cnn_str,L(lid));
    end
    fprintf(datafile_id,'%s\n',cnn_str);
end
fclose(datafile_id);