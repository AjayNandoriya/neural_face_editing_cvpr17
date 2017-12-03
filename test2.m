basedir = 'C:\Users\qcri\Documents\Ajay\Face\SOA\cvpr17\cvpr17_relit_test2\test\images_source';
refdir = 'C:\Users\qcri\Documents\Ajay\Face\SOA\cvpr17\cvpr17_relit_test2\test\images_ref';
flist = dir([basedir '\*outputs.png']);

img_R = im2double(imread(fullfile(flist(k).folder,flist(k).name)));
for k=1:length(flist)
    fname = fullfile('C:\Users\qcri\Documents\Ajay\Face\SOA\cvpr17',[flist(k).name(10:15) '.jpg']);
    img_in = im2double(imread(fname));
    img_inR = im2double(imread(fullfile(flist(k).folder,flist(k).name)));
    img_inR = imresize(img_inR,size(img_in(:,:,1)));
    
    fname = fullfile(refdir,flist(k).name);
    img_outR = im2double(imread(fname));
    
    img_outR = imresize(img_outR,size(img_in(:,:,1)));
    img_out = img_outR.*(img_in+0.01)./(img_inR+0.01);
    
    fname = fullfile('C:\Users\qcri\Documents\Ajay\Face\SOA\cvpr17',sprintf('%s_ours.jpg',flist(k).name(10:15)));
    imwrite(img_out,fname);
end
    