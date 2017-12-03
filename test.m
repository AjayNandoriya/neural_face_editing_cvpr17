img = im2double(imread('C:\Users\qcri\Documents\Ajay\Face\SOA\cvpr17\201398.jpg'));
img1 = im2double(imread('C:\Users\qcri\Documents\Ajay\Face\SOA\cvpr17\cvpr17_relit_test\test\images\img_fg_reconstruct\00000000-201398-outputs.png'));
img2 = im2double(imread('C:\Users\qcri\Documents\Ajay\Face\SOA\cvpr17\cvpr17_relit_test\test\images\img_fg_relit\00000000-201398-outputs.png'));
img1 = imresize(img1,size(img(:,:,1)));
img2 = imresize(img2,size(img(:,:,1)));
img_out = (img2.*(img+0.01))./(img1+0.01);
subplot(221);imshow(img);
subplot(222);imshow(img1);
subplot(223);imshow(img2);
subplot(224);imshow(img_out,[]);
imwrite(img_out,'C:\Users\qcri\Documents\Ajay\Face\SOA\cvpr17\200645_out.png');



img = im2double(imread('C:\Users\qcri\Documents\Ajay\Face\SOA\cvpr17\201398.jpg'));
img1 = im2double(imread('C:\Users\qcri\Documents\Ajay\Face\SOA\cvpr17\cvpr17_relit_test\test\images\img_fg_reconstruct\00000001-201398-outputs.png'));
img2 = im2double(imread('C:\Users\qcri\Documents\Ajay\Face\SOA\cvpr17\cvpr17_relit_test\test\images\img_fg_relit\00000001-201398-outputs.png'));
img1 = imresize(img1,size(img(:,:,1)));
img2 = imresize(img2,size(img(:,:,1)));
img_out = (img2.*(img+0.01))./(img1+0.01);

subplot(221);imshow(img);
subplot(222);imshow(img1);
subplot(223);imshow(img2);
subplot(224);imshow(img_out,[]);
imwrite(img_out,'C:\Users\qcri\Documents\Ajay\Face\SOA\cvpr17\192124_out.png');