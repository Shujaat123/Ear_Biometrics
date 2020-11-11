clc
clear all
close all

src_dir = ['C:\Users\Shujaat Khan\Downloads\AMI\images'];
images = dir([src_dir,'\*.jpg']);

for img_ind = 1:numel(images)
    subjects(img_ind) = str2num(extractBefore(images(img_ind).name,'_'));
end
subjects = unique(subjects);
img_ind = 0;
ear_images = [];
sub_labels = [];

for sub_ind = 1:numel(subjects)
    for sample_ind = 1:numel(images)/numel(subjects)
        img_ind = img_ind +1;
        ear_img = single(imread([src_dir,'\',images(img_ind).name]))./255;
        ear_images(:,:,:,img_ind) = permute(ear_img,[3,2,1]);
        sub_labels(img_ind) = sub_ind; 
    end
end

