clc;close;clear;
image = zeros(361,500);

file_path =  'train/';% 图像文件夹路径  
img_path_list = dir(strcat(file_path,'*.jpg'));%获取该文件夹中所有bmp格式的图像  
img_num = length(img_path_list);%获取图像总数量 

if img_num > 0 %有满足条件的图像  
        for j = 1:img_num %逐一读取图像  
            image_name = img_path_list(j).name;% 图像名  
            image_temp =  imread(strcat(file_path,image_name));  
            image(:,j)=reshape(image_temp,[],1);
            fprintf('%d %d %s\n',i,j,strcat(file_path,image_name));% 显示正在处理的图像名  
        end  
end   

avg = mean(image, 2);   
avg_face = mean(image,2);
avg = reshape(avg,19,19);
X = bsxfun(@minus, image, avg_face); 


sigma = X*X';
%amgis = image*image'/size(image,2);
[V1,D1] = eig(sigma);
%[V2,D2] = eig(amgis);

%V1 = image' * V2;
% aaa = reshape(V1(:,35),108,75);
% imagesc(aaa);
% colormap('gray');  

eigenfaces = V1;
%eigenfaces = eigenfaces ./ (ones(size(eigenfaces,1),1) * sqrt(sum(eigenfaces.*eigenfaces)));

 
%% visualize the average face
P = sqrt(numel(avg_face));
Q = numel(avg_face) / P;
avg_face2 = reshape(avg_face, P, Q);
imagesc(reshape(avg_face, P, Q)); title('Mean face');
colormap('gray');
 
%% visualize some eigenfaces
figure;
num_eigenfaces_show = 9;
for i = 1:num_eigenfaces_show
	subplot(3, 3, i)
	imagesc(reshape(eigenfaces(:, end-i+1), P, Q));
	title(['Eigenfaces ' num2str(i)]);
end
colormap('gray');


err = [];

Image_Sample =  imread(strcat(file_path,'face00002.jpg'));
Image_Sample = reshape(Image_Sample,[],1);
la = double(Image_Sample);
Image_Minus = double(Image_Sample)- avg_face;
Image_restruct = avg_face;
for i = 1:361
    v = eigenfaces(:, end-i+1);
    a = Image_Minus'*v;
    value = a*v;
    Image_restruct = value+Image_restruct;
    err1 = sum((la-Image_restruct).^2);
    err = [err,err1];
end
figure;
plot(err)
figure;
imagesc(reshape(Image_restruct, P, Q));
colormap('gray');
err = [];
test_path = 'test/';
Image_Sample =  imread(strcat(test_path,'face.jpg'));
Image_Sample = reshape(Image_Sample,[],1);
la = double(Image_Sample);
Image_Minus = double(Image_Sample)- avg_face;
Image_restruct = avg_face;
for i = 1:361
    v = eigenfaces(:, end-i+1);
    a = Image_Minus'*v;
    value = a*v;
    Image_restruct = value+Image_restruct;
    err1 = sum((la-Image_restruct).^2);
    err = [err,err1];
end
figure;
plot(err)
figure;
imagesc(reshape(Image_restruct, P, Q));
colormap('gray');

err = [];
test_path = 'test/';
Image_Sample =  imread(strcat(test_path,'nonface.jpg'));
Image_Sample = reshape(Image_Sample,[],1);
la = double(Image_Sample);
Image_Minus = double(Image_Sample)- avg_face;
Image_restruct = avg_face;
for i = 1:361
    v = eigenfaces(:, end-i+1);
    a = Image_Minus'*v;
    value = a*v;
    Image_restruct = value+Image_restruct;
    err1 = sum((la-Image_restruct).^2);
    err = [err,err1];
end
figure;
plot(err)
figure;
imagesc(reshape(Image_restruct, P, Q));
colormap('gray');



