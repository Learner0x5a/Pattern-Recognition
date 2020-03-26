clc
clear all
train_dir='F:\A1\Spring\模式识别\第五周作业\第五周作业\face\train';
train_num=500;
test_num=2;
energy=0.95;

row=19;
column=19;
picsize=row*column;
train_data=zeros(train_num,picsize);%预分配数据可以加速数据读取，矩阵的行数是训练数据的个数，列数是图片的维度
train_files=dir(train_dir);%获取训练目录下的所有文件，获得的每一个文件都是一个结构体，我们需要的是其中的name属性。第一个和第二个文件分别表示当前目录和父目录，需要跳过
for i=1:train_num
    file_name=sprintf('%s\\%s',train_dir,train_files(i+2).name);%这里需要加双斜杠
    face_data=imread(file_name);
    [row,column]=size(face_data);
    face_data=face_data(1:picsize);%将读取的数据转成一个行向量
    train_data(i,:)=face_data;%将该行向量添加到训练集中
end
%% 求平均脸
facemean=mean(train_data);
figure
imshow(reshape(facemean,[row column])/256)
%facemean=repmat(facemean,train_num,1);
train_sub=train_data-facemean;
train_Qsub=train_sub';
%% 读取测试数据集
test_data=zeros(test_num,row*column);%预分配数据可以加速数据读取
%test_files=dir(test_dir);%获取训练目录下的所有文件，获得的每一个文件都是一个结构体，我们需要的是其中的name属性。第一个和第二个文件分别表示当前目录和父目录，需要跳过
for i=1:test_num
    file_name=['test/' num2str(i) '.jpg'];%这里需要加双斜杠
    face_data=imread(file_name);
    [row2,column2]=size(face_data);
    face_data=face_data(1:row2*column2);%将读取的数据转成一个行向量
    test_data(i,1:row2*column2)=face_data;%将该行向量添加到训练集中
end
test_Qsub=(test_data-facemean)';

%% 对train_Qsub进行SVD分解

[U S V]=svd(train_Qsub,0); %S对角元为相应特征值


%% a）
x=train_Qsub(:,99);
proj=x'*U;
err=zeros(picsize,1);
picr=zeros(picsize,1);

for i=1:picsize
    picr=picr+proj(i)*U(:,i);
    err(i)=sum((picr-x).^2)/picsize;
end
figure;
hold on
plot(1:picsize,err);
%% b)
x=test_Qsub(:,1);
proj=x'*U;
err=zeros(picsize,1);
picr=zeros(picsize,1);

for i=1:picsize
    picr=picr+proj(i)*U(:,i);
    err(i)=sum((picr-x).^2)/picsize;
end

hold on
plot(1:picsize,err);
%% c)
x=test_Qsub(:,2);
proj=x'*U;
err=zeros(picsize,1);
picr=zeros(picsize,1);

for i=1:picsize
    picr=picr+proj(i)*U(:,i);
    err(i)=sum((picr-x).^2)/picsize;
end

hold on
plot(1:picsize,err);

xlim([1 361]);
xlabel('K');
ylabel('MSE');
legend('a','b','c');


