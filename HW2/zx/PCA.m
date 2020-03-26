clc
clear all
train_dir='F:\A1\Spring\ģʽʶ��\��������ҵ\��������ҵ\face\train';
train_num=500;
test_num=2;
energy=0.95;

row=19;
column=19;
picsize=row*column;
train_data=zeros(train_num,picsize);%Ԥ�������ݿ��Լ������ݶ�ȡ�������������ѵ�����ݵĸ�����������ͼƬ��ά��
train_files=dir(train_dir);%��ȡѵ��Ŀ¼�µ������ļ�����õ�ÿһ���ļ�����һ���ṹ�壬������Ҫ�������е�name���ԡ���һ���͵ڶ����ļ��ֱ��ʾ��ǰĿ¼�͸�Ŀ¼����Ҫ����
for i=1:train_num
    file_name=sprintf('%s\\%s',train_dir,train_files(i+2).name);%������Ҫ��˫б��
    face_data=imread(file_name);
    [row,column]=size(face_data);
    face_data=face_data(1:picsize);%����ȡ������ת��һ��������
    train_data(i,:)=face_data;%������������ӵ�ѵ������
end
%% ��ƽ����
facemean=mean(train_data);
figure
imshow(reshape(facemean,[row column])/256)
%facemean=repmat(facemean,train_num,1);
train_sub=train_data-facemean;
train_Qsub=train_sub';
%% ��ȡ�������ݼ�
test_data=zeros(test_num,row*column);%Ԥ�������ݿ��Լ������ݶ�ȡ
%test_files=dir(test_dir);%��ȡѵ��Ŀ¼�µ������ļ�����õ�ÿһ���ļ�����һ���ṹ�壬������Ҫ�������е�name���ԡ���һ���͵ڶ����ļ��ֱ��ʾ��ǰĿ¼�͸�Ŀ¼����Ҫ����
for i=1:test_num
    file_name=['test/' num2str(i) '.jpg'];%������Ҫ��˫б��
    face_data=imread(file_name);
    [row2,column2]=size(face_data);
    face_data=face_data(1:row2*column2);%����ȡ������ת��һ��������
    test_data(i,1:row2*column2)=face_data;%������������ӵ�ѵ������
end
test_Qsub=(test_data-facemean)';

%% ��train_Qsub����SVD�ֽ�

[U S V]=svd(train_Qsub,0); %S�Խ�ԪΪ��Ӧ����ֵ


%% a��
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


