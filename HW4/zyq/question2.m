clc;clear;close all;
data = load('trainData.txt');
data11 = data(1:100,:);
data22 = data(101:end,:);
data1 = data11(:,1:2);
data2 = data22(:,1:2);

pointx = [-6:0.1:6]';
pointy = pointx;
point = [pointx,pointx];
num = 14641;
point = zeros(num,2);

id = 1;
for i =1:121
    for j = 1:121
        point(id,:) = [pointx(i),pointy(j)];
        id = id+1;
    end
end
figure;
scatter(data1(:,1),data1(:,2),'b','filled');
hold on;
scatter(data2(:,1),data2(:,2),'r','filled');
hold on;

color = ['m','c','g','k'];

iter = 1;
for k=1:2:7
area = zeros(num,1);


for i =1:num
    p = point(i,:);
    minus_M = bsxfun(@minus,data1,p );
    dis1 = sqrt(sum(minus_M.*minus_M,2));
    minus_M = bsxfun(@minus,data2,p );
    dis2 = sqrt(sum(minus_M.*minus_M,2));  
    dis1 = [dis1,zeros(100,1)];
    dis2 = [dis2,ones(100,1)];
    dis = [dis1;dis2];
    sorted_dis = sortrows(dis,1);
    sortd_dis_k =  sorted_dis(1:k,:);
    score = sum(sortd_dis_k,1);
    if score(1,2) >= (k+1)/2
        %hold on;
       % scatter(p(1),p(2),'r');
       area(i) = 0;
    else
        %hold on;
       % scatter(p(1),p(2),'b');     
       area(i) = 1;
    end
    
    
    
end
hold on;
area = reshape(area,[121,121]);

[c,h] = contour(pointx,pointy,area,1,color(iter));
h.LineWidth = 0.5;
iter = iter+1;
end
hold on;
legend('���1','���2','k=1','k=3','k=5','k=7');
%axis([-1,4,-1,4]);
%scatter(point(:,1),point(:,2),'y');