clc;close all;clear;


%point1 = [[2,0];[2,2];[2,4];[3,3]];
%point2 = [[0,3];[-2,2];[-1,-1];[1,-2];[3,-1]];

point1 = [[1,1];[2,0];[2,1];[0,2];[1,3]];
point2 = [[-1,2];[0,0];[-1,0];[-1,-1];[0,-2]];

l1 =size(point1,1);
l2 =size(point2,1);

m1 = mean(point1)';
m2 = mean(point2)';

s1 = cov(point1)*(l1-1);
s2 = cov(point2)*(l2-1);
sw = s1+s2;
w = inv(sw)*(m1-m2);
w = w/sqrt(sum(w.^2));
disp(w(1)^2+w(2)^2)
y1 = mean(w'*m1);
y2 = mean(w'*m2);
y0 = (l1*y1+l2*y2)/(l1+l2);

figure;
scatter(point1(:,1),point1(:,2),'b');
hold on;
scatter(point2(:,1),point2(:,2),'r');
figure;
H = [];






h = scatter(point1(:,1),point1(:,2),'b');
H = [H,h];
hold on;
h = scatter(point2(:,1),point2(:,2),'r');
H = [H,h];
hold on;
syms y
syms x
digits(4)
y = vpa(w(2))/vpa(w(1))*x;
hold on;
h = ezplot(y);
H = [H,h];
theta = atan(w(2)/w(1));
title('');

px = cos(theta)*y0;
py = sin(theta)*y0;
h =scatter(px,py,'g','filled');
H = [H,h];


for i =1 :l1
    p = point1(i,:)';
    ro = w'*p;
    hold on;
    px = cos(theta)*ro;
    py = sin(theta)*ro;
    h = scatter(px,py,'b','filled');
    H = [H,h];
    h= plot([px,p(1)],[py,p(2)],'b');
    H = [H,h];
end


for i =1 :l2
    p = point2(i,:)';
    ro = w'*p;
    hold on;
    px = cos(theta)*ro;
    py = sin(theta)*ro;
    h = scatter(px,py,'r','filled');
    H = [H,h]; 
    h = plot([px,p(1)],[py,p(2)],'r');
    H = [H,h];
end
%axis([-2,4,-2,4]);
axis equal;
axis([-2,3,-2,3]);
legend(H([1,2,3,4,5,6,17,18]),'类别1','类别2','Fisher判别面','最佳判别点','类别1判别面投影点','类别1投影线','类别2判别面投影点','类别2投影线');

