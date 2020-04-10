clc;clear;close all;
point1 = [[3,4];[3,8];[2,6];[4,6]];
point2 = [[3,0];[3,-4];[1,-2];[5,-2]];

figure;
scatter(point1(:,1),point1(:,2),'b');
hold on;
scatter(point2(:,1),point2(:,2),'r');
hold on;

len1 = size(point1,1);
len2 = size(point2,1);
M1 = sum(point1)'/len1;
M2 = sum(point2)'/len2;

sigma1 = cov(point1);
sigma2 = cov(point2);

W1 = -0.5*inv(sigma1);
W2 = -0.5*inv(sigma2);


w1 = inv(sigma1)*M1;
w2 = inv(sigma2)*M2;

w10 = -0.5*w1'*M1-0.5*log(det(sigma1));
w20 = -0.5*w2'*M2-0.5*log(det(sigma2));

digits(4)
W_minus =vpa( W1-W2);
w_minus = vpa(w1-w2);
w0_minus = vpa(w10-w20);
syms x1
syms x2

ezplot(W_minus(1,1)*x1^2+(W_minus(1,2)+W_minus(2,1))*x1*x2+W_minus(2,2)*x2^2+w_minus(1,1)*x1+w_minus(2,1)*x2+w0_minus==0);

axis([0,6,-5,9]);

