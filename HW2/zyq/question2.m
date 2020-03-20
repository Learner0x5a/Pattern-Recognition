close all;clc;clear;

w1_x = [2,2,2,3,3,2.5,1.5,3.5,4,0.5];
w1_y = [3,2,4,3,4,3,2,2.5,4,0.5];

figure;
scatter(w1_x,w1_y,'r');
hold on;

w2_x = [0,-2,-1,1,3,-2,-3,-5,4];
w2_y = [2.5,2,-1,-2,0,-2,-4,-2,-1];
scatter(w2_x,w2_y,'b');
hold on;
legend('W1','W2');
p1_x = mean(w1_x);
p1_y = mean(w1_y);

p2_x = mean(w2_x);
p2_y = mean(w2_y);
scatter(p1_x,p1_y,'r','fill');
scatter(p2_x,p2_y,'b','fill');
hold on;

k_1 = (p2_y-p1_y)/(p2_x-p1_x);
k = -1/k_1;
mid = [(p2_x+p1_x)/2,(p2_y+p1_y)/2];
scatter(mid(1),mid(2),'k');
legend('W1_M','W2_M','M','fill');
b = mid(2)-k*mid(1);
refline(k,b)

legend('W1','W2','W1\_M','W2\_M','M','sperate line');



