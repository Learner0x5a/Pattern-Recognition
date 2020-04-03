clc;close all;clear;
point1 = [[1,1];[2,0];[2,1];[0,2];[1,3]];
point2 = [[-1,2];[0,0];[-1,0];[-1,-1];[0,-2]];


figure;
scatter(point1(:,1),point1(:,2),'b');
hold on;
scatter(point2(:,1),point2(:,2),'r');


mixed = [point1;-point2];
mixed = [mixed,[ones(5,1);-ones(5,1)]];
W = [1,-1,1];
c =1;
flag = 0;

score = 0;
iter = 0;
while(score<10)
    score = 0;
    for i =1:10
        p = W*mixed(i,:)';
        iter = iter+1;
        if p>0
            score  = score+1;
        else 
            W = W+c*mixed(i,:);
        end
    end
end
syms f(x,y)
f(x,y) = W(1)*x+W(2)*y+W(3);
disp(f(x,y));
hold on;

ezplot(f(x,y));
axis([-1.5,2.5,-2,3.5]);
disp(iter);
title('');

legend('类别1','类别2',['分界面',char(f),'=0']);
