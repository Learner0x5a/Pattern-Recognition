clear;clc;close all;
point = [[1,1];[1,3];[3,1];[2,2];[3,3];[1,2];[2,1];[2,3];[3,2]];

figure;


scatter(point(1:5,1),point(1:5,2),'b');
hold on;
scatter(point(6:9,1),point(6:9,2),'r');
hold on;
syms x y
h = ezplot('(x-2)^2-(y-2)^2=1/2',[1,3]);
set(h,'color',[0,1,1],'LineWidth',2);
hold on;
h = ezplot('(y-2)^2-(x-2)^2=1/2',[1,3]);
set(h,'color',[0,1,1],'LineWidth',2);
legend('类别1','类别2','分界线');