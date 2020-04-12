clc
clear all
w1=[2,3;2,2;2,4;3,3;3,4;2.5,3;
    1.5,2;4,4;0.5,0.5;];
w2=[0,2.5;-2,2;-1,-1;1,-2;3,0;-2,-2;
    -3,-4;-5,-2;4,-1;];
w=[w1;w2];
c1=mean(w1,1);
c2=mean(w2,1);
c=(c1+c2)/2;
%% �д���
x=-5:0.1:5;
y=(x-c(1))*-(c1(2)-c2(2))/(c1(1)-c2(1))+c(2);
figure;
hold on;
plot(x,y);
%% ����
for i=1:size(w,1)
    if(w(i,2)>(w(i,1)-c(1))*-(c1(2)-c2(2))/(c1(1)-c2(1))+c(2))
        scatter(w(i,1),w(i,2),'g');
    else
        scatter(w(i,1),w(i,2),'r');
    end
end
scatter(w1(:,1), w1(:,2), '*');
scatter(w2(:,1), w2(:,2), 'v');