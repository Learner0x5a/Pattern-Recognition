clc
clear all
%% load 模板
m=[0,1,2,3,4,6,8,9];
%m=[1,8];
mL=length(m);
%f1：二值化一阶x,y原点矩，二阶x，y中心矩特征
f1=zeros(mL,4);
for i=1:mL
    file_name=['train/' num2str(m(i)) '.bmp'];
    data=double(imread(file_name));
    %avg=mean(mean(data));
    %data=(data>avg);
    tmplate{i}=data;
    %f1
    [loc_x,loc_y]=find(data>=mean(mean(data)));
    f1(i,1)=mean(loc_x);
    f1(i,2)=mean(loc_y);
    f1(i,3)=var(loc_x);
    f1(i,4)=var(loc_y);
    
    %imshow(data);
end
for i=1:4
    v1(i)=var(f1(:,i));
    f1(:,i)=f1(:,i)/sqrt(v1(i));
end


%% load 测试文档  1~9
t=8;
tL=length(t);
d=8;

for i=1:tL
    file_name=['test/' num2str(t(i)) '.bmp'];
    
    data=double(imread(file_name));
    
    file2=[];
    for j=1:d
        subfile=data(:,round((j-1)*size(data,2)/d+1):round(j*size(data,2)/d));
        avg=mean(mean(mean(subfile)));
        data2=mean(subfile,3);
        data2=(data2>0.98*avg);
        file2=[file2,data2];
    end
    
    %{
	[a,b]=size(file2);
	gray_count1=zeros(a,1);
	for k=1:a
		gray_count1(k)=sum(sum(double(file2(k,:))));
	end
	[aa,bb]=find(gray_count1==min(gray_count1));
	gray_count11=gray_count1(aa:end,:);
	[cc,dd]=find(gray_count11==max(gray_count11));
    cc=max(cc);
	
    %}

    ts{i}=file2;
    %{
    m_tf=data(1:cc+aa-2,:);
	n_tf=data(cc+aa-1:end,:);
	ts_m{i}=m_tf;
	ts_n{i}=n_tf;
	ts_count(i)=cc+aa-2;
	%}
    
    %imshow(file2);
    %{
    avg=mean(mean(mean(data)));
    data=mean(data,3);
    data=(data>0.9*avg);
    
    imshow(data);
    ts{i}=data;
    %}
end
%%
e=0.001;
step=0.5;
th=0.95;
for i=1:tL
    tf=ts{i};
    tf_size=size(tf);
    %{
    tf2=ts_n{i};
    tf_size2=size(tf2);
    %}
    figure;
    imshow(tf);
    center=[];
    for j=1:mL
        m0=tmplate{j};
        m_size=size(m0);
        i0=0.6;
        for k=1:3
            tpl=interp2(1:m_size(2),1:m_size(1),double(m0),linspace(1,m_size(2),round(m_size(2)*i0)),linspace(1,m_size(1),round(m_size(1)*i0))','Linear');
            %[hog_tpl, vis_tpl]=extractHOGFeatures(tpl,'CellSize',[4 4]);
            move_1=tf_size(1)-size(tpl,1)+1;
            move_2=tf_size(2)-size(tpl,2)+1;
            count=zeros(move_1,move_2);
            %{
            move_11=tf_size2(1)-size(tpl,1)+1;
            move_22=tf_size2(2)-size(tpl,2)+1;
            count2=zeros(move_11,move_22);
            %}
            for u=1:move_1
                for v=1:move_2
                    test_area=tf(u:u+size(tpl,1)-1,v:v+size(tpl,2)-1);
                    
                    [loc_x,loc_y]=find(test_area>=mean(mean(test_area)));
                    f11(u,v,1)=mean(loc_x)/sqrt(v1(1))-f1(j,1);
                    f11(u,v,2)=mean(loc_y)/sqrt(v1(2))-f1(j,2);
                    f11(u,v,3)=var(loc_x)/sqrt(v1(3))-f1(j,3);
                    f11(u,v,4)=var(loc_y)/sqrt(v1(4))-f1(j,4);
                    %{
                    if(count(u,v)>0.95&&j==2)
                        imshow(and(test_area,tpl))
                    end
                    %}
                end
            end
            %{
            for u=1:move_11
                for v=1:move_22
                    test_area=tf2(u:u+size(tpl,1)-1,v:v+size(tpl,2)-1);
                    
                    [loc_x,loc_y]=find(test_area>=mean(mean(test_area)));
                    f12(u,v,1)=mean(loc_x)/sqrt(v1(1))-f1(j,1);
                    f12(u,v,2)=mean(loc_y)/sqrt(v1(2))-f1(j,2);
                    f12(u,v,3)=var(loc_x)/sqrt(v1(3))-f1(j,3);
                    f12(u,v,4)=var(loc_y)/sqrt(v1(4))-f1(j,4);
                end
            end 
%}
            f=sum(f11.^2,3);
            %f12=sum(f12.^2,3);
            th1=1.2*min(min(f))+e;
            %th2=1.2*min(min(f12))+e;
            f_pre=f;
            %f12_pre=f12;
            f=(f<th1);
            %f12=(f12<th2);


            
            [L,L_num]=bwlabel(f,8);
%            [L2,L2_num]=bwlabel(f12,8);
            
            for w=1:L_num
                [loc1,loc2]=find(L==w);
                if(length(loc1)>=1)
                    loc=round(mean([loc1,loc2],1));
                    center=[center;[loc,f_pre(loc(1),loc(2)),m(j),i0]];
                end
            end 
            %{
            for w=1:L2_num
                [loc1,loc2]=find(L2==w);
                if(length(loc1)>=1)
                    loc=round(mean([loc1,loc2],1));
                    loc_2(1)=loc(1)+ts_count(i);
                    loc_2(2)=loc(2);
                    center=[center;[loc_2,f12_pre(loc(1),loc(2)),m(j),i0]];
                end
            end      \
            %}       
            i0=i0-step;
        end
    end
    del=[];
    for u=1:size(center,1)
        for v=u+1:size(center,1)
            d=norm(center(u,1:2)-center(v,1:2));
            if(d<size(tf,2)/10)
                if(center(u,3)>center(v,3))
                    del=[del v];
                else
                    del=[del u];
                end
            end
        end
    end
    del=sort(unique(del));
    for u=length(del):-1:1
        center(del(u),:)=[];
    end
    for u=1:size(center,1)
        m_num=find(m==center(u,4));
        %rectangle('Position',[center(p,2),center(p,1),size(tmplate{m_num},1)*center(p,5),size(tmplate{m_num},2)*center(p,5)],'LineWidth',1.5,'EdgeColor','r');
        rectangle('Position',[center(u,2),center(u,1),size(tmplate{m_num},1)*center(u,5),size(tmplate{m_num},2)*center(u,5)],'LineWidth',1.5,'EdgeColor','r');
        text(center(u,2)+round(size(tmplate{m_num},1)*center(u,5)/2),center(u,1)+round(size(tmplate{m_num},2) * center(u,5)/2),num2str(center(u,4)),'horiz','left','color','r');
    end
end



