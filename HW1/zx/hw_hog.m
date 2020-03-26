
clc
clear all
%% load Ä£°å
m=[0,1,2,3,4,6,8,9];
%m=[1,8];
mL=length(m);
for i=1:mL
    file_name=['train/' num2str(m(i)) '.bmp'];
    data=double(imread(file_name));
    avg=mean(mean(data));
    data=(data>avg);
    tmplate{i}=data;
    %imshow(data);
end

%% load ²âÊÔÎÄµµ  1~9
t=7:8;
tL=length(t);
d=16;

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
    ts{i}=file2;
	[a,b]=size(file2);
	gray_count1=zeros(a,1);
	for k=1:a
		gray_count1(k)=sum(sum(double(file2(k,:))));
	end
	[aa,bb]=find(gray_count1==min(gray_count1));
	gray_count11=gray_count1(aa:end,:);
	[cc,dd]=find(gray_count11==max(gray_count11));
    cc=max(cc);
	m_tf=file2(1:cc+aa-2,:);
	n_tf=file2(cc+aa-1:end,:);
    
	ts_m{i}=m_tf;
	ts_n{i}=n_tf;
	ts_count(i)=cc+aa-2;
	
    %imshow(file2);
    %{
    avg=mean(mean(mean(data)));
    data=mean(data,3);
    data=(data>0.9*avg);
    
    imshow(data);
    ts{i}=data;
    %}
end

%% Ä£°åÆ¥Åä
e=0.0001;
step=0.5;
th=0.95;
for i=1:tL
    tf=ts_m{i};
    tf_size=size(tf);
    
    tf2=ts_n{i};
    tf_size2=size(tf2);
    
    figure;
    imshow(ts{i});
    center=[];
    for j=1:mL
        m0=tmplate{j};
        m_size=size(m0);
        i0=2;
        for k=1:3
            tpl=interp2(1:m_size(2),1:m_size(1),double(m0),linspace(1,m_size(2),round(m_size(2)*i0)),linspace(1,m_size(1),round(m_size(1)*i0))','Linear');
            [hog_tpl, vis_tpl]=extractHOGFeatures(tpl,'CellSize',[4 4]);
            move_1=tf_size(1)-size(tpl,1)+1;
            move_2=tf_size(2)-size(tpl,2)+1;
            count=zeros(move_1,move_2);
            
            move_11=tf_size2(1)-size(tpl,1)+1;
            move_22=tf_size2(2)-size(tpl,2)+1;
            count2=zeros(move_11,move_22);
            
            for u=1:move_1
                for v=1:move_2
                    test_area=tf(u:u+size(tpl,1)-1,v:v+size(tpl,2)-1);
                    [hog_test, vis_test]=extractHOGFeatures(test_area,'CellSize',[4 4]);
                    %count(u,v)=sum(sum(and(test_area,tpl)))/((e+norm(tpl))*(e+norm(test_area)));
                    count(u,v)=sum(sum(hog_test.*hog_tpl))/((e+norm(hog_tpl))*(e+norm(hog_test)));
                    %{
                    if(count(u,v)>0.95&&j==2)
                        imshow(and(test_area,tpl))
                    end
                    %}
                end
            end
            
            for u=1:move_11
                for v=1:move_22
                    test_area=tf2(u:u+size(tpl,1)-1,v:v+size(tpl,2)-1);
                    
                    [hog_test, vis_test]=extractHOGFeatures(test_area,'CellSize',[4 4]);
                    %count(u,v)=sum(sum(and(test_area,tpl)))/((e+norm(tpl))*(e+norm(test_area)));
                    count(u,v)=sum(sum(hog_test.*hog_tpl))/((e+norm(hog_tpl))*(e+norm(hog_test)));
                    %count(u,v)=sum(sum(and(test_area,tpl)))/((e+norm(tpl))*(e+norm(test_area)));
                    count2(u,v)=sum(sum(test_area.*tpl))/((e+norm(tpl))*(e+norm(test_area)));
                    %{
                    if(count(u,v)>0.95&&j==2)
                        imshow(and(test_area,tpl))
                    end
                    %}
                end
            end 
            
            count_th=th*max(max(count));
            count_pre=count;
            count=(count>count_th);
            
            count2_th=th*max(max(count2));
            count2_pre=count2;
            count2=(count2>count2_th);
            
            [L,L_num]=bwlabel(count,8);
            [L2,L2_num]=bwlabel(count2,8);
            
            for w=1:L_num
                [loc1,loc2]=find(L==w);
                if(length(loc1)>=1)
                    loc=round(mean([loc1,loc2],1));
                    center=[center;[loc,count_pre(loc(1),loc(2)),m(j),i0]];
                end
            end 
            
            for w=1:L2_num
                [loc1,loc2]=find(L2==w);
                if(length(loc1)>=1)
                    loc=round(mean([loc1,loc2],1));
                    loc_2(1)=loc(1)+ts_count(i);
                    loc_2(2)=loc(2);
                    center=[center;[loc_2,count2_pre(loc(1),loc(2)),m(j),i0]];
                end
            end             
            i0=i0*step;
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



        
    
    




    

    
