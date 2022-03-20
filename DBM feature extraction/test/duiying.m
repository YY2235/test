clear all
GG=xlsread('d:/E/shujuku/DBM feature extraction/yundongshengxue/data.csv');
A=xlsread('d:/E/shujuku/DBM feature extraction/yundongshengxue/indices.csv');
melq=zeros(1127,150);
for i=1:1127
    ind=find(A==i-1);
    melq(i,:)=GG(ind,:);
end
% save('D:/E/shujuku/SAVEE/mel45图/melq.mat','melq'); 
% save('D:/E/shujuku/SAVEE/mel45图/mellabelsq.mat','mellabelsq'); 
csvwrite('d:/E/shujuku/DBM feature extraction/yundongshengxue/data1.csv',melq); 
% save('D:/E/shujuku/DBM feature extraction/saver/deep_bolin_mel_200_150_qzl.mat','melq'); 
