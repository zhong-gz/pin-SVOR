clc
close all
clear all

x=1:5;
color =linspecer(size(x,2));
marker=['o'; '^'; 's';'d';'p'];
markersize = 10;
linewidth = 1.5;
fz=12;

n=1; %mae
m=300;
y_1=[0.002500 0.002533 0.002917 0.003250 0.003583];
y_2=[0.003167 0.003333 0.004000 0.005000 0.005667];
y_3=[0.003250 0.003500 0.004250 0.005000 0.005583];
y_4=[0.003167 0.003750 0.004417 0.005000 0.005417];
y_5=[0.003767 0.003833 0.004500 0.005250 0.005583];
y = [y_1;y_2;y_3;y_4;y_5];
h1 = figure(1);
visulization(x,y,m,color,marker,markersize,linewidth,n,fz);
set(gcf,'Position',[50 550 600 400]);

% n=2; %mze
% m=300;
% y_1=[0.002500 0.002533 0.002917 0.003250 0.003583];
% y_2=[0.003167 0.003333 0.004000 0.005000 0.005667];
% y_3=[0.003250 0.003500 0.004250 0.005000 0.005583];
% y_4=[0.003167 0.003750 0.004417 0.005000 0.005417];
% y_5=[0.003767 0.003833 0.004500 0.005250 0.005583];
% y = [y_1;y_2;y_3;y_4;y_5];
% h2 = figure(2);
% visulization(x,y,m,color,marker,markersize,linewidth,n,fz);
% set(gcf,'Position',[800 550 600 400]);

n=1; %mae
m=600;
y_1=[0.002750 0.002833 0.003083 0.003750 0.004167 ];
y_2=[0.004000 0.003667 0.003667 0.004167 0.005333 ];
y_3=[0.003083 0.003500 0.003917 0.004333 0.005000 ];
y_4=[0.003500 0.003833 0.004250 0.005000 0.005417 ];
y_5=[0.003583 0.004000 0.004333 0.004917 0.005583 ];
y = [y_1;y_2;y_3;y_4;y_5];
h3 = figure(3);
visulization(x,y,m,color,marker,markersize,linewidth,n,fz);
set(gcf,'Position',[50 300 600 400]);

% n=2; %mae
% m=600;
% y_1=[0.002750 0.002833 0.003083 0.003750 0.004167 ];
% y_2=[0.004000 0.003667 0.003667 0.004167 0.005333 ];
% y_3=[0.003083 0.003500 0.003917 0.004333 0.005000 ];
% y_4=[0.003500 0.003833 0.004250 0.005000 0.005417 ];
% y_5=[0.003583 0.004000 0.004333 0.004917 0.005583 ];
% y = [y_1;y_2;y_3;y_4;y_5];
% h4 = figure(4);
% visulization(x,y,m,color,marker,markersize,linewidth,n,fz);
% set(gcf,'Position',[800 300 600 400]);

n=1; %mae
m=1500;
y_1=[0.002000 0.002000 0.002500 0.004000 0.004667 ];
y_2=[0.003333 0.004167 0.005833 0.008333 0.012500 ];
y_3=[0.002833 0.003167 0.005000 0.007000 0.010333 ];
y_4=[0.002500 0.003333 0.004333 0.005333 0.008833 ];
y_5=[0.003000 0.003667 0.003333 0.004833 0.007500 ];
y = [y_1;y_2;y_3;y_4;y_5];
h5 = figure(5);
visulization(x,y,m,color,marker,markersize,linewidth,n,fz);
set(gcf,'Position',[50 100 600 400]);

% n=2; %mae
% m=1500;
% y_1=[0.002000 0.002000 0.002500 0.004000 0.004667 ];
% y_2=[0.003333 0.004167 0.005833 0.008333 0.012500 ];
% y_3=[0.002833 0.003167 0.005000 0.007000 0.010333 ];
% y_4=[0.002500 0.003333 0.004333 0.005333 0.008833 ];
% y_5=[0.003000 0.003667 0.003333 0.004833 0.007500 ];
% y = [y_1;y_2;y_3;y_4;y_5];
% h6 = figure(6);
% visulization(x,y,m,color,marker,markersize,linewidth,n,fz);
% set(gcf,'Position',[800 100 600 400]);

function [h] = visulization(x,y,m,color,marker,markersize,linewidth,n,fz)
    for i = 1:size(x,2)
        plot(x, y(i,:),'LineWidth',linewidth,'LineStyle','-','MarkerSize',markersize,...
            'Marker',marker(i),'Color',color(i,:),'MarkerEdgeColor','k',...
            'MarkerFaceColor',color(i,:));
        hold on
    end
    hold on

    temp_up=max(max(y));
    temp_down=min(min(y));
    t=temp_up-temp_down;    
    if n ==1
        n='MAE';
        ylim([temp_down-0.05*t temp_up+0.05*t]); 
    else
        n='MZE';
        ylim([temp_down-0.3*t temp_up+0.3*t]); 
    end
    
    xlim([0.5 5.5]); 
    plot([5.49 5.49],[temp_down-0.5*t temp_up+0.5*t],'k','LineWidth',1);
    hold on
    xlabel('Ratio of Noise','FontSize',fz); 
    ylabel(n,'FontSize',fz);
    title(['m=',num2str(m)],'FontSize',fz)
    legend({'pin-SVOR','SVOR-EXC','SVOR-IMC','RED-SVM','KDLOR'},'Location','northwest','FontSize',fz); %Í¼ÀýµÄÉèÖÃ
    set(gca,'xtick',1:5);
    set(gca,'xticklabel',{'0','0.1','0.2','0.3','0.4'}); 
    grid on
    box on
    hold off
end