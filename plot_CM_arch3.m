clear all;
clc;
close all;
load('cm_arch3.mat');
rm_dia = cm-diag(diag(cm));
axes = plot(sum(rm_dia,2),'*','markers',8);
classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 24, 25, 27, 28, 30, 32, 35, 36, 40, 42, 45, 48, 49, 54, 56, 63, 64, 72, 81];
xlabel('Class Index','fontweight','bold','fontsize',16);
title('Wrong predictions by Arch3','fontweight','bold','fontsize',16);
ylabel('Number of wrong predictions','fontweight','bold','fontsize',16);
x = [1:40];
dx = 0.2; dy = 0.2; % displacement so the text does not overlay the data points
b = strings(1,40);
s = sum(rm_dia,2);
for i =1:40
    b(i)=['(' num2str(classes(i)) ',' num2str(s(i)) ')'];
end

c = cellstr(b);
text(x+dx, sum(rm_dia,2)+dy, c,'FontSize', 14);
set(gca,'fontsize',20)