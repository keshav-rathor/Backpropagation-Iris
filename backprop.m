close all; 
clear all; 
clc
%% load divided input data set
load fisheriris
% coding (+1/-1) of 3 classes
a = [-1 -1 +1]';
b = [-1 +1 -1]';
c = [+1 -1 -1]';
% define training inputs
rand_ind = randperm(50);
trainSeto = meas(rand_ind(1:35),:);
trainSeto=trainSeto';
trainVers = meas(50 + rand_ind(1:35),:);
trainVers=trainVers';
trainVirg = meas(100 + rand_ind(1:35),:);
trainVirg=trainVirg';
trainInp = [trainSeto trainVers trainVirg];
% define targets
tmp1 = repmat(a,1,length(trainSeto));
tmp2 = repmat(b,1,length(trainVers));
tmp3 = repmat(c,1,length(trainVirg));
T = [tmp1 tmp2 tmp3];
%% network training
trainCor = zeros(10,10);
valCor = zeros(10,10);
Xn = zeros(1,10);
Yn = zeros(1,10);
for k = 1:10,
Yn(1,k) = k;
for n = 1:10,
Xn(1,n) = n; 
net = newff(trainInp,T,[k n],{},'trainbfg'); 
net = init(net); 
net.divideParam.trainRatio = 1; 
net.divideParam.valRatio = 0; 
net.divideParam.testRatio = 0; 
net.trainParam.show = NaN; 
net.trainParam.max_fail = 2; 
rand_ind = randperm(50);
valSeto = meas(rand_ind(1:20),:);
valSeto= valSeto';
valVers = meas(50 + rand_ind(1:20),:);
valVers=valVers';
valVirg = meas(100 + rand_ind(1:20),:);
valVirg=valVirg';
valInp = [valSeto valVers valVirg]; 
VV.P = valInp; 
tmp1 = repmat(a,1,length(valSeto));
tmp2 = repmat(b,1,length(valVers));
tmp3 = repmat(c,1,length(valVirg));
valT = [tmp1 tmp2 tmp3];
net = train(net,trainInp,T,[],[],VV);%,TV); 
Y = sim(net,trainInp);
[Yval,Pfval,Afval,Eval,perfval] = sim(net,valInp,[],[],valT);
% calculate [%] of correct classifications
trainCor(k,n) = 100 * length(find(T.*Y > 0)) / length(T);
valCor(k,n) = 100 * length(find(valT.*Yval > 0)) / length(valT);
end
end
figure
surf(Xn,Yn,trainCor/3);
view(2)
figure
surf(Xn,Yn,valCor/3);
view(2)
%% final training
k = 3;
n = 3;
fintrain = [trainInp valInp];
finT = [T valT];
net = newff(fintrain,finT,[k n],{},'trainbfg');
net.divideParam.trainRatio = 1;
net.divideParam.valRatio = 0;
net.divideParam.testRatio = 0;
net = train(net,fintrain,finT);
finY = sim(net,fintrain);
finCor = 100 * length(find(finT.*finY > 0)) / length(finT);
fprintf('Num of neurons in 1st layer = %d\n',net.layers{1}.size)
fprintf('Num of neurons in 2nd layer = %d\n',net.layers{2}.size)
fprintf('Correct class = %.3f %%\n',finCor/3)
%% Testing
rand_ind = randperm(50);
testSeto = meas(rand_ind(36:50),:);
testSeto=testSeto';
testVers = meas(50 + rand_ind(36:50),:);
testVers=testVers';
testVirg = meas(100 + rand_ind(36:50),:);
testVirg=testVirg';


% define test set
testInp = [testSeto testVers testVirg];
testT = [repmat(a,1,length(testSeto)) repmat(b,1,length(testVers))
repmat(c,1,length(testVirg))];
testOut = sim(net,testInp);
testCor = 100 * length(find(testT.*testOut > 0)) / length(testT);
fprintf('Correct class = %.3f %%\n',testCor/3)
% plot targets and network response
figure;
plot(testT')
xlim([1 21])
ylim([0 2])
set(gca,'ytick',[1 2 3])
hold on
grid on
plot(testOut','r')
legend('Targets','Network response')
xlabel('Sample No.')