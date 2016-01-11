load('dataset.mat') % load dataset into workspace 
X=Xtrain;           %Changing name of Xtrain to X
Y=Ytrain;           %Changing name of Ytrain to Y
X_test=Xtest;       %Changing name of Xtest to X_test
Y_test=Ytest;       %Changing name of Ytest to Y_test
g=groups;

[~,nc_X]=size(X);   % Number of input variables (size)
[ng,~]=size(g);     % Number of groups (size)


lambda=0.07;    % Calculated by taking different values on  Xtrain and checking error
eps=1e-1;       % User defined accuracy (epsilon)
n_iter=100;     % Number of iterations for computing gradient step
[B_hat,C,L,mu]=proxgrdopt(X,Y,g,lambda,eps,n_iter); % Perform proximal-gradient-optimization
Y_pred=X_test*B_hat;    % Predicting Y using X_test as input

error=(Y_test-Y_pred);  % Error (Given y - Found y)
SS_res=sum(error.^2);   % Sum of squares of residuals
SS_tot=sum((Y_test-mean(Y_test)).^2); % sum of squares total
R_sqr=1-SS_res/SS_tot;  % Prediction accuracy R-squared
% Plotting results of Xtest against the results we got
subplot(1,2,1)
plot(Y_test,Y_test,'-r',Y_test,Y_pred,'o','linewidth',2);
    set(gca,'fontsize',10);
    xlabel('y, response');
    ylabel('y, response');
    title(strcat('R^2= ',num2str(R_sqr)));
    legend('Y_{test} vs. Y_{test}','Y_{test} vs. Y_{pred}','location','northwest');
    axis square
    grid on
subplot(1,2,2)
plot(Y_test,error,'om','linewidth',2);
    set(gca,'fontsize',10); % gca is used to get current axis on which graph is drawn.
    xlabel('y_{test}');
    ylabel('prediction error/residuals');
    axis square
    grid on