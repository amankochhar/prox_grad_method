load('dataset.mat') % load dataset into workspace 
X=Xtrain;           
Y=Ytrain;           
X_test=Xtest;      
Y_test=Ytest;      
g=groups;

[~,nc_X]=size(X);  
[ng,~]=size(g);    


lambda=0.07;    
eps=1e-1;       % User defined accuracy (epsilon)
n_iter=100;     % Number of iterations for computing gradient step
[B_hat,C,L,mu]=proxgrdopt(X,Y,g,lambda,eps,n_iter);
Y_pred=X_test*B_hat;    

error=(Y_test-Y_pred);  
SS_res=sum(error.^2);   
SS_tot=sum((Y_test-mean(Y_test)).^2); 
R_sqr=1-SS_res/SS_tot;  

% Plotting results of Xtest against the results we get
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
    set(gca,'fontsize',10); 
    xlabel('y_{test}');
    ylabel('prediction error/residuals');
    axis square
    grid on
