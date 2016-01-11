function [B_hat,C,L,mu] = proxgrdopt(X,Y,g,lambda,eps,n_iter) % B_hat = proxgrdopt(X,y,g,lambda,eps,n_iter)

% Process supplied inputs
[nr_X,nc_X]=size(X);
[nr_Y,nc_Y]=size(Y);
[~,nc_g]=size(g);

% Basic checks for the size of matrices
if nr_X ~= nr_Y
    fprintf('Error: number of rows in X and Y are not same.\n');
    return
elseif nc_Y > 1
    fprintf('Error: number of columns in Y must be 1.\n');
    return
elseif nc_g~=nc_X
    fprintf('Error: number of columns in X and g must be same.\n');
    return
end

% Initialization
% Step 1: Constructing C matrix from eq(4)
    [~,nc_X]=size(X);  % number of input variables
    [ng,~]=size(g); % number of groups (rows) ng = 98
    n_el_g=sum(g,2); % number of non-zero elements in each group (rows) = 2450
    wg=sqrt(n_el_g); % weights for each group(rows)
    C=zeros(sum(n_el_g),nc_X); % initialize C matrix filled with zeros of size 2450x50
    k=0;
    
    for i=1:ng      % for ith group ng = 98
        nz_index=find(g(i,:)==1); % index of non-zero elements in each group (1-49-1)
        nel_nz_index=numel(nz_index);

        for j=1:nel_nz_index   % for jth element in group
           k=k+1;
           C(k,nz_index(j))=wg(i);    % Assigning wg values to non zero elements     
        end
   end
    
       C=C*lambda; % Final C-matrix
       
% Step 2: Compute L from eq (10)
    d= size(g);
    mu=eps/(2*sqrt(d(:,1)));
    gamma_func=lambda*max(sqrt(sum((C/lambda).^2)),[],2); % from eq 8
    L=max(eig(X'*X))+gamma_func^2/mu; % from eq 10
    w0=zeros(nc_X,1); % fix set of coefficients 
    wt=w0; 
    
% Iterations    
    zt=0; 
for t=0:n_iter-1 % number of iterations set in test code
% Step 1: Compute gradient from step 1 of pseudocode
    % Step 1a: Compute shrinkage operator S from lemma 1
    u=C*wt;
    u_l2_norm=norm(u,2);% maximum singular value of matrix
    if u_l2_norm>1
        S_u=u/u_l2_norm;
    elseif u_l2_norm<=1
        S_u=u;
    end
    alpha_star=S_u;
    df_wt=X'*(X*wt-Y)+C'*alpha_star; % from lemma 2
   
% Step 2: Perform gradient descent step 2 of pseudocode
    Bt=wt-(1/L)*df_wt;
% Step 3: Set zt from step 3 of pseudocode  
    zt=-(1/L)*((t+1)/2)*df_wt+zt;
% Step 4: Set wt from step 4 of pseudocode
    wt=((t+1)/(t+3))*Bt+(2/(t+3))*zt;
end

    B_hat=Bt;    
end
