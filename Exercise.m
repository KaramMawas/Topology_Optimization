% Topology Exercise 2
%Worked By Karam MAWAS MTN(2946939)

close all
clear all
clc


% initializing
n = 21;
x = -1:0.1:1; y=3*x +2 ;
original_x=x;original_y=y;


y = y+0.2*randn(n,1)'; x = x+0.2*randn(n,1)';
% scatter (x,y)
x=x'; y=y';

% applying the leaest square method.
k=length(x);

%Linear Regrission
A_L=[x ones(k,1)];
Delta_X_L_hat=A_L\y;
Y_L_hat=A_L*Delta_X_L_hat;
e_L_hat=y-Y_L_hat;
square_sum_L=e_L_hat'*e_L_hat;

%Compute r(redundancy)
n=length(Delta_X_L_hat);
r=k-n;

Scaled_Square_L=square_sum_L/r;

%The Test of Orthogonality
test = A_L'*e_L_hat;
m = round(test *1e+8)*1e-8;

if m == 0
    display('The Test of Orthogonality of A_Model is True')
else
    display('The Test of Orthogonality of A_Model is False')
end


%Plot Polynomial

% for estimated points

Y_L_Points=Delta_X_L_hat(2,:)+Delta_X_L_hat(1,:)*x(:,1);

%for linspace (for the Polynomial)
X_linspace=linspace(min(x),max(x),100)';

Y_L_Polynomial=Delta_X_L_hat(2,:)+Delta_X_L_hat(1,:)*X_linspace;

a = 40;

% For the Line

% subplot(1,2,1)
scatter(x,y,'*','r') %Plot the Data Points
hold on
scatter(x,Y_L_Points,a,'MarkerEdgeColor',[0 .5 .5],...
              'MarkerFaceColor',[0 .7 .7],...
              'LineWidth',1.5)        %for estimated points  

plot(X_linspace,Y_L_Polynomial,'-.') %for polynomial

grid on
for i=1:length(x)
    line([x(i) x(i)],[y(i) Y_L_Points(i)])
end
xlabel('X')
ylabel('Y')
str=sprintf('Line regression by the Least Square Method');
title(str);
l = legend('Data Points','estimated points','polynomial','estimated residuals');
set(l,'Location','NorthWest')
% scatter(original_x,original_y,'g+')
%% Solving the problem by Total Least Square

A_augmented = [A_L y];
[U,S,V] = svd(A_augmented);
S_hat = S;
S_hat(3,3)=0;
A_augmented_hat = U*S_hat*V';
A_delta = A_augmented_hat - A_augmented;

% x_hat
xx_hat_TLS = reshape(V(:,3),3,1);
xx_hat_TLS = xx_hat_TLS./(-xx_hat_TLS(3));

% y_hat or l hat 
y_hat_TLS = A_augmented_hat(:,1:2)*xx_hat_TLS(1:2,1);

% x_hat
x_hat_TLS = (y_hat_TLS - xx_hat_TLS(2))/xx_hat_TLS(1);

% errorneous vector
e_hat_TLS = y - y_hat_TLS;
square_sum_TLS = e_hat_TLS'*e_hat_TLS;


%Plot Polynomial

% for estimated points


a = 40;

% For the Line

% subplot(1,2,1)
figure
scatter(x,y,'*','r') %Plot the Data Points
hold on
scatter(x_hat_TLS,y_hat_TLS,a,'MarkerEdgeColor',[0 .5 .5],...
              'MarkerFaceColor',[0 .7 .7],...
              'LineWidth',1.5)        %for estimated points  
          
Y_Polynomial=xx_hat_TLS(2)+xx_hat_TLS(1)*X_linspace;
plot(X_linspace,Y_Polynomial,'-.') %for polynomial

grid on
for i=1:length(x)
    line([x(i) x_hat_TLS(i,1)],[y(i) y_hat_TLS(i)]);
end
xlabel('X')
ylabel('Y')
str=sprintf('Line regression by the Total Least Square Method');
title(str);
l = legend('Data Points','estimated points','polynomial','estimated residuals');
set(l,'Location','NorthWest')
%% Gauﬂ-Helmert model

% computing the taylor points for the line regression (a,b)
a_TP = (original_y(1)-original_y(2))/(original_x(1)-original_x(2));
b_TP = (original_y(2)*original_x(1)-original_y(1))/(original_x(1)-original_x(2));
% e_x_TP = 0; e_y_TP = 0;
% b_TP = b_TP*ones(length(x),1); a_TP = a_TP*ones(length(x),1);
e_x_TP = zeros(length(x),1); e_y_TP = zeros(length(x),1);
e_TP = [e_x_TP; e_y_TP];



Nor=inf ;eps=1e-20;
Counter=0 ; iteration=100 ;

r = zeros(2*length(x),1);t = zeros(2,1);

while Nor>eps && iteration>Counter


    
    A = -1*[ones(length(x),1) x(:)-e_TP(1:length(x),1)];
    B = [b_TP*eye(length(x),length(x)) -1*eye(length(x),length(x))];
    W = y(:) - a_TP - b_TP*x(:);
    
    K = [eye(2*length(x),2*length(x)) zeros(2*length(x),2) B';
        zeros(2,2*length(x)) zeros(2,2) A';
        B A zeros(length(x),length(x))];
    
   vector_hat = inv(K)*[r;t;-1*W]; 
   
   % 1st. cond.
   e_hat = vector_hat(1:length(x));
   e_hat = e_hat - e_TP(1:length(x));
   Nor=norm(e_hat);
   
   % 2nd. cond.
   delta_xx_hat_GH = vector_hat(2*length(x)+1:2*length(x)+2);
   delta_xx_hat_GH_cond = norm(delta_xx_hat_GH);
   
   e_TP = e_TP(1:length(x),1) - e_hat(1:length(x),1);
     
    Counter=Counter+1;
     
end
e_x_GH = vector_hat(1:length(x)); 
e_y_GH = vector_hat(length(x)+1:2*length(x)); 
delta_xx_hat_GH = vector_hat(2*length(x)+1:2*length(x)+2);
lagrange = vector_hat(2*length(x)+3:3*length(x)+2);

% Extracting the observations on x and y and unkown(a,b)
y_hat_GH = y(:) - e_y_GH; x_hat_GH = x(:) - e_x_GH;
xx_hat_GH = delta_xx_hat_GH + [a_TP;b_TP];

%Plot Polynomial

% for estimated points


a = 40;

% For the Line

% subplot(1,2,1)
figure
scatter(x,y,'*','r') %Plot the Data Points
hold on
scatter(x_hat_GH,y_hat_GH,a,'MarkerEdgeColor',[0 .5 .5],...
              'MarkerFaceColor',[0 .7 .7],...
              'LineWidth',1.5)        %for estimated points  
          
Y_Polynomial=xx_hat_GH(2)+xx_hat_GH(1)*X_linspace;
plot(X_linspace,Y_Polynomial,'-.') %for polynomial

grid on
for i=1:length(x)
    line([x(i) x_hat_GH(i,1)],[y(i) y_hat_GH(i)]);
end
xlabel('X')
ylabel('Y')
str=sprintf('Line regression by Gauﬂ Helmert Model');
title(str);
l = legend('Data Points','estimated points','polynomial','estimated residuals');
set(l,'Location','NorthWest')