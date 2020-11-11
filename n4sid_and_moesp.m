%% 2.160 FA20 PS5 Problem 2
% Authors: H. Harry Asada and Nicholas S. Selby
% 
% We aim to apply subspace system identification algorithms, MOESP and/or
% N4SID, to a realistic data set. You first create data from a given true system, and then
% apply the subspace algorithms. So, you know the correct answer.

clear;clc;close all;rng(1)

%% (a)
% The system we consider is a fourth-order, discrete-time, linear time-invariant system.
% The system matrices of the true system are given by
A = [0.603 0.603 0 0;-0.603 0.603 0 0;0 0 -0.603 -0.603;0 0 0.603 -0.603];
B = [1.1650,-0.6965;0.6268 1.6961;0.0751,0.0591;0.3516 1.7971];
C = [0.2641,-1.4462,1.2460,0.5774;0.8717,-0.7012,-0.6390,-0.3600];
D = [-0.1356,-1.2704;-1.3493,0.9846];

% Note that the system is a 2-input, 2-output system. We take White noise random
% sequences of 1,000 points as input u. In MatLab,
N = 1000;
u = randn(N,2);

% Create input and output data sequences.
DIM_X = length(A);
[DIM_Y,DIM_U] = size(D);
TS = 1;

sys = ss(A,B,C,D,TS);
[y,t,x] = lsim(sys,u,0:(N-1));

figure
for i=1:DIM_X
    subplot(2,DIM_X,i)
    plot(t,x(:,i),'k')
    xlabel('t')
    ylabel(['x_' num2str(i)])
end
for i=1:2
    subplot(2,DIM_X,5+i)
    plot(t,y(:,i),'k')
    xlabel('t')
    ylabel(['y_' num2str(i)])
end
sgtitle('(a) Input and Output Data Sequences')

[M,P,W] = deal(cell(DIM_U,1));
for iu=1:DIM_U
    [M{iu}, P{iu}, W{iu}] = dbode(A,B,C,D,TS,iu);
end

%% (b) 
% Apply N4SID and/or MOESP algorithms to the data generated. You need to specify
% an approximate system order when using these algorithms. Try out different system
% orders, and discuss the results.
data = iddata(y,u,1);
for nx = 1:4
    % N4SID
    sys_n4sid = n4sid(data,nx);
    figure
    for iu=1:DIM_U
        [m,p,w] = dbode(sys_n4sid.A,sys_n4sid.B,sys_n4sid.C,sys_n4sid.D,1,iu);
        for iy=1:DIM_Y
            subplot(2*DIM_Y, DIM_U, 2*(iy-1)*DIM_U+iu)
            loglog(W{iu},M{iu}(:,iy),'k--', w,m(:,iy),'k')
            if iu==1, ylabel('Mag.'); end
            subplot(2*DIM_Y, DIM_U, 2*(iy-1)*DIM_U+iu+DIM_U)
            semilogx(W{iu},P{iu}(:,iy),'k--', w,p(:,iy),'k')
            if iy==DIM_Y, xlabel('\omega'); end
        end
    end
    sgtitle(['N4SID Bode Plots for NX=' num2str(nx)])
    
    % MOESP
    [~,sys_moesp] = moesp(y,u,4);
    [a,b,c,d]=sys_moesp(nx);
    figure
    for iu=1:DIM_U
        [m,p,w] = dbode(a,b,c,d,1,iu);
        for iy=1:DIM_Y
            subplot(2*DIM_Y, DIM_U, 2*(iy-1)*DIM_U+iu)
            loglog(W{iu},M{iu}(:,iy),'k--', w,m(:,iy),'k')
            if iu==1, ylabel('Mag.'); end
            subplot(2*DIM_Y, DIM_U, 2*(iy-1)*DIM_U+iu+DIM_U)
            semilogx(W{iu},P{iu}(:,iy),'k--', w,p(:,iy),'k')
            if iy==DIM_Y, xlabel('\omega'); end
        end
    end
    sgtitle(['MOESP Bode Plots for NX=' num2str(nx)])
end

%% (c)
% Since state space representation is not unique, the system matrices obtained in part b)
% may not agree with the true system in part a). For the purpose of verifying the identified
% system with system matrices, A, B, C, and D, obtain a 2 by 2 transfer matrix representing
% the input-output relationship of the system in part b) and compare it to the transfer
% function associated with the system in part a).
syms q
transfer_matrix = @(A,B,C,D) C*inv(q*eye(size(A))-A)*B+D;

tm_a = transfer_matrix(A,B,C,D)
tm_n4sid = transfer_matrix(sys_n4sid.A,sys_n4sid.B,sys_n4sid.C,sys_n4sid.D)
tm_moesp = transfer_matrix(a,b,c,d)