clc;
clear al;
close all;

%%
Ts = 0.1;
Nsim = 30; %sec
tvec = 0:Ts:Nsim;

% Laplace Domain Transfer function of the system
G = tf(4,[1,3,10]);
yr = tf(2,[1,0]);
yrt = 2*ones(length(tvec));

g1 = 0.5; % g1 = Ap/Ai
g2 = 0.05; % g2 = Ad/Ai

G1 = tf([g2*4 g1*4 4],[1 3 10 0]);
rlocus(G1)

Ai = 10; % from root locus plot

% system step response without controller
figure;
step(G)
title("Step Response of G(s) = 4/(s^2 + 3s + 10)")


C = Ai*(g1 + tf(1,[1,0]) + g2*tf([1,0],1));
% final system output in Laplace domain
y = (C*G/(1 + C*G))*yr;

syms t s H(s)
[Num,Den] = tfdata(y);
y_syms = poly2sym(cell2mat(Num),s)/poly2sym(cell2mat(Den),s);
H(s) = y_syms;
Hpf = partfrac(H);
h(t) = vpa(ilaplace(Hpf,s,t));
y_time = h(t);

previous_error = 0;
integral = 0;

for i=1:length(tvec)
    y_store(i) = subs(y_time,t,tvec(i));
    err(i) = yrt(i) - y_store(i);
    proportional = err(i);
    integral = integral + err(i)*Ts;
    derivative = (err(i) - previous_error) / Ts;
    output = (Ai*g1)*proportional+Ai*integral+(Ai*g2)*derivative;
    previous_error = err(i);
end

figure;
plot(tvec,y_store)
title("Y(s) = 4/(s^2 + 3s + 10) tracking yr(t) = 2")
xlim([0,6])
grid on

figure;
plot(tvec,err)
title("error")
