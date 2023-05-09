function fx = Bickley1(t, x)
% function fx = Bickley1(x)
%
% Computes the rhs of the quasiperidically driven Bickley jet in the state
% space augmented by the time component ("autonomizes" the non-autonomous
% system).
%
% Input:    x:  2 x k array of input vectors (column wise)
%           t: scalar value
% Output:   fx: 2 x k array of rhs evaluations (column wise)
%
% System taken from:
% Alireza Hadjighasem, Daniel Karrasch, Hiroshi Teramoto, George Haller:
% A Spectral Clustering Approach to Lagrangian Vortex Detection, 2015

% parameters
U0 = 5.4138;                        % Mm/day
L0 = 1.77;                          % Mm
r0 = 6.371;                         % Mm
c1 = 0.1446*U0;
c2 = 0.2053*U0;
c3 = 0.4561*U0;
% lx = 6.371e6*pi;
% ly = 1.77e6;
ep1 = 0.075;
ep2 = 0.4;
ep3 = 0.3;

k1 = 2/r0;
k2 = 4/r0;
k3 = 6/r0;

% temporary variables
this = (ep1*exp(-1i*k1*c1*t).*exp(1i*k1*x(1)) + ep2*exp(-1i*k2*c2*t).*exp(1i*k2*x(1)) + ep3*exp(-1i*k3*c3*t).*exp(1i*k3*x(1)))
forcing = real(ep1*exp(-1i*k1*c1*t).*exp(1i*k1*x(1)) + ep2*exp(-1i*k2*c2*t).*exp(1i*k2*x(1)) + ep3*exp(-1i*k3*c3*t).*exp(1i*k3*x(1)));
dxforcing = real(ep1*exp(-1i*k1*c1*t)*1i*k1.*exp(1i*k1*x(1)) + ep2*exp(-1i*k2*c2*t)*1i*k2.*exp(1i*k2*x(1)) + ep3*exp(-1i*k3*c3*t)*1i*k3.*exp(1i*k3*x(1)));

syL = sech(x(2)/L0).^2;

fx = [U0*syL + 2*U0*tanh(x(2)/L0).*syL.*forcing, U0*L0*syL.*dxforcing]';
