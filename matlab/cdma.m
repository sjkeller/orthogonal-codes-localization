
figure('DefaultAxesFontSize',16)

fs = 1e6; % sampling freq
T = 100e-3; % length of the sequence (probe signal) in seconds
r = 15000; % rate
% filter
rolloff = 1/8;


%%
% method for code generation
codeMethod = "gold";

M = r*T;    % number of symbols per sequence
%corr peak increases with code length (?)
% cinit seed used for localiszation
%cinit = 1;
cinit = 1;
anchor1 = genBasebandSig(cinit,M,T,fs,rolloff,codeMethod);

%cinit = 9;
cinit = 9;
anchor2 = genBasebandSig(cinit,M,T,fs,rolloff,codeMethod);

%cinit = 11;
cinit = 11;
anchor3 = genBasebandSig(cinit,M,T,fs,rolloff,codeMethod);

cinit = 14;
anchor4 = genBasebandSig(cinit,M,T,fs,rolloff, codeMethod);

n = size(anchor1, 2);
time = linspace(0, n / fs * 1000, n);
subplot(2,2,1); plot(time, anchor1 / max(abs(anchor1)), "g", "LineWidth",1.5); title("Anchor 1");
xlabel("time [ms]"); ylabel("amplitude");
subplot(2,2,2); plot(time, anchor2 / max(abs(anchor2)), "r", "LineWidth",1.5); title("Anchor 2")
xlabel("time [ms]"); ylabel("amplitude");
subplot(2,2,3); plot(time, anchor3 / max(abs(anchor3)), "b", "LineWidth",1.5); title("Anchor 3");
xlabel("time [ms]"); ylabel("amplitude");
subplot(2,2,4); plot(time, anchor4 / max(abs(anchor4)), "m", "LineWidth",1.5); title("Anchor 4");
xlabel("time [ms]"); ylabel("amplitude");

%subplot(2,3,4); histogram(anchor1, "Normalization", "probability"); title("Anchor 1");

%subplot(2,3,5); histogram(anchor2, "Normalization", "probability"); title("Anchor 2")

%subplot(2,3,6); histogram(anchor3, "Normalization", "probability"); title("Anchor 3");

%[anchor_diff, x] = xcorr(anchor2, anchor1);
%subplot(1,1,1); plot(x, anchor_diff); title("Anchor 1/2 corr");
%hold on;
%[anchor_diff, x] = xcorr(anchor3, anchor2);
%plot(x, anchor_diff); title("Anchor 3/2 corr");

%[anchor_diff, x] = xcorr(anchor1, anchor3);
%plot(x, anchor_diff); title("Anchor 1/3 corr");
%hold off;

%%

% received signal
rx = zeros(1,2*numel(anchor1));
% an dieser stelle werden die signale von den verschiedenen ankern gemixt.
% alle haben unterschiedliche laufzeiten
delay1 = 10e-3; % delay in seconds
idx = (floor(delay1*fs) + 1):(floor(delay1*fs) + numel(anchor1) );
rx(idx) = rx(idx) + anchor1;
delay2 = 20e-3; % delay in seconds
idx = (floor(delay2*fs) + 1):(floor(delay2*fs) + numel(anchor2));
rx(idx) = rx(idx) + anchor2;
delay3 = 45e-3; % delay in seconds
idx = (floor(delay3*fs) + 1):(floor(delay3*fs) + numel(anchor3));
rx(idx) = rx(idx) + anchor3;
delay4 = 50e-3; % delay in seconds
idx = (floor(delay4*fs) + 1):(floor(delay4*fs) + numel(anchor4));
rx(idx) = rx(idx) + anchor4;
%%

% empfänger sieht nun nur den mix aus verschiedenen signalen:
%t = linspace(0, fs, size(rx,1))
n = size(rx, 2);
time = linspace(0, n / fs, n);
subplot(1,1,1)
plot(time * 1000, rx / max(abs(rx)), "k", "LineWidth",1.5);
xlabel("time [ms]"); ylabel("amplitude");
hold on;
xline(delay1 * 1000, "--g", "LineWidth",3.0);
xline(delay2 * 1000, "--r", "LineWidth",3.0);
xline(delay3 * 1000, "--b", "LineWidth",3.0);
xline(delay4 * 1000, "--m", "LineWidth",3.0);
hold off;
%%

% aus diesem signal muss man nun die einzelnen signallaufzeiten
% herausfinden. dafür benutzt man die kreuzkorrelation r(tau).
%
% die laufzeiten ergeben sich nun aus:
% estimated_delay = arg max(r(tau))
% anchor 1
subplot(1,1,1)

[r1,tau1] = xcorr(rx,anchor1);
[ar1,atau1] = xcorr(anchor1,anchor2);
tau1(tau1 < 0) = NaN;
plot(tau1 / fs * 1000, abs(r1) / max(r1), "g", "LineWidth",1.5)
hold on;
[r2,tau2] = xcorr(rx,anchor2);
[ar2,atau2] = xcorr(anchor2,anchor2);
tau2(tau2 < 0) = NaN;
plot(tau2 / fs * 1000, abs(r2) / max(r2), "r", "LineWidth",1.5)
[r3,tau3] = xcorr(rx,anchor3);
[ar3,atau3] = xcorr(anchor3,anchor3);
tau3(tau3 < 0) = NaN;
plot(tau3 / fs * 1000, abs(r3) / max(r3), "b", "LineWidth",1.5)
[r4,tau4] = xcorr(rx,anchor4);
[ar4,atau4] = xcorr(anchor4,anchor4);
tau4(tau4 < 0) = NaN;
plot(tau4 / fs * 1000, abs(r4) / max(r4), "m", "LineWidth",1.5)
hold off;
xlabel("delay [ms]");
ylabel("abs corr");

subplot(2,2,1); plot(tau1 / fs * 1000, abs(r1) / max(r1), "g", "LineWidth",1.5); title("Correlation with Anchor 1");
xlabel("delay [ms]");
ylabel("abs corr");

subplot(2,2,2); plot(tau2 / fs * 1000, abs(r2) / max(r2), "r", "LineWidth",1.5); title("Correlation with Anchor 2");
xlabel("delay [ms]");
ylabel("abs corr");

subplot(2,2,3); plot(tau3 / fs * 1000, abs(r3) / max(r3), "b", "LineWidth",1.5); title("Correlation with Anchor 3");
xlabel("delay [ms]");
ylabel("abs corr");

subplot(2,2,4); plot(tau4 / fs * 1000, abs(r4) / max(r4), "m", "LineWidth",1.5); title("Correlation with Anchor 4");
xlabel("delay [ms]");
ylabel("abs corr");
%%
%%% Autocorrelation
subplot(2,2,1); plot(atau1 / fs * 1000, abs(ar1), "k", "LineWidth",1.5); title("Autocorrelation of Anchor 1");
xlabel("tau [ms]");
ylabel("abs corr");

subplot(2,2,2); plot(atau2 / fs * 1000, abs(ar2), "k", "LineWidth",1.5); title("Autocorrelation of Anchor 2");
xlabel("tau [ms]");
ylabel("abs corr");

subplot(2,2,3); plot(atau3 / fs * 1000, abs(ar3), "k", "LineWidth",1.5); title("Autocorrelation of Anchor 3");
xlabel("tau [ms]");
ylabel("abs corr");

subplot(2,2,4); plot(atau4 / fs * 1000, abs(ar4), "k", "LineWidth",1.5); title("Autocorrelation of Anchor 4");
xlabel("tau [ms]");
ylabel("abs corr");
%%
[~,idx] = max(r1);
delay1 = tau1(idx)/fs

[~,idx] = max(r2);
delay2 = tau2(idx)/fs

[~,idx] = max(r3);
delay3 = tau3(idx)/fs

[~,idx] = max(r4);
delay4 = tau4(idx)/fs