function p = genBasebandSig(cinit,M,T,fs,rolloff)
    %ltePSSCHPRBS(ue,n)
    c = ltePRBS(cinit,M,'signed'); % generates pseudorandom bin sequence of length M
    %c = comm.GoldSequence(cinit, M);
    p = [];
    Nsym = floor(T/M*fs); % samples per symbol
    for m = 1:M
        p = [p c(m)*ones(1,Nsym)]; % create 1x(NSym*M) vector
    end
    b = rcosdesign(rolloff,4*Nsym,Nsym); % get coefficients of cosinus FIR filter
    % add samples
    delay = floor(length(b)/2); % group delay
    p = [p zeros(1,delay)];
    p = filter(b,1,p);

    % remove filter delay
    p = p(delay+1:end);
end