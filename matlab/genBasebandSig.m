function p = genBasebandSig(cinit,M,T,fs,rolloff)
    c = ltePRBS(cinit,M,'signed');
    p = [];
    Nsym = floor(T/M*fs); % samples per symbol
    for m = 1:M
        p = [p c(m)*ones(1,Nsym)];
    end
    b = rcosdesign(rolloff,4*Nsym,Nsym);
    % add samples
    delay = floor(length(b)/2); % group delay
    p = [p zeros(1,delay)];
    p = filter(b,1,p);
    % remove filter delay
    p = p(delay+1:end);
end