function p = genBasebandSig(cinit,M,T,fs,rolloff,method)
    
    switch method
        case "prbs"
            c = ltePRBS(cinit,M,'signed');
        case "gold"
            gold = comm.GoldSequence("FirstPolynomial",[16 2 0], ...
                    "SecondPolynomial",[16 8 5 2 0], ...
                    'FirstInitialConditions',[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1],...
                    'SecondInitialConditions',[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1],...
                    'SamplesPerFrame',M);
            c = gold();
        case "walsh"
            walsh = comm.WalshCode;
            walsh.Length = 64;
            walsh.SamplesPerFrame = M;
            c = walsh();
        case "kasami"
            kasami = comm.KasamiSequence('SamplesPerFrame',M,...
                        'Polynomial',[16 8 4 3 2 0],...
                        'InitialConditions',[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1]);
            c = kasami();
        case "lfsr"
            pnlfsr = comm.PNSequence('Polynomial',[3 2 0], ...
                        'SamplesPerFrame',M,'InitialConditions',[0 0 1]);
            c = pnlfsr();
        otherwise
            disp("method not supported");
            return;
    end
    c(c==0) = -1;
    Nsym = floor(T/M*fs); % samples per symbol
    %disp(c);
     % generates pseudorandom bin sequence of length M
    %
    p = [];
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