function p = genBasebandSig(cinit, deg, method, poly)
                            %'Polynomial',,... [1 1 1 0 1 1 0 1 1 0 1 1 0]
                            %[13 12 10 9 7 6 5 1 0]
                            %[13 4 3 1 0]
    switch method

        case "gold"
            gold = comm.GoldSequence('SamplesPerFrame', (2^deg - 1), ...
                    "FirstPolynomial", fpoly,...
                    "SecondPolynomial", spoly,...
                    'FirstInitialConditions', int2bit(cinit, max(fpoly)),...
                    'SecondInitialConditions', int2bit(cinit, max(spoly)));
            c = gold();

        case "kasami"
            kasami = comm.KasamiSequence('SamplesPerFrame', (2^deg - 1),...
                        'Polynomial', poly, ...
                        'InitialConditions', int2bit(cinit, deg));
            c = kasami();
        case "lfsr"
            pnlfsr = comm.PNSequence('SamplesPerFrame', (2^deg - 1),...
                        'Polynomial', poly, ...
                        'InitialConditions', int2bit(cinit, deg));
            c = pnlfsr();
        otherwise
            disp("method not supported");
            return;
    end
    c(c==0) = -1;
    %Nsym = floor(T/M*fs); % samples per symbol 
    %disp(c);
     % generates pseudorandom bin sequence of length M
    %
    % remove cosine FIR for evaluation
    %p = [];
    %for m = 1:M
    %    p = [p c(m)*ones(1,Nsym)]; % create 1x(NSym*M) vector
    %end
    %b = rcosdesign(rolloff,4*Nsym,Nsym); % get coefficients of cosinus FIR filter
    % add samples
    %delay = floor(length(b)/2); % group delay
    %p = [p zeros(1,delay)];
    %p = filter(b,1,p);

    % remove filter delay
    %p = p(delay+1:end);
    p = c;
end