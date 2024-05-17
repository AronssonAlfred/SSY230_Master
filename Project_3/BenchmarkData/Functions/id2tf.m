function transfer = id2tf(model)
    if strcmp(model.type, 'ARX')
        A = [1, -model.theta(1:model.na)'];
        B = model.theta(model.na+1:end)';
    elseif strcmp(model.type, 'OE')
        B = model.theta(model.na+1:end)';
        F = [1, -model.theta(1:model.na)'];
        A = F;  % In OE, the transfer function denominator is F(q)
    else
        error('Unknown model type');
    end
    transfer = idtf(B, A, -1, 'InputDelay', model.nk);
end

%function transfer = id2tf(arx)
%
%    A = [1,-arx.theta(1:arx.na)'];
%    B = arx.theta(arx.na+1:end)';
%    transfer = tf(B,A,-1);
%    
%end