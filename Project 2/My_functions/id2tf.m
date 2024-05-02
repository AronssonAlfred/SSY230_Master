function transfer = id2tf(arx)

    A = [1,arx.theta(1:arx.na)]';
    B = [arx.theta(arx.na+1:end)]';
    transfer = tf(idpoly(A,B));
    
end 