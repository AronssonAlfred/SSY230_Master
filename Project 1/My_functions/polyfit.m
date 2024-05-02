function m = polyfit(x,y,lambda, n)

    x2 = poly_x2(x,n);
    
    m = linRegressRegul(x2,y,lambda);
    m.n = n;
    m.model = 'POLY';
    m.x = x2;

end