function m = linRegressRegul(x, y, lambda)

    x_aug = [x; sqrt(lambda) * eye(size(x))];
    y_aug = [y; zeros(size(y))];

    m = LinRegress(x_aug, y_aug);

end