function m = knnRegress(x, y, k)
    % KNN model
    m.x = x;
    m.y = y;
    m.k = k;
    m.model = 'knn'; 
end