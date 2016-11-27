function [ purities, best_centroids, best_pred ] = KMeans( X, y, k )
% KMEANS 
% k - number of clusters

[n, d] = size(X);
NUM_RUNS = 10;
MAX_KMEANS_ITERS = 50;

best_SSE = Inf;
for r = 1:NUM_RUNS
   centroids = X(randi(n, k, 1),:);
   iter = 0;
   while iter < MAX_KMEANS_ITERS
       iter = iter + 1;
       
       % Assign each data point to its nearest cluster center
       clusters = zeros(n, 1, 'uint16');
       for i = 1:n
          xi = X(i,:);
          nearest = inf;
          for j = 1:k
              dist = norm(xi - centroids(j,:));
              if dist < nearest
                  nearest = dist;
                  clusters(i) = j;
              end
          end
       end
       
       % Re?estimate the cluster center
       %oldCentroids = centroids;
       clusterCounts = zeros(k,1);
       centroids = zeros(k,d);
       for i = 1:n
          clusterCounts(clusters(i)) = clusterCounts(clusters(i)) + 1;
          centroids(clusters(i),:) = centroids(clusters(i),:) + X(i,:);
       end
       centroids = centroids ./ (clusterCounts * ones(1,d));
   end
   
   % determine class for each cluster
   countForClusters = zeros(k,k);
   for i = 1:n
      countForClusters(clusters(i), y(i)) = countForClusters(clusters(i), y(i)) + 1;
   end
   [~, mapCluster2Label] = max(countForClusters, [], 2);
   
   % convert cluster index to class label
   pred = zeros(n, 1);
   for i = 1:n
      pred(i) = mapCluster2Label(clusters(i)); 
   end
   diff = double(y ~= pred);
   SSE = diff' * diff;
   if SSE < best_SSE
      best_SSE = SSE;
      best_centroids = centroids;
      best_pred = pred;
   end
   purities = zeros(1,k);
   countForLabels = zeros(1,k);
   for i = 1:n
      purities(y(i)) = purities(y(i)) + (y(i) == pred(i));
      countForLabels(y(i)) = countForLabels(y(i)) + 1;
   end
   purities = purities ./ countForLabels;
end
fprintf('Best SSE: %f \n   Purity: Class 0: %f \t Class 1: %f \n\n',...
       best_SSE, purities(1)*100, purities(2)*100);

end

