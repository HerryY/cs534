clc, clear;

%% Settings
X = dlmread('walking-data/walking.train.data');
y = dlmread('walking-data/walking.train.labels') + 1;
[n, d] = size(X);


%% 1. K-Means
fprintf('1. ');
KMeans(X, y, 2);


%% 2. PCA
S = zeros(d, d);
x_bar = mean(X, 1);
for i = 1:n
    temp = X(i,:) - x_bar;
    S = S + temp' * temp;
end
S = S / (n-1);
[V, D] = eig(S);
E = diag(D);
[PC, I] = sort(E, 'descend');

% Find smallest d for each threshold
thresholds = [80 90];
smallest_dims = zeros(size(thresholds));
total_variance = sum(PC);
for i = 1:numel(thresholds)
    min_d = 0;
    s = 0;
    sum_var = total_variance * thresholds(i) / 100;
    while s < sum_var
        min_d = min_d + 1;
        s = s + PC(min_d);
    end
    smallest_dims(i) = min_d;
end
fprintf('2. Smallest dimension we can reduce if we wish to retain: \n');
for i = 1:numel(thresholds)
   fprintf('%d%% of the total variance: %d \n', thresholds(i), smallest_dims(i)); 
end
fprintf('\n');

% plot first 10 principle components
B = 1:10;
Percentage_PC = PC(B) / sum(PC) * 100;
bar(B, Percentage_PC);
xlabel('PC');
ylabel('Variance (%)');
    
    
%% 3. Apply PCA
choices = [1 2 3];
fprintf('3. Applay PCA:\n');

Z = X * V(:,I(1:3));
for c = choices
    % Apply KMeans with k=2 to the reduced data Z
    fprintf('%d dimension(s): \n', c);
    KMeans(Z(:,1:c), y, 2);
end


%% 4. Apply LDA for 2 classes

% compute m1, m2
M = zeros(2,d);
N = zeros(2,d);
for i = 1:n
    N(y(i),:) = N(y(i),:) + 1;
    M(y(i),:) = M(y(i),:) + X(i,:);
end
M = M ./ N;

% compute S
S = 0;
for i = 1:n
    temp = X(i,:)  - M(y(i),:);
    S = S + temp * temp';
end

% compute w = S^-1 * (m1 - m2)
w = (M(1) - M(2)) \ S;

% project to 1D-data and apply K-Means
KMeans(X * w, y, 2);




