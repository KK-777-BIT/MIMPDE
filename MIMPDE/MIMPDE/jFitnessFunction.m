function cost = jFitnessFunction(feat,label,X,opts)
% Default of [alpha; beta]
ws = [0.95; 0.05];
if isfield(opts,'ws'), ws = opts.ws; end

% Check if any feature exist
if sum(X == 1) == 0
  cost = 1;
else
  % Error rate
  error    = jwrapper_KNN(feat(:,X == 1),label,opts);
  % Number of selected features   
  num_feat = sum(X == 1);
  % Total number of features       
  max_feat = length(X); 
  % Set alpha & beta
  alpha    = ws(1); 
  beta     = ws(2);
  % Cost function 
  cost     = alpha * error + beta * (num_feat / max_feat); 
end
end

%---------------------Call Functions----------------------
function error = jwrapper_KNN(sFeat,label,opts)
if isfield(opts,'k'), k = opts.k; end               % k=5
if isfield(opts,'Model'), Model = opts.Model; end   

foldErr = zeros(1, Model.NumTestSets);
for f = 1:Model.NumTestSets
  trainIdx = training(Model, f);    testIdx  = test(Model, f);
  xtrain   = sFeat(trainIdx,:);     ytrain   = label(trainIdx);
  xvalid   = sFeat(testIdx,:);      yvalid   = label(testIdx);
  My_Model = fitcknn(xtrain,ytrain,'NumNeighbors',k);
  pred     = predict(My_Model,xvalid);
  Acc      = sum(pred == yvalid) / length(yvalid);
  foldErr(f) = 1 - Acc;
end
error = mean(foldErr);
end








