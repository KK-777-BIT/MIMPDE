% Evaluate the accuracy of the classification

function Acc = jknn(feat,label,opts)
% Default of k-value
k = 5;

if isfield(opts,'k'), k = opts.k; end
if isfield(opts,'Model'), Model = opts.Model; end

foldAcc = zeros(1, Model.NumTestSets);
for f = 1:Model.NumTestSets
  trainIdx = training(Model, f);    testIdx  = test(Model, f);
  xtrain   = feat(trainIdx,:);      ytrain   = label(trainIdx);
  xvalid   = feat(testIdx,:);       yvalid   = label(testIdx);
  My_Model = fitcknn(xtrain,ytrain,'NumNeighbors',k);
  pred     = predict(My_Model,xvalid);
  foldAcc(f) = sum(pred == yvalid) / length(yvalid);
end
Acc = mean(foldAcc);

 fprintf('\n Accuracy: %g %%\n',100 * Acc);
end


