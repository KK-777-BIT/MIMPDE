function [MIfeat, selected_mi_values, selected_original_indices] = MI_TOOL(X,y,K) 
 num_features = size(X,2);
 mi_values = zeros(1, num_features);

 for i = 1:num_features
    mi_values(i) = mi(X(:,i), y, 10);
 end

 [sorted_mi, sorted_indices] = sort(mi_values, 'descend');

 k = round(num_features*K);
 selected_features_indices = sorted_indices(1:k);
 MIfeat = X(:, selected_features_indices);
 
 selected_mi_values = sorted_mi(1:k); % Return the sorted MI values of selected features
 selected_original_indices = selected_features_indices; % Return the original indices of selected features
end

function MI = mi(X, Y, numBins)
    % Calculate joint probability matrix of X and Y
    set(0, 'DefaultFigureVisible', 'off'); 
    h = histogram2(X, Y, numBins, 'Normalization', 'probability', 'Visible', 'off');
    jointXY = h.Values;

    
    % Calculate marginal probabilities of X and Y
    pX = sum(jointXY, 2);
    pY = sum(jointXY, 1);

    % Calculate mutual information
    MI = 0;
    for i = 1:length(pX)
        for j = 1:length(pY)
            if jointXY(i, j) > 0
                MI = MI + jointXY(i, j) * log2(jointXY(i, j) / (pX(i) * pY(j)));
            end
        end
    end
end