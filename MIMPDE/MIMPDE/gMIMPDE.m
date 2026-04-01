function MIMPDE = gMIMPDE(feat, label, opts)
    % Parameters
    lb = 0;
    ub = 1;
    thres = 0.5;
    
    if isfield(opts, 'N')
        N = opts.N;
    end
    if isfield(opts, 'T')
        max_Iter = opts.T;
    end
    if isfield(opts, 'thres')
        thres = opts.thres;
    end
    
    % Get MI values from opts
    if isfield(opts, 'mi_values')
        mi_values_sorted = opts.mi_values;
    end
    
    % Function
    fun = @jFitnessFunction;
    
    % Dimension
    dim = size(feat, 2);
    
    % MI-related initialization
    mi_max = max(mi_values_sorted);
    mi_min = min(mi_values_sorted);
    if mi_max == mi_min
        normalized_mi = ones(1, dim) * 0.5;
    else
        normalized_mi = (mi_values_sorted - mi_min) / (mi_max - mi_min);
    end
    
    num_F1 = round(dim * 0.3);
    num_F3 = round(dim * 0.3);
    num_F2 = dim - num_F1 - num_F3;
    
    F1_indices = 1:num_F1;
    F2_indices = (num_F1 + 1):(num_F1 + num_F2);
    F3_indices = (num_F1 + num_F2 + 1):dim;
    
    Groups = {F1_indices, F2_indices, F3_indices};
    
    X = zeros(N, dim);
    for i = 1:N
        k = randi(3);
        selected_group_indices = Groups{k};
        for d = 1:dim
            is_in_selected_group = ismember(d, selected_group_indices);
            if is_in_selected_group
                if rand() < normalized_mi(d)
                    X(i, d) = 1;
                else
                    X(i, d) = 0;
                end
            else
                if rand() < 0.5
                    X(i, d) = 1;
                else
                    X(i, d) = 0;
                end
            end
        end
    end
    % End of MI initialization
    
    function r = cauchy_random(mu, gamma)
    r = mu + gamma * tan(pi * (rand - 0.5));
    while r <= 0
        r = mu + gamma * tan(pi * (rand - 0.5));
    end
    end

    % Dynamic strategy and parameter initialization
    ng = 25; 
    if isfield(opts, 'ng')
       ng = opts.ng; 
    end
    p_rate = 0.1; 
    if isfield(opts, 'p_rate')
       p_rate = opts.p_rate; 
    end
    delta_sharing = 0.04; 
    if isfield(opts, 'delta_sharing')
       delta_sharing = opts.delta_sharing; 
    end
    c_learning_rate = 0.1;
    
    numStrategies = 3;
    s1_Num = round(N * 0.3);
    s3_Num = round(N * 0.3);
    s2_Num = N - s1_Num - s3_Num;
    NP_j = [s1_Num, s2_Num, s3_Num];
    strategy_ranks = [1, 2, 3];
    dominant_strategy_idx = strategy_ranks(1);
    
    delta_f = zeros(1, numStrategies);
    success_rates = zeros(1, numStrategies);
    
    % JADE parameters
    muF = ones(1, numStrategies) * 0.5;
    muCR = ones(1, numStrategies) * 0.5;
    % == == == == == == == == == == == == == == == == == == == == == == == %
    
    OU = zeros(N, dim);
    fit = zeros(1, N);
    fitG = inf;
    Xgb = zeros(1, dim);
    
    for i = 1:N
        fit(i) = fun(feat, label, X(i, :) > thres, opts);
        if fit(i) < fitG
            fitG = fit(i);
            Xgb = X(i, :);
        end
    end
    
    U = zeros(N, dim);
    V = zeros(N, dim);
    curve = zeros(1, max_Iter);
    curve(1) = fitG;
    t = 2;
    
    % Record Fitness Improvement for each strategy per generation
    strategy_fi_sum = zeros(max_Iter, numStrategies);   % Total improvement per generation
    strategy_fi_avg = zeros(max_Iter, numStrategies);   % Average improvement per individual per generation
    strategy_pop_ratio = zeros(max_Iter, numStrategies); % Population ratio of each strategy per generation
    strategy_pop_ratio(1, :) = NP_j ./ N;
    
    % Initialize stagnation counter
    stagnation_counter = 0;
    last_fitG = fitG;
    
    % Must initialize indices before the loop so that the MI calculation module can find individuals
    [~, index_init] = sort(fit);
    pop_indices_S1 = index_init(1:s1_Num);
    pop_indices_S2 = index_init(s1_Num + 1 : s1_Num + s2_Num);
    pop_indices_S3 = index_init(s1_Num + s2_Num + 1 : N);
    
    while t <= max_Iter
        
        % Dynamic resource allocation
        if mod(t, ng) == 0 && t > ng
            % Stagnation intervention check 
            if stagnation_counter > ng * 1.5
                NP_j_new = zeros(1, numStrategies);
                NP_j_new(3) = round(N * 0.6);
                NP_j_new(2) = round(N * 0.3);
                NP_j_new(1) = N - NP_j_new(3) - NP_j_new(2);
                stagnation_counter = 0;
            else
                
                % Calculate fitness score
                for j = 1:numStrategies
                    if NP_j(j) > 0
                        success_rates(j) = delta_f(j) / (ng * NP_j(j));
                    else
                        success_rates(j) = 0;
                    end
                end
                sum_rate = sum(success_rates);
                if sum_rate == 0
                    score_fitness = ones(1, numStrategies) / numStrategies;
                else
                    score_fitness = success_rates / sum_rate;
                end
                
                % Calculate MI score
                avg_mi_strategies = zeros(1, numStrategies);
                % Use updated indices from the previous generation
                pop_cells = {pop_indices_S1, pop_indices_S2, pop_indices_S3};
                
                for k = 1:numStrategies
                    curr_indices = pop_cells{k};
                    if isempty(curr_indices)
                        avg_mi_strategies(k) = 0;
                        continue;
                    end
                    
                    sum_mi_group = 0;
                    valid_inds = 0;
                    for idx = curr_indices
                        % Find which features this individual selected
                        sel_feat_idx = find(X(idx, :) > thres);
                        if ~isempty(sel_feat_idx)
                            % Calculate average MI
                            sum_mi_group = sum_mi_group + mean(normalized_mi(sel_feat_idx));
                            valid_inds = valid_inds + 1;
                        end
                    end
                    
                    if valid_inds > 0
                        avg_mi_strategies(k) = sum_mi_group / valid_inds;
                    else
                        avg_mi_strategies(k) = 0;
                    end
                end
                
                % Normalize MI scores
                sum_mi_val = sum(avg_mi_strategies);
                if sum_mi_val == 0
                    score_mi = ones(1, numStrategies) / numStrategies;
                else
                    score_mi = avg_mi_strategies / sum_mi_val;
                end
                
                % Hybrid weighted calculation of final quotas
                w_fitness = 0.7; 
                w_mi = 1 - w_fitness;
                
                final_scores = w_fitness * score_fitness + w_mi * score_mi;
                
                % Update rankings to determine Dominant
                [~, strategy_ranks] = sort(final_scores, 'descend');
                
                % Calculate proportions and allocate
                rho = final_scores / sum(final_scores);
                NP_j_new = round(rho * N);
                
                % Minimum allocation guarantee to prevent any strategy from disappearing completely
                min_NP = max(2, round(N * 0.1));
                for j = 1:numStrategies
                    if NP_j_new(j) < min_NP
                        NP_j_new(j) = min_NP;
                    end
                end
                
                % Renormalize to ensure the sum equals N
                total_NP = sum(NP_j_new);
                if total_NP > N
                    over_alloc = total_NP - N;
                    [~, max_idx] = max(NP_j_new);
                    NP_j_new(max_idx) = NP_j_new(max_idx) - over_alloc;
                end
                NP_j_new(end) = N - sum(NP_j_new(1:end-1));
                
            end
            NP_j = NP_j_new;
            delta_f = zeros(1, numStrategies);
        end
        s1_Num = NP_j(1);
        s2_Num = NP_j(2);
        s3_Num = NP_j(3);
        [~, index] = sort(fit);
        
        % Leader-follower information sharing
        dominant_strategy_idx = strategy_ranks(1);
        slave_strategy_indices = strategy_ranks(2:3);
        
        pop_indices_S1 = index(1 : s1_Num);
        pop_indices_S2 = index(s1_Num + 1 : s1_Num + s2_Num);
        pop_indices_S3 = index(s1_Num + s2_Num + 1 : N);
        all_pop_indices = {pop_indices_S1, pop_indices_S2, pop_indices_S3};
        non_master = setdiff(1:3, dominant_strategy_idx);
      
        if ~isempty(all_pop_indices{non_master(1)})
            x_lbest_slave1 = X(all_pop_indices{non_master(1)}(1), :);
        else
            x_lbest_slave1 = Xgb;
        end
        if ~isempty(all_pop_indices{non_master(2)})
            x_lbest_slave2 = X(all_pop_indices{non_master(2)}(1), :);
        else
            x_lbest_slave2 = Xgb;
        end
        x_bar = (x_lbest_slave1 + x_lbest_slave2) / 2;
        eX = X(pop_indices_S1, :);
        
        % Parameter recording initialization
        S_F = cell(1, numStrategies);
        S_CR = cell(1, numStrategies);
        for j = 1:numStrategies, S_F{j} = []; S_CR{j} = []; end
        current_F = zeros(1, N);
        current_CR = zeros(1, N);
        individual_strategy = zeros(1, N);
        delta_f_new = zeros(1, numStrategies);
        
        % Strategy S1 
        for i = 1:s1_Num
            current_idx = pop_indices_S1(i);
            individual_strategy(current_idx) = 1;
            k = randi([1 s1_Num]);
            k_idx = pop_indices_S1(k);

            if fit(current_idx) <= fit(k_idx)
               sigma = 1;
            else
               sigma = exp((fit(k_idx) - fit(current_idx)) / (abs(fit(current_idx)) + realmin));
            end
            V(current_idx, :) = X(current_idx, :) .* (1 + normrnd(0, sigma, 1, dim));
            if dominant_strategy_idx == 1
               V(current_idx, :) = V(current_idx, :) + delta_sharing .* (x_bar - X(current_idx, :)); 
            end
            U(current_idx, :) = V(current_idx, :);
        end
        
        % Strategy S2 
        for i = 1:s2_Num
            current_idx = pop_indices_S2(i);
            individual_strategy(current_idx) = 2;
            
            F_i = cauchy_random(muF(2), 0.1);
            F_i = min(1, max(0.01, F_i));
            CR_i = normrnd(muCR(2), 0.1);
            CR_i = min(1, max(0, CR_i));
            current_F(current_idx) = F_i;
            current_CR(current_idx) = CR_i;
            
            if isempty(eX)
               r1_idx = index(1);
            else
               r1 = randi([1 size(eX, 1)]);
               r1_idx = pop_indices_S1(r1);
            end
            r2 = randi([1 s2_Num]);
            r2_idx = pop_indices_S2(r2);
            
            V(current_idx, :) = X(current_idx, :) + F_i .* (X(r1_idx, :) - X(current_idx, :)) ...
                              + (1 - F_i) .* (X(r2_idx, :) - X(current_idx, :));
            if dominant_strategy_idx == 2
               V(current_idx, :) = V(current_idx, :) + delta_sharing .* (x_bar - X(current_idx, :)); 
            end
            
            rnbr = randi([1, dim]);
            for d = 1:dim
                if rand() <= CR_i || d == rnbr
                   U(current_idx, d) = V(current_idx, d);
                else 
                   U(current_idx, d) = X(current_idx, d); 
                end
            end
        end
        
        % Strategy S3 
        p_num = ceil(N * p_rate);

        if p_num == 0
           p_num = 1; 
        end

        if p_num > N
           p_num = N; 
        end
        
        for i = 1:s3_Num
            current_idx = pop_indices_S3(i);
            individual_strategy(current_idx) = 3;
            
            F_i = cauchy_random(muF(3), 0.1);
            F_i = min(1, max(0.01, F_i));
            CR_i = normrnd(muCR(3), 0.1);
            CR_i = min(1, max(0, CR_i));
            current_F(current_idx) = F_i;
            current_CR(current_idx) = CR_i;
            
            pbest_idx = index(randi(p_num));
            pbad_idx = index(N - randi(p_num) + 1);
            
            V(current_idx, :) = X(current_idx, :) + F_i .* (X(pbest_idx, :) - X(pbad_idx, :)) ...
                               + rand .* (Xgb - X(pbest_idx, :));
            if dominant_strategy_idx == 3
               V(current_idx, :) = V(current_idx, :) + delta_sharing .* (x_bar - X(current_idx, :)); 
            end
            
            rnbr = randi([1, dim]);
            for d = 1:dim
                if rand() <= CR_i || d == rnbr
                   U(current_idx, d) = V(current_idx, d); 
                else U(current_idx, d) = X(current_idx, d); 
                end
            end
        end
        
        % Selection and update 
        for i = 1:N
            XB = U(i, :);
            XB(XB > ub) = ub;
            XB(XB < lb) = lb;
            U(i, :) = XB;
            OU(i, :) = U(i, :);
            OU(i, OU(i, :) > ub) = ub;
            OU(i, OU(i, :) < lb) = lb;
            Fnew = fun(feat, label, (U(i, :) > thres), opts);
            OFnew = fun(feat, label, (OU(i, :) > thres), opts);
            fit_old = fit(i);
            temp = fit(i);
            
            winner_strat = 0; % 0=None, 1=U, 2=OU
            
            if Fnew < temp
                X(i, :) = U(i, :);
                fit(i) = Fnew;
                temp = Fnew;
                winner_strat = 1;
            end
            if OFnew < temp
                X(i, :) = OU(i, :);
                fit(i) = OFnew;
                winner_strat = 2;
            end
            if fit(i) < fit_old
                strat_idx = individual_strategy(i);
                if strat_idx > 0
                    delta_f_new(strat_idx) = delta_f_new(strat_idx) + (fit_old - fit(i));
                    if strat_idx == 2 || strat_idx == 3
                        S_F{strat_idx}(end + 1) = current_F(i);
                        S_CR{strat_idx}(end + 1) = current_CR(i);
                    end
                end
            end
            if fit(i) < fitG
                fitG = fit(i);
                Xgb = X(i, :);
            end
        end
        
        delta_f = delta_f + delta_f_new;
        strategy_fi_sum(t, :) = delta_f_new;
        strategy_fi_avg(t, :) = delta_f_new ./ max(1, NP_j);
        strategy_pop_ratio(t, :) = NP_j ./ N;
        
        % Update muF and muCR 
        for j = 1:numStrategies
            if j == 1
               continue; 
            end
            if ~isempty(S_F{j})
                sum_CR_sq = sum(S_CR{j} .^ 2);
                sum_CR = sum(S_CR{j});
                if sum_CR == 0
                    mean_L_CR = 0;
                else
                    mean_L_CR = sum_CR_sq / sum_CR;
                end
                muCR(j) = (1 - c_learning_rate) * muCR(j) + c_learning_rate * mean_L_CR;
                muCR(j) = min(1, max(0, muCR(j)));
                mean_A_F = mean(S_F{j});
                muF(j) = (1 - c_learning_rate) * muF(j) + c_learning_rate * mean_A_F;
                muF(j) = min(1, max(0.01, muF(j)));
            else
                muCR(j) = (1 - c_learning_rate) * muCR(j) + c_learning_rate * rand;
                muF(j) = (1 - c_learning_rate) * muF(j) + c_learning_rate * rand;
                muCR(j) = min(1, max(0, muCR(j)));
                muF(j) = min(1, max(0.01, muF(j)));
            end
        end
        
        % Update stagnation counter
        if fitG < last_fitG
            stagnation_counter = 0;
            last_fitG = fitG;
        else
            stagnation_counter = stagnation_counter + 1;
        end
        
        curve(t) = fitG;
        t = t + 1;
    end
    
    fprintf('\n Best fitness (MIMPDE): %f', fitG);
    
    Pos = 1:dim;
    Sf = Pos((Xgb > thres) == 1);
    sFeat = feat(:, Sf);
    
    MIMPDE.gb = fitG;
    MIMPDE.sf = Sf;
    MIMPDE.ff = sFeat;
    MIMPDE.nf = length(Sf);
    MIMPDE.c = curve;
    MIMPDE.f = feat;
    MIMPDE.l = label;
    MIMPDE.strategy_fi_sum = strategy_fi_sum;
    MIMPDE.strategy_fi_avg = strategy_fi_avg;
    MIMPDE.strategy_fi_cum = cumsum(strategy_fi_sum, 1);
    MIMPDE.strategy_pop_ratio = strategy_pop_ratio;
end