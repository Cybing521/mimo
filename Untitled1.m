A_lambda_values = 1:1:8;  % 横轴归一化区域大小的取值范围
num_trials = 1000;          % 每个横轴值进行20次试验
Lr_values = [15,10];       % Lr的两个不同值
lambda = 1;
% 初始化用于存储平均容量的矩阵
average_capacity_Proposed = zeros(length(A_lambda_values), length(Lr_values));

% 计算容量并取平均值
for j = 1:length(Lr_values)
    Lr = Lr_values(j);
    for u = 1:length(A_lambda_values)
        A_lambda = A_lambda_values(u);
        square_size=A_lambda*lambda;
        capacity_trials = zeros(1, num_trials);
        for trial = 1:num_trials
            N = 4;        %发送
            M = 4;        %接收天线
            Lt = 10;
            power=10;     %噪声功率
            SNR_dB = 15;
            SNR_linear = 10^(SNR_dB / 10);
            sigma = power / SNR_linear;
            lambda = 1;
            D = lambda/2;                   %天线最小距离
            xi=1e-3;      %收敛容差
            xii=1e-4;
            r = zeros(2,M);
            attempts = 0;
            while true
                candidates = rand(2, 1000) .* square_size; % 随机候选点
                valid = true(1,1000);
                for k = 2:1000
                    if min(vecnorm(candidates(:,1:k-1)-candidates(:,k),2,1)) < D
                        valid(k) = false;
                    end
                end
                valid_points = candidates(:,valid);
                if size(valid_points,2) >= M
                    r = valid_points(:,1:M);
                    break;
                end
                attempts = attempts + 1;
                if attempts > 400, error('无法初始化合法位置'); end
            end
            
            P = Lt;
            theta_p = rand(P, 1) * pi;  % 仰角
            phi_p = rand(P, 1) * pi;    % 方向角
            G = zeros(P, N);
            for p = 1:P
                for n = 1:N
                    G(p, n) = exp(1i * pi * sin(theta_p(p)) * cos(phi_p(p)) * (n - 1));
                end
            end
            theta_q = rand(Lr, 1) * pi;
            phi_q = rand(Lr, 1) * pi;
            Sigma = zeros(Lr, Lt);
            diag_len = min(Lr, Lt);
            diag_elements = (randn(diag_len,1) + 1i*randn(diag_len,1)) / sqrt(2*Lr);
            Sigma(1:diag_len, 1:diag_len) = diag(diag_elements);
            rho_q=rand(Lr,1);
            f_r_m = zeros(Lr, 1);
            converged = false;
            channel_capacity_prev = 0;
            [F,f_r_m]  = calculate_F(theta_q,phi_q,lambda,r,f_r_m);
            H_r = F' * Sigma * G;
            
            for iter = 1:50
                cvx_begin
                variable Q(4,4)
                maximize( log_det( eye(M) + (1/sigma) * H_r * Q * H_r' ) )
                subject to
                trace(Q) <= power;  % 能量约束
                Q >= 0;             % Q是半正定的
                cvx_end
                m0=0;
                for i = 1:4
                    m0 = m0 + 1;
                    r_mi= r(:,m0);                % 初始化变量
                    current_objective_value =0;
                    max_sca_iter = 60;            % 最大迭代次数
                    for sca_iter = 1:max_sca_iter
                        previous_objective_value=current_objective_value;
                        [F,f_r_m]  = calculate_F(theta_q,phi_q,lambda,r,f_r_m);
                        [f_rm, ~]=compute_field_response(r_mi, theta_q, phi_q, lambda);
                        H_r = F' * Sigma * G;
                        B_m= calculate_B_m(G,Q,H_r,m0,sigma,Sigma);
                        [grad_g, delta_m]=calculate_gradients(B_m,f_rm,r_mi,lambda,theta_q,phi_q);
                        r_mii=grad_g/delta_m+r_mi;
                        is_feasible =true;
                        if any(r_mii < 0 | r_mii > square_size)
                            is_feasible = false;
                        else
                            for k = 1:M
                                if k ~= m0
                                    distance = norm(r_mii - r(:, k), 2);
                                    if distance < D
                                        is_feasible = false;
                                        break;
                                    end
                                end
                            end
                        end
                        if(is_feasible)
                            r_mi = r_mii;
                            % 满足条件，更新 r
                        else
                            H = delta_m*eye(2);
                            f = -(grad_g + delta_m * r_mi);
                            % 定义不等式约束（距离约束）
                            A = [];
                            b1 = [];
                            for k = 1:M
                                if k ~= m0
                                    r_k = r(:, k);
                                    a_k = (r_mi - r_k)/norm(r_mi - r_k);
                                    b_row = -D-a_k'*r_k;
                                    A = [A; -a_k'];
                                    b1 = [b1;b_row];
                                    lb = zeros(2,1);
                                    ub = square_size*ones(2,1);
                                end
                            end
                            % 使用 quadprog 进行求解
                            %options = optimset('Display', 'off');  % 不显示迭代信息
                            options = optimset('Display', 'off', 'TolFun', 1e-8, 'TolX', 1e-8);
                            r_mnew = quadprog(H, f, A,b1, [], [], lb, ub, r_mi, options);
                            if isempty(r_mnew)
                                warning('SCA 优化失败，保留原位置');
                                r_mnew = r_mi;  % 保留原位置
                            end
                            r_mi = r_mnew;
                        end
                        r(:,m0) = r_mi;
                        [F,f_r_m]  = calculate_F(theta_q,phi_q,lambda,r,f_r_m);
                        H_r = F' * Sigma * G;
                        B_m = calculate_B_m(G,Q,H_r,m0,sigma,Sigma);
                        [f_r_new, ~]=compute_field_response(r_mi, theta_q, phi_q, lambda);
                        current_objective_value = real(f_r_new'*B_m*f_r_new)
                        if(abs(current_objective_value-previous_objective_value)<xii)
                            break;
                        end
                    end
                    r(:,m0)=r_mi;
                end
                [F,f_r_m]  = calculate_F(theta_q,phi_q,lambda,r,f_r_m);
                H_r = F' * Sigma * G;
                H_rQH = H_r * Q * H_r';
                channel_capacity_current =log2(det( eye(M) + (1/sigma) * H_rQH ))
                % 更新信道容量
                if (abs(channel_capacity_current - channel_capacity_prev) <xi)
                    break;
                end
                channel_capacity_prev = channel_capacity_current;
            end
            capacity_trials(trial) = channel_capacity_current;
        end
        % 计算平均容量
        average_capacity_Proposed(u, j) = mean(capacity_trials)
    end
end

% 绘图
figure;
hold on;
plot(A_lambda_values, average_capacity_Proposed(:, 1), 'b-', 'LineWidth', 1.5); % Proposed, Lr=15
plot(A_lambda_values, average_capacity_Proposed(:, 2), 'r--', 'LineWidth', 1.5); % Proposed, Lr=10

% 设置图例和标签
legend('Proposed, Lr=15', 'Proposed, Lr=10');
xlabel('Normalized region size A/\lambda');
ylabel('Capacity (bps/Hz)');
title('Capacity vs. Normalized Receive Region Size for Proposed Method');
grid on
hold off;
function [grad_g, delta_m] = calculate_gradients( B_m,f_rm,r_mi,lambda,theta_qi,phi_qi)
b=B_m*f_rm;
amplitude_bq = abs(b);     % 幅度 |bq|
phase_bq = angle(b);       % 相位 ∠bq
x = r_mi(1); y = r_mi(2);
rho_qi=x*sin(theta_qi).*cos(phi_qi) + y*cos(theta_qi);
kappa = (2 * pi / lambda) *rho_qi - phase_bq;
term_x = amplitude_bq .* sin(theta_qi) .* cos(phi_qi) .* sin(kappa);  % Lr x 1
term_y = amplitude_bq .* cos(theta_qi) .* sin(kappa);                % Lr x 1
grad_x = - (2 * pi / lambda) * sum(term_x);
grad_y = - (2 * pi / lambda) * sum(term_y);
grad_g = [grad_x; grad_y];
sum_abs_b = sum(abs(b));
delta_m = (8 * pi^2) / (lambda^2) * sum_abs_b;
end

function [F,f_r_m] = calculate_F(theta_q,phi_q,lambda,r,f_r_m)
M=4;
for m = 1:M
    x = r(1,m);
    y = r(2,m);
    phase = 2*pi/lambda * (x*sin(theta_q).*cos(phi_q) + y*cos(theta_q));
    f_r_m = exp(1i*phase);
    F(:, m) = f_r_m;
end
end
function B_m = calculate_B_m(G,Q,H_r,m0,sigma,Sigma)
N=4;
[U_Q, V_Q] = eig(Q);
Lambda_Q_sqrt = sqrt(V_Q);
W_r = H_r * U_Q * Lambda_Q_sqrt;
W_r_H=W_r';
W_r_H(:, m0) = [];
A_m = inv(eye(N) + (1/sigma^2) * (W_r_H * W_r_H'));
B_m=Sigma * G * U_Q * Lambda_Q_sqrt * A_m * Lambda_Q_sqrt * U_Q' * G' * Sigma';
end
function [f_r, phase] = compute_field_response(r, theta_q, phi_q, lambda)
x = r(1); y = r(2);
phase = 2*pi/lambda * (x*sin(theta_q).*cos(phi_q) + y*cos(theta_q));
f_r = exp(1i*phase);
end

