% Initialize parameters
clear
clc
close all


%% Parameters
g = 9.81;   % Gravity constant
m = 74;           % Pendulum mass
% I = 62.91;         % Pendulum Inertia
% L = sqrt(I/m);    % Pendulum length
c = 5.73;          % Pendulum Viscosity
L = 1.7*0.52;
I = 0.971 * m * L^2;
T_r_delay = 0;
        
% State Space
AG = [0 1; 0.3*m*g*L/I -c/I];
BG = [0; 1/I];
CG = eye(2);
DG = 0;
NameData = 'CTypeC,RGain,MusOff,Paper';

A = [0 1; m*g*L/I -c/I];
B = [0; 1/I];
C = eye(2);
D = 0;


T_d_input = 0.06; % Input delay in seconds to plant (Sensory delay)
T_d_output = 0.06; % Output delay in seconds from plant (Motor delay)
dt = 0.004; % Time step sec
T = 60; % Total simulation time sec
N = T/dt; % Number of time steps
Fs = 250;           % Sampling rate [Hz]
Ns = T*Fs;          % Number of samples in segment duration
f = Fs*(0:(Ns/2))/Ns;

% Noise Coefficients
mn_coeff_base = 1;
mn_coeff = mn_coeff_base;
thn_coeff = 0.002;
thdotn_coeff = 0.0036;

motor_noiseTest  = readmatrix("NoiseP1.txt");
th_noiseTest  = readmatrix("NoiseP2.txt");
thdot_noiseTest  = readmatrix("NoiseP3.txt");

mainAddress = pwd;
mainAddress = horzcat(mainAddress,'\');
% For Linux
%mainAddress = horzcat(mainAddress,'/');


train_delays = [0 ];
Gain_delays = [0];
alpha_list = [0];
% robotic_delays = 0:20:300;
robotic_delays = 0;

beta_list = [0 0]; % K coeff

Q_options = 1;
% R_options = 10.^(-1.6);
% Qt_options = 1;
R_options = 10.^(-4:0.1:3)


prev_act_u_hat_error = 0;
prev_act_u_hat = 0;
prev_act_u = 0;
%% Simulation

% It's not important for this code, mostly used when we want 
iteration = 1; 
% Simulation
for iQ = 1:numel(Q_options)   
    for iR = 1:numel(R_options)

        for iTrain = 1:size(robotic_delays,2)

            close all
            alpha = alpha_list(iTrain);
            beta = beta_list(iTrain);
            T_r_delay = robotic_delays(iTrain);
            
            
            % Transfer function for designing the LQR With Robotic delay
            sys_delayed = ss(AG, BG, CG, DG, 'InputDelay', T_d_input + Gain_delays(iTrain), 'OutputDelay', T_d_output);
            % % Natural delay
            % sys_delayed = ss(A, B, C, D, 'InputDelay', T_d_input, 'OutputDelay', T_d_output);
            
            % Padé approximation for the delay
            order = 2; % Order of the Padé approximation (adjust as needed)
            sys_delayed_approx = pade(sys_delayed, order);
            
            [A_d, B_d, C_d, D_d] = ssdata(sys_delayed_approx);
            
            
            % Define the state and control weight matrices Q and R
            Q_d = eye(size(A_d)) .* Q_options(iQ); % You may need to adjust the weights based
            
            Q = eye(2) .* Q_options(iQ); % State weighting matrix
            R = R_options(iR); % Control weighting matrix
            
            % Solve the continuous-time Algebraic Riccati Equation (CARE)
            [P, ~, ~] = care(A, B, Q, R);
            
            % Compute the LQR gain
            K2 = R \ B' * P; % Equivalent to K = inv(R) * B' * P
            
            K = lqr(A_d, B_d, Q_d, R);
        
            % The gains for the real dynamics
            K = K(5:6);
        
            % Time to predict (forward prediction)
            T_d_forward = T_d_output + T_d_input + train_delays(iTrain);
            train_delays(iTrain)
            nForward = (round(T_d_forward/dt));
        
            Description = horzcat('LQR','ITR',num2str(iteration), ...
                                    'DN',num2str((T_d_input + T_d_output)*1000), ...
                                    'DR',num2str((T_r_delay)*1000), ...
                                    'DT',num2str(train_delays(iTrain)*1000), ...
                                    'DG',num2str(Gain_delays(iTrain)*1000), ...
                                    'MN',num2str(mn_coeff), ...
                                    'TN',num2str(thn_coeff), ...
                                    'TDN',num2str(thdotn_coeff), ...
                                    'R',num2str(R), ...
                                    'QT',num2str(Q(1,1)), ...
                                    'QTD',num2str(Q(2,2)), ...
                                    'alpha',num2str(alpha), ...
                                    'beta',num2str(beta), ...
                                    NameData);
                
            mkdir(Description);
        
        
            for iTest = 1:30
                
                iTest
        
                prevReversal = 0;
                prevPStorque = 0;
                prevPStorqueHistory = 0;

                % Initialize variables
                init_angle_rad = 0 * pi / 180; 
                state = [init_angle_rad; 0]; % [theta; theta_dot]
                state_hat = [init_angle_rad; 0]; % Estimated state for Smith Predictor
                u = zeros(1, N); % Control input
                state_hat_history = zeros(2, N);
                prev_state = state;        

                for k = 1:N
                
                    % State and torque for the predictor
                    state_hat = zeros(2,round(T_d_input/dt) + round(T_d_output/dt) + round(T_r_delay/dt));
                    u_hat = zeros(1,round(T_d_input/dt) + round(T_d_output/dt) + round(T_r_delay/dt));
                    state_hat(:,1) = state(:,k);
                    state_hat(1,1) = state_hat(1,1) + thn_coeff * th_noiseTest(k,iTest); 
                    state_hat(2,1) = state_hat(2,1) + thdotn_coeff * thdot_noiseTest(k,iTest);
        
                    state_hat_error(1,1) = thn_coeff;
                    state_hat_error(2,1) = thdotn_coeff; 
                    now_state = state(:,k);
                    % error to actual feedback (the beta factor will be multiplied
                    % later)
                    if k>(round(T_d_input/dt) + round(T_d_output/dt) + round(T_r_delay/dt)) && k<N
                        error_prediction = state_hat(:,1) - state_hat_history(:,k+1);                
                    else
                        error_prediction = 0;
                    end
                    
                    nForward = 1;
                    % Forward Simulation for Smith Predictor
                    for iForward = 1:nForward
                        if (k+iForward-2)>=(round(T_d_input/dt) + round(T_d_output/dt) + round(T_r_delay/dt))
                            u_hat(iForward) = -K * (state_hat(:, iForward));
                            u_hat_error(iForward) = abs(u_hat(iForward)) * mn_coeff;
                            
                        else
                            u_hat(iForward) = 0;
                            u_hat_error(iForward) = 0;
                            
                        end
                        state_hat_dot = A * state_hat(:,iForward) + B * u_hat(iForward);
                        state_hat(:, iForward+1) = state_hat(:, iForward) + state_hat_dot * dt;
        
                        state_hat_dot_error = A * state_hat_error(:,iForward) + B * u_hat_error(iForward);
                        state_hat_error(:, iForward+1) = state_hat_error(:, iForward) + state_hat_dot_error * dt;
                
                    end   
                    
                    % Worst case scenario - a correction for the prediction will be
                    % added 
                    correction_state_hat(1,1) = sign(state_hat(1,iForward)) * thn_coeff * alpha;
                    correction_state_hat(2,1) = sign(state_hat(2,iForward)) * thdotn_coeff * alpha;
                    
        
        %             correction_state_hat = correction_state_hat *(1-beta) + error_prediction * beta;
                    correction_state_hat = correction_state_hat + error_prediction * beta;
        
                    % LQR implementation
                    correction_u_hat = -K * correction_state_hat;
        
                    % LQR control based on Smith Predictor
                    if k <= (N-(round(T_d_input/dt) + round(T_d_output/dt) + round(T_r_delay/dt)))
                %         u(k) = -K * (state_hat(:, end));        
                        u(k + (round(T_d_input/dt) + round(T_d_output/dt) + round(T_r_delay/dt))-1) = u_hat(iForward) + correction_u_hat ;
                        state_hat_history(:,k+ (round(T_d_input/dt) + round(T_d_output/dt) + round(T_r_delay/dt))) = state_hat(:, iForward);
                    end
                
        
                    torque_final = u(k);
                
                    % Muscle Dynamics
                    [torque_final, prev_act_u] = ...
                            muscle_dyn(prev_act_u, torque_final, dt);
                    [psTorque, prevReversal, prevPStorque, prevPStorqueHistory] = ...
                            passive_torque(now_state, prev_state, prevReversal, ...
                            prevPStorque, prevPStorqueHistory, m, g, L);
        
        %             psTorque = 0;
                    prev_state = now_state;
                    
                    % % Assymetry in motor noise
                    % if torque_final<0 
                    %     mn_coeff = mn_coeff_base * 0.77;
                    % else
                    %     mn_coeff = mn_coeff_base;
                    % end

                    if torque_final<-256
                        torque_final = -256;
                    end
                    if torque_final>84
                        torque_final = 84;                
                    end

                    finaltorque(k) = torque_final * (1 + custom_TN(torque_final) * mn_coeff * motor_noiseTest(k,iTest)) - psTorque;
                    
                    % Dynamic of the system
                    state_dot = A * state(:, k) + B * finaltorque(k);
                    state(:, k+1) = state(:, k) + state_dot * dt;
                    acc_temp(k) = state_dot(2);
                
                end
                
                torque_temp = finaltorque;
                pos_temp = state(1,1:end-1);
                vel_temp = state(2,1:end-1);
                
                specturmfirst = fft(vel_temp*180/pi);
                P2first = abs(specturmfirst/Ns);
                P1first = P2first(1:Ns/2+1);
                P1first(2:end-1) = 2*P1first(2:end-1);
                figure('visible','off');
                plot(f(1:100),P1first(1:100),'b*') 
                title('Amplitude spectrum of angle signal')
                xlabel('f (Hz)')
                ylabel('Amplitudes (deg)')
                nameFigure = horzcat('Pendulum_FreqSpec_',num2str(iTest));
                formatFile = horzcat(mainAddress, Description, '\', ...
                    nameFigure);
                saveas(gcf,horzcat(formatFile,'.png')) 
                totalSpectrum (iTrain,iTest,:) = P1first;
        
        
                
                specturmfirst = fft(torque_temp);
                P2first = abs(specturmfirst/Ns);
                P1first = P2first(1:Ns/2+1);
                P1first(2:end-1) = 2*P1first(2:end-1);
                figure('visible','off');
                plot(f(1:100),P1first(1:100),'b*') 
                title('Amplitude spectrum of torque signal')
                xlabel('f (Hz)')
                ylabel('Amplitudes (N.m)')
                nameFigure = horzcat('Torque_FreqSpec_',num2str(iTest));
                formatFile = horzcat(mainAddress, Description, '\', ...
                    nameFigure);
                saveas(gcf,horzcat(formatFile,'.png')) 
                totalSpectrumTorque (iTrain,iTest,:) = P1first;
        
                
                
                
                torque_NOposvel = torque_temp - ...
                    ( pos_temp .* (m*g*L) ) - vel_temp*c;
                
                
                torque_NOvelaccel = torque_temp - acc_temp*I - vel_temp*c;
                
                % Plot results
                sampleShow = N;
                samplePrediction = size(state_hat_history,2);
                time = dt:dt:(sampleShow)*dt;
                timePridiction = dt:dt:(samplePrediction)*dt;
                
                
                if max(state(1,:)) *180/pi > 6 || ...
                        min(state(1,:)) * 180/pi < -3
                    result_char = 'f';
                else
                    result_char = 's';
                end
        
                History_result(iTest) = result_char;
                
                figure('visible','off');
                subplot(3, 1, 1);
                plot(time, state(1, 1:sampleShow).*180/pi, 'r', timePridiction, state_hat_history(1, :).*180/pi, 'b');
                legend('Actual theta', 'Estimated theta (Smith Predictor)');
                xlim ([0 60])
        %         ylim ([-3 6])
                xlabel('Time (s)');
                ylabel('Angle (deg)');
                title('Inverted Pendulum with Smith Predictor and LQR');
                nameFigure = horzcat('Pendulum_Behaviour_',num2str(iTest),...
                    result_char);
                subplot(3, 1, 2);
                plot(time, finaltorque(1:sampleShow), 'g');
                legend('Control Input (N.m)');
                xlabel('Time (s)');
                ylabel('Control Input');
                subplot(3, 1, 3);
                plot(time, state(1, 1:sampleShow).*180/pi, 'r');
                legend('Actual theta');
                xlim ([0 60])
        %         ylim ([-3 6])
                xlabel('Time (s)');
                ylabel('Angle (deg)');
        
                formatFile = horzcat(mainAddress, Description, '\', ...
                    nameFigure);
                saveas(gcf,horzcat(formatFile,'.png')) 
        
        
        
                state_torque(1:2,:) = state(:,1:end-1);
                state_torque(3,:) = finaltorque;
                save(formatFile,'state_torque')
                
                signal1 = acc_temp((floor(T_r_delay/dt))+1:sampleShow)*180/pi;
                signal2 = torque_NOposvel(1:sampleShow-floor(T_r_delay/dt));
                
                corrSignal = xcorr(pos_temp(floor(T_r_delay/dt)+1:sampleShow)*180/pi, ...
                    torque_NOvelaccel(1:sampleShow-floor(T_r_delay/dt)),'unbiased');
                corrSignal2 = xcorr(signal1, signal2,'unbiased');
        
        %         if Gain_delays(iTrain) == Gain_delays(1) && iTest == 10
        %             sample1Trace = pos_temp;
        %         end
        % 
        %         if Gain_delays(iTrain) == Gain_delays(end) && iTest == 10
        %             sample2Trace = pos_temp;
        %         end
        
                figure('visible','off');
                x0=300;
                y0=300;
                width=500;
                height=400;  
                set(gcf,'position',[x0,y0,width,height])   
                plot(acc_temp,torque_NOposvel)
        
                nameFigure = horzcat('TorqueAcceleration',num2str(iTest),...
                    result_char);
                title(nameFigure,'fontsize',15)
                xlabel('Acceleration (deg/sec^2)','fontsize',12)
                ylabel('Torque (N.m)','fontsize',12) 
                %                     xlim([-0.8 1])
                %                     ylim([-30 30])
                formatFile = horzcat(mainAddress, Description, '\', ...
                    nameFigure);
                saveas(gcf,horzcat(formatFile,'.png'))  
        
                tCorr = numel(corrSignal);
                time = dt:dt:dt*tCorr;
                time = time - dt*tCorr/2;
                figure('visible','off');
                x0=300;
                y0=300;
                width=800;
                height=350;  
                set(gcf,'position',[x0,y0,width,height])   
                plot(time,corrSignal)
                [maxCorr, Ind] = min(corrSignal);
                Ind = (Ind - tCorr/2)*dt;
                text(Ind, maxCorr, ['(', num2str(Ind), ', ', num2str(maxCorr), ')'], 'VerticalAlignment', 'bottom');
                nameFigure = horzcat('CrossCorrTorqueAngleCTorqueOut',num2str(iTest),...
                    result_char);
                title(nameFigure,'fontsize',15)
                xlabel('Time samples','fontsize',12)
                ylabel('CrossCorrTorqueAngle','fontsize',12) 
                formatFile = horzcat(mainAddress, Description, '\', ...
                    nameFigure);
                saveas(gcf,horzcat(formatFile,'.png'))  
                
                
                tCorr = numel(corrSignal2);
                time = dt:dt:dt*tCorr;
                time = time - dt*tCorr/2;
                figure('visible','off');
                x0=300;
                y0=300;
                width=800;
                height=350;  
                set(gcf,'position',[x0,y0,width,height])
                plot(time,corrSignal2)
                [maxCorr, Ind] = max(corrSignal2);
                Ind = (Ind-tCorr/2)*dt;
                text(Ind, maxCorr, ['(', num2str(Ind), ', ', num2str(maxCorr), ')'], 'VerticalAlignment', 'top');
                nameFigure = horzcat('CrossCorr-TorqueAcceleration-',num2str(iTest),...
                    result_char);
                title(nameFigure,'fontsize',15)
                xlabel('Time samples','fontsize',12)
                ylabel('CrossCorrTorqueAcc','fontsize',12) 
                formatFile = horzcat(mainAddress, Description, '\', ...
                    nameFigure);
                saveas(gcf,horzcat(formatFile,'.png')) 
        
        
                window_size = 10/dt;
        
                % Number of windows
                num_windows = floor(length(signal1) / window_size);
        
                cc_results = zeros( num_windows, 2*window_size-1);
                cc_results_norm = zeros( num_windows, 2*window_size-1);
               
                % Compute cross-correlation for each window
                for i = 1:num_windows
                    start_idx = (i-1)*window_size + 1;
                    end_idx = i*window_size;
                    
                    segment1 = signal1(start_idx:end_idx);
                    segment2 = signal2(start_idx:end_idx);
                    
                    cc_results(i, :) = xcorr(segment1, segment2,'unbiased');
                    cc_results_norm(i, :) = cc_results(i, :) ./ max(cc_results(i,:));
                end
                CCNormal(iTest,:) = cc_results_norm(i, :) ;
                CCNonNormal(iTest,:) = cc_results(i, :) ;
                coRRLength = floor(numel(corrSignal2)/2);
        
                corrSignals(iTest,:) = corrSignal2(coRRLength-4000:coRRLength+4000);
        
            end
            corrSignalsMean(iTrain,:) = mean(corrSignals);
            corrSignalsLast(iTrain,:) = corrSignals(iTest,:);
            TrainCC(iTrain,:) = mean(CCNormal(find(History_result == 's'),:));
        %     TrainCC(iTrain,:) = CCNormal(iTest,:);
        
            TrainCCNonNormal(iTrain,:)=mean(CCNonNormal(find(History_result == 's'),:));
            TrainCCstdNonNormal(iTrain,:)=std(CCNonNormal(find(History_result == 's'),:));
            ssStudy = CCNonNormal(find(History_result == 's'),:);
            
            
            nameFigure = 'TestResults';
            formatFile = horzcat(mainAddress, Description, '\', ...
                    nameFigure);
            writematrix(History_result, horzcat(formatFile, 'HR.txt'))

        
        end

    end
end



%%
function [torque_final, prev_act] = muscle_dyn(prevAct, true_torque, dt)

    if sign(true_torque) == -1
        MVC = 256;
    else
        MVC = 84;
    end


    activation = true_torque/MVC;
    if sign(true_torque) ~= sign(prevAct)
        prevAct = 0;
    end

    if abs(activation) >= abs(prevAct)
        tau = 0.01 * (0.5 + 1.5 * abs(prevAct));
    else
        tau = 0.04 / (0.5 + 1.5 * abs(prevAct));
    end

    output = (abs(activation) - abs(prevAct)) / tau * dt ...
        + abs(prevAct);
    
    torque_final = sign(true_torque) * output * MVC;

    prev_act = activation;
end

function [psTorque, prevReversal, prevPStorque, prevPStorqueHistory] = ...
    passive_torque(now_state, prev_state, prevReversal, ...
    prevPStorque, prevPStorqueHistory, m, g, L)
    
    th = now_state(1);
    thdot = now_state(2);

%     prevTh = prev_state(1);
    prevThDot = prev_state(2);

    prevVelocitySign  = sign(prevThDot);
    if prevVelocitySign == 0
        prevVelocitySign = 1;
    end
    
    % Computing the rotation parameter. Rotation is the angle that
    % pendulum traveled from the previous point that it had zero
    % velocity. 
    rotationSign = sign(thdot);
    if rotationSign == 0
        rotationSign = 1;
    end
    if rotationSign ~= prevVelocitySign 
        prevReversal = th;
        prevPStorque = prevPStorqueHistory;
    end
    
    if abs(th - prevReversal) <= 0.03 / 180 * pi              
        rotation = rotationSign * 0.03 / 180 * pi;
    else
        rotation = th - prevReversal;
    end
    
    % This is the formula for computing the passive stiffness based
    % on the formula reported by Loram (Loram et al )
    coeffT = 0.467 * abs(rotation)^(-0.334) * m * 180 / pi * 2 *...
        g * L / (11 * 180/pi);

    psTorque = coeffT * rotation + prevPStorque;

    prevPStorqueHistory = psTorque;
    
end

function TNoise = custom_TN(TorqueInput)
    if TorqueInput <0
        sign = -1;
    else
        sign = 1;
    end
    if sign == -1
        MVC = TorqueInput/256;
        if  MVC < 10
            TNoise = 0.021 * MVC + 0.121;
        else
            TNoise = 0.042 * MVC - 0.109;
        end
    else
        MVC = TorqueInput/84;
        if  MVC < 30
            TNoise = 0.013 * MVC + 0.058;
        else
            TNoise = 0.021 * MVC - 0.182;
        end
    end
end