% Q-learning AP RL model for standing balance.
% Writer: Amin Nasr - SMPLab - amin.nasr@ubc.ca

% This is the environment and the physiological bits of the model. 
classdef Environment < handle

    properties (Access = public)

        % Mechanical characteristics of the inverted pendulum. 
        m;
        h;
        I;
        g;
        c;
        
        % Number of samples in each simulation. 
        num_samples;
        

                % The range of the theta thetadot and torqus that the system can
        % operate or choose.
        th_range = [-3, 6];
        thdot_range = [-15, 15];
        act_range = [-100, 50];


        % The nonlinear passive stiffness suggested by Loram (Loram et al
        % 2007)  
        prevVelocitySign = 1;
        prevPStorque = 0;
        prevReversal = 0;
        prevPStorqueHistory = 0;
        prev_act = 0;
        
        % These values are set according to the physiology of human. 
        % MN: paper (Tracy et al 2007)
        % TN: Paper (Tisserand et all 2022, Fitzpatrick 1994)
        % TDN: Paper (Tisserand 2022, Fitzpatrick 1994)
        mn_coeff = 0.02;
        thn_coeff = 0.003;
        thdotn_coeff = 0.001;

                % building up the noises using pinknoise function of the matlab.
        % Why Pink noise? The pink noise has a frequency spectrum in which
        % the amp is related to the freq with the 1/f function. It's like
        % how human is doing the standing balance. (Paper) 
        m_noise;
        t_noise;
        td_noise;
         
        % Loading the noises for the tests. 
        motor_noiseTest  = readmatrix("Noise1Test.txt");
        th_noiseTest  = readmatrix("Noise2Test.txt");
        thdot_noiseTest  = readmatrix("Noise3Test.txt");

        motor_noise  = readmatrix("Noise1Train.txt");
        th_noise = readmatrix("Noise2Train.txt");
        thdot_noise  = readmatrix("Noise3Train.txt");

    end

    methods

        % In this function the passive stiffness on the ankle will be
        % computed. It is based on the states that the pendulum has, and a
        % history of events which will define the rotation paremeter. 
        
        function setMyConstant(obj, itr, ThRange, ThDotRange)
            
            rng(itr);

            obj.th_range = ThRange;
            obj.thdot_range = ThDotRange;

            obj.m = 74;
            obj.h = 1.70*0.52;
            obj.I = obj.m*obj.h*obj.h;
            obj.g = 9.81;
            obj.c = 5.73;

            obj.num_samples = 250*60*1;

        end

        function psTorque = passive_torque(obj, now_state )
            
            th = now_state(2);
            thdot = now_state(1);
            
            % Computing the rotation parameter. Rotation is the angle that
            % pendulum traveled from the previous point that it had zero
            % velocity. 
            rotationSign = sign(thdot);
            if rotationSign == 0
                rotationSign = 1;
            end
            if rotationSign ~= obj.prevVelocitySign 
                obj.prevReversal = th;
                obj.prevVelocitySign = rotationSign;
                obj.prevPStorque = obj.prevPStorqueHistory;
            end
            
            if abs(th - obj.prevReversal) <= 0.03 / 180 * pi              
                rotation = rotationSign * 0.03 / 180 * pi;
            else
                rotation = th - obj.prevReversal;
            end
            
            % This is the formula for computing the passive stiffness based
            % on the formula reported by Loram (Loram et al )
            coeffT = 0.467 * abs(rotation)^(-0.334) * 2 * obj.m * 180 / pi * ...
                obj.g * obj.h / (11 * 180/pi);

            psTorque = coeffT * rotation + obj.prevPStorque;

            obj.prevPStorqueHistory = psTorque;
            
        end

        % For new simualtions we need to erase the history of passive
        % stiffness from the previous simulation. 
        function reset_pt(obj, now_state)

            obj.prevVelocitySign = sign(now_state(1));
            obj.prevPStorque = 0;
            obj.prevReversal = 0;
            obj.prevPStorqueHistory = 0;
            obj.prev_act = 0;
            
        end
        
        function TNoise = custom_TN(obj, TorqueInput)
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

        % This function will perform the dynamic of the simulation based on
        % the state space model on the inverted pendulum. 
        function [perceived_state, next_state] = dynamics(obj, ...
                true_state, true_torque, ps_torque, t, num_episode, dt)
        
            % note that the noise for the torque are proportional to theta.
            % And the coefficient of the noise for the plantar flexion is
            % 0.77 based on the (Tracy et al 2007)
            % if true_torque < 0 
            %    mn_added = obj.mn_coeff * abs(true_torque) * 0.77 * ...
            %        obj.motor_noise(t,(mod(num_episode,1000)+1));
            %  else
                mn_added = obj.mn_coeff * custom_TN(obj,true_torque) * ... % abs(true_torque) * ...
                    obj.motor_noise(t,(mod(num_episode,1000)+1));
            % end
            
            exerted_torque = true_torque + mn_added;

            A = [-obj.c/obj.I, obj.m*obj.g*obj.h/obj.I; 1, 0];
            B = [1/obj.I; 0];
            C = [-1/obj.I; 0];
            
            diff_state = A * true_state + B * exerted_torque + C * ps_torque;
        
            next_state = diff_state * dt + true_state;
            
            tdn_added = obj.thdotn_coeff * ...
                obj.thdot_noise(t,(mod(num_episode,1000)+1));

            perceived_state(1,1) = next_state(1) + tdn_added;

            tn_added = obj.thn_coeff * ...
                obj.th_noise(t,(mod(num_episode,1000)+1));

            perceived_state(2,1) = next_state(2) + tn_added;


        end

        % running the dynamic of the model and for tests. The only
        % difference is in the noise part. 
        function [perceived_state, next_state, exerted_torque] = ...
                dynamics_test(obj, ...
                true_state, true_torque, ps_torque, t, num_episode, dt)
            
            %if true_torque < 0 
            %    mn_added = obj.mn_coeff * abs(true_torque) * 0.77 * ...
            %        obj.motor_noiseTest(t,(mod(num_episode,1000)));
            %else
                mn_added = obj.mn_coeff * custom_TN(obj,true_torque) * ... % abs(true_torque) * ...
                    obj.motor_noise(t,(mod(num_episode,1000)+1));
            %end
            
            exerted_torque = true_torque + mn_added;

            A = [-obj.c/obj.I, obj.m*obj.g*obj.h/obj.I; 1, 0];
            B = [1/obj.I; 0];
            C = [-1/obj.I; 0];
            
            diff_state = A * true_state + B * exerted_torque + C * ps_torque;
        
            next_state = diff_state * dt + true_state;
            
            tdn_added = obj.thdotn_coeff * ...
                obj.thdot_noiseTest(t,(mod(num_episode,1000)));

            perceived_state(1,1) = next_state(1) + tdn_added;

            tn_added = obj.thn_coeff * ...
                obj.th_noiseTest(t,(mod(num_episode,1000)));

            perceived_state(2,1) = next_state(2) + tn_added;

        end

        % This will check if the simulation finished or not. 
        function failure_flag = failure_check(obj, now_state)
            failure_flag = 0;
            if now_state(2) > obj.th_range(2) /180*pi || now_state(2) < ...
                    obj.th_range(1)/180*pi
                failure_flag = 1;
            end

        end

        % This function will apply the muscle dynamics based on the formula
        % used in OpenSim.
        function torque_final = muscle_dyn(obj, true_torque, MVC, dt)

            activation = true_torque/MVC;
            if sign(true_torque) ~= obj.prev_act
                prevAct = 0;
            else
                prevAct = obj.prev_act;
            end

            if abs(activation) >= abs(prevAct)
                tau = 0.01 * (0.5 + 1.5 * abs(prevAct));
            else
                tau = 0.04 / (0.5 + 1.5 * abs(prevAct));
            end

            output = (abs(activation) - abs(prevAct)) / tau * dt ...
                + abs(prevAct);
            
            torque_final = sign(true_torque) * output * MVC;

            obj.prev_act = activation;
        end

        % This function will initiate the tests and randomly assign a theta
        % and theta dot to the inverted pendulum. This will assure that the
        % initiation dynamically will not result in the failure. 
        function state = initiate_test(obj, totalDelay) 
            
            % In this part, the offset angle because of delay will be
            % computed in the worst case scenario. 
            gTmax = obj.m * obj.g * obj.h * obj.th_range(2) * pi / 180;            
            galpha = gTmax/obj.I;
            Vmax = galpha * totalDelay / 1000;
            deltaThDelay = Vmax ^ 2 / 2 / galpha;
            TBackward = abs(obj.act_range(1))-gTmax;
            balpha = TBackward/obj.I;
            deltaThTorque = Vmax ^ 2 / 2 / balpha;
            dThTot1 = (deltaThTorque + deltaThDelay)* 180/pi;            
            thetaRange = [obj.th_range(1) + dThTot1, obj.th_range(2) - dThTot1];
            
            % In this part, the offset for the lunching the pendulum
            % through the edge (velocity to fall) will be computed. 
            gTmin = obj.m * obj.g * obj.h * (obj.th_range(2)-1) * pi / 180;
            TBackwardMax = abs(obj.act_range(1))-gTmin;
            balphaMax = TBackwardMax/obj.I;
            dThMaxDot = (obj.thdot_range(2)*pi/180) ^ 2 / 2 / balphaMax;
            dThTot2 = (deltaThDelay + dThMaxDot) * 180/pi;
            thInnerLim = [obj.th_range(1) + dThTot2, ...
                obj.th_range(2) - dThTot2];
            
            % Now I will randomize the theta. The theta will be in the
            % range in the thetaRange variable. 
            th_rand = mean(thetaRange) + ...
                (max(thetaRange) - min(thetaRange)) / 8 * randn(1,1);

            % Based on the theta, the theta dot will be initiated. Theta
            % dot will not be initiated in the edge direction at the end of
            % the range theta. The ability to have velocities in the
            % direction of falling will come back linearly when the theta is far
            % from the edge. In the thInnerLim, the theta that the theta
            % dot could be initiated in the full range is shown. 
            if th_rand < thInnerLim(2)
                vel_range(2) = obj.thdot_range(2);
            else
                mLine = obj.thdot_range(2)/(thInnerLim(2)-thetaRange(2));
                bLine = - mLine * thetaRange(2);
                vel_range(2) = mLine * th_rand + bLine;
            end

            if th_rand > thInnerLim(1)
                vel_range(1) = obj.thdot_range(1);
            else
                mLine = obj.thdot_range(1)/(thInnerLim(1)-thetaRange(1));
                bLine = - mLine * thetaRange(1);
                vel_range(1) = mLine * th_rand + bLine;
                
            end
            
            % Randomized theta dot based on the range that is possible
            % according to the theta. 
            thDot_rand = mean(vel_range) + ...
                (max(vel_range) - min(vel_range)) / 6 * randn(1,1);

            state = [thDot_rand; th_rand];

        end

    end

end
