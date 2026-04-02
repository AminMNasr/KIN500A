% Q-learning AP RL model for standing balance.
% Writer: Amin Nasr - SMPLab - amin.nasr@ubc.ca

% This is the Q-Learning method to solve the MDP problem of standing 
% balance. 
classdef QAgent < handle


    properties (Access = public)

        % The number of discretization of the states and the action. 
        n_theta ;
        n_thetadot ;
        n_action;

        % The range of the theta thetadot and torqus that the system can
        % operate or choose.
        th_range ;
        thdot_range;
        act_range ;
             
        % Learning constants
        gam ;
        penBack ;
        penForw ;
        rewBack ; 
        rewForw ;
        alphDenum ;
        expDenum ;

        rew ;
        pen ;

        % Linear descretization of the states and actions.
        n_th ;
        n_thdot ;
        n_acts ;


        % The action which results in torque zero for the start of
        % simulations.
        action_zero ;

        % Qvalues in the simulation
        Q_vals ;
        
        % The number of visits for each state in the simulation, It will be
        % used to determine the exploration rate and alpha factor.
        num_visits ;
        
        % Metabolic cost
        cst ;
        cstc ;
        
    end

    methods

        function setMyConstant(obj, itr, ThRange, ThDotRange,thetaNState, thetaDNState, ...
                reward, punish, gamma, alpha, exp)
                    % The number of discretization of the states and the action. 

            rng(itr);
            obj.n_theta = thetaNState;
            obj.n_thetadot = thetaDNState;
            obj.n_action = 16;

            obj.penBack = punish(1);
            obj.penForw = punish(2);

            obj.rewBack = reward(1);
            obj.rewForw = reward(2);

            obj.gam = gamma;
            obj.alphDenum = alpha;
            obj.expDenum = exp ;

    
            % The range of the theta thetadot and torqus that the system can
            % operate or choose.
            obj.th_range = ThRange;
            obj.thdot_range = ThDotRange;
            obj.act_range = [-100, 50];
                 
    
            % Linear descretization of the states and actions.
            obj.n_th = linspace( obj.th_range(1)*pi/180, ...
                obj.th_range(2)*pi/180, obj.n_theta );
            obj.n_thdot = linspace(obj.thdot_range(1)*pi/180, ...
                obj.thdot_range(2)*pi/180, ...
                obj.n_thetadot);
            obj.n_acts = linspace( obj.act_range(1), obj.act_range(2), ...
                obj.n_action );
    
    
            % The action which results in torque zero for the start of
            % simulations.
            obj.action_zero = find(obj.n_acts == 0);
    
            % Qvalues in the simulation
            obj.Q_vals = zeros( obj.n_theta * obj.n_thetadot , ...
                obj.n_action );
            
            % The number of visits for each state in the simulation, It will be
            % used to determine the exploration rate and alpha factor.
            obj.num_visits = zeros(obj.n_theta * obj.n_thetadot , 1 );
            
            % Metabolic cost


        end
        
        % The function to define index of the Q in the Qval matrix. 
        function Qindx = qind(obj, disc_thdot, disc_th)
            Qindx = (disc_th-1)* obj.n_thetadot + disc_thdot;
        end
        
        % This function will choose an action based on the exploration
        % rate. The exploration rate is based on the number of the visits
        % for that state. 
        function now_act = action(obj, q_ind)
            
            rand_det = obj.num_visits( q_ind );
            
            if (rand < 1/max(1,floor(rand_det/obj.expDenum)))
                % Random action
                now_act = randsample( 1:obj.n_action, 1 );
            else
                % Choose the optimal action
                [~,now_act] = max( obj.Q_vals( q_ind , : ) );                
            end
            obj.num_visits( q_ind ) = obj.num_visits( q_ind ) + 1;
        end

        % This function will be used in the tests, when the system is
        % learned and the policy is greedy. 
        function now_act = policy(obj, q_ind)
            if all(obj.Q_vals( q_ind , : ) == obj.Q_vals( q_ind , 1 ))
                now_act = randsample( 1:obj.n_action, 1 );
            else
                [~,now_act] = max( obj.Q_vals( q_ind , : ));
            end
        end

        
        % These functions will output the reward for the step that is under
        % observation and will update the q-values. This is for failure and
        % will punish the system. 
        
        function instantPunish = qPunish (obj, now_state)
            
            if now_state(2)<0 
                obj.pen = obj.penBack * (1 - now_state(1) * 180/pi) ;
            else
                obj.pen = obj.penForw * (1 + now_state(1) * 180/pi);
            end
            instantPunish = obj.pen;
        end     
               
        function reward = qUpdate_failure (obj, q_ind, act_prev, now_act)
            rand_crit = obj.num_visits( q_ind );
            reward = obj.pen - abs(obj.cst*abs(now_act)) ...
                - obj.cstc * abs(obj.n_acts(now_act)- obj.n_acts(act_prev));
            obj.Q_vals( q_ind, now_act ) = ...
                obj.Q_vals( q_ind, now_act ) + ...
                (1/(1+floor(rand_crit/obj.alphDenum))) * ( reward ...
                - obj.Q_vals( q_ind, now_act ) );
                       
        end

        % These functions will output the reward for the step that is under
        % observation and will update the q-values. This is for standing 
        % upright and will reward the system based on the metabolic cost. 
        
        function instantReward = qReward (obj, now_state)
            
            if now_state(2)<0 
                obj.rew = obj.rewBack;
            else
                obj.rew = obj.rewForw;
            end
            instantReward = obj.rew;
        end
        
        
        function reward = qUpdate(obj, q_ind_prev, q_ind_next, act_prev, ...
                now_act)
            rand_det = obj.num_visits( q_ind_prev );
            reward = obj.rew - abs(obj.cst*abs(obj.n_acts(now_act))) ...
                - obj.cstc * abs(obj.n_acts(now_act)- obj.n_acts(act_prev));
            obj.Q_vals( q_ind_prev, act_prev ) = ...
                obj.Q_vals( q_ind_prev, act_prev ) + ...
                (1/(2+floor(rand_det/obj.alphDenum))) * ( reward ...
                + obj.gam * max( obj.Q_vals( q_ind_next, : ) )...
                - obj.Q_vals( q_ind_prev, act_prev ) );
            
            
        end
                
        % Find states from the theta and theta dot.
        function [disc_th, disc_thdot] = find_state(obj, now_state)          
            disc_th = find( obj.n_th > now_state(2), 1, 'first' );
	        disc_thdot = find( obj.n_thdot > now_state(1) , 1, 'first' );
            if isempty( disc_th )
                disc_th = obj.n_theta;
            end
            if isempty( disc_thdot )
                disc_thdot = obj.n_thetadot;
            end

        end

    end

end
