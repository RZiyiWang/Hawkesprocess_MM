from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import logging
import tensorflow as tf
from tf_agents.specs import BoundedArraySpec, BoundedTensorSpec
from tf_agents.trajectories.time_step import StepType
from tf_agents.trajectories.time_step import TimeStep
from tf_agents.environments import tf_py_environment
from tf_agents.environments import *
import time
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import actor_distribution_rnn_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.policies.policy_saver import PolicySaver
from tf_agents.utils import common
from tf_agents.environments import *
from tf_agents.policies import *
import os
import functools
from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import sequential
from tf_agents.policies import random_tf_policy
from tf_agents.keras_layers import dynamic_unroll_layer
from tf_agents.drivers import dynamic_step_driver
from tqdm import tqdm 


#Define: DQN agent
class QuoteAgent:
    def __init__(self,env,name,num_iterations=10000,continue_training=False,continue_saved_policy=None):

        train_sequence_length=1
        # Params for QNetwork
        fc_layer_params=(100,) #20change to 100
        # Params for QRnnNetwork
        input_fc_layer_params=(50,) #10
        lstm_size=(20,) #4
        output_fc_layer_params=(20,) #4

        # Params for collect
        initial_collect_steps=2000
        collect_steps_per_iteration=1
        epsilon_greedy=0.1 
        replay_buffer_capacity=100000
        # Params for target update
        target_update_tau=0.05
        target_update_period=5
        # Params for train
        train_steps_per_iteration=1
        batch_size=64
        learning_rate=1e-3
        n_step_update=1
        gamma=0.99
        reward_scale_factor=1.0
        gradient_clipping=None
        use_tf_functions=True
        # Params for eval
        num_eval_episodes=10 #better 200
        eval_interval=200 #1000
        # Params for checkpoints
        train_checkpoint_interval=10000
        policy_checkpoint_interval=5000
        rb_checkpoint_interval=20000
        # Params for summaries and logging
        log_interval=200 #1000
        summary_interval=200 #1000
        summaries_flush_secs=10
        debug_summaries=False
        summarize_grads_and_vars=False
        eval_metrics_callback=None
        KERAS_LSTM_FUSED = 2

        self.name=name
        self.env=env

        logits = functools.partial(
            tf.keras.layers.Dense,
            activation=None,
            kernel_initializer=tf.random_uniform_initializer(minval=-0.03, maxval=0.03),
            bias_initializer=tf.constant_initializer(-0.2))

        dense = functools.partial(
            tf.keras.layers.Dense,
            activation=tf.keras.activations.relu,
            kernel_initializer=tf.compat.v1.variance_scaling_initializer(
            scale=2.0, mode='fan_in', distribution='truncated_normal'))

        fused_lstm_cell = functools.partial(
            tf.keras.layers.LSTMCell, implementation=KERAS_LSTM_FUSED)


        def create_feedforward_network(fc_layer_units, num_actions):
            return sequential.Sequential(
            [dense(num_units) for num_units in fc_layer_units]
            + [logits(num_actions)])

        def create_recurrent_network(
            input_fc_layer_units,
            lstm_size,
            output_fc_layer_units,
            num_actions):
            rnn_cell = tf.keras.layers.StackedRNNCells(
                [fused_lstm_cell(s) for s in lstm_size])
            return sequential.Sequential(
                [dense(num_units) for num_units in input_fc_layer_units]
                + [dynamic_unroll_layer.DynamicUnroll(rnn_cell)]
                + [dense(num_units) for num_units in output_fc_layer_units]
                + [logits(num_actions)])
        

        root_dir = os.path.expanduser(self.name)
        train_dir = os.path.join(root_dir, 'train')
        eval_dir = os.path.join(root_dir, 'eval')

        train_summary_writer = tf.compat.v2.summary.create_file_writer(train_dir, flush_millis=summaries_flush_secs * 1000)
        train_summary_writer.set_as_default()

        eval_summary_writer = tf.compat.v2.summary.create_file_writer(eval_dir, flush_millis=summaries_flush_secs * 1000)
        
        eval_metrics = [
            tf_metrics.NumberOfEpisodes(),
            tf_metrics.EnvironmentSteps(),
            tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
            tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes)] 
        
        global_step = tf.compat.v1.train.get_or_create_global_step()

        with tf.compat.v2.summary.record_if(
            lambda: tf.math.equal(global_step % summary_interval, 0)):

            tf_env = tf_py_environment.TFPyEnvironment(self.env)
            eval_tf_env = tf_py_environment.TFPyEnvironment(self.env)

            action_spec = tf_env.action_spec()
            num_actions = action_spec.maximum - action_spec.minimum + 1

            if train_sequence_length != 1 and n_step_update != 1:
                raise NotImplementedError(
                'train_eval does not currently support n-step updates with stateful '
                'networks (i.e., RNNs)')
            
            if train_sequence_length > 1:
                q_net = create_recurrent_network(
                input_fc_layer_params,
                lstm_size,
                output_fc_layer_params,
                num_actions)
            else:
                q_net = create_feedforward_network(fc_layer_params, num_actions)
                train_sequence_length = n_step_update
            
            tf_agent = dqn_agent.DqnAgent(
                tf_env.time_step_spec(),
                tf_env.action_spec(),
                q_network=q_net,
                optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate),
                target_update_tau=target_update_tau,
                target_update_period=target_update_period,
                td_errors_loss_fn=common.element_wise_squared_loss,
                gamma=gamma,
                train_step_counter=global_step,
                epsilon_greedy=epsilon_greedy,

                n_step_update=n_step_update,
                reward_scale_factor=reward_scale_factor,
                gradient_clipping=gradient_clipping,
                debug_summaries=debug_summaries,
                summarize_grads_and_vars=summarize_grads_and_vars)
            
            tf_agent.initialize()

            replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
                data_spec=tf_agent.collect_data_spec,
                batch_size=tf_env.batch_size,
                max_length=replay_buffer_capacity)

            replay_buffer_observer = replay_buffer.add_batch 

            train_metrics = [
                tf_metrics.NumberOfEpisodes(),
                tf_metrics.EnvironmentSteps(),
                tf_metrics.AverageReturnMetric(),
                tf_metrics.AverageEpisodeLengthMetric(),
            ]
            
            eval_policy = tf_agent.policy
            collect_policy = tf_agent.collect_policy

            collect_driver = dynamic_step_driver.DynamicStepDriver(
                tf_env,
                collect_policy,
                observers=[replay_buffer_observer] + train_metrics,
                num_steps=collect_steps_per_iteration) # collect x steps for each training iteration 

            train_checkpointer = common.Checkpointer(
                ckpt_dir=train_dir,
                agent=tf_agent,
                global_step=global_step,
                metrics=metric_utils.MetricsGroup(train_metrics, 'train_metrics'))
            policy_checkpointer = common.Checkpointer(
                ckpt_dir=os.path.join(train_dir, 'policy'),
                policy=eval_policy,
                global_step=global_step)
            rb_checkpointer = common.Checkpointer(
                ckpt_dir=os.path.join(train_dir, 'replay_buffer'),
                max_to_keep=1,
                replay_buffer=replay_buffer)
            
            train_checkpointer.initialize_or_restore()
            rb_checkpointer.initialize_or_restore()
            
            # Dataset generates trajectories with shape [Bx2x...]
            dataset = replay_buffer.as_dataset(
                num_parallel_calls=3,
                sample_batch_size=batch_size,
                num_steps=train_sequence_length + 1).prefetch(3)
        

            if use_tf_functions:
                # To speed up collect use common.function.
                collect_driver.run = common.function(collect_driver.run)
                tf_agent.train = common.function(tf_agent.train)
            
            logging.info(
                '----------------------- %s : Initializing replay buffer by collecting experience for %d steps with '
                'a random policy.', self.name, initial_collect_steps)

            initial_collect_policy = random_tf_policy.RandomTFPolicy(
                tf_env.time_step_spec(), tf_env.action_spec())
            
            dynamic_step_driver.DynamicStepDriver(
                tf_env,
                initial_collect_policy,
                observers=[replay_buffer_observer] + train_metrics,
                num_steps=initial_collect_steps).run()    
            
            
            results = metric_utils.eager_compute(
                eval_metrics,
                eval_tf_env,
                eval_policy,
                num_episodes=num_eval_episodes,
                train_step=global_step,
                summary_writer=eval_summary_writer,
                summary_prefix='Metrics',
            )
            
            if eval_metrics_callback is not None:
                eval_metrics_callback(results, global_step.numpy())
            metric_utils.log_metrics(eval_metrics)
            
            
            time_step = None
            policy_state = collect_policy.get_initial_state(tf_env.batch_size) #policy_state = ()
            iterator = iter(dataset)

            timed_at_step = global_step.numpy()
            time_acc = 0
            
            def train_step(iterator):
                experience, _ = next(iterator)
                return tf_agent.train(experience)
            
            if use_tf_functions:
                train_step = common.function(train_step)                

            for _ in tqdm(range(num_iterations)):
                start_time = time.time()
                time_step, policy_state = collect_driver.run(time_step=time_step,policy_state=policy_state)
                for _ in range(train_steps_per_iteration):
                    train_loss = train_step(iterator)
                time_acc += time.time() - start_time

                if global_step.numpy() % log_interval == 0:
                    logging.info('step = %d, loss = %f', global_step.numpy(),
                                train_loss.loss)
                    steps_per_sec = (global_step.numpy() - timed_at_step) / time_acc
                    logging.info('%.3f steps/sec', steps_per_sec)
                    tf.compat.v2.summary.scalar(
                        name='global_steps_per_sec', data=steps_per_sec, step=global_step)
                    timed_at_step = global_step.numpy()
                    time_acc = 0

                for train_metric in train_metrics:
                    train_metric.tf_summaries(
                        train_step=global_step, step_metrics=train_metrics[:2])

                if global_step.numpy() % train_checkpoint_interval == 0:
                    train_checkpointer.save(global_step=global_step.numpy())

                if global_step.numpy() % policy_checkpoint_interval == 0:
                    policy_checkpointer.save(global_step=global_step.numpy())

                if global_step.numpy() % rb_checkpoint_interval == 0:
                    rb_checkpointer.save(global_step=global_step.numpy())

                if global_step.numpy() % eval_interval == 0:
                    results = metric_utils.eager_compute(
                        eval_metrics,
                        eval_tf_env,
                        eval_policy,
                        num_episodes=num_eval_episodes,
                        train_step=global_step,
                        summary_writer=eval_summary_writer,
                        summary_prefix='Metrics',
                    )
                    if eval_metrics_callback is not None:
                        eval_metrics_callback(results, global_step.numpy())
                    metric_utils.log_metrics(eval_metrics)

        saver = PolicySaver(eval_policy, batch_size = None)
        saver.save(self.name + '_saved_policy')