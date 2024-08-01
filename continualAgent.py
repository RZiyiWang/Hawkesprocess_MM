import time
from absl import logging
import tensorflow as tf
from tf_agents.specs import BoundedArraySpec, BoundedTensorSpec
from dynamics import *
from strategies import *
from constants import *

from tf_agents.agents.ddpg import critic_rnn_network
from tf_agents.agents.sac import sac_agent
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import actor_distribution_rnn_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.policies.policy_saver import PolicySaver
from tf_agents.utils import common
import time
from tqdm import tqdm
from absl import logging


class Agent: 
    def __init__(self, tf_env, eval_tf_env, name="agent",continue_saved_policy=None, initialize_replay_buffer=True):
        self.tf_env = tf_env
        self.eval_tf_env = eval_tf_env
        self.time_step_spec = tf_env.time_step_spec()
        self.observation_spec = self.time_step_spec.observation
        self.action_spec = tf_env.action_spec()
        self.name = name
        
        self.rewards = []
        self.episodes = []
        
        actor_net = actor_distribution_rnn_network.ActorDistributionRnnNetwork(
            self.observation_spec,
            self.action_spec,
            input_fc_layer_params=actor_fc_layers,
            lstm_size=actor_lstm_size,
            output_fc_layer_params=actor_output_fc_layers,
            activation_fn=tf.keras.activations.relu,
            dtype=tf.float32,
            continuous_projection_net=tanh_normal_projection_network.TanhNormalProjectionNetwork)

        critic_net = critic_rnn_network.CriticRnnNetwork(
            (self.observation_spec, self.action_spec),
            observation_fc_layer_params=critic_obs_fc_layers,
            action_fc_layer_params=critic_action_fc_layers,
            joint_fc_layer_params=critic_joint_fc_layers,
            lstm_size=critic_lstm_size,
            output_fc_layer_params=critic_output_fc_layers,
            kernel_initializer='glorot_uniform',
            last_kernel_initializer='glorot_uniform')

        self.tf_agent = sac_agent.SacAgent(
            self.time_step_spec,
            self.action_spec,
            actor_network=actor_net,
            critic_network=critic_net,
            actor_optimizer=tf.compat.v1.train.AdamOptimizer(
                learning_rate=actor_learning_rate),
            critic_optimizer=tf.compat.v1.train.AdamOptimizer(
                learning_rate=critic_learning_rate),
            alpha_optimizer=tf.compat.v1.train.AdamOptimizer(
                learning_rate=alpha_learning_rate),
            target_update_tau=target_update_tau,
            target_update_period=target_update_period,
            td_errors_loss_fn=td_errors_loss_fn,
            gamma=gamma,
            reward_scale_factor=reward_scale_factor,
            gradient_clipping=gradient_clipping,
            debug_summaries=debug_summaries,
            summarize_grads_and_vars=summarize_grads_and_vars,
            train_step_counter=global_step)
        
        self.tf_agent.initialize()
        
        self.eval_metrics = [
          tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
          tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes)
        ]
        self.train_metrics = [
            tf_metrics.NumberOfEpisodes(),
            tf_metrics.EnvironmentSteps(),
            tf_metrics.AverageReturnMetric(),
            tf_metrics.AverageEpisodeLengthMetric(),
        ]
        
        self.train_summary_writer = tf.compat.v2.summary.create_file_writer(
          name + "_train", flush_millis=summaries_flush_secs * 1000)
        self.train_summary_writer.set_as_default()

        self.eval_summary_writer = tf.compat.v2.summary.create_file_writer(
          name + "_eval", flush_millis=summaries_flush_secs * 1000)


        self.eval_policy = self.tf_agent.policy
        # Initialize collect_policy with saved_policy
        if continue_saved_policy is not None:
            self.collect_policy = tf.saved_model.load(continue_saved_policy)
        else:
            self.collect_policy = self.tf_agent.collect_policy
        
        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            self.tf_agent.collect_data_spec,
            batch_size=tf_env.batch_size,
            max_length=replay_buffer_capacity)
        
        if initialize_replay_buffer:
            self.initial_collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
                self.tf_env,
                self.collect_policy,
                observers=[self.replay_buffer.add_batch] + self.train_metrics,
                num_episodes=initial_collect_episodes)
            
            if use_tf_functions:
                self.initial_collect_driver.run = common.function(self.initial_collect_driver.run)
            
            logging.info(
                '----------------------- %s : Initializing replay buffer by collecting experience for %d episodes '
                'with a random policy.', self.name, initial_collect_episodes)
            self.initial_collect_driver.run()
        
        self.collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
            self.tf_env,
            self.collect_policy,
            observers=[self.replay_buffer.add_batch] + self.train_metrics,
            num_episodes=collect_episodes_per_iteration)        

        if use_tf_functions:
            self.collect_driver.run = common.function(self.collect_driver.run)
            self.tf_agent.train = common.function(self.tf_agent.train)
            
        self.results = metric_utils.eager_compute(
            self.eval_metrics,
            self.eval_tf_env,
            self.eval_policy,
            num_episodes=num_eval_episodes,
            train_step=global_step,
            summary_writer=self.eval_summary_writer,
            summary_prefix='Metrics',
        )
        
        if eval_metrics_callback is not None:
            eval_metrics_callback(self.results, global_step.numpy())
        metric_utils.log_metrics(self.eval_metrics)

        
        self.dataset = self.replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=batch_size,
            num_steps=train_sequence_length + 1).prefetch(3)
        self.iterator = iter(self.dataset)
        
        if use_tf_functions:
            self.train_step = common.function(self.train_step)
        
    def train_step(self):
        experience, _ = next(self.iterator)
        return self.tf_agent.train(experience)
    
    def train(self,num_iterations):
        time_step = None
        policy_state = self.collect_policy.get_initial_state(self.tf_env.batch_size)

        timed_at_step = global_step.numpy()
        time_acc = 0
        for _ in tqdm(range(num_iterations)):
            start_time = time.time()
            time_step, policy_state = self.collect_driver.run(
                time_step=time_step,
                policy_state=policy_state,
            )
            for _ in range(train_steps_per_iteration):
                train_loss = self.train_step()
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

            for train_metric in self.train_metrics:
                train_metric.tf_summaries(
                    train_step=global_step, step_metrics=self.train_metrics[:2])

            if global_step.numpy() % eval_interval == 0:
                self.results = metric_utils.eager_compute(
                    self.eval_metrics,
                    self.eval_tf_env,
                    self.eval_policy,
                    num_episodes=num_eval_episodes,
                    train_step=global_step,
                    summary_writer=self.eval_summary_writer,
                    summary_prefix='Metrics',
                )
                self.rewards.append(float(self.results['AverageReturn']))
                self.episodes.append(_)
            if eval_metrics_callback is not None:
                eval_metrics_callback(self.results, global_step.numpy())
            metric_utils.log_metrics(self.eval_metrics) 
        
        self.save()
            
    def save(self):
        saver = PolicySaver(self.eval_policy, batch_size = None)
        saver.save(self.name+'_saved_policy')

