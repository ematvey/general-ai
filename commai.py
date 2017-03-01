import os
import sys
import operator
import importlib

sys.path.append(os.path.expandvars('../Round1/src/'))

# tasks_config_file = os.path.expandvars('../Round1/src/tasks_config.challenge.json')

import learners
from core.serializer import StandardSerializer
from core.environment import Environment
from core.config_loader import JSONConfigLoader
from core.session import Session
from view.win_console import StdOutView
import logging


class SessionLogger():
    def __init__(self, session_log='./session.csv'):
        self.reward = 0
        self.total_reward = 0
        self.token_in = None
        self.token_out = ""
        self.token_prev = ""
        self.logger = logging.getLogger("session_logger")
        self.logger.propagate = False
        # create file handler which logs session 
        fh = logging.FileHandler(session_log, mode='w')
        fh.setLevel(logging.INFO)
        self.logger.addHandler(fh)
    
    def env_token_update(self, token):
        self.token_in = token

    def learner_token_update(self, token):
        self.token_prev = self.token_out
        self.token_out = token

    def total_reward_update(self, total_reward):
        self.reward = total_reward - self.total_reward
        self.total_reward = total_reward

    def total_time_update(self, total_time):
        if total_time > 1:
            self.logger.info("{}\t{}\t{}\t{}".format(total_time - 1, self.token_in, self.token_prev, self.reward))
            self.reward = 0


def train_agent(agent, id, tasks_config='tasks/micro1.json', output_file_template='./output-{}',
        session_log='./session.csv'):
    serializer = StandardSerializer()
    tasks_config_file = os.path.expandvars(tasks_config)
    task_scheduler = JSONConfigLoader().create_tasks(tasks_config_file)
    env = Environment(serializer, task_scheduler, scramble=False,
                        max_reward_per_task=1000, byte_mode=True)

    session_logger = SessionLogger(session_log)
    session = Session(env, agent)
    session.env_token_updated.register(session_logger.env_token_update)
    session.learner_token_updated.register(session_logger.learner_token_update)
    session.total_time_updated.register(session_logger.total_time_update)
    session.total_reward_updated.register(session_logger.total_reward_update)
    view = StdOutView(env, session)
    try:
        view.initialize()
        session.run()
    except BaseException:
        view.finalize()
        output_file_template = os.path.expandvars(output_file_template)
        save_results(session, output_file_template.format(id))
        raise
    else:
        view.finalize()


def save_results(session, output_file):
    if session.get_total_time() == 0:
        # nothing to save
        return
    with open(output_file, 'w') as fout:
        print('* General results', file=fout)
        print('Average reward: {avg_reward}'.format(
            avg_reward=session.get_total_reward() / session.get_total_time()),
            file=fout)
        print('Total time: {time}'.format(time=session.get_total_time()),
               file=fout)
        print('Total reward: {reward}'.format(
            reward=session.get_total_reward()),
            file=fout)
        print('* Average reward per task', file=fout)
        for task, t in sorted(session.get_task_time().items(),
                              key=operator.itemgetter(1)):
            r = session.get_reward_per_task()[task]
            print('{task_name}: {avg_reward}'.format(
                task_name=task, avg_reward=r / t),
                file=fout)
        print('* Total reward per task', file=fout)
        for task, r in sorted(session.get_reward_per_task().items(),
                              key=operator.itemgetter(1), reverse=True):
            print('{task_name}: {reward}'.format(task_name=task, reward=r),
                  file=fout)
        print('* Total time per task', file=fout)
        for task, t in sorted(session.get_task_time().items(),
                              key=operator.itemgetter(1)):
            print('{task_name}: {time}'.format(task_name=task, time=t),
                  file=fout)
        print('* Number of trials per task', file=fout)
        for task, r in sorted(session.get_task_count().items(),
                              key=operator.itemgetter(1)):
            print('{task_name}: {reward}'.format(task_name=task, reward=r),
                  file=fout)
