import os
import sys
import operator
import importlib

sys.path.append(os.path.expandvars('../Round1/src/'))

tasks_config_file = os.path.expandvars('../Round1/src/tasks_config.challenge.json')
output_file_template = os.path.expandvars('./output-{}')

import learners
from core.serializer import StandardSerializer
from core.environment import Environment
from core.config_loader import JSONConfigLoader
from core.session import Session
from view.win_console import StdOutView

from aux import save_results
from reinforce_commai import Agent as agent

def train_agent(agent, id):
    serializer = StandardSerializer()
    task_scheduler = JSONConfigLoader().create_tasks(tasks_config_file)
    env = Environment(serializer, task_scheduler, scramble=False,
                        max_reward_per_task=1000, byte_mode=True)

    session = Session(env, agent)
    view = StdOutView(env, session)
    try:
        view.initialize()
        session.run()
    except BaseException:
        view.finalize()
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