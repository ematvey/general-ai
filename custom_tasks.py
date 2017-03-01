import os
import sys
import operator
import importlib

sys.path.append(os.path.expandvars('../Round1/src/'))

from tasks.challenge.round1.challenge_micro import MicroBase
from tasks.challenge.round1.task_generator import TaskGenerator
from core.task import on_start
import string
import random

class Micro1Task(MicroBase):
    ALPHABET_SIZE = 10

    def __init__(self):
        self.base_alphabet = string.ascii_letters + string.digits + ' ,.!;?-'
        self.alphabet = random.sample(self.base_alphabet, self.ALPHABET_SIZE)
        self.correct_answer = random.choice(self.alphabet)
        super(Micro1Task, self).__init__()

    @on_start()
    def micro1_on_start(self, event):
        self.remaining_options = len(self.base_alphabet)
        self.should_know = False

    def agent_should_know_answers(self):
        return self.should_know

    def question_answered(self, is_correct):
        super(Micro1Task, self).question_answered(is_correct)
        if is_correct or self.remaining_options == 0:
            self.should_know = True
        if not is_correct:
            self.remaining_options -= 1

    def get_task_generator(self):
        alphabet = self.alphabet
        correct_answer = self.correct_answer

        def micro1_question(self):
            return random.choice(alphabet), correct_answer
        return TaskGenerator(micro1_question)

