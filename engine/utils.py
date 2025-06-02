import os
from PIL import Image
from openai import OpenAI
import numpy as np
import copy

from .step_interpreters import register_step_interpreters, parse_step

# Set OpenAI API key
OPENAI_API_KEY = "your API key"

class Program:
    def __init__(self, prog_str, init_state=None):
        self.prog_str = prog_str
        self.state = init_state if init_state is not None else dict()
        self.instructions = self.prog_str.split('\n')


class ProgramInterpreter:
    def __init__(self, dataset='nlvr'):
        self.step_interpreters = register_step_interpreters(dataset)

    def execute_step(self, prog_step, inspect):
        step_name = parse_step(prog_step.prog_str, partial=True)['step_name']
        print(step_name)
        result = self.step_interpreters[step_name].execute(prog_step, inspect)
        
        if inspect:
            if not isinstance(result, tuple) or len(result) != 2:
                raise ValueError(f"[execute_step] Expected (output, html_str) from '{step_name}', got: {result}")
            return result
        else:
            return result if not isinstance(result, tuple) else result[0]

    def execute(self, prog, init_state, inspect=False):

        if isinstance(prog, str):
            prog = Program(prog, init_state)
        else:
            assert isinstance(prog, Program)

        prog_steps = [Program(instruction, init_state=prog.state) for instruction in prog.instructions]

        html_str = '<hr>'
        for prog_step in prog_steps:
            if inspect:
                step_output, step_html = self.execute_step(prog_step, inspect)
                html_str += step_html + '<hr>'
            else:
                step_output = self.execute_step(prog_step, inspect)

        if inspect:
            return step_output, prog.state, html_str
        return step_output, prog.state


class ProgramGenerator():
    def __init__(self, prompter):
        self.prompter = prompter
        self.client = OpenAI(
            api_key=OPENAI_API_KEY,
            project="proj_1mWjP874mS8VC8zmtH4ri4KV"
        )

    def generate(self, inputs):
        prompt = self.prompter(inputs)
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=1024
        )
        prog = response.choices[0].message.content.strip()
        return prog, prompt
