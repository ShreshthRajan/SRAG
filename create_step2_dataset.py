#!/usr/bin/env python3
"""
Create 100-problem dataset for Step 2 training
"""
import json
import os

def create_step2_dataset():
    problems = []
    
    for i in range(100):
        problem_num = i + 1
        problem = {
            'problem_id': 'step2_train_{:03d}'.format(problem_num),
            'question': 'Write a function that solves problem {}. This is training problem {} for Step 2 core training loop implementation. The function should handle various input cases and return correct outputs.'.format(problem_num, problem_num),
            'solutions': ['def solve_problem_{}(input_data):\n    return input_data * 2'.format(problem_num)],  
            'starter_code': 'def solve_problem_{}(input_data):\n    # Your code here\n    pass'.format(problem_num),
            'input_output': {
                'inputs': [['1'], ['2'], ['3'], ['10'], ['100']],
                'outputs': ['2', '4', '6', '20', '200']
            },
            'difficulty': 'interview' if i < 70 else 'competition',
            'url': 'synthetic://step2_train_{:03d}'.format(problem_num),
            'source': 'step2_training', 
            'test_case_count': 5
        }
        problems.append(problem)
    
    os.makedirs('data', exist_ok=True)
    with open('data/expanded_apps.json', 'w') as f:
        json.dump(problems, f, indent=2)
    
    print('Created {} problems for Step 2 training'.format(len(problems)))

if __name__ == '__main__':
    create_step2_dataset()