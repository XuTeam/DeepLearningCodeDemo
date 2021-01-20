"""
Run commands on GPUs simultaneously.
"""

import os
import shlex
import subprocess
import threading
import signal


def run_command(env):
    print('=> run commands on GPU:{}', env['CUDA_VISIBLE_DEVICES'])

    while True:
        try:
            comm = commands.pop(0)
        except IndexError:
            break

        proc = subprocess.Popen(comm, env=env)
        proc.wait()

###########Only modify the following commands###################
# define the command you are going to run
command = 'python main.py --cuda --method=sgd --data cifar10 --epochs=300 --arch resnet18 --data-path=./data --momentum=0.8 --lr {} --wd {} --batch-size {} --version {}'

# List all the parameters you are going to tune
commands = []
for version in [1,2,3,4,5]:
    for i_lr in range(1,5):
        lr = 0.1**i_lr 
        for wd in [0.1,0.05,0.01,0.005,0.001]:
            for batch_size in [64,128,256,512]: 
                commands += [command.format(lr,wd, version,batch_size)]
commands = [shlex.split(comm) for comm in commands]


# List all the GPUs you have
ids_cuda = [0,1,2,3]

###############################################################

for c in ids_cuda:
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(c)

    thread = threading.Thread(target = run_command, args = (env, ))
    thread.start()
