import subprocess
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-in', '--input_path', type=str, default=None)
args = parser.parse_args()
print(args)

start = time.time()
subprocess.check_call('sh ./run_models5.sh {}'.format(args.input_path), shell=True)
end = time.time()
cost_time = end - start
minute = cost_time / 60
minute = int(minute)
second = cost_time % 60
print('COST TIME: {} min {} sec.'.format(minute, second))