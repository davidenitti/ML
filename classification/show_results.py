import json,glob2
import os,sys

def get_best(root_dir,task):
    json_files = glob2.glob(os.path.join(root_dir,'**/*'+task+'*.json'))
    full_res = {}
    for file in json_files:
        with open(file, 'r') as f:
            data = json.load(f)
            #print(file)
            for exp in data['res']:
                print(exp[1],exp[0]['net_params'],exp[0])
                for param in exp[0]['net_params']:
                    if param not in full_res:
                        full_res[param]={}
                    if exp[0]['net_params'][param] not in full_res[param]:
                        full_res[param][exp[0]['net_params'][param]] = []
                    full_res[param][exp[0]['net_params'][param]].append(exp[1])
    for param in full_res:
        print(param)
        for value in full_res[param]:
            best_acc = max([e['best'] for e in full_res[param][value]])
            print(param,"=",value,'acc',best_acc,full_res[param][value])
if __name__ == '__main__':
    get_best("../../../results/",'cifar10')