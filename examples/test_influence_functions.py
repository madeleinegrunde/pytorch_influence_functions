#### Make theis a comment instead! /usr/bin/env python3
import sys
sys.path.append('/scratch/mgrunde/pytorch_influence_functions/')
import pytorch_influence_functions as ptif
from train_influence_functions import load_model, load_data

if __name__ == "__main__":
    config = ptif.get_default_config()
    #config['gpu'] = -1
    print(config)

    model = load_model()
    trainloader, testloader = load_data()
    ptif.init_logging('logfile.log')
    ptif.calc_img_wise(config, model, trainloader, testloader)
