#### Make theis a comment instead! /usr/bin/env python3
import sys
sys.path.append('/gscratch/krishna/mgrunde/pytorch_influence_functions/')
import pytorch_influence_functions as ptif
from train_influence_functions import load_model, load_data
import numpy as np
import json
import argparse
import os

def makePath(path):
    if not os.path.exists(path):
        print('making directory to', path)
        os.makedirs(path)


def main(args):
    config = ptif.get_default_config()
    #config['gpu'] = -1
    outfolder = '%s-%s-%s' % (args.output_dir, args.start_idx, args.end_idx)
    config['save_pth'] = 'outdir/%s/save_pth/' % outfolder
    config['outdir'] = 'outdir/%s/outdir/' % outfolder
    config['log_filename'] = 'outdir/%s/logs/log.txt' % outfolder
    config['test_sample_num'] = args.end_idx - args.start_idx
    config['test_sample_start_per_class'] = args.start_idx
    for i in config:
        print(i, config[i])

    # make output dirs if don't exist
    makePath(config['save_pth'])
    makePath(config['outdir'])
    makePath(config['log_filename'])


    # get influence functions
    model = load_model()
    trainloader, testloader = load_data()
    ptif.init_logging('logfile.log')
    ptif.calc_img_wise(config, model, trainloader, testloader)


    # get preds
    preds = []
    model.eval()
    for batch in testloader:
        batch = batch[0].cuda()
        out = model.forward(batch)

        out = out.cpu()
        out = out.detach().numpy()
        output = np.argmax(out, axis=1)
        output = [int(i) for i in output]
        preds += output

    print('Got %s outputs, first one is %s' % (len(preds), preds[0]))
    with open('%s%s' % (config['save_pth'], 'predictions.json'), 'w+') as f:
        json.dump(preds, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='basic',
                                                    help='folder in output')
    parser.add_argument('--start_idx', type=int, default=0,
                                                    help='First class index to generate functions for')
    parser.add_argument('--end_idx', type=int, default=1,
                                            help='Last class index to generate functions for')
    args = parser.parse_args()
    print("Saving outputs to %s", args.output_dir)
    main(args)
                                                       
