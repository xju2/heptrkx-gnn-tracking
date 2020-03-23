#!/usr/bin/env python
import argparse
from heptrkx.dataset import event as master

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summary of the event")
    add_arg = parser.add_argument
    add_arg('--input-dir', 
            default='/global/cscratch1/sd/xju/heptrkx/codalab/inputs/train_all',
            help='input trackML data')
    add_arg("--evtid", default=21001, help='event id', type=int)
    args = parser.parse_args()
    
    event = master.Event(args.input_dir)
    event.read(args.evtid)
    out_str = "Summary\n"
    out_str += '{} hits\n'.format(event.hits.shape[0])
    out_str += "{} particles\n".format(event.particles.shape[0])

    print(out_str)