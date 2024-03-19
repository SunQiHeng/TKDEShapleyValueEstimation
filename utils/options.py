#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # model arguments                 
    parser.add_argument('--samples', type=int, default=0,
                        help="sample number of complementary contributions.")
    parser.add_argument('--ori_players',type=int,default=0,
                        help="the number of players before adding/deleting players")                    
    parser.add_argument('--new_players', type=int, default=0,
                        help="the number of players after adding/deleting players.")
    parser.add_argument('--all_players', type=int, default=0,
                        help="the number of players all players.")                    
    parser.add_argument('--game', type=str, default="voting",
                        help="the number of players after adding/deleting players.")
    args = parser.parse_args()
    return args
