#!/bin/bash


# for expert 

python main.py --task disjoint --n_tasks 1 --dataset fashion
python main.py --task disjoint --n_tasks 1 --dataset mnist

n_tasks=10

echo '1'
#python main.py --task rotations --n_tasks $n_tasks --dataset fashion
#python main.py --task rotations --n_tasks $n_tasks --dataset mnist
echo '2'
#python main.py --task permutations --n_tasks $n_tasks --dataset fashion
#python main.py --task permutations --n_tasks $n_tasks --dataset mnist
echo '3'
python main.py --task disjoint --n_tasks $n_tasks --dataset fashion
python main.py --task disjoint --n_tasks $n_tasks --dataset mnist
echo '4'
python main.py --task disjoint --n_tasks $n_tasks --dataset fashion --upperbound True
python main.py --task disjoint --n_tasks $n_tasks --dataset mnist --upperbound True
