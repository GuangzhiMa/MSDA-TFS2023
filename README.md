# Multi-source Domain Adaptation with Interval-Valued Target Data via Fuzzy Neural Networks 
This is the official site for the paper "Multi-source Domain Adaptation with Interval-Valued Target Data via Fuzzy Neural Networks". This work is done by 

- Guangzhi Ma (UTS), Guangzhi.Ma@student.uts.edu.au
- Prof. Jie Lu (UTS), Jie.Lu@uts.edu.au
- A/Prof. Guangquan Zhang (UTS), Guangquan.Zhang@uts.edu.au

# Software version
PyTorch 1.9.0. Python version is 3.9.7. CUDA version is 11.2.

These python files require some basic scientific computing python packages, e.g., numpy. I recommend users to install python via Anaconda (python 3.9.7), which can be downloaded from https://www.anaconda.com/distribution/#download-section . If you have installed Anaconda, then you do not need to worry about these basic packages.

After you install anaconda and PyTorch, you can run codes successfully. Good luck!

# Code
You can run 
```
python main_select.py 
```
--> get Fig. 3.

You can run 
```
python main.py --dset 'synthetic' --gpu_id 0 --batch_size 100 --max_epoch 500
```
--> get results on the synthetic dataset.

You can run 
```
python main.py --dset 'weather' --gpu_id 0 --s 'OW' --t 'S' --batch_size 500 --max_epoch 100 
```
--> get results on the weather dataset of task 'OW to S'.


# Acknowledgment
GM, JL, and GZ were supported by the Australian Research Council (ARC) under FL190100149.
