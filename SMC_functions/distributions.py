# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 13:16:35 2022

@author: amill212
"""
#basic pdf creator for gaussians 
import scipy.stats as stats

def distribution(mu, sigma):
    distribution = stats.norm(mu, sigma)
    return distribution 