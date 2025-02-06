#Proof of concept implementation of 11-round key recovery attack

import speck as sp
import numpy as np
from os import urandom
from math import log2
from time import time

import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(3)#

import tensorflow as tf
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth=True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

from tensorflow.keras.models import model_from_json


WORD_SIZE = sp.WORD_SIZE();



#load distinguishers
json_file = open('./neural networks/single_block_resnet.json','r');
json_model = json_file.read();


net7 = model_from_json(json_model);
net6 = model_from_json(json_model);

net7.load_weights('./neural networks/net7_small.h5');
net6.load_weights('./neural networks/net6_small.h5');



m7 = np.load('./wrong key response/data_wrong_key_mean_7r.npy');
s7 = np.load('./wrong key response/data_wrong_key_std_7r.npy'); s7 = 1.0/s7;
m6 = np.load('./wrong key response/data_wrong_key_mean_6r.npy');
s6 = np.load('./wrong key response/data_wrong_key_std_6r.npy'); s6 = 1.0/s6;

#binarize a given ciphertext sample
#ciphertext is given as a sequence of arrays
#each array entry contains one word of ciphertext for all ciphertexts given
def convert_to_binary(l):
    n = len(l);
    k = WORD_SIZE * n;
    X = np.zeros((k, len(l[0])),dtype=np.uint8);
    for i in range(k):
        index = i // WORD_SIZE;
        offset = WORD_SIZE - 1 - i%WORD_SIZE;
        X[i] = (l[index] >> offset) & 1;
    X = X.transpose();
    return(X);

def hw(v):
    res = np.zeros(v.shape,dtype=np.uint8);
    for i in range(16):
        res = res + ((v >> i) & 1)
    return(res);

low_weight = np.array(range(2**WORD_SIZE), dtype=np.uint16);
low_weight = low_weight[hw(low_weight) <= 2];

#make a plaintext structure
#takes as input a sequence of plaintexts, a desired plaintext input difference, and a set of neutral bits
def make_structure(pt0, pt1, diff,neutral_bits):
    p0 = np.copy(pt0); p1 = np.copy(pt1);
    p0 = p0.reshape(-1,1); p1 = p1.reshape(-1,1);
    for i in neutral_bits:
        d = 1 << i; d0 = d >> 16; d1 = d & 0xffff
        p0 = np.concatenate([p0,p0^d0],axis=1);
        p1 = np.concatenate([p1,p1^d1],axis=1);
    p0b = p0 ^ diff[0]; p1b = p1 ^ diff[1];
    return(p0,p1,p0b,p1b);

#generate a Speck key, return expanded key
def gen_key(nr):
    key = np.frombuffer(urandom(8),dtype=np.uint16);
    ks = sp.expand_key(key, nr);
    return(ks);

def gen_plain(n):
    pt0 = np.frombuffer(urandom(2*n),dtype=np.uint16);
    pt1 = np.frombuffer(urandom(2*n),dtype=np.uint16);
    return(pt0, pt1);

def gen_challenge(n, nr ):
    diff=(0x211, 0xa04)
    neutral_bits = [20,21,22,14,15,23]
    
    pt0, pt1 = gen_plain(n);
    pt0a, pt1a, pt0b, pt1b = make_structure(pt0, pt1, diff=diff, neutral_bits=neutral_bits);
    pt0a, pt1a = sp.dec_one_round((pt0a, pt1a),0);
    pt0b, pt1b = sp.dec_one_round((pt0b, pt1b),0);
    key = gen_key(nr);
    #if (keyschedule is 'free'): key = np.frombuffer(urandom(2*nr),dtype=np.uint16);
    ct0a, ct1a = sp.encrypt((pt0a, pt1a), key);
    ct0b, ct1b = sp.encrypt((pt0b, pt1b), key);
    return([ct0a, ct1a, ct0b, ct1b], key);

def find_good(cts, key, nr=3, target_diff = (0x0040,0x0)):
    pt0a, pt1a = sp.decrypt((cts[0], cts[1]), key[nr:]);
    pt0b, pt1b = sp.decrypt((cts[2], cts[3]), key[nr:]);
    diff0 = pt0a ^ pt0b; diff1 = pt1a ^ pt1b;
    d0 = (diff0 == target_diff[0]); d1 = (diff1 == target_diff[1]);
    d = d0 * d1;
    v = np.sum(d,axis=1);
    return(v);

#having a good key candidate, exhaustively explore all keys with hamming distance less than two of this key
def verifier_search(cts, best_guess, net = net6):
    #print(best_guess);
    use_n=len(cts[0])
    
    ck1 = best_guess[0] ^ low_weight;
    ck2 = best_guess[1] ^ low_weight;
    n = len(ck1);
    ck1 = np.repeat(ck1, n); keys1 = np.copy(ck1);
    ck2 = np.tile(ck2, n); keys2 = np.copy(ck2);
    ck1 = np.repeat(ck1, use_n);
    ck2 = np.repeat(ck2, use_n);
    ct0a = np.tile(cts[0][0:use_n], n*n);
    ct1a = np.tile(cts[1][0:use_n], n*n);
    ct0b = np.tile(cts[2][0:use_n], n*n);
    ct1b = np.tile(cts[3][0:use_n], n*n);
    pt0a, pt1a = sp.dec_one_round((ct0a, ct1a), ck1);
    pt0b, pt1b = sp.dec_one_round((ct0b, ct1b), ck1);
    pt0a, pt1a = sp.dec_one_round((pt0a, pt1a), ck2);
    pt0b, pt1b = sp.dec_one_round((pt0b, pt1b), ck2);
    X = sp.convert_to_binary([pt0a, pt1a, pt0b, pt1b]);
    Z = net.predict(X, batch_size=2**15);
    Z = Z / (1 - Z);
    Z = np.log2(Z);
    Z = Z.reshape(-1, use_n);
    v = np.mean(Z, axis=1) * len(cts[0]);
    m = np.argmax(v); val = v[m];
    key1 = keys1[m]; key2 = keys2[m];
    return(key1, key2, val);


#here, we use some symmetries of the wrong key performance profile
#by performing the optimization step only on the 14 lowest bits and randomizing the others
#on CPU, this only gives a very minor speedup, but it is quite useful if a strong GPU is available
#In effect, this is a simple partial mitigation of the fact that we are running single-threaded numpy code here
tmp_br = np.arange(2**14, dtype=np.uint16);
tmp_br = np.repeat(tmp_br, 32).reshape(-1,32);
#32 is related the number of the key candidates, So Num_cand=32

def bayesian_rank_kr(cand, emp_mean, m=m7, s=s7):
    global tmp_br;
    n = len(cand);
    if (tmp_br.shape[1] != n):
        tmp_br = np.arange(2**14, dtype=np.uint16);
        tmp_br = np.repeat(tmp_br, n).reshape(-1,n);
    tmp = tmp_br ^ cand;
    v = (emp_mean - m[tmp]) * s[tmp];
    v = v.reshape(-1, n);
    scores = np.linalg.norm(v, axis=1);
    return(scores);




def bayesian_key_recovery_early_stop(cts, net, m , s , num_cand , min_num_iter,max_num_iter,early_Net):
    n = len(cts[0]);
    keys = np.random.choice(2**(WORD_SIZE-2),num_cand,replace=False); 
    scores = 0; 

    ct0a, ct1a, ct0b, ct1b = np.tile(cts[0],num_cand), np.tile(cts[1], num_cand), np.tile(cts[2], num_cand), np.tile(cts[3], num_cand);
    scores = np.zeros(2**(WORD_SIZE-2));
    used = np.zeros(2**(WORD_SIZE-2));
    all_keys = np.zeros(num_cand * max_num_iter,dtype=np.uint16);
    all_v = np.zeros(num_cand * max_num_iter);
    
    for i in range(max_num_iter):
        k = np.repeat(keys, n);
        c0a, c1a = sp.dec_one_round((ct0a, ct1a),k); c0b, c1b = sp.dec_one_round((ct0b, ct1b),k);
        X = sp.convert_to_binary([c0a, c1a, c0b, c1b]);
        Z = net.predict(X,batch_size=2**15);
        Z = Z.reshape(num_cand, -1);
        means = np.mean(Z, axis=1);
        Z = Z/(1-Z); Z = np.log2(Z); v =np.sum(Z, axis=1); 
        all_v[i * num_cand:(i+1)*num_cand] = v;
        all_keys[i * num_cand:(i+1)*num_cand] = np.copy(keys);
        
        if(i>=min_num_iter-1) and (i<max_num_iter-1):
            
            is_add=early_Net[i+1-min_num_iter].predict(np.array([all_v[:(i+1)*num_cand]]))
            is_add=is_add[0]
            
            #print(i)
            #print(all_v[:(i+1)*num_cand])
            
            if(is_add<0.5):
                break
        
        scores = bayesian_rank_kr(keys, means, m=m, s=s);
        tmp = np.argpartition(scores+used, num_cand)
        keys = tmp[0:num_cand];
        r = np.random.randint(0,4,num_cand,dtype=np.uint16); 
        r = r << 14; keys = keys ^ r;
    return(all_keys[:(i+1)*num_cand], scores, all_v[:(i+1)*num_cand]);






def test_bayes(cts,early_net_7r,early_net_6r, C1,C2,
               Num_cand1,Num_iter1_max ,Num_iter1_min,
               Num_cand2,Num_iter2_max ,Num_iter2_min,
               it=500,  net=net7, net_help=net6, m_main=m7, m_help=m6, s_main=s7, s_help=s6):
    n = len(cts[0]);

    alpha = 10.0

    best_val = -100.0; best_key = (0,0); best_pod = 0;
    keys = np.random.choice(2**WORD_SIZE, 32, replace=False);
    
    eps = 0.001; local_best = np.full(n,-10); num_visits = np.full(n,eps);
    
    cutoff1=C1
    cutoff2=C2
    
    
    for j in range(it):
        print('\r' +str(int((j))),end='', flush=True)
        priority = local_best + alpha * np.sqrt(log2(j+1) / num_visits); i = np.argmax(priority);
        num_visits[i] = num_visits[i] + 1;
        if (best_val > cutoff2):
            improvement = True
            while improvement:
                k1, k2, val = verifier_search([cts[0][best_pod], cts[1][best_pod], cts[2][best_pod], cts[3][best_pod]], best_key,net=net_help)
                improvement = (val > best_val);
                if (improvement):
                    best_key = (k1, k2); best_val = val;
            print('')  
            return(best_key, j);
        #bayesian_key_recovery(cts, net, m , s , num_cand , num_iter):
        keys, scores, v = bayesian_key_recovery_early_stop([cts[0][i], cts[1][i], cts[2][i], cts[3][i]], 
                                                           net=net,m=m_main, s=s_main,num_cand=Num_cand1, 
                                                           min_num_iter=Num_iter1_min,max_num_iter=Num_iter1_max,
                                                           early_Net=early_net_7r )
        vtmp = np.max(v);
        if (vtmp > local_best[i]): 
            local_best[i] = vtmp;

        if (vtmp > cutoff1):
            l2 = [i for i in range(len(keys)) if v[i] > cutoff1];
            for i2 in l2:
                c0a, c1a = sp.dec_one_round((cts[0][i],cts[1][i]),keys[i2]);
                c0b, c1b = sp.dec_one_round((cts[2][i],cts[3][i]),keys[i2]);         
                keys2,scores2,v2 = bayesian_key_recovery_early_stop([c0a, c1a, c0b, c1b],net=net_help,m=m_help,s=s_help,
                                                                    num_cand=Num_cand2, min_num_iter=Num_iter2_min,max_num_iter=Num_iter2_max,
                                                                    early_Net=early_net_6r);
                vtmp2 = np.max(v2);
                if (vtmp2 > best_val):
                    best_val = vtmp2; best_key = (keys[i2], keys2[np.argmax(v2)]); best_pod=i;
    
    improvement = True
    while improvement:
        k1, k2, val = verifier_search([cts[0][best_pod], cts[1][best_pod], cts[2][best_pod], cts[3][best_pod]], best_key, net=net_help);
        improvement = (val > best_val);
        if (improvement):
            best_key = (k1, k2); best_val = val;
    print('')        
    
    return(best_key, it);


def load_Net_for_EarlyStoppint(nr,num_cand,Min_inter,Max_inter):
    diff=(0x40,0x0)
    num_rounds=nr
    #num_cand=32
    
    wdir = './neural networks_diff_distinguish_structure'
    if(num_rounds==6):
        wdir=wdir+'_6r'
    wdir=wdir+'/'
    #print(wdir)
    Net=[]
    for step in range(Min_inter,Max_inter):
        net_name='res_SPECK32_round='+str(num_rounds)+'_'+str(diff)+str(num_cand)+'_'+str(step)
        json_file=open(wdir+net_name+'model'+'.json','r')
        json_model = json_file.read()
        net = model_from_json(json_model)
        net.load_weights(wdir+net_name+'.h5')
        Net.append(net)
    return Net
    
    


def test(n, nr=11, num_structures=100, it=500, 
         net=net7, net_help=net6, 
         m_main=m7, s_main=s7,  m_help=m6, s_help=s6):
    
    neutral_bits = [20,21,22,14,15,23]
    
    Num_cand1=32
    Num_iter1_max=5
    Num_iter1_min=4
    #C1=3.302297592
    
    Num_cand2=32
    Num_iter2_max=5
    Num_iter2_min=2
    
    C1=4.9
    C2=0.0

    print(str(Num_cand1)+'_'+str(Num_iter1_max)+'_'+str(Num_iter1_min))
    print(str(Num_cand2)+'_'+str(Num_iter2_max)+'_'+str(Num_iter2_min))
    print('C1='+str(C1)+'_C2='+str(C2)+'_')
    
    All_net_7r=load_Net_for_EarlyStoppint(7,Num_cand1,Num_iter1_min,Num_iter1_max)
    All_net_6r=load_Net_for_EarlyStoppint(6,Num_cand2,Num_iter2_min,Num_iter2_max)
    
    
    print("Checking Speck32/64 implementation.");
    if (not sp.check_testvector()):
        print("Error. Aborting.");
        return(0);
    arr1 = np.zeros(n, dtype=np.uint16); arr2 = np.zeros(n, dtype=np.uint16);
    time_attack=np.zeros(n, dtype=np.uint16)

    
    
    t0 = time();
    data = 0;  
    #good = np.zeros(n, dtype=np.uint8);

    for i in range(n):
        print("Test:",i);
        ct, key = gen_challenge(num_structures,nr);
        #ct,key=all_ct[i],all_key[i]
        #g = find_good(ct, key); g = np.max(g); good[i] = g;
        guess, num_used = test_bayes(ct,All_net_7r,All_net_6r,C1,C2,
                             Num_cand1,Num_iter1_max ,Num_iter1_min,
                             Num_cand2,Num_iter2_max ,Num_iter2_min);
        time_attack[i]=num_used
        num_used = min(num_structures, num_used); 
        data = data + 2 * (2 ** len(neutral_bits)) * num_used;
        arr1[i] = guess[0] ^ key[nr-1]; arr2[i] = guess[1] ^ key[nr-2];
        print("Difference between real key and key guess: ", hex(arr1[i]), hex(arr2[i]));
    t1 = time();
    print("Done.");
    d1 = [hex(x) for x in arr1]; d2 = [hex(x) for x in arr2];
    
    print("Differences between guessed and last key:", d1);
    print("Differences between guessed and second-to-last key:", d2);
    print("Wall time per attack (average in seconds):", (t1 - t0)/n);
    print("Data blocks used (average, log2): ", log2(data) - log2(n));
    
    print(str(Num_cand1)+'_'+str(Num_iter1_max)+'_'+str(Num_iter1_min))
    print(str(Num_cand2)+'_'+str(Num_iter2_max)+'_'+str(Num_iter2_min))
    print('C1='+str(C1)+'_C2='+str(C2)+'_')
    path='./attack_result_early_stopping/'
    name=str(Num_cand1)+'_'+str(Num_iter1_max)+'_'+str(Num_iter1_min)+'_'+str(Num_cand2)+'_'+str(Num_iter2_max)+'_'+str(Num_iter2_min)+'_'+'C1='+str(C1)+'_C2='+str(C2)+'_'
    
    np.save(path+name+'Time_attack_end.npy',time_attack)
    np.save(path+name+'run_sols1_11r.npy', arr1);
    np.save(path+name+'run_sols2_11r.npy', arr2);
    return(arr1, arr2)
    #return(arr1, arr2, good);

arr1, arr2 = test(1000);

num_correct=0

for i in range(len(arr1)):
    if((arr1[i]==0) and (arr2[i] in [0<<14,1<<14,2<<14,3<<14])):
        num_correct=num_correct+1

print('Early Stopping',num_correct/len(arr1))

