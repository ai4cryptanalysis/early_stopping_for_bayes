#Proof of concept implementation of 16-round key recovery attack of simon32/64

import simon_acc as cipher
import numpy as np


from os import urandom
from math import log2
from time import time

import os
NUM=0
os.environ["CUDA_VISIBLE_DEVICES"] = str(NUM)#

import tensorflow as tf
'''
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth=True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)
'''
from tensorflow.keras.models import model_from_json

WORD_SIZE = cipher.WORD_SIZE()

#binarize a given ciphertext sample
#ciphertext is given as a sequence of arrays
#each array entry contains one word of ciphertext for all ciphertexts given
def convert_to_binary(l):
    n = len(l);
    k = WORD_SIZE * n
    X = np.zeros((k, len(l[0])),dtype=np.uint8)
    for i in range(k):
        index = i // WORD_SIZE
        offset = WORD_SIZE - 1 - i%WORD_SIZE
        X[i] = (l[index] >> offset) & 1
    X = X.transpose();
    return(X)

def hw(v):
    res = np.zeros(v.shape,dtype=np.uint8)
    for i in range(16):
        res = res + ((v >> i) & 1)
    return(res);

low_weight = np.array(range(2**WORD_SIZE), dtype=np.uint16)
low_weight = low_weight[hw(low_weight) <= 2]

#make a plaintext structure
#takes as input a sequence of plaintexts, a desired plaintext input difference, and a set of neutral bits

def generate_nb_diff(nb):
    index=0
    for i in nb:
        index=index^(1<<i)
    return index

def make_structure(pt0, pt1, diff,neutral_bits):
    
    p0 = np.copy(pt0); p1 = np.copy(pt1);
    p0 = p0.reshape(-1,1); p1 = p1.reshape(-1,1)
    for i in neutral_bits:
        d = generate_nb_diff(i); 
        d0 = d >> 16; 
        d1 = d & 0xffff
        p0 = np.concatenate([p0,p0^d0],axis=1);
        p1 = np.concatenate([p1,p1^d1],axis=1);
    p0b = p0 ^ diff[0]; p1b = p1 ^ diff[1];
    return(p0,p1,p0b,p1b);

#generate a Speck key, return expanded key
def gen_key(nr):
    key = np.frombuffer(urandom(8),dtype=np.uint16).reshape(4,-1)
    ks = cipher.expand_key(key, nr)
    return(ks)

def gen_plain(n):
    pt0 = np.frombuffer(urandom(2*n),dtype=np.uint16);
    pt1 = np.frombuffer(urandom(2*n),dtype=np.uint16);
    return(pt0, pt1);

def find_good(cts, key, nr=1+3, target_diff = (0x0,0x40)):
    pt0a, pt1a = cipher.decrypt((cts[0], cts[1]), key[nr:]);
    pt0b, pt1b = cipher.decrypt((cts[2], cts[3]), key[nr:]);
    diff0 = pt0a ^ pt0b; diff1 = pt1a ^ pt1b;
    d0 = (diff0 == target_diff[0]); d1 = (diff1 == target_diff[1]);
    d = d0 * d1;
    v = np.sum(d,axis=1);
    return(v)
def gen_challenge():
    
    n=2**7
    nr=1+3+11+1
    
    diff=(0x440, 0x1000)
    neutral_bits = [[2], [3], [4], [6], [8], [9],
                    [10], [18], [22], [0, 24],[12, 26]]
    
    
    key = gen_key(nr)#
    flag=True
    
    
    value=0xfff0^0x5
    while(flag):
        pt0, pt1 = gen_plain(n)
        
        pt0=pt0&value#choose the subpart 
        
        
        pt0a, pt1a, pt0b, pt1b = make_structure(pt0, pt1, diff=diff, neutral_bits=neutral_bits)
        pt0a, pt1a = cipher.dec_one_round((pt0a, pt1a),key[0][0])
        pt0b, pt1b = cipher.dec_one_round((pt0b, pt1b),key[0][0])
        
  
        ct0a, ct1a = cipher.encrypt((pt0a, pt1a), key)
        ct0b, ct1b = cipher.encrypt((pt0b, pt1b), key)
        
        v=find_good([ct0a, ct1a, ct0b, ct1b], key)
        if(max(v)==2**len(neutral_bits)):flag=False
    '''
    old=np.where(v==2048)[0][0]
    
    for index in range(5,40):
        ct0a[index], ct1a[index], ct0b[index], ct1b[index]=ct0a[old], ct1a[old], ct0b[old], ct1b[old]
    '''
    return([ct0a, ct1a, ct0b, ct1b], key)







#here, we use some symmetries of the wrong key performance profile
#by performing the optimization step only on the 14 lowest bits and randomizing the others
#on CPU, this only gives a very minor speedup, but it is quite useful if a strong GPU is available
#In effect, this is a simple partial mitigation of the fact that we are running single-threaded numpy code here


def bayesian_rank_kr(tmp_br,cand, emp_mean, m, s):
    n = len(cand)
    tmp = tmp_br ^ cand
    v = (emp_mean - m[tmp]) * s[tmp]
    v = v.reshape(-1, n)
    scores = np.linalg.norm(v, axis=1)
    return(scores)



def bayesian_key_recovery(cts, num_cand,num_iter,net, pre_mean, pre_std,tmp_br,best_key_cand1):

    #tmp_br = np.arange(2**14, dtype=np.uint16)
    #tmp_br = np.repeat(tmp_br, num_cand).reshape(-1,num_cand)
    
    #num_cand = 32
    #num_iter=10#
    num_cipher=len(cts[0])
    keys = np.random.choice(2**(WORD_SIZE-2),num_cand,replace=False)
    if(best_key_cand1!=None):
        for num_cand_key in range(len(best_key_cand1)):
            keys[num_cand_key]=best_key_cand1[num_cand_key]
    
    c0l, c0r = np.tile(cts[0], num_cand),np.tile(cts[1], num_cand)
    c1l, c1r = np.tile(cts[2], num_cand),np.tile(cts[3], num_cand)


    scores = np.zeros(2**(WORD_SIZE-2))
    all_keys = np.zeros(num_cand * num_iter,dtype=np.uint16)
    all_v = np.zeros(num_cand * num_iter)
    for i in range(num_iter):
        k = np.repeat(keys, num_cipher)
        ctdata0l, ctdata0r = cipher.dec_one_round((c0l, c0r), k)
        ctdata1l, ctdata1r = cipher.dec_one_round((c1l, c1r), k)

        X=cipher.convert_to_binary([ctdata0l, ctdata0r,ctdata1l, ctdata1r])

        Z = net.predict(X,batch_size=2**16)
        Z = Z.reshape(num_cand, -1)
        means = np.mean(Z, axis=1)
        Z = Z/(1-Z)
        Z = np.log2(Z)
        v =np.sum(Z, axis=1) 
        #print('Checking'+str(i),max(v),end='##')
        all_v[i * num_cand:(i+1)*num_cand] = v
        all_keys[i * num_cand:(i+1)*num_cand] = np.copy(keys)
        scores = bayesian_rank_kr(tmp_br,keys, means, pre_mean, pre_std)
        tmp = np.argpartition(scores, num_cand)
        keys = tmp[0:num_cand]
        r = np.random.randint(0,4,num_cand,dtype=np.uint16)
        r = r << 14
        keys = keys ^ r
    #print('')
    return(all_keys, all_v)



    



def test_bayes(cts, net, net_help, m_main,s_main, m_help,  s_help):
    

    cutoff1=25.0
    cutoff2=100.0
    
  
    print('C1='+str(cutoff1))
    
    print('C2='+str(cutoff2))
    
    tmp_br_last = np.arange(2**14, dtype=np.uint16)
    tmp_br_last = np.repeat(tmp_br_last, 32).reshape(-1,32)
    
    tmp_br_second = np.arange(2**14, dtype=np.uint16)
    tmp_br_second = np.repeat(tmp_br_second, 32).reshape(-1,32)
    
    
    
    n = len(cts[0]);
    use_n=len(cts[0][0])
  
    alpha = np.sqrt(n)

    best_val = -1000.0; 
    best_key = (0,0); 

    best_pod = 0; 

    keys = np.random.choice(2**WORD_SIZE, 32, replace=False);
    keys_best_cand1=[i for i in keys]
    eps = 0.00001; 
    local_best = np.full(n,-1000); 
    num_visits = np.full(n,eps);
    
    it=2**9
    
    for j in range(it):
        priority = local_best + alpha * np.sqrt(log2(j+1) / num_visits) 
        i = np.argmax(priority)
        num_visits[i] = num_visits[i] + 1
        if (best_val > cutoff2):
            return (j,best_pod)
            
            
        print('\r' +'Loop='+str(j)+'---Choosing '+str(i)+' Cipher',end='', flush=True)
        #bayesian_key_recovery_forC1(cts, net, pre_mean, pre_std,num_cand,tmp_br,ALL_NET)
        keys, v = bayesian_key_recovery([cts[0][i], cts[1][i], cts[2][i], cts[3][i]],                                              
                                               num_cand=32, num_iter=5,
                                               net=net, pre_mean=m_main, pre_std=s_main,
                                               tmp_br=tmp_br_second,
                                               best_key_cand1=None
                                               
                                               );
        vtmp = np.max(v);
        if (vtmp > local_best[i]): 
            local_best[i] = vtmp

        if (vtmp > cutoff1):
            l2 = [index for index in range(len(keys)) if v[index] > cutoff1]
            for i2 in l2:
                c0a, c1a = cipher.dec_one_round((cts[0][i],cts[1][i]),keys[i2])
                c0b, c1b = cipher.dec_one_round((cts[2][i],cts[3][i]),keys[i2])       
                #bayesian_key_recovery(cts, num_cand,num_iter,net, pre_mean, pre_std,tmp_br)
                keys2,v2 = bayesian_key_recovery([c0a, c1a, c0b, c1b],
                                                 num_cand=32, num_iter=5,
                                                 net=net_help, pre_mean=m_help, pre_std=s_help,
                                                 tmp_br=tmp_br_second,
                                                 best_key_cand1=None)
                vtmp2 = np.max(v2);
                if (vtmp2 > best_val):
                    best_val = vtmp2; 
                    best_key = (keys[i2], keys2[np.argmax(v2)]); 
                    best_pod=i



    
    return (j,best_pod )


def make_fileFolder(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print("Folder created")
    else:
        print("Folder already exists")




def test(n):
    
    
    
    #load distinguishers
    wdir = './neural networks/'
    net_name='new_ND_VV_Simon32_11R'
    net = tf.keras.models.load_model( wdir+net_name)
        
    net_name='new_ND_VV_Simon32_10R'
    net_help = tf.keras.models.load_model( wdir+net_name)
        
    path='./wrong key response/'
    m_main=np.load(path+"ND_VV_Simon32_11R_mean.npy")
    s_main=np.load(path+"ND_VV_Simon32_11R_std.npy")
    s_main=1.0/s_main
        
    m_help=np.load(path+"ND_VV_Simon32_10R_mean.npy")
    s_help=np.load(path+"ND_VV_Simon32_10R_std.npy")
    s_help=1.0/s_help
    
    
    
    cutoff1=25
    cutoff2=100
    

    

    
    pre_file_path='./key_recovery_result_Bao/'
    key_attack_path='32_5_c1='+str(cutoff1)+'_32_5_c2='+str(cutoff2)
    make_fileFolder(pre_file_path+key_attack_path)
    key_attack_path=pre_file_path+key_attack_path+'/'
    
    
    
    print("Checking Simon32/64 implementation.");
    if (not cipher.check_testvector()):
        print("Error. Aborting.");
        return(0);
    

    choose_plain=[]
    
    num_right_over_c2=0


    
    
    t0 = time()
    for i in range(n):
        print("Test:",i);
        ct, key = gen_challenge()
        
        v=find_good(ct, key)
        print('Generating Ciphertext Data: ',np.where(v==2048)[0])
        
        #test_bayes(cts, net, net_help, m_main,s_main, m_help,  s_help,num_cand4last,num_inter4last,cutoff1,num_cand4second,num_inter4second,cutoff2,max_num_key,rk1,rk2,all_net)
        num_used,best_index = test_bayes(ct,  net=net, net_help=net_help, m_main=m_main, s_main=s_main, m_help=m_help, s_help=s_help)

        choose_plain.append([num_used,v[best_index]])

        print('Choose the plaintext structure',best_index,v[best_index],max(v))
        t1=time()
        print("Wall time per attack (average in seconds):", (t1 - t0)/(i+1))
        if(v[best_index]==2048):
            num_right_over_c2=num_right_over_c2+1
        print(num_right_over_c2/(i+1))
        
        np.save(key_attack_path+'Using_choose_plain.npy',np.array(choose_plain))

    
    t1 = time()
    print("Done.")
   
    print("Wall time per attack (average in seconds):", (t1 - t0)/n)

    
    pre=str(NUM)
    np.save(key_attack_path+'Using_choose_plain.npy',np.array(choose_plain))
    np.save(key_attack_path+'Using_Time_'+pre+'v3_using_patch_3.npy',np.array([(t1 - t0)/n]))
    
    
    return(choose_plain)

choose_plain= test(99)#


