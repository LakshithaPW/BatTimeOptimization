from ActorCritic import *
import numpy as np
from GASSOM import *
from scipy.io import loadmat
from Util import *
import time
import pickle


state_dim=518
n_actions=1

actor = Actor(state_dim, n_actions, activation=nn.Tanh)
critic = Critic(state_dim, activation=nn.Tanh)
ac_learner=A2CLearner(actor,critic)
eps = np.finfo(np.float32).eps.item()
transProb = loadmat('trans_prob_16.mat')        # Load transition probablity matrix
tp=transProb['trans_prob']
PARAM_GASSOM = {
        "win_size": 50,                     #The width of one small patch
        "topo_subspace": [16,16],           #The shape of the topological subspaces
        "transProb": tp,                    #The transition probability matrix
    }
gassommodel = GASSOM(PARAM_GASSOM)

HRTFs = loadmat('HRTF_4_final.mat')
HRTFl=HRTFs['HRTFl'][0][0]
HRTFr=HRTFs['HRTFr'][0][0]
azimuth=HRTFs['azm']
elevation=HRTFs['elv']
lateral=np.arcsin(np.sin(azimuth*np.pi/180)*np.cos(elevation*np.pi/180))*180/np.pi

t_dur=2e-3
offset=1600
stride=5
window=50
N_episodes=50000
eps_interval=100
running_reward = 0
Fs=1000.0
T_eps=3

capture_state=np.zeros((1,N_episodes))
for ep in range(N_episodes):
  N_calls=10
  trajectory=[]
  all_actions=[]
  state = np.zeros((1,256))
  ep_reward = 0

  t_vec=np.arange(0,T_eps-1/Fs,1/Fs)
  N=len(t_vec)

  #initialize the predator, evader location and direction
  theta_p_init=30*(2*np.random.rand()-1)
  theta_p=theta_p_init
  v_p=4
  pos_p=np.zeros((1,2))
  R_W_B=get_R_from_angles(theta_p,0,0)
  t_W_B=np.zeros((3,1))
  T_W_B=np.vstack((np.hstack((R_W_B, t_W_B)), [0, 0, 0 ,1]))

  R_B_H=get_R_from_angles(0,0,0)
  t_B_H=np.zeros((3,1))
  T_B_H=np.vstack((np.hstack((R_B_H, t_B_H)), [0, 0, 0 ,1]))

  T_W_H=np.matmul(T_W_B,T_B_H)
  T_H_W=np.linalg.pinv(T_W_H)

  v_e=2
  theta_e=160*(2*np.random.rand()-1)+theta_p_init
  radius_e=(3-2)*np.random.rand()+2
  pos_e=np.array([radius_e*np.cos(theta_p_init*np.pi/180),radius_e*np.sin(theta_p_init*np.pi/180)])
  
  call_index=0
  prop_time=np.zeros((1,N))
  distance=np.zeros((1,N))
  body_head_angle=np.zeros((1,N))
  pos_e_all=np.zeros((2,N))
  pos_p_all=np.zeros((2,N))
  rel_angle=np.zeros((1,N))

  prev_state=np.zeros((256,))
  prev_body_head_angle_hit=0.0
  prev_distance=0.0
  

  command=0
  call_count=0
  sparse_reward_capture=0
  IPI,tau,k = distance_based_ipi(2,Fs)
  prev_IPI=IPI*(1/Fs)
  prev_ech_rec=0
  prev_echo_hit=0
  for index,t_val in enumerate(t_vec):

    #move the evader
    pos_e = pos_e + np.array([v_e*np.cos(theta_e*np.pi/180),v_e*np.sin(theta_e*np.pi/180)])*(1/Fs)
    rel_p_e = (pos_e-pos_p)
    distance_to_evader=np.sqrt(np.sum((pos_p-pos_e)**2))
    distance[0,index]=np.sqrt(np.sum((pos_p-pos_e)**2))
    t_prop=distance[0,index]/343.0
    prop_time[0,index]=round(t_prop*Fs)

    if index==(call_index+2*prop_time[0,call_index]):
        call_count=call_count+1
        ech_hit=int(call_index+prop_time[0,call_index])
        ech_rec=int(call_index+2*prop_time[0,call_index])
        IPI,tau,k = distance_based_ipi(distance[0,ech_hit],Fs)
        call_index=call_index+IPI;

        if(distance[0,index]<0.2):
          print('Captured')
          sparse_reward_capture=1 

        coor_target_echo_hit=[pos_e_all[0,ech_hit],pos_e_all[1,ech_hit],0,1];
        source_vec=np.matmul(T_H_W,coor_target_echo_hit);
        [source_azim,source_elev,r] = cart2sph(source_vec[0],source_vec[1],source_vec[2]);
        source_azim=source_azim*180/np.pi;
        source_elev=source_elev*180/np.pi;  
        source_lat=np.arcsin(np.sin(source_azim*np.pi/180)*np.cos(source_elev*np.pi/180))*180/np.pi

        t,x=fm_sweep_generation(t_dur)
        HRTFl_interp,HRTFr_interp=HRTF_interp_func(HRTFl,HRTFr,lateral,elevation,source_lat,source_elev)
        HRTF_interp=[HRTFl_interp,HRTFr_interp]
        IR_lr=imp_res(HRTF_interp)
        IR_l=IR_lr[0]
        IR_r=IR_lr[1]
        x_l=np.convolve(IR_l.flatten(),x.flatten(),'full');
        x_r=np.convolve(IR_r.flatten(),x.flatten(),'full');
        x_l=x_l[offset:]
        x_r=x_r[offset:]
        x_win_norm=form_feature_vec(x_l,x_r,window,stride)

        #Encode the input signal
        coef,error = gassommodel.sparseEncode(x_win_norm.T)
        #Update the GASSOM feature extractors
        gassommodel.updateBasis(x_win_norm.T)

        state=coef.copy()
        body_head_angle_hit=body_head_angle[0,ech_hit]*np.pi/180
        cur_IPI=IPI*(1/Fs)
        temp_vec=np.zeros((6,))
        temp_vec[0]=prev_body_head_angle_hit
        temp_vec[1]=body_head_angle_hit
        temp_vec[2]=prev_IPI
        temp_vec[3]=cur_IPI
        temp_vec[4]=1e-3/prev_IPI
        temp_vec[5]=1e-3/cur_IPI        

        feature_vec=np.concatenate((prev_state,state,temp_vec))
        feature_vec=to_ten(feature_vec)
        reward = -(distance[0,ech_hit]-prev_distance)*Fs/(ech_hit-prev_echo_hit)
        if(call_count<2):
          action=ac_learner.learn(feature_vec,reward,0)
          action=0.0
        else:
          action=ac_learner.learn(feature_vec,reward,1)        
        all_actions.append(action)
        ep_reward += reward
        command=action/IPI

        prev_state=state.copy()
        prev_IPI=cur_IPI
        prev_ech_rec=ech_rec
        prev_body_head_angle_hit=body_head_angle_hit
        prev_distance=distance[0,ech_hit]
        prev_echo_hit=ech_hit
        if(distance[0,index]<0.68):
          v_p=v_p-0.1*(v_p-(abs(v_e)+0.5))
        if(sparse_reward_capture==1 or call_index>=N):
          break

    del_theta_B_H=get_R_from_angles(command,0,0)
    T_B_H=np.matmul(T_B_H,np.vstack((np.hstack((del_theta_B_H, t_B_H)), [0, 0, 0 ,1])))
    T_W_H=np.matmul(T_W_B,T_B_H)
    T_H_W=np.linalg.pinv(T_W_H)
    theta_b_h,phi_b_h,gamma_b_h=get_angles_from_R(T_B_H[0:3,0:3])
    body_head_angle[0,index]=theta_b_h*180/np.pi

    if(index>tau):
        theta_p=theta_p+k*body_head_angle[0,index-tau]*(1/Fs)
    else:
        theta_p=theta_p

    [del_bx,del_by,del_bz]=sph2cart(theta_p,0,v_p)
    R_W_B=get_R_from_angles(theta_p,0,0)
    delt_W_B=np.zeros((1,3))
    delt_W_B[0,0]=del_bx
    delt_W_B[0,1]=del_by
    delt_W_B[0,2]=del_bz
    T_W_B[0:3,3]=T_W_B[0:3,3]+delt_W_B*(1/Fs)
    T_W_B[0:3,0:3]=R_W_B
    pos_p[0,0]=T_W_B[0,3]
    pos_p[0,1]=T_W_B[1,3]

    coor_target=[pos_e[0],pos_e[1],0,1]
    source_vec=np.matmul(T_H_W,coor_target)
    [source_azim,source_elev,r] = cart2sph(source_vec[0],source_vec[1],source_vec[2])
    source_azim=source_azim*180/np.pi
    source_elev=source_elev*180/np.pi

    rel_angle[0,index]=source_azim
    pos_e_all[0,index]=pos_e[0]
    pos_e_all[1,index]=pos_e[1]
    pos_p_all[0,index]=pos_p[0,0]
    pos_p_all[1,index]=pos_p[0,1]

    if(abs(source_azim)>90):
        break
  
  # update cumulative reward
  running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
  capture_state[0,ep]=sparse_reward_capture
  capture_percentage=(np.sum(capture_state)/N_episodes)*100
  # log results
  if ep % eps_interval == 0:
    print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}\tCapture percentage: {:.2f}'.format(ep, ep_reward, running_reward,capture_percentage))
    timenow = time.strftime('%Y%m%d-%H%M%S',time.localtime(time.time()))
    filename = 'Models//Time_Opt_AC_'+ str(ep) + '.pkl'  
    Save_model=[actor,critic,gassommodel]
    pickle.dump(Save_model,open(filename,"wb"))

print('Training done...')