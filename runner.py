import numpy as np
import torch
from IPython import embed
class Runner(object):
    def __init__(self, env, s_norm, actor, critic, sample_size, gamma, lam, exp_rate, max_steps=500):
        self.env = env
        self.s_norm = s_norm
        self.actor = actor
        self.critic = critic
        self.obs = np.asarray(env.reset().reshape((1, -1)), dtype=np.float32)
        self.obs[:] = env.reset()
        self.toTorch = lambda x: torch.Tensor(x).float()
        self.sample_size = sample_size
        self.dones = None

        # lambda used in GAE
        self.lam = lam

        # discount rate
        self.gamma = gamma
        self.v_min = self.critic.v_min
        self.v_max = self.critic.v_max
      

        # exploration rate
        self.exp_rate = exp_rate

        #parameters for curriculum learning
        self.max_steps = max_steps
        self.current_step = 0

    def run(self):
        self.current_step = 0
        n_steps = self.sample_size
        self.s_norm.update()
        self.obs[:] = self.env.reset()
        mb_obs=[]
        mb_acs=[]
        mb_rwds=[]
        mb_vpreds=[]
        mb_alogps=[]
        mb_obs_next=[]
        mb_dones=[]
        mb_vpreds_next=[]
        mb_fails=[]
        for _ in range(n_steps):
            obst = self.toTorch(self.obs)
            self.s_norm.record(obst)
            obst_norm = self.s_norm(obst)
            
            with torch.no_grad():
                # with probability exp_rate to act stochastically
                m = self.actor.act_distribution(obst_norm)
                #exp = np.random.rand() < self.exp_rate
                acs = m.sample() #if exp else m.mean
                alogps = torch.sum(m.log_prob(acs), dim=1).numpy()
                acs = acs.view(-1).numpy()
                vpreds = self.critic(obst_norm).view(-1).numpy()
                #print("vpreds ：{}".format(vpreds.shape))
               
          
            mb_obs.append(self.obs.copy().reshape(-1))
            mb_acs.append(acs.reshape(-1))
            mb_vpreds.append(vpreds)
            mb_alogps.append(alogps)


            self.obs[:], rwds, self.dones, infos = self.env.step(acs)
            self.current_step+=1
            if(self.dones==False):
                    mb_fails.append(False)
            else:
                    mb_fails.append(True)
            if(self.current_step > self.max_steps):
                self.dones = True
            

            mb_dones.append(self.dones)
            mb_obs_next.append(self.obs.copy().reshape(-1))
            if(self.dones == True):
                if(mb_fails[-1]==False):
                     with torch.no_grad():
                        obst = self.toTorch(self.obs)
                        self.s_norm.record(obst)
                        obst_norm = self.s_norm(obst)
                        vpreds_next = self.critic(obst_norm).view(-1).numpy()
                        #print("vpreds_next:{}".format(vpreds_next.shape))
                        mb_vpreds_next.append(vpreds_next.reshape(-1))
                else:
                    mb_vpreds_next.append(np.asarray([0], dtype=np.float32))
                self.obs[:] = self.env.reset()
                self.current_step = 0
            else:
                with torch.no_grad():
                        obst = self.toTorch(self.obs)
                        self.s_norm.record(obst)
                        obst_norm = self.s_norm(obst)
                        vpreds_next = self.critic(obst_norm).view(-1).numpy()
                        #print("vpreds_next:{}".format(vpreds_next.shape))
                        mb_vpreds_next.append(vpreds_next.reshape(-1))
            mb_rwds.append([rwds])
           


        mb_acs = np.asarray(mb_acs, dtype = np.float32)
        mb_obs = np.asarray(mb_obs, dtype = np.float32)
        mb_dones = np.asarray(mb_dones, dtype = np.bool)
        mb_obs_next = np.asarray(mb_obs_next, dtype = np.float32)
        mb_rwds = np.asarray(mb_rwds, dtype = np.float32)
        mb_vpreds = np.asarray(mb_vpreds, dtype = np.float32)
        mb_vpreds_next = np.asarray(mb_vpreds_next, dtype = np.float32)
        mb_alogps = np.asarray(mb_alogps, dtype = np.float32)


        mb_advs = np.zeros_like(mb_rwds)
        mb_vtars = np.zeros_like(mb_rwds)

        mb_delta = mb_rwds + self.gamma * mb_vpreds_next - mb_vpreds
     
        #compute the GAE and TD(lamda) advantage and value function
        lastgaelam = 0
        lastvtar=0
        for t in reversed(range(n_steps)):
            if(t==n_steps-1):
                coe = 1.0
            else:
                coe = 1-mb_dones[t]              
            mb_advs[t] = lastgaelam =mb_delta[t] + coe*self.gamma*self.lam*lastgaelam
            mb_vtars[t] = mb_advs[t]+mb_vpreds[t]
        #mb_vtars = mb_advs + mb_vpreds
        #mb_advs = mb_delta
        #mb_vtars = mb_delta + mb_vpreds


        #mb_vtars = np.clip(mb_vtars, self.v_min, self.v_max)
        rollouts={}
        rollouts["acs"] = mb_acs
        rollouts["obs"] = mb_obs
        rollouts["obs_next"] = mb_obs_next
        rollouts["rwds"] = mb_rwds
        rollouts["advs"] = mb_advs
        rollouts["vtars"] = mb_vtars
        rollouts["alogps"] = mb_alogps
        rollouts["dones"] = mb_dones
        rollouts["vpreds"] = mb_vpreds


        return rollouts
    def setExpRate(self, exp_rate):
        self.exp_rate = exp_rate

    def testModel(self, num_epoch =10,render=False, mode=0):
       
        if(mode==0):
            self.env.pdCon.kp[0, 0]=0
            self.env.pdCon.kp[2, 2]=0
            self.env.pdCon.kd[0, 0]=0
            self.env.pdCon.kd[2, 2]=0
        #print(self.env.pdCon.kp)
        #print(self.env.pdCon.kd)
        with torch.no_grad():
            rwd_acc=0
            test_step=0
            rwd_action=0
            rwd_alive =0
            rwd_jump =0
            rwd_vel =0
            rwd_foot = 0
            rwd_upright = 0
            for i in range(num_epoch):
                self.current_step = 0
                obs = self.env.reset()
                obs= torch.Tensor(obs).float()#.view(-1,1)
                #M = torch.Tensor(self.env.M_state)
                #N = torch.Tensor(self.env.M_action)
                #obs = torch.mm(M, obs).view(-1)
                done = False
                obs_norm = self.s_norm(obs)
                while(not done):
                    ac = self.actor(obs_norm).numpy()
                    '''print(obs)
                    print(ac1)
                    obs = torch.mm(M, obs.view(-1, 1)).view(-1)
                    obs_norm = self.s_norm(obs)
                    ac2 = self.actor(obs_norm)#.numpy()
                    print(obs)
                    print(ac2)
                    print(torch.mm(N, ac1.view(-1,1)) - ac2.view(-1,1))
                    print("#####")
                    ac = ac2.view(-1).numpy()'''


                    obs, rwd, done , info = self.env.step(ac)
                    #print(obs[-1])
                    if(render==True):
                        self.env.render()
                    rwd_acc += rwd
                    rwd_alive += info["rwd_live"]
                    rwd_action += info["rwd_action"]
                    rwd_upright += info["rwd_upright"]
                    rwd_foot += info["rwd_foot"]
                    rwd_vel += info["rwd_vel"]
                    rwd_jump += info["rwd_jump"]
                    test_step+=1
                    obs = torch.Tensor(obs).float()#.view(-1,1)
                    #obs = torch.mm(M, obs).view(-1)
                    obs_norm = self.s_norm(obs)
                    self.current_step+=1
                    if(self.current_step>self.max_steps):
                        break
            print("rwd_acc:{}".format(rwd_acc/num_epoch))
            print("test_step:{}".format(test_step/num_epoch))
            print("rwd_action:{}".format(rwd_action/test_step))
            print("rwd_upright:{}".format(rwd_upright/test_step))
            print("rwd_jump:{}".format(rwd_jump/test_step))
            print("rwd_foot:{}".format(rwd_foot/test_step))
            print("rwd_vel:{}".format(rwd_vel/test_step))
            print("rwd_alive:{}".format(rwd_alive/test_step))
            print("kp_start:{}".format(self.env.kpandkd_start))
            print("kp_end:{}".format(self.env.kpandkd_end))


            return rwd_acc/num_epoch, test_step/num_epoch, rwd_action/test_step, rwd_upright/test_step, rwd_foot/test_step, rwd_vel/test_step, rwd_alive/test_step
            

        


if __name__ == "__main__":
    #import pybullet_envs
    import gym
    from model import *
    from gym_biped.envs.bipedEnv import *
    #env = gym.make("Walker2d-v2")
    #env.isRender = False
    env = bipedEnv(True)
    env.reset()
   
    s_norm, actor, critic = load_model("./Walker2d-Bullet-curriculum-5-50/checkpoint_1920.tar")
    #actor = Actor(env.observation_space.shape[0], env.action_space.shape[0], env.action_space, hidden=[128, 64])
    #critic = Critic(env.observation_space.shape[0], 0, 3/(1-0.99), hidden =[128, 64])
    #s_norm = Normalizer(env.observation_space.shape[0])
    runner =  Runner(env, s_norm, actor, critic, 4096, 0.99, 0.95, 1)
    #runner.env.pdCon.kp[0, 0]=0
    #runner.env.pdCon.kd[0, 0]=1000
    #runner.env.pdCon.kp[2, 2]=1000
    #runner.env.pdCon.kd[2, 2]=100
    runner.env.setKpandKd(0)
    print(runner.testModel(mode=1))


      

      