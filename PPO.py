<<<<<<< HEAD
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from runner import *
import os
from tensorboardX import SummaryWriter

def adjust_learning_rate(optimizer, num_sample, num_samples, init_lr):
    lr = init_lr*(1-num_sample/num_samples)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
def adjust_KpandKd(env, alpha):
    kp = env.pdCon.init_kp.copy()
    env.setKpandKd(kp[2, 2]*alpha)


def PPO(save_path,
        env,
        actor,
        critic,
        s_norm,
        clip_param,
        ppo_epoch,
        num_mini_batch,
        lr_actor=None,
        lr_critic=None,
        max_grad_norm=None,
        use_clipped_value_loss=False
        ):

        if(not os.path.isdir(save_path)):
            os.mkdir(save_path)

        writer = SummaryWriter()

        runner =  Runner(env, s_norm, actor, critic, 2048, 0.99, 0.95, 1)
        optimizer_actor = optim.Adam(actor.parameters(), lr=lr_actor)
        optimizer_critic = optim.Adam(critic.parameters(), lr=lr_critic)
        num_samples = ppo_epoch* runner.sample_size
        num_sample = 0

        M_action = torch.Tensor(runner.env.M_action).float()
        M_state = torch.Tensor(runner.env.M_state).float()

        rwd_threshold = 280
        full_assist_flag = False


        for i in range(ppo_epoch):
            num_sample += runner.sample_size
            #clip_param*=exp_rate
            #adjust_learning_rate(optimizer_actor, num_sample, num_samples, lr_actor)
            #adjust_learning_rate(optimizer_critic, num_sample, num_samples, lr_critic)
            #adjust_KpandKd(runner.env, 0, num_samples, np.array([0,1000,1000, 1000]))
            #print(runner.env.pdCon.kp[2,2])
            #print(runner.env.pdCon.kd)

           
            rollouts = runner.run()
            obs = rollouts["obs"] 
            acs = rollouts["acs"]
            obs_next = rollouts["obs_next"] 
            rwds = rollouts["rwds"]
            dones = rollouts["dones"] 
            vtars = rollouts["vtars"]
            advs = rollouts["advs"]
            alogps = rollouts["alogps"]
            vpreds = rollouts["vpreds"]

            #normalize advantage
            advs = (advs-advs.mean())/(advs.std() + 0.0001)
            advs = np.clip(advs, -4, 4)

            value_loss_epoch = 0
            action_loss_epoch = 0
            symmetry_loss_epoch = 0

            num_epoch = int(obs.shape[0] / num_mini_batch) 

            #print(id_list)
            for epoch in range(10):

                id_list = np.arange(obs.shape[0])
                np.random.shuffle(id_list)

                for idx in range(num_epoch):
                 


                    idx_range  = np.random.choice(obs.shape[0], num_mini_batch, replace=False)

                    obs_batch = torch.Tensor(obs[idx_range, :]).float()
                  
                    acs_batch = torch.Tensor(acs[idx_range, :]).float()
                    vtars_batch = torch.Tensor(vtars[idx_range, :]).float()
                    alogps_old_batch = torch.Tensor(alogps[idx_range, :]).float()
                    advs_batch = torch.Tensor(advs[idx_range, :]).float()
                    vpreds_old_batch = torch.Tensor(vpreds[idx_range, :]).float()


                    optimizer_actor.zero_grad()
                    optimizer_critic.zero_grad()

                    obs_normed_batch = s_norm(obs_batch)
                    m = actor.act_distribution(obs_normed_batch)
                    alogps_batch = m.log_prob(acs_batch).sum(dim=1).view(-1,1)
                    vpreds_batch = critic(obs_normed_batch) 
                    ratio = torch.exp(alogps_batch -alogps_old_batch)
                    
                    surr1 = ratio * advs_batch
                    surr2 = torch.clamp(ratio, 1.0 - clip_param,
                                        1.0 + clip_param) * advs_batch
                    action_loss = -torch.min(surr1, surr2).mean()
                    
                    if(use_clipped_value_loss==False):
                        value_loss = 0.5 * ((vpreds_batch - vtars_batch).pow(2)).mean()
                    else:
                        vpreds_clipped_batch = vpreds_old_batch + (vpreds_batch-vpreds_old_batch).clamp(-clip_param, clip_param)
                        value_loss_clipped = 0.5 * ((vpreds_clipped_batch - vtars_batch).pow(2)/(critic.v_std)).mean()
                        value_loss = 0.5 * ((vpreds_batch - vtars_batch).pow(2)/(critic.v_std)).mean()
                        value_loss = torch.min(value_loss, value_loss_clipped)


                    #motion symmetry loss

                    acs_mirror_batch = torch.mm(acs_batch, M_action)
                    obs_mirror_batch = torch.mm(obs_batch, M_state)
                    symmetry_loss = 4*((acs_mirror_batch - actor(s_norm(obs_mirror_batch))).pow(2)).mean()

                    symmetry_loss_epoch += symmetry_loss.item()

                    value_loss += symmetry_loss


                    
                    value_loss.backward()
                    action_loss.backward()

                    nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
                    nn.utils.clip_grad_norm_(critic.parameters(),max_grad_norm)

                    optimizer_actor.step()
                    optimizer_critic.step()


                    

                    value_loss_epoch += value_loss.item()
                    action_loss_epoch += action_loss.item()
                

            num_updates = num_epoch * num_mini_batch

            value_loss_epoch /= num_updates
            action_loss_epoch /= num_updates
            symmetry_loss_epoch /= num_updates
            #print("iter:{}".format(i))
            #print("action_loss:{}".format(action_loss_epoch))
            #print("value_loss:{}".format(value_loss_epoch))

            if(i%10==0):
              data = {"actor": actor.state_dict(),
              "critic": critic.state_dict(),
              "s_norm": s_norm.state_dict()}

              torch.save(data, save_path+"/checkpoint_"+str(i)+".tar")
            if(i%10==0):
              print("iter:{}".format(i))
              rwd_acc, test_step, rwd_action, rwd_upright, rwd_foot, rwd_vel, rwd_alive = runner.testModel(mode=1)
              print("symmetry loss:{}".format(symmetry_loss_epoch))
              print("kp_init:{}".format(runner.env.pdCon.init_kd[0, 0]))
              if(rwd_acc>rwd_threshold):
                  if(full_assist_flag == False):
                      full_assist_flag = True
                      rwd_threshold*=0.7
                  adjust_KpandKd(runner.env, 0.75)
              #rwd_test , step_test= runner.testModel(mode=0)
              #print("test_rwd:{}".format(rwd_test))
              #print("test_step:{}".format(step_test))
              print("actor_std:{}".format(torch.exp(actor.a_std).detach().numpy()))
              print("######################")
              #debug log
              writer.add_scalar("data/rwd_acc", rwd_acc, i)
              writer.add_scalar("data/test_step", test_step, i)
              writer.add_scalar("data/rwd_action", rwd_action, i)
              writer.add_scalar("data/rwd_upright", rwd_upright, i)
              writer.add_scalar("data/rwd_foot", rwd_foot, i)
              writer.add_scalar("data/rwd_vel", rwd_vel, i)
              writer.add_scalar("data/rwd_alive", rwd_alive, i)
              writer.add_scalar("data/symmetry", symmetry_loss_epoch, i)
              writer.add_scalar("data/init_kp", runner.env.pdCon.init_kd[0, 0], i)

            #embed()
=======
import  numpy as np 
import pybullet as p
print("China")
>>>>>>> 9036abb4b2a9ac72d646e0fe6d04b7a9aa2f386e
