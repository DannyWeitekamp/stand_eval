import os
import pickle
import numpy as np
from eval_funcs import eval_holdout_stats, eval_total_cert_stats, avg_stats
from tutorenvs.fractions_std import FractionArithmetic
from tutorenvs.multicolumn_std import MultiColumnAddition
from tutorenvs.trainer import Trainer, AuthorTrainer
from agent_configs import make_agent



def make_completeness_profile(env, n=100, name=""):
    problems = []
    for i in range(100):
        env.set_random_problem()
        problems.append(env.problem)
    env.make_completeness_profile(problems, name)

def make_env(domain):
    if(domain == "frac"):
        env = FractionArithmetic(check_how=False, check_args=True,
                             demo_args=True, demo_how=True,
                             problem_types=["AD","AS","M"], n_fracs=2)
    elif(domain == "mc"):
        env = MultiColumnAddition(check_how=False, check_args=True,
            demo_args=True, demo_how=True, n_digits=3,
            carry_zero=False, random_n_digits=False)
    elif(domain == "mc-zero-carry"):
        env = MultiColumnAddition(check_how=False, check_args=True,
            demo_args=True, demo_how=True, n_digits=3,
            carry_zero=True, random_n_digits=False)

    return env

mc_start_probs = [
    ["574", "798"],
    ["248", "315"],
    ["872", "371"],
    ["394", "452"],
    ["252", "533"],
    ["334", "943"],
    ["189", "542"],
]


def run_training(agent, domain, n=100, eval_kwargs={}):
    env = make_env(domain)
    profile = f"gt-{domain}.txt"
    if(not os.path.exists(profile)):
        make_completeness_profile(env, 100, profile)

    problems = []
    if(domain == "mc"):
        problems = mc_start_probs

    trainer = AuthorTrainer(agent, env, 
        problem_set=problems, n_problems=n)

    # Set up 
    c_log = []
    skill_app_map = {}
    def on_problem_end():
        c_log.append(
            eval_holdout_stats(agent, profile, skill_app_map, 
                return_cert_stats=True,
                return_state_stats=False,
                **eval_kwargs)
        )
        if(getattr(agent, 'process_lrn_mech', None) is not None):
            print(agent.process_lrn_mech.grammar)
        # print("---------------")
        # for skill in agent.skills.values():
        #     print()
        #     print(skill)
        #     print(skill.when_lrn_mech)
        # print("---------------")

    trainer.on_problem_end = on_problem_end 

    trainer.start()

    # Translate from list of dicts (c_log) to dict of numpy arrays (stats)
    stats = {}
    for k in c_log[0]:
        lst = [d[k] for d in c_log]
        if(isinstance(lst[0], (float,int))):
           lst = np.array(lst, dtype=np.float64) 
        stats[k] = lst

    return skill_app_map, stats

def train_or_load_rep(domain, when, use_proc, n_prob=100, rep=0,
                        force_run=False):
    dir_name = f"saves/{domain}-{when}{'-proc' if use_proc else ''}-{n_prob}-reps/" 
    file_name = dir_name + f"{rep}.pickle"
    if(not os.path.exists(file_name) or force_run):
        agent = make_agent(domain, when, use_proc)
        skill_app_map, stats = run_training(agent, domain, n=n_prob)

        wp_stats = eval_total_cert_stats(skill_app_map, stats['when_preds'])
        cert_stats = eval_total_cert_stats(skill_app_map, stats['certainties'])
        tup = (stats, wp_stats, cert_stats)


        os.makedirs(dir_name, exist_ok=True)
        with open(file_name, 'wb+') as f:
            pickle.dump(tup, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print("SKIP REP: ", rep)
        with open(file_name, 'rb') as f:
            tup = pickle.load(f)

    return tup


from multiprocessing import Process, Queue
def train_reps(domain, when, use_proc, n_prob=100, reps=100,
                force_run=False):
    from agent_configs import make_agent
    stat_lst = []
    wp_stat_lst = []
    cert_stat_lst = []
    for i in range(reps):

        # Run train_or_load_rep in seperate process 
        def run(queue, *args):
            tup = train_or_load_rep(*args)
            queue.put(tup)
        queue = Queue()
        p = Process(target=run,
         args=(queue, domain, when, use_proc, n_prob, i, force_run))
        p.start()

        # These block until the process terminates
        tup = queue.get()
        print("AFTER GET")
        p.join() 
        

        print("DONE!")
        
        # stats, wp_stats, cert_stats = train_or_load_rep(
        #     domain, when, use_proc, n_prob, i, force_run, add_cert_stats)
        stats, wp_stats, cert_stats = tup
        stat_lst.append(stats)
        wp_stat_lst.append(wp_stats)
        cert_stat_lst.append(cert_stats)

    return stat_lst, wp_stat_lst, cert_stat_lst

def train_or_load_condition(domain, when, use_proc_lrn,
                            reps=100, n_prob=100, force_run=False):
    file_name = f"saves/{domain}-{when}{'-proc' if use_proc_lrn else ''}-{n_prob}x{reps}.pickle"
    
    if(not os.path.exists(file_name) or force_run):
        stat_lst, wp_stat_lst, cert_stat_lst = train_reps(
            domain, when, use_proc_lrn, reps=reps, n_prob=n_prob
        )
        stats = avg_stats(stat_lst)
        wp_stats = avg_stats(wp_stat_lst)
        cert_stats = avg_stats(cert_stat_lst)
        tup = (stats, wp_stats, cert_stats)
        
        os.makedirs("saves", exist_ok=True)
        with open(file_name, 'wb+') as f:
            pickle.dump(tup, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(file_name, 'rb') as f:
            tup = pickle.load(f)
        stats, wp_stats, cert_stats = tup
    
    return stats, wp_stats, cert_stats



if __name__ == "__main__":
    

    # train_or_load_condition("mc", "decision_tree", False, n_prob=100, reps=3)
    # train_or_load_condition("mc", "random_forest", False, n_prob=100, reps=1)
    # train_or_load_condition("mc", "stand", False, n_prob=100, reps=3)
    # train_or_load_condition("mc", "stand", True, n_prob=100, reps=3)

    # train_or_load_condition("frac", "decision_tree", False, n_prob=100, reps=3)
    # train_or_load_condition("frac", "random_forest", False, n_prob=100, reps=3)
    # train_or_load_condition("frac", "stand", False, n_prob=100, reps=3)
    #train_or_load_condition("frac", "stand", True, n_prob=100, reps=3)
    
    
    # train_or_load_condition("mc", "decision_tree", False, n_prob=100, reps=10)
    train_or_load_condition("mc", "stand", False, n_prob=100, reps=10)
    # train_or_load_condition("mc", "stand-relaxed", False, n_prob=100, reps=10)
    # train_or_load_condition("mc", "stand", True, n_prob=100, reps=20)
    #train_or_load_condition("mc", "random_forest", False, n_prob=100, reps=3)
    
    train_or_load_condition("mc", "decision_tree", False, n_prob=100, reps=20)
    train_or_load_condition("mc", "stand", False, n_prob=100, reps=20)
    #train_or_load_condition("mc", "stand-relaxed", False, n_prob=100, reps=10)
    #train_or_load_condition("mc", "stand", True, n_prob=100, reps=20)
    train_or_load_condition("mc", "random_forest", False, n_prob=100, reps=6)

    train_or_load_condition("frac", "decision_tree", False, n_prob=100, reps=10)
    train_or_load_condition("frac", "stand", False, n_prob=100, reps=10)
    #train_or_load_condition("frac", "stand", True, n_prob=100, reps=10)
    train_or_load_condition("frac", "random_forest", False, n_prob=100, reps=3)


    train_or_load_condition("frac", "decision_tree", False, n_prob=100, reps=20)
    train_or_load_condition("frac", "stand", False, n_prob=100, reps=20)
    #train_or_load_condition("frac", "stand", True, n_prob=100, reps=20)
    train_or_load_condition("frac", "random_forest", False, n_prob=100, reps=6)

    
    
