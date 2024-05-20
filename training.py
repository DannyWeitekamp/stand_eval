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

# mc_start_probs = [
#     ["574", "798"],
#     ["248", "315"],
#     ["872", "371"],
#     ["394", "452"],
#     ["252", "533"],
#     ["334", "943"],
#     ["189", "542"],
# ]

def gen_eval_callback(agent, profile, eval_kwargs):
    # Set up 
    c_log = []
    skill_app_map = {}
    def eval_callback():
        c_log.append(
            eval_holdout_stats(agent, profile, skill_app_map, 
                return_cert_stats=True,
                return_state_stats=False,
                **eval_kwargs)
        )
        if(getattr(agent, 'process_lrn_mech', None) is not None):
            print(agent.process_lrn_mech.grammar)
    return c_log, skill_app_map, eval_callback

def c_log_to_stats(c_log):
    stats = {}
    for k in c_log[0]:
        lst = [d[k] for d in c_log]
        if(isinstance(lst[0], (float,int))):
           lst = np.array(lst, dtype=np.float64) 
        stats[k] = lst
    return stats

def resample_problem_pool(env, problem_pool=[], inds=None, pool_size=100):
    if(inds is None):
        inds = np.arange(0, pool_size)

    for i in inds:
        prob_config = env.set_random_problem()
        if(i < len(problem_pool)):
            problem_pool[i] = prob_config
        else:
            problem_pool.append(prob_config)

    print("LEN", len(problem_pool))
    return problem_pool

def run_active_training(agent, domain, n=100, resample_prop=.5, eval_kwargs={}):

    env = make_env(domain)
    profile = f"gt-{domain}.txt"
    if(not os.path.exists(profile)):
        make_completeness_profile(env, 100, profile)

    #############################    

    c_log, skill_app_map, eval_callback = (
        gen_eval_callback(agent, profile, eval_kwargs)
    ) 
    
    problem_pool = resample_problem_pool(env, pool_size=100)
    p = problem_pool[0]
    
    
    for i in range(n):
        # Train one problem 'p'
        trainer = AuthorTrainer(agent, env, problem_set=[p])#, n_problems=n)
        trainer.on_problem_end = eval_callback
        trainer.start() 

        print("+" * 100)
        print(f"Finished problem {i} of {n}")
        
        # Eval average rollout certainty for each problem in pool 
        certainties = []
        for _p in problem_pool:
            env.set_problem(*_p)
            out = agent.act_rollout(env.get_state(),
                # add_out_of_process=True,
                # eval_mode=False,
                ignore_filter=True
            )


            # print(f"CERTAINTY {_p}", out['avg_certainty'])
            certainties.append(out['min_pos_certainty'])
            
        import numpy as np        
        # New 'p' is least certain problem, resample the 50% most certain ones
        order = np.argsort(certainties)
        p = problem_pool[order[0]]
        print("Select", p, certainties[order[0]])
        resample_problem_pool(env, problem_pool, [order[0]]+list(order[int(n*(1-resample_prop)):]))
        # resample_inds = [order[0]]+list(np.random.choice(order[1:],size=int(n*resample_prop), replace=False))
        # resample_problem_pool(env, problem_pool, resample_inds)

    stats = c_log_to_stats(c_log)

    return skill_app_map, stats

def run_training(agent, domain, n=100, eval_kwargs={}):
    env = make_env(domain)
    profile = f"gt-{domain}.txt"
    if(not os.path.exists(profile)):
        make_completeness_profile(env, 100, profile)

    problems = []
    trainer = AuthorTrainer(agent, env, 
        problem_set=problems, n_problems=n)

    c_log, skill_app_map, eval_callback = (
        gen_eval_callback(agent, profile, eval_kwargs)
    ) 

    trainer.on_problem_end = eval_callback 
    trainer.start()

    # Translate from list of dicts (c_log) to dict of numpy arrays (stats)
    stats = c_log_to_stats(c_log)

    return skill_app_map, stats

def train_or_load_rep(domain, when, use_proc, active=False, n_prob=100, rep=0,
                        force_run=False):
    ps = '-proc' if use_proc else ''
    _as = '-active' if active else ''
    dir_name = f"saves/{domain}-{when}{ps}{_as}-{n_prob}-reps/" 
    file_name = dir_name + f"{rep}.pickle"
    if(not os.path.exists(file_name) or force_run):
        agent = make_agent(domain, when, use_proc)

        if(active):
            skill_app_map, stats = run_active_training(agent, domain, n=n_prob)
        else:
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
def train_reps(domain, when, use_proc, active=False, n_prob=100, reps=20,
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
         args=(queue, domain, when, use_proc, active, n_prob, i, force_run))
        p.start()

        # These block until the process terminates
        tup = queue.get()
        p.join() 
        
        # stats, wp_stats, cert_stats = train_or_load_rep(
        #     domain, when, use_proc, n_prob, i, force_run, add_cert_stats)
        stats, wp_stats, cert_stats = tup
        stat_lst.append(stats)
        wp_stat_lst.append(wp_stats)
        cert_stat_lst.append(cert_stats)

    return stat_lst, wp_stat_lst, cert_stat_lst

def train_or_load_condition(domain, when, use_proc, active=False,
                            reps=100, n_prob=100, force_run=False):
    ps = '-proc' if use_proc else ''
    _as = '-active' if active else ''
    file_name = f"saves/{domain}-{when}{ps}{_as}-{n_prob}x{reps}.pickle"
    
    if(not os.path.exists(file_name) or force_run):
        stat_lst, wp_stat_lst, cert_stat_lst = train_reps(
            domain, when, use_proc, active, reps=reps, n_prob=n_prob
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
        
    if(True):
        # For timing
        results = []
        def run_it(domain, when, use_proc, n_prob=100):
            agent = make_agent(domain, when, use_proc)
            skill_app_map, stats = run_active_training(agent, domain, n=n_prob)
            # skill_app_map, stats = run_training(agent, domain, n=n_prob)
            fit_elapse_logger = agent.when_cls.ifit_elapse_logger
            predict_elapse_logger = agent.when_cls.predict_elapse_logger
            results.append(f"{when}: {str(fit_elapse_logger)}")
            results.append(f"{when}: {str(predict_elapse_logger)}")
            # print("Predict:\n", predict_elapse_logger)

        # run_it("mc", "decision_tree", False, n_prob=100)
        run_it("mc", "stand", False, n_prob=100)
        run_it("mc", "random_forest", False, n_prob=10)
        run_it("mc", "xg_boost", False, n_prob=20)

    # for res in results:
    #     print(res)

    # if(False):
    # train_or_load_condition("mc", "decision_tree", False, n_prob=100, reps=3)
    # train_or_load_condition("mc", "random_forest", False, n_prob=100, reps=1)
    # train_or_load_condition("mc", "stand", False, n_prob=100, reps=3)
    # train_or_load_condition("mc", "stand", True, n_prob=100, reps=3)

    # train_or_load_condition("frac", "decision_tree", False, n_prob=100, reps=3)
    # train_or_load_condition("frac", "random_forest", False, n_prob=100, reps=3)
    # train_or_load_condition("frac", "stand", False, n_prob=100, reps=3)
    #train_or_load_condition("frac", "stand", True, n_prob=100, reps=3)
    
    
    # train_or_load_condition("mc", "decision_tree", False, n_prob=100, reps=10)
    # train_or_load_condition("mc", "stand", False, n_prob=100, reps=10)
    # train_or_load_condition("mc", "stand-relaxed", False, n_prob=100, reps=10)
    # train_or_load_condition("mc", "stand", True, n_prob=100, reps=20)
    #train_or_load_condition("mc", "random_forest", False, n_prob=100, reps=3)
    
    train_or_load_condition("mc", "decision_tree", False, n_prob=100, reps=40)
    train_or_load_condition("mc", "stand", False, n_prob=100, reps=40)
    train_or_load_condition("mc", "random_forest", False, n_prob=100, reps=20)
    train_or_load_condition("mc", "xg_boost", False, n_prob=100, reps=20)


    train_or_load_condition("frac", "decision_tree", False, n_prob=100, reps=10)
    train_or_load_condition("frac", "stand", False, n_prob=100, reps=10)
    train_or_load_condition("frac", "random_forest", False, n_prob=100, reps=6)


    train_or_load_condition("frac", "decision_tree", False, n_prob=100, reps=40)
    train_or_load_condition("frac", "stand", False, n_prob=100, reps=40)
    #train_or_load_condition("frac", "stand", True, n_prob=100, reps=20)
    train_or_load_condition("frac", "random_forest", False, n_prob=100, reps=20)
    train_or_load_condition("frac", "xg_boost", False, n_prob=100, reps=20)

    
    
