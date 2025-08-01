import os, sys
import pickle
import numpy as np
from eval_funcs import eval_holdout_stats, eval_total_cert_stats, avg_stats
#from tutorenvs.fractions_std import FractionArithmetic
#from tutorenvs.multicolumn_std import MultiColumnAddition
#from tutorenvs.trainer import Trainer, AuthorTrainer
from tutorgym.env_classes.misc.fraction_arith.fractions import FractionArithmetic
from tutorgym.env_classes.misc.mc_addition.multicolumn import MultiColumnAddition, Action, ProblemState
from tutorgym.trainer import Trainer, AuthorTrainer
import tutorgym.helpers.ai2t_helpers # Registers SkillApplication -> Action
from agent_configs import make_agent
from tutorgym.evaluator import CompletenessEvaluator



# def make_completeness_profile(env, n=100, name=""):
#     problems = []
#     for i in range(100):
#         env.set_random_problem()
#         problems.append(env.problem)
#     env.make_rand_compl_prof(problems, name)



def make_env(domain):
    if(domain == "frac"):
        env = FractionArithmetic(demo_annotations=["arg_foci", "how_help"],
                                check_annotations=["arg_foci"],
                             problem_types=["AD","AS","M"], n_fracs=2)
    elif(domain == "mc"):
        env = MultiColumnAddition(demo_annotations=["arg_foci", "how_help"],
                                  check_annotations=["arg_foci"], n_digits=3,
            carry_zero=False, random_n_digits=False)
    elif(domain == "mc-zero-carry"):
        env = MultiColumnAddition(demo_annotations=["arg_foci", "how_help"],
                                  check_annotations=["arg_foci"],  n_digits=3,
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
    def eval_callback(context):
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

    # print("LEN", len(problem_pool))
    # print(problem_pool[0])
    return problem_pool

def run_active_training(agent, domain, n=100, prof_n=100, resample_prop=.5, eval_kwargs={}, start_probs=[]):

    env = make_env(domain)
    profile = f"gt-{domain}.txt"
    if(not os.path.exists(profile)):
        env.make_rand_compl_prof(profile, prof_n)

        # make_completeness_profile(env, 100, profile)

    #############################    

    c_log, skill_app_map, eval_callback = (
        gen_eval_callback(agent, profile, eval_kwargs)
    )


    if(len(start_probs) > 0):
        trainer = AuthorTrainer(agent, env, 
            problem_set=start_probs,
            # evaluators=[compl_evaluator])
            problem_end_callbacks=[eval_callback])

        trainer.on_problem_end = eval_callback 
        trainer.start()

    
    problem_pool = resample_problem_pool(env, pool_size=100)
    p = problem_pool[0]
    
    
    for i in range(len(start_probs), n):
        # Train one problem 'p'
        trainer = AuthorTrainer(agent, env, 
            problem_set=[p], problem_end_callbacks=[eval_callback])#, n_problems=n)
        # trainer.on_problem_end = eval_callback
        trainer.start() 

        print("+" * 100)
        print(f"Finished problem {i+1} of {n}")
        
        # Eval average rollout certainty for each problem in pool 
        certainties = []
        for _p in problem_pool:
            print(_p)
            env.set_problem(**_p)
            out = agent.act_rollout(env.get_state().objs,
                # add_out_of_process=True,
                # eval_mode=False,
                ignore_filter=True
            )


            print(f"CERTAINTY {_p}", out['min_pos_certainty'])
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

def run_training(agent, domain, n=100, n_prof=100, eval_kwargs={}, start_probs=[]):
    env = make_env(domain)
    profile = f"gt-{domain}.txt"
    if(not os.path.exists(profile)):
        env.make_rand_compl_prof(profile, n_prof)

    c_log, skill_app_map, eval_callback = (
        gen_eval_callback(agent, profile, eval_kwargs)
    ) 

    # compl_evaluator = CompletenessEvaluator(
    #     eval_freq="problem_end", 
    #     compl_prof = profile, 
    #     print_htn=True, print_diff=True,
    #     check_annotations=[])

    trainer = AuthorTrainer(agent, env, 
        problem_set=start_probs, n_problems=n,
        # evaluators=[compl_evaluator])
        problem_end_callbacks=[eval_callback])

    trainer.start()
    # c_log = compl_evaluator.log

    # Translate from list of dicts (c_log) to dict of numpy arrays (stats)
    # print("LOG")
    # print(c_log)
    stats = c_log_to_stats(c_log)
    # print("STATS")
    # print(stats)
    # raise ValueError()

    return skill_app_map, stats


def load_rep(domain, when, use_proc, active=False, n_prob=100, rep=0):
    ps = '-proc' if use_proc else ''
    _as = '-active' if active else ''
    dir_name = f"saves/{domain}-{when}{ps}{_as}-{n_prob}-reps/" 
    file_name = dir_name + f"{rep}.pickle"
    if(os.path.exists(file_name)):
        with open(file_name, 'rb') as f:
            stats = pickle.load(f)
        return stats
    return None


def train_or_load_rep(domain, when, use_proc, active=False, n_prob=100, rep=0,
                        force_run=False):
    ps = '-proc' if use_proc else ''
    _as = '-active' if active else ''
    dir_name = f"saves/{domain}-{when}{ps}{_as}-{n_prob}-reps/" 
    file_name = dir_name + f"{rep}.pickle"
    if(not os.path.exists(file_name) or force_run):
        print("RUN REP", rep, file_name)
        agent = make_agent(domain, when, use_proc)

        start_probs = []
        
        # An exception for Fractions, always start with one problem 
        #  of each type
        if(domain == "frac"):
            pass
            # env = make_env(domain)
            # start_probs = [
            #     env.set_random_problem("AD"),
            #     env.set_random_problem("AS"),
            #     env.set_random_problem("M"),
            #     env.set_random_problem("AD"),
            #     env.set_random_problem("AS"),
            #     env.set_random_problem("M"),
            # ]

            
        
            # An exception for MC when using process-learning, use curated set
        if(domain == "mc"):
            # print("BLARGS")
            start_probs = [
                    {"upper": "574", "lower": "798"},
                    {"upper": "872", "lower": "371"},
                    {"upper": "913", "lower": "522"},
                    {"upper": "248", "lower": "315"},
                    
                    {"upper": "394", "lower": "452"},
                    {"upper": "252", "lower": "533"},
                    {"upper": "334", "lower": "943"},
                    {"upper": "189", "lower": "542"},
            ]
        if(active):
            skill_app_map, stats = run_active_training(agent, domain, n=n_prob, start_probs=start_probs)
        else:
            skill_app_map, stats = run_training(agent, domain, n=n_prob, start_probs=start_probs)

        # print(skill_app_map)
        corrs = np.array([rew > 0 for ind, rew in skill_app_map.values()], dtype=np.bool_)

        wp_stats = eval_total_cert_stats(corrs, stats['when_preds'])
        # raise ValueError()
        #cert_stats = eval_total_cert_stats(corrs, stats['certainties'])
        stats = {**stats, **wp_stats}


        os.makedirs(dir_name, exist_ok=True)
        with open(file_name, 'wb+') as f:
            pickle.dump(stats, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print("SKIP REP: ", rep)
        with open(file_name, 'rb') as f:
            stats = pickle.load(f)

    return stats


from multiprocessing import Process, Queue
def train_reps(domain, when, use_proc, active=False, n_prob=100, reps=40, start=0,
                force_run=False, sep_process=False):
    from agent_configs import make_agent
    stat_lst = []
    wp_stat_lst = []
    cert_stat_lst = []
    for i in range(start, reps):
        args = (domain, when, use_proc, active, n_prob, i, force_run)
        if(sep_process):
            # Run train_or_load_rep in seperate process 
            def run(queue, *args):
                stats = train_or_load_rep(*args)
                queue.put(stats)
            queue = Queue()
            p = Process(target=run,
             args=(queue, *args))
            p.start()

            # These block until the process terminates
            stats = queue.get()
            p.join() 
        else:
            print("THIS HAPPNED")
            stats = train_or_load_rep(*args)
        
        # stats, wp_stats, cert_stats = train_or_load_rep(
        #     domain, when, use_proc, n_prob, i, force_run, add_cert_stats)
        # stats, wp_stats, cert_stats = tup
        stat_lst.append(stats)
        # wp_stat_lst.append(wp_stats)
        # cert_stat_lst.append(cert_stats)

    return stat_lst

def train_or_load_condition(domain, when, use_proc, active=False,
                            reps=40, start=0, n_prob=100, force_run=False, sep_process=False):
    ps = '-proc' if use_proc else ''
    _as = '-active' if active else ''
    file_name = f"saves/{domain}-{when}{ps}{_as}-{n_prob}x{reps}.pickle"
    
    if(not os.path.exists(file_name) or force_run):
        stat_lst = train_reps(
            domain, when, use_proc, active, reps=reps, n_prob=n_prob, start=start,
            force_run=force_run, sep_process=sep_process
        )

        # Don't average stats for fill-in runs that don't cover all reps
        if(start != 0):
            return
        print(stat_lst)
        stats = avg_stats(stat_lst)
        # wp_stats = avg_stats(wp_stat_lst)
        # cert_stats = avg_stats(cert_stat_lst)
        # stats = {**stats, **wp_stats}
        # tup = (stats, wp_stats, cert_stats)
        
        os.makedirs("saves", exist_ok=True)
        with open(file_name, 'wb+') as f:
            pickle.dump(stats, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(file_name, 'rb') as f:
            stats = pickle.load(f)
        # stats, wp_stats, cert_stats = tup
    
    return stats



if __name__ == "__main__":
        
    if(False):
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

        run_it("mc", "decision_tree", False, n_prob=100)
        run_it("mc", "stand", False, n_prob=100)
        run_it("mc", "random_forest", False, n_prob=10)
        run_it("mc", "xg_boost", False, n_prob=20)



    active = "a" in sys.argv

    domain = "mc"
    if("m" in sys.argv):
        domain = "mc"
    elif("f" in sys.argv):
        domain = "frac"

    model = "stand"
    if("s" in sys.argv):
        model = "stand"
    elif("h" in sys.argv):
        model = "stand_hs"
    elif("r" in sys.argv):
        model = "random_forest"
    elif("x" in sys.argv):
        model = "xg_boost"
    elif("d" in sys.argv):
        model = "decision_tree"

    use_proc = False
    if("p" in sys.argv):
        use_proc = True

    start = 0
    if("s10" in sys.argv):
        start = 10
    elif("s20" in sys.argv):
        start = 20
    elif("s30" in sys.argv):
        start = 30

    force_run = False
    if("force" in sys.argv):
        force_run = True

    sep_process = False
    if("sepp" in sys.argv):
        sep_process = True

    train_or_load_condition(domain, model, use_proc, active=active,
        n_prob=25, reps=40, start=start,
        force_run=force_run, sep_process=sep_process)
    

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
    
    #train_or_load_condition("mc", "decision_tree", False, n_prob=100, reps=20)
    

    # if("blehh" in sys.argv):
    #     train_or_load_condition("mc", "decision_tree", False, active=False, n_prob=100, reps=20)
    # elif("active" in sys.argv):
    #     train_or_load_condition("mc", "stand", False, active=True, n_prob=100, reps=20)
    #     train_or_load_condition("mc", "random_forest", False, active=True, n_prob=100, reps=20)
    #     train_or_load_condition("mc", "xg_boost", False, active=True, n_prob=100, reps=20)
    # else:
    #     train_or_load_condition("mc", "stand", False, active=False, n_prob=100, reps=20)
    #     train_or_load_condition("mc", "random_forest", False, active=False, n_prob=100, reps=20)
    #     train_or_load_condition("mc", "xg_boost", False, active=False, n_prob=100, reps=20)


    #train_or_load_condition("frac", "decision_tree", False, n_prob=100, reps=10)
    #train_or_load_condition("frac", "stand", False, n_prob=100, reps=10)
    #train_or_load_condition("frac", "random_forest", False, n_prob=100, reps=6)


    #train_or_load_condition("frac", "decision_tree", False, n_prob=100, reps=40)
    #train_or_load_condition("frac", "stand", False, n_prob=100, reps=40)
    #train_or_load_condition("frac", "stand", True, n_prob=100, reps=20)
    #train_or_load_condition("frac", "random_forest", False, n_prob=100, reps=20)
    #train_or_load_condition("frac", "xg_boost", False, n_prob=100, reps=20)

    
    
