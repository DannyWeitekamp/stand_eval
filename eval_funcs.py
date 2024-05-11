import json
from apprentice.shared import SAI
import numpy as np

def _load_profile_line(agent, line):
    item = json.loads(line)
    profile_sais = [SAI(x) for x in item['sais']]
    state = item['state']
    state = agent.standardize_state(state)
        
    return state, item, profile_sais


# ----------------------------------------------
#  Evaluate Step Level Diffs (Agent Predictions vs Ground Truth)

def _eval_diffs(agent, state, item, profile_sais):    
    # Run act_all in eval_mode to give its most conservative guess of the conflict set
    is_start = len(item['hist'])==0
    prob_uid = state.get("__uid__")                
    agent_sais = agent.act_all(state, return_kind='sai',
     is_start=is_start, prob_uid=prob_uid, eval_mode=True)


    # Find the difference of the sets 
    set_agent_sais = set(agent_sais)
    set_profile_sais = set(profile_sais)
    missing = set_profile_sais - set_agent_sais
    extra = set_agent_sais - set_profile_sais
    correct = set_agent_sais.intersection(set_profile_sais)
    n_diff = len(missing) + len(extra)
    first_correct = len(agent_sais) > 0 and agent_sais[0] in set_profile_sais; 
    diff_stats = {"-": list(missing), "+": list(extra), "=" : list(correct), "first_correct": first_correct}
    return diff_stats

def _print_diffs(state_stats, print_correct=False):
    for prob_stats in state_stats:
        n_diffs = len(prob_stats['-']) + len(prob_stats['+'])
        
        if(n_diffs != 0):
            print(f"--DIFF: {prob_stats['problem']} {prob_stats['hist']} --")
            for m in prob_stats['-']:
                print("  -", m['selection'], m['inputs']['value'])
            for m in prob_stats['+']:
                print("  +", m['selection'], m['inputs']['value'])
        if(print_correct == True or 
           print_correct=="when_diff" and n_diffs > 0):
            for m in prob_stats['=']:
                print("  =", m['selection'], m['inputs']['value'])   

# ----------------------------------------------
#  Evaluate Certainty 

def _eval_cert(agent, state, item, profile_sais, skill_app_map, when_preds, certainties):
    # Run act_all so that it gives skill apps including negative ones
    is_start = len(item['hist'])==0
    prob_uid = state.get("__uid__")
    agent_skill_apps = agent.act_all(state, return_kind='skill_app',
        is_start=is_start, prob_uid=prob_uid,
        add_out_of_process=True,
        eval_mode=False,
        ignore_filter=True
    )

    # Ensure that skill_app_map is up to date    
    set_profile_sais = set(profile_sais)
    for sa in agent_skill_apps:
        if(sa not in skill_app_map):
            # Insert (index, correct) into skill_app_map
            skill_app_map[sa] = (len(skill_app_map), sa.sai in set_profile_sais)
            when_preds.append(0)
            certainties.append(0)

        # Insert when_pred and certainty
        index, correct = skill_app_map[sa]

        when_preds[index] = sa.when_pred
        if(hasattr(sa, 'certainty')):
            certainties[index] = sa.certainty

# ----------------------------------------------
#  Evaluate Performance Totals

def _eval_totals(state_stats):
    n_total_correct, total = 0, 0
    n_first_correct, total_states = 0, 0

    completeness = 0
    step_score = 0
    correctness = 0
    omission_rate = 0
    comission_rate = 0
    omission_score = 0
    comission_score = 0

    for prob_stats in state_stats:
        n_missing, n_extra, n_corr = len(prob_stats['-']),len(prob_stats['+']),len(prob_stats['='])
        # NOTE: max() calls here prevent divide-by-zero if for some reason there 
        #  are no correct next actions (which really shouldn't ever be the case)
        step_score += n_corr / max(n_missing+n_extra+n_corr, 1)
        completeness += (n_missing == 0 and n_extra == 0)
        correctness += prob_stats['first_correct']
        omission_rate += n_missing > 0;
        comission_rate += n_extra > 0;
        omission_score += n_missing / max(n_missing+n_corr, 1)
        comission_score += n_extra / max(n_extra+n_corr, 1)

    step_score /= len(state_stats)    
    completeness /= len(state_stats)    
    correctness /= len(state_stats)
    omission_rate /= len(state_stats)
    comission_rate /= len(state_stats)
    omission_score /= len(state_stats)
    comission_score /=  len(state_stats)
    return {"step_score" : step_score, "completeness" : completeness, "correctness" : correctness,
            "omission_rate" : omission_rate, "comission_rate" : comission_rate,
            "omission_score" : omission_score, "comission_score" : comission_score,
            }

def _print_totals(total_stats, to_print=[]):
    if(not isinstance(to_print, list)):
        to_print = list(total_stats.keys())
    for meausre_name in to_print:
        value = total_stats[meausre_name];
        title = " ".join([x.capitalize() for x in meausre_name.split("_")])
        print(f"{title} : {value*100:.2f}%")


def eval_holdout_stats(agent, profile, 
                       skill_app_map={},
                       return_state_stats=False,
                       return_cert_stats=False,
                       print_diffs=False, 
                       print_correct=False,
                       print_totals=True,
                     **kwargs):
    ''' Evaluates an agent's correctness and completeness against a holdout 
        completeness profile consisting of states and correct actions at those states.
    '''
    import json, os
    print("START EVAL COMPLETENESS")

    state_stats = []
    when_preds = [0]*len(skill_app_map)
    certainties = [0]*len(skill_app_map)
    with open(profile, 'r') as profile_f:
        for line_ind, line in enumerate(profile_f):
            # Read a line from the profile
            state, item, profile_sais = _load_profile_line(agent, line)
            prob_stats = {"problem": item['problem'], 'hist' : item['hist']}

            # Calc stats of how agent's proposed actions on 'state' differ from ground-truth  
            diff_stats = _eval_diffs(agent, state, item, profile_sais)
            prob_stats.update(diff_stats)
            state_stats.append(prob_stats)

            if(return_cert_stats):
                # Update skill_app_map, when_preds, certainties
                _eval_cert(agent, state, item, profile_sais, skill_app_map, when_preds, certainties)
    
    # print("WWWW", when_preds)
    total_stats = _eval_totals(state_stats)

    if(print_diffs):
        _print_diffs(state_stats, print_correct)

    if(print_totals):
        _print_totals(total_stats, print_totals)

    out = {**total_stats}

    # These are optional returns since they are very large and can
    #  eat up lots of memory in a long-running program
    if(return_state_stats):
        out['state_stats'] = state_stats
    if(return_cert_stats):
        out['when_preds'] = when_preds
        out['certainties'] = certainties

    return out

# def collate_stats(stats_list):
#     return {k: np.array([dic[k] for dic in LD],dtype=np.float64) for k in LD[0]}

# -----------------------------------------------------------
# Certainty Stats for Thrashing, TPR etc. 
def eval_total_cert_stats(skill_app_map, holdout_certs, cert_threshs=[.9,1.0]):
    corrs = np.array([rew > 0 for ind, rew in skill_app_map.values()], dtype=np.bool_)
    incorrs = ~corrs
    L = len(holdout_certs)
    
    # Vacillation Rate : Proportion of errors that reoccur
    FP_reocc    = np.zeros(L-1, dtype=np.float64)
    FN_reocc    = np.zeros(L-1, dtype=np.float64)
    error_reocc = np.zeros(L-1, dtype=np.float64)
    total_FP_reocc, total_pTPs = 0, 0 
    total_FN_reocc, total_pTNs = 0, 0 
    total_error_reocc, total_pTs = 0, 0 

    # Productive Monotonicity: Proportion of certainty changes
    #  that correctly move toward 1.0 if correct or -1.0 if incorrect. 
    prod_monot  = np.zeros(L-1, dtype=np.float64)
    total_prod_d, total_d  = 0, 0
    
    # Certainty Preds @ thresh 
    precisions_at_thresh  = [np.zeros(L, dtype=np.float64) for _ in range(len(cert_threshs))]
    total_TP_at_thresh = [0] * len(cert_threshs)
    total_PP_at_thresh = [0] * len(cert_threshs)

    for i, certs in enumerate(holdout_certs):
        # Fill in missing certs with 0 predictions
        certs = np.pad(certs, (0, len(skill_app_map)-len(certs)), constant_values=(0, 0))
        TPs = certs[corrs] > 0.0
        TNs = certs[incorrs] < 0.0

        # Calc delta based measures
        if(i != 0):
            k = i-1

            # -- Error Re-occurance ---
            FP_reoccs = ~TPs[prev_TPs]
            FN_reoccs = ~TNs[prev_TNs]
            n_FP_reoccs, n_pTPs = np.count_nonzero(FP_reoccs), len(prev_TPs)
            n_FN_reoccs, n_pTNs = np.count_nonzero(FN_reoccs), len(prev_TNs)
            n_reoccs, n_pTs = n_FP_reoccs + n_FN_reoccs, n_pTPs + n_pTNs

            FP_reocc[k] = (n_FP_reoccs / n_pTPs) if n_pTPs > 0 else 0.0
            FN_reocc[k] = (n_FN_reoccs / n_pTNs) if n_pTNs > 0 else 0.0
            error_reocc[k] = (n_reoccs / n_pTs) if n_pTs > 0 else 0.0

            total_FP_reocc += n_FP_reoccs
            total_pTPs += n_pTPs
            total_FN_reocc += n_FN_reoccs
            total_pTNs += n_pTNs
            total_error_reocc += n_reoccs
            total_pTs += n_pTs

            # --- Productive Monotonicity ---             
            d_certs = certs-prev_certs
            prod_pos = (d_certs > 0) & corrs
            prod_neg = (d_certs < 0) & incorrs
            prods = (prod_pos | prod_neg)[d_certs != 0]
            n_prod_d, n_d = np.count_nonzero(prods), len(prods)

            prod_monot[k] = (n_prod_d / n_d) if n_d > 0 else 1.0
            total_prod_d += n_prod_d
            total_d += n_d

        # Update Precision
        for t, thresh in enumerate(cert_threshs):
            pred = certs >= thresh;

            # True positives over predicted positives
            TP = np.count_nonzero(pred & corrs)
            PP = np.count_nonzero(pred)
            total_TP_at_thresh[t] += TP
            total_PP_at_thresh[t] += PP
            precisions_at_thresh[t][i] = (TP / PP) if PP > 0 else 1.0

        prev_certs = certs
        prev_TPs = TPs
        prev_TNs = TNs

    out = {}
    for t, thresh in enumerate(cert_threshs):
        total_TP = total_TP_at_thresh[t]
        total_PP = total_PP_at_thresh[t]

        out[("total_precision", thresh)] = (total_TP / total_PP) if total_PP > 0 else 1.0
        out[("precision", thresh)] = precisions_at_thresh[t]

    out.update({
        "FP_reocc" : FP_reocc,
        "FN_reocc" : FN_reocc,
        "error_reocc" : error_reocc,
        "total_FP_reocc" : total_FP_reocc / total_pTPs if total_pTPs > 0 else 1.0,
        "total_FN_reocc" : total_FN_reocc / total_pTNs if total_pTPs > 0 else 1.0,
        "total_error_reocc" : total_error_reocc / total_pTs if total_pTs > 0 else 1.0,

        "prod_monot" : prod_monot,
        "total_prod_monot" : (total_prod_d / total_d) if total_d > 0 else 1.0,
    })

    return out

# -------------
# Extra Utilities

def avg_stats(stats_list, ignore=['certainties', 'when_preds']):
    keys = [k for k in stats_list[0].keys() if k not in ignore]
    accum_d = {}
    for k in keys:
        accum_d[k] = [stats_list[0][k]]
        
    for i in range(1,len(stats_list)):
        stats_d = stats_list[i]
        for k in keys:
            accum_d[k].append(stats_d[k]) 
    
    tot_stat_d = {}
    for k,v in accum_d.items():
        print(k, v)
        tot_stat_d[k] = {"avg" : np.mean(v,axis=0),
                         "std" : np.std(v,axis=0),
                         "N"  : len(v)}
    return tot_stat_d
