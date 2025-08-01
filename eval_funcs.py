import json
# from apprentice.shared import SAI
import numpy as np
from tutorgym.evaluator import ProfileIterator
from tutorgym.shared import Action

# def _load_profile_line(agent, line):
#     item = json.loads(line)
#     # print(item['sais'])
#     profile_actions = [SAI(x) for x in item['correct_actions']]
#     state = item['state']
#     state = agent.standardize_state(state)
        
#     return state, item, profile_sais


prob_uid_dict = {}
def resolve_prob_uid(agent, state, item):
    is_start = len(item['action_hist'])==0

    # TODO: this is a bit of a hack
    prob_uid = None
    if(is_start and hasattr(agent, "standardize_state")):
        state = agent.standardize_state(state)
        prob_uid = state.get("__uid__")
        prob_uid_dict[item['problem']] = prob_uid
    else:
        prob_uid = prob_uid_dict.get(item['problem'], None)

    return is_start, prob_uid


# ----------------------------------------------
#  Evaluate Step Level Diffs (Agent Predictions vs Ground Truth)
def _eval_diffs(agent, state, item, profile_actions, check_annotations=[]):   
    # print("N profile actions", len(profile_actions))
    
    is_start, prob_uid = resolve_prob_uid(agent, state, item)

    agent_actions = agent.act_all(state, return_kind='skill_app',
     is_start=is_start, prob_uid=prob_uid, eval_mode=True)

    # Copy and filter each set of actions to only include annotations
    #  included in `check_annotations`
    conv_profile_actions = []#[Action(p_act) for p_act in profile_actions]
    for p_act in profile_actions:
        p_act = Action(p_act)
        p_act = Action(p_act.as_tuple(), **{k:v for k,v in p_act.annotations.items() 
                                        if k in check_annotations})
        conv_profile_actions.append(p_act)

    conv_agent_actions = []#[Action(a_act) for a_act in agent_actions]
    for a_act in agent_actions:
        a_act = Action(a_act)
        a_act = Action(a_act.as_tuple(), **{k:v for k,v in a_act.annotations.items() 
                                        if k in check_annotations})
        conv_agent_actions.append(a_act)

    set_agent_actions = set(conv_agent_actions)
    set_profile_actions = set(conv_profile_actions)
    missing = set_profile_actions - set_agent_actions
    incorrect = set_agent_actions - set_profile_actions
    correct = set_agent_actions.intersection(set_profile_actions)
    first_correct = len(agent_actions) > 0 and agent_actions[0] in set_profile_actions; 
    n_diff = len(missing) + len(incorrect)

    step_score = max(0.0, len(set_profile_actions)-n_diff) / len(set_profile_actions)

    diffs = {"problem": item['problem'], 'action_hist' : item['action_hist'],
             "-": list(missing), "+": list(incorrect), "=" : list(correct),
             "first_correct" : first_correct,
             "step_score" : step_score
            }
    return diffs

def _print_diffs(state_stats, print_correct=False):
    for prob_stats in state_stats:
        n_diffs = len(prob_stats['-']) + len(prob_stats['+'])
        
        if(n_diffs != 0):
            print(f"--DIFF: {prob_stats['problem']} {prob_stats['action_hist']} --")
            for m in prob_stats['-']:
                print(m)
                print("  -", m)
            for m in prob_stats['+']:
                print("  +", m)
        if(print_correct == True or 
           print_correct=="when_diff" and n_diffs > 0):
            for m in prob_stats['=']:
                print("  =", m)   

# ----------------------------------------------
#  Evaluate Certainty 

def _eval_cert(agent, state, item, profile_actions, skill_app_map, when_preds, certainties):
    # Run act_all so that it gives skill apps including negative ones

    is_start, prob_uid = resolve_prob_uid(agent, state, item)
    # is_start = len(item['action_hist'])==0
    # prob_uid = state.get("__uid__") if is_start else None
    agent_skill_apps = agent.act_all(state, return_kind='skill_app',
        is_start=is_start, prob_uid=prob_uid,
        add_out_of_process=True,
        eval_mode=False,
        ignore_filter=True
    )

    # Ensure that skill_app_map is up to date    
    set_profile_sais = set([a.as_tuple() for a in  profile_actions])
    for sa in agent_skill_apps:
        if(sa not in skill_app_map):
            # Insert (index, correct) into skill_app_map
            # print(sa.as_tuple() in set_profile_sais)
            skill_app_map[sa] = (len(skill_app_map), sa.as_tuple() in set_profile_sais)
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
    # print("N LINES:", len(state_stats))
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
                       print_diffs=True, 
                       print_correct=False,
                       print_totals=True,
                     **kwargs):
    ''' Evaluates an agent's correctness and completeness against a holdout 
        completeness profile consisting of states and correct actions at those states.
    '''
    import json, os
    print("START EVAL COMPLETENESS")

    # out = agent.eval_completeness(profile, print_diffs=False)
    # print("COMPL:", out['completeness'])
    # print("------------")
    profile_iter = ProfileIterator(profile)

    state_stats = []
    when_preds = [0]*len(skill_app_map)
    certainties = [0]*len(skill_app_map)
    # with open(profile, 'r') as profile_f:
    # for line_ind, line in enumerate(profile_f):
    for state, item, profile_actions in profile_iter:
        # Read a line from the profile
        # state, item, profile_sais = _load_profile_line(agent, line)
        prob_stats = {"problem": item['problem'], 'action_hist' : item['action_hist']}

        # Calc stats of how agent's proposed actions on 'state' differ from ground-truth  
        diff_stats = _eval_diffs(agent, state, item, profile_actions)
        # print(diff_stats)
        prob_stats.update(diff_stats)
        state_stats.append(prob_stats)

        # raise ValueError()

        if(return_cert_stats):
            # Update skill_app_map, when_preds, certainties
            _eval_cert(agent, state, item, profile_actions, skill_app_map, when_preds, certainties)

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
DEFAULT_CERT_BINS = [( .5+(.02*(i)), .5+(.02*(i+1)) ) for i in range(25)]+[(1.0,1.0)]
def eval_total_cert_stats(corrs, holdout_certs, cert_bins=DEFAULT_CERT_BINS, diff_thresh=0.02):
    # corrs = np.array([rew > 0 for ind, rew in skill_app_map.values()], dtype=np.bool_)
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
    w_prod_monot  = np.zeros(L-1, dtype=np.float64)
    total_prod_d, total_w_prod_d, total_d, total_abs_d_cert  = 0, 0.0, 0, 0.0
    
    # Certainty Preds @ thresh 
    precisions_in_bin  = [np.zeros(L, dtype=np.float64) for _ in range(len(cert_bins))]
    total_TP_at_thresh = [0] * len(cert_bins)
    total_PP_at_thresh = [0] * len(cert_bins)

    for i, certs in enumerate(holdout_certs):
        # Fill in missing certs with 0 predictions
        certs = np.pad(certs, (0, len(corrs)-len(certs)), constant_values=(0, 0))
        # print(certs)
        TPs = certs[corrs] > 0.0
        TNs = certs[incorrs] < 0.0

        # Calc delta based measures
        if(i != 0):
            k = i-1

            # -- Error Re-occurance ---
            FP_reoccs = ~TNs[prev_TNs] # Neg that were ok then pos
            FN_reoccs = ~TPs[prev_TPs] # Pos that were ok then neg
            n_FP_reoccs, n_pTNs = np.count_nonzero(FP_reoccs), len(prev_TNs)
            n_FN_reoccs, n_pTPs = np.count_nonzero(FN_reoccs), len(prev_TPs)
            n_reoccs, n_pTs = n_FP_reoccs + n_FN_reoccs, n_pTPs + n_pTNs

            FP_reocc[k] = (n_FP_reoccs / n_pTNs) if n_pTNs > 0 else 0.0
            FN_reocc[k] = (n_FN_reoccs / n_pTPs) if n_pTPs > 0 else 0.0
            error_reocc[k] = (n_reoccs / n_pTs) if n_pTs > 0 else 0.0

            total_FP_reocc += n_FP_reoccs
            total_pTPs += n_pTPs
            total_FN_reocc += n_FN_reoccs
            total_pTNs += n_pTNs
            total_error_reocc += n_reoccs
            total_pTs += n_pTs

            # --- Productive Monotonicity ---             
            d_certs = certs-prev_certs

            # print("::", certs.shape, prev_certs.shape)

            # print(certs)
            prod_pos = (d_certs > 0) & corrs
            prod_neg = (d_certs < 0) & incorrs
            prod_mask = (prod_pos | prod_neg)
            prods = prod_mask[np.abs(d_certs) > diff_thresh]
            n_prod_d, n_d = np.count_nonzero(prods), len(prods)            
            prod_monot[k] = (n_prod_d / n_d) if n_d > 0 else np.nan


            abs_d_certs = np.abs(d_certs)
            sum_abs_d_certs = np.sum(abs_d_certs)
            w_prod_d = (np.sum(prod_mask*abs_d_certs) - np.sum((~prod_mask)*abs_d_certs))
            w_prod_monot[k] = w_prod_d / sum_abs_d_certs if sum_abs_d_certs != 0.0 else np.nan
            total_w_prod_d += w_prod_d #/// len(d_certs)
            total_abs_d_cert += sum_abs_d_certs
            # print("::", w_prod_d, sum_abs_d_certs)


            # print(prod_monot[k], np.mean(np.abs((prod_pos | prod_neg)*d_certs))-np.mean(np.abs((~(prod_pos | prod_neg))*d_certs)), n_d)
            total_prod_d += n_prod_d
            total_d += n_d

        # Update Precision
        for t, c_bin in enumerate(cert_bins):
            c_min, c_max = c_bin
            if(c_min != c_max):
                pred = (certs >= c_min) & (certs < c_max);
            else:
                pred = (certs == c_min)

            # print("c_min, c_max", c_min, c_max)
            # print("pred", np.sum(pred))
            # print("corrs", np.sum(corrs))
            # print("certs")
            # print(certs)
            

            # True positives over predicted positives
            TP = np.count_nonzero(pred & corrs)
            PP = np.count_nonzero(pred)
            total_TP_at_thresh[t] += TP
            total_PP_at_thresh[t] += PP
            precisions_in_bin[t][i] = (TP / PP) if PP > 0 else 1.0

        prev_certs = certs
        prev_TPs = TPs
        prev_TNs = TNs

    # print("certs", certs)
    out = {}
    for t, c_bin in enumerate(cert_bins):
        total_TP = total_TP_at_thresh[t]
        total_PP = total_PP_at_thresh[t]

        out[("total_precision", c_bin)] = (total_TP / total_PP) if total_PP > 0 else 1.0
        out[("precision", c_bin)] = precisions_in_bin[t]
        out[("bin_n", c_bin)] = total_PP
        out[("TP_n", c_bin)] = total_TP
        # print(":", c_bin, total_TP, total_PP, out[("total_precision", c_bin)])

    out.update({
        "FP_reocc" : FP_reocc,
        "FN_reocc" : FN_reocc,
        "error_reocc" : error_reocc,
        "total_FP_reocc" : total_FP_reocc / total_pTNs if total_pTPs > 0 else 1.0,
        "total_FN_reocc" : total_FN_reocc / total_pTPs if total_pTPs > 0 else 1.0,
        "total_error_reocc" : total_error_reocc / total_pTs if total_pTs > 0 else 1.0,
        "prod_monot" : prod_monot,
        "w_prod_monot" : w_prod_monot,
        "total_prod_monot" : (total_prod_d / total_d) if total_d > 0 else 1.0,
        "total_w_prod_monot" : (total_w_prod_d / total_abs_d_cert) if total_d > 0 else 1.0,
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
        # print(k, v)
        tot_stat_d[k] = {"avg" : np.nanmean(v,axis=0),
                         "std" : np.nanstd(v,axis=0),
                         "N"  : len(v)}
    return tot_stat_d
