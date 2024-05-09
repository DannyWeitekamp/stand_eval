def _load_profile_line(agent, line):
    item = json.loads(line)
    profile_sais = [SAI(x) for x in item['sais']]
    state = item['state']
    if(is_start):
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

def _print_diffs(state_stats):
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
        when_preds[index] = sa.certainty
        certainties[index] = sa.when_pred

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
    return {"step_score" : step_score, "completeness" : completeness, "correctness" : correctness
            "ommision_rate" : omission_rate, "commision_rate" : comission_rate,
            "ommision_score" : omission_score, "commision_score" : comission_score,
            }

def _print_totals(total_stats, to_print=[]):
    if(not isinstance(to_print, list)):
        to_print = list(total_stats.keys())
    for meausre_name in to_print:
        value = total_stats[value];
        title = " ".join([x.capitalize() for x in meausre_name.split("_")])
        print(f"{cap} : {value*100:.2f}")


def eval_holdout_stats(agent, profile, 
                       skill_app_map={}, skill_app_records=[],
                       return_state_stats=False,
                       return_cert_stats=False,
                       print_diffs=True, 
                       print_totals=True,
                       print_correct=False, 
                     **kwargs):
    ''' Evaluates an agent's correctness and completeness against a holdout 
        completeness profile consisting of states and correct actions at those states.
    '''
    import json, os
    print("START EVAL COMPLETENESS")

    state_stats, when_preds, certainties = [], [], []
    with open(profile, 'r') as profile_f:
        for line_ind, line in enumerate(profile_f):
            # Read a line from the profile
            state, item, profile_sais = agent._load_profile_line(line)
            prob_stats = {"problem": item['problem'], 'hist' : item['hist']}

            # Calc stats of how agent's proposed actions on 'state' differ from ground-truth  
            diff_stats = _eval_diffs(agent, state, item, profile_sais)
            prob_stats.update(diff_stats)
            state_stats.append(prob_stats)

            if(return_cert_stats):
                # Update skill_app_map, when_preds, certainties
                _eval_cert(agent, state, item, profile_sais, skill_app_map, when_preds, certainties)
    
    total_stats = _eval_totals(state_stats)

    if(print_diffs):
        _print_diffs(state_stats)

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



# -----------------------------------------------------------
# Certainty Stats for Thrashing, TPR etc. 

def eval_total_cert_stats(skill_app_map, holdout_certs, holdout_when_preds, cert_threshs=[.9,1.0]):
    corrs = np.array([rew > 0 for ind, rew in coskill_app_map.values()], dtype=np.float64)
    incorrs = ~corrs
    L = len(holdout_certs)

    # Thrashing : error re-occurance rates
    cert_FP_reocc    = np.zeros(L-1,dtype=np.float64)
    cert_FN_reocc    = np.zeros(L-1,dtype=np.float64)
    cert_error_reocc = np.zeros(L-1,dtype=np.float64)

    wp_FP_reocc    = np.zeros(L-1,dtype=np.float64)
    wp_FN_reocc    = np.zeros(L-1,dtype=np.float64)
    wp_error_reocc = np.zeros(L-1,dtype=np.float64)
    

    # Productive Monotonicity: Proportion of certainty changes
    #  that correctly move toward 1.0 if correct or -1.0 if incorrect. 
    cert_prod_monot  = np.zeros(L-1, dtype=np.bool)
    wp_prod_monot  = np.zeros(L-1, dtype=np.bool)
    total_d_certs, total_prod_d_certs = 0, 0
    total_d_wps, total_prod_d_wps = 0, 0
    
    # Certainty Preds @ thresh 
    cert_preds_at_thresh  = [np.zeros(L, dtype=np.float64) for _ in range(cert_threshs)]  

    for i, (certs, wps) in enumerate(zip(holdout_certs, holdout_when_preds)):
        cert_TPs = certs[corrs] > 0
        cert_TNs = certs[incorrs] < 0

        wp_TPs = wps[corrs] > 0
        wp_TNs = wps[incorrs] < 0

        # Update delta based measures
        if(i != 0):
            k = i-1

            # -- Error Re-occurance ---
            #   certainty
            cert_FP_reoccs = ~cert_TPs[prev_cert_TPs]
            cert_FN_reoccs = ~cert_TNs[prev_cert_TNs] 
            n_cert_FP_reoccs, n_cert_FPs = np.count_nonzero(cert_FP_reoccs), len(cert_FP_reoccs)
            n_cert_FN_reoccs, n_cert_FNs = np.count_nonzero(cert_FN_reoccs), len(cert_FN_reoccs)

            #   when_pred
            wp_FP_reoccs = ~wp_TPs[prev_wp_TPs]
            wp_FN_reoccs = ~wp_TNs[prev_wp_TNs]
            n_wp_FP_reoccs, n_wp_FPs = np.count_nonzero(wp_FP_reoccs), len(wp_FP_reoccs)
            n_wp_FN_reoccs, n_wp_FNs = np.count_nonzero(wp_FN_reoccs), len(wp_FN_reoccs)

            # --- Productive Monotonicity --- 
            
            #   certainty
            d_certs = certs-prev_certs
            prod_pos_certs = (d_certs > 0) & corrs
            prod_neg_certs = (d_certs < 0) & incorrs
            prod_certs = (prod_pos_certs | prod_neg_certs)[d_certs != 0]
            n_prod_d_certs, n_d_certs = np.count_nonzero(prod_certs), len(prod_certs)
            cert_prod_monot[k] = (n_prod_d_certs / n_d_certs) if n_d_certs > 0 else 1.0
            total_prod_d_certs += n_prod_d_certs
            total_d_certs += n_d_certs

            #   when_pred
            d_wps = wps-prev_wps
            prod_pos_wps = (d_wps > 0) & corrs
            prod_neg_wps = (d_wps < 0) & incorrs
            prod_wps = (prod_pos_wps | prod_neg_wps)[d_wps != 0]
            n_prod_d_wps, n_d_wps = np.count_nonzero(prod_wps), len(prod_wps)
            wp_prod_monot[k] = (n_prod_d_wps / n_d_wps) if n_d_wps > 0 else 1.0
            total_prod_d_wps += n_prod_d_wps
            total_d_wps += n_d_wps

        prev_certs, prev_wps = certs, wps
        prev_cert_TPs, prev_wp_TPs = cert_TPs, wp_TPs
        prev_wp_TNs, prev_wp_TNs = cert_TNs, wp_TNs

    corr = np.array(len(skill_app_map), dtype=np.float64)
    corr = np.array(len(skill_app_map), dtype=np.float64)


