from apprentice.agents.cre_agents.cre_agent import CREAgent

common = {
    "where_learner": "mostspecific",
    "planner" : "setchaining",
    "explanation_choice" : "least_operations",
    "find_neighbors" : True,
    "when_args": {
        "encode_relative" : True,
        "check_sanity" : False
    },
}

mc_basic = {
    "search_depth" : 2,
    "function_set" : ["OnesDigit","TensDigit","Add","Add3"],
    "extra_features" : ["SkillCandidates","Match"],
}

mc_proc_lrn = {
    "extra_features" : ["RemoveAll","SkillCandidates","Match"],
}

frac_basic = {
    "function_set": ['Multiply', 'Add'],
    "feature_set": ['Equals'],
    "one_skill_per_match" : True,
    "extra_features" : ["Match"],
}

frac_basic = {
    "function_set": ['Multiply', 'Add'],
    "feature_set": ['Equals'],
}

proc_lrn = {
    "process_learner": "htnlearner",
    "track_rollout_preseqs" : True,
    "action_filter_args" : {"thresholds": [.3, 0, -.5, -.75]},
    "implicit_reward_kinds" : ["unordered_groups"]
}

STAND = {
    "when_learner": "stand",
    "which_learner": "when_prediction",
    "action_chooser" : "max_which_utility",
    "suggest_uncert_neg" : True,
}

DT = {
    "when_learner": "decision_tree",
}

RF = {
    "when_learner": "random_forest",   
}

agent_configs = {
    # MC
    ("mc", "decision_tree", False) : {
        **common,
        **mc_basic,
        **DT,
    },
    ("mc", "random_forest", False) : {
        **common,
        **mc_basic,
        **RF,
    },
    ("mc", "stand", False) : {
        **common,
        **mc_basic,
        **STAND,  
    },
    ("mc", "stand", True) : {
        **common,
        **mc_basic,
        **mc_proc_lrn,
        **STAND,  
        **proc_lrn
    },

    # Frac
    ("frac", "decision_tree", False) : {
        **common,
        **frac_basic,
        **DT,
    },
    ("frac", "random_forest", False) : {
        **common,
        **frac_basic,
        **RF,
    },
    ("frac", "stand", False) : {
        **common,
        **frac_basic,
        **STAND,  
    },
    ("frac", "stand", True) : {
        **common,
        **frac_basic,
        **STAND,  
        **proc_lrn
    }
}

def make_agent(domain, when, use_proc_lrn):
    config = agent_configs[(domain, when, use_proc_lrn)]
    print(config)
    return CREAgent(**config)
    



agent_args = {
                "search_depth" : 2,
                # "where_learner": "generalize",
                "where_learner": "mostspecific",

                # "when_learner": "sklearndecisiontree",
                # "when_learner": "decisiontree",
                                
                # For STAND
                "when_learner": "stand",
                "which_learner": "when_prediction",
                "action_chooser" : "max_which_utility",
                "suggest_uncert_neg" : True,

                # "when_args" : {},
                # "when_args" : {},

                # "explanation_choice" : "least_operations",
                "planner" : "setchaining",
                # // "when_args" : {"cross_rhs_inference" : "implicit_negatives"},
                "function_set" : ["OnesDigit","TensDigit","Add","Add3"],
                # "feature_set" : [],
                # "feature_set" : ['Equals'],
                "extra_features" : ["RemoveAll", "SkillCandidates","Match"],
                # "extra_features" : ["SkillValue"],#,"Match"],
                "find_neighbors" : True,
                # "strip_attrs" : ["to_left","to_right","above","below","type","id","offsetParent","dom_class"],
                # "state_variablization" : "metaskill",
                "when_args": {
                    "encode_relative" : True,
                    # "rel_enc_min_sources": 1,
                    "check_sanity" : False
                },

                "process_learner": "htnlearner",
                "track_rollout_preseqs" : True,
                "action_filter_args" : {"thresholds": [.3, 0, -.5, -.75]},

                "implicit_reward_kinds" : ["unordered_groups"]
            }
