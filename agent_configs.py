from apprentice.agents.cre_agents.cre_agent import CREAgent

common = {
    "where_learner": "mostspecific",
    "planner" : "setchaining",
    "explanation_choice" : "least_operations",
    # "find_neighbors" : True,
    # "error_on_bottom_out" : False,
    # "when_args": {
    #     "encode_relative" : True,
    #     "check_sanity" : False,
    #     "one_hot" : True
    # },
    # "when_learner": "stand",
    "which_learner": "when_prediction",
    "action_chooser" : "max_which_utility",
    "suggest_uncert_neg" : True,

    # "when_learner" : 'sklearndecisiontree',
    
    "error_on_bottom_out" : False,
    "one_skill_per_match" : True,
    
    "extra_features" : ["Match"],
    # "when_args" : {"encode_relative" : True, },
    
    "should_find_neighbors" : True,

    "when_args": {
        "encode_relative" : False,
        # "one_hot" : True,
        # "rel_enc_min_sources": 1,
        "check_sanity" : False
    },

    "process_learner": "htnlearner",
    "track_rollout_preseqs" : True,
    "action_filter_args" : {"thresholds": [0.3, 0, -0.5, -0.75]}
}

mc_basic = {
    "search_depth" : 2,
    "function_set" : ["OnesDigit","TensDigit","Add","Add3"],
    "extra_features" : ["SkillCandidates","Match"],
}

mc_proc_lrn = {
    "search_depth" : 2,
    "extra_features" : ["RemoveAll","SkillCandidates","Match"],
    "when_args": {
        "encode_relative" : True, 
        "check_sanity" : False
    },
    # "implicit_reward_kinds" : ["unordered_groups"]
    "implicit_reward_kinds" : None,
}

frac_basic = {
    "search_depth" : 1,
    "function_set": ['Multiply', 'Add'],
    "feature_set": ['Equals'],
    "one_skill_per_match" : True,
    "extra_features" : ["Match"],

    # Not sure why but in version ran with first participants
    #  in Feb. used encode_relative=False
     "when_args": {
        "encode_relative" : False,
        "check_sanity" : False,
        "one_hot" : True
    },
}

# frac_basic = {
#     "function_set": ['Multiply', 'Add'],
#     "feature_set": ['Equals'],
# }

proc_lrn = {
    "process_learner": "htnlearner",
    "track_rollout_preseqs" : True,
    "action_filter_args" : {"thresholds": [.3, 0, -.5, -.75]},
    "implicit_reward_kinds" : ["unordered_groups"],
}

DT = {
    "when_learner": "decision_tree",
}

STAND = {
    "when_learner": "stand",
    "which_learner": "when_prediction",
    "action_chooser" : "max_which_utility",
    "suggest_uncert_neg" : True,
    "when_args" : {
        **common['when_args'],
        "split_choice" : "dyn_all_near_max",
        "slip" : 0.1,
        "w_path_slip" : True,
    }
}

STAND_HS = {
    "when_learner": "stand",
    "which_learner": "when_prediction",
    "action_chooser" : "max_which_utility",
    "suggest_uncert_neg" : True,
    "when_args" : {
        **common['when_args'],
        "lam_p" : 25.0,
        "lam_e" : 25.0,
        "lam_l" : 50.0,
        "split_choice" : "dyn_all_near_max",
        "slip" : 0.1,
        "w_path_slip" : True,
    }
}

STAND_Relaxed ={
    "when_learner": "stand",
    "which_learner": "when_prediction",
    "action_chooser" : "max_which_utility",
    "suggest_uncert_neg" : True,
    "when_args" : {
        **common['when_args'],
        "split_choice" : "all_near_max"
    }
}

RF = {
    "when_learner": "random_forest",   
    "which_learner": "when_prediction",
    "action_chooser" : "max_which_utility",
    "suggest_uncert_neg" : True,
}

XGB = {
    "when_learner": "xg_boost",   
    "which_learner": "when_prediction",
    "action_chooser" : "max_which_utility",
    "suggest_uncert_neg" : True,
    "when_args" : {
        **common['when_args'],
        "one_hot" : True
    }
}

agent_configs = {
    # MC
    ("mc", "decision_tree", True) : {
        **common,
        **mc_basic,
        **mc_proc_lrn,
        **DT,
    },
    ("mc", "random_forest", True) : {
        **common,
        **mc_basic,
        **mc_proc_lrn,
        **RF,
    },
    ("mc", "xg_boost", True) : {
        **common,
        **mc_basic,
        **mc_proc_lrn,
        **XGB,
    },
    ("mc", "stand", True) : {
        **common,
        **mc_basic,
        **mc_proc_lrn,
        **STAND,  
    },
    ("mc", "stand_hs", True) : {
        **common,
        **mc_basic,
        **mc_proc_lrn,
        **STAND_HS,  
    },
    ("mc", "stand-relaxed", False) : {
        **common,
        **mc_basic,
        **STAND_Relaxed,  
    },
    ("mc", "stand", True) : {
        **common,
        **mc_basic,
        **mc_proc_lrn,
        **STAND,  
        **proc_lrn,

        # Turn off unordered groups... causing BAD ACTION error
        # "implicit_reward_kinds" : None,
        # Slim down features
        # "extra_features" : ["RemoveAll", "SkillCandidates","Match"],

    },

    ("mc", "stand_hs", True) : {
        **common,
        **mc_basic,
        **mc_proc_lrn,
        **STAND,  
        **proc_lrn,

        # Turn off unordered groups... causing BAD ACTION error
        # "implicit_reward_kinds" : None,
        # Slim down features
        # "extra_features" : ["RemoveAll", "SkillCandidates","Match"],

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
    ("frac", "xg_boost", False) : {
        **common,
        **frac_basic,
        **XGB,
    },
    ("frac", "stand", False) : {
        **common,
        **frac_basic,
        **STAND,  
    },
    ("frac", "stand_hs", False) : {
        **common,
        **frac_basic,
        **STAND_HS,  
    },
    ("frac", "stand-relaxed", False) : {
        **common,
        **frac_basic,
        **STAND_Relaxed,  
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
