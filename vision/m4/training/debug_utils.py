# [MODULE DOC] Debug helpers meant to be called with a Trainer instance as `self` (they read self.vl_optim) to
# inspect
# Adam-style optimizer state.
""" Trainer debug utils """


def dump_optim_states(self):
    # [FUNCTION DOC] Print, for the first N entries of the optimizer state (N = number of param groups), the
    # step count and
    # whether exp_avg and exp_avg_sq are all zeros.
    # KNOWN ISSUE: optimizer state is keyed per parameter, not per param group, so slicing it to the number of
    # param groups inspects the first few parameters only, mislabelled as "param group"; also the builtin
    # all() over
    # a tensor only works elementwise for 1-D tensors.
    """dumps basic information about the state of the optimizer"""

    print("*** Optim States Dump:")
    param_groups_cnt = len(self.vl_optim.param_groups)
    # state dict has more than param_groups info, so extract only the param groups
    param_group_states = list(self.vl_optim.state.values())[:param_groups_cnt]
    for i, state in enumerate(param_group_states):
        print(f"param group: {i}")
        print(f"  step={state['step']}")
        print(f"  exp_avg    all_zero={all(state['exp_avg'] == 0)}")
        print(f"  exp_avg_sq all_zero={all(state['exp_avg_sq'] == 0)}")

    # can also dump LR state if need be
    # print(f"LR={self.vl_scheduler.get_last_lr()}")


def validate_optim_states_are_reset(self):
    # [FUNCTION DOC] Check that a freshly reset optimizer has step == 1 and all-zero exp_avg / exp_avg_sq for
    # the same first N
    # state entries; raise ValueError naming the offending entry otherwise.
    # KNOWN ISSUE: same per-parameter vs per-group slicing issue as dump_optim_states; the error messages
    # label
    # the exp_avg / exp_avg_sq tensors as "step".
    """
    for a new or fully reset optimizer we expect all zeros `exp_avg` and `exp_avg_sq` state tensors and step=1
    """

    param_groups_cnt = len(self.vl_optim.param_groups)
    param_group_states = list(self.vl_optim.state.values())[:param_groups_cnt]
    for i, state in enumerate(param_group_states):
        if state["step"] != 1:
            raise ValueError(f"optimizer reset didn't seem to work: state={i} step={state['step']}")
        if not all(state["exp_avg"] == 0):
            raise ValueError(f"optimizer reset didn't seem to work: state={i} step={state['exp_avg']}")
        if not all(state["exp_avg_sq"] == 0):
            raise ValueError(f"optimizer reset didn't seem to work: state={i} step={state['exp_avg_sq']}")
