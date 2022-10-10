def col_groups_fn(params):
    return [param.unsqueeze(0) for param in params]


def linear_groups(model, weight_decay):
    to_prox_cols, not_to_decay = [], []

    for name, param in model.named_parameters():
        if 'linear1.weight' in name:
            to_prox_cols.append(param)
        else:
            not_to_decay.append(param)

    optimizer_grouped_parameters = [
        {
            'params': to_prox_cols,
            'weight_decay': weight_decay,
            'groups_fn': col_groups_fn
        },
        {
            'params': not_to_decay,
            'weight_decay': 0.0,
            'groups_fn': None
        }
    ]

    return optimizer_grouped_parameters
