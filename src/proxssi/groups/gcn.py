# 1, n_col, n_row
# when norm (dim=(0,1)) is called later, will be equal to l2 norm over rows
def row_groups_fn(params):
    return [param.T.unsqueeze(0) for param in params]


def gcn_groups(model, weight_decay):
    to_prox_rows, not_to_decay = [], []

    for name, param in model.named_parameters():
        if 'conv1.weight' in name:
            to_prox_rows.append(param)
        else:
            not_to_decay.append(param)

    optimizer_grouped_parameters = [
        {
            'params': to_prox_rows,
            'weight_decay': weight_decay,
            'groups_fn': row_groups_fn
        },
        {
            'params': not_to_decay,
            'weight_decay': 0.0,
            'groups_fn': None
        }
    ]

    return optimizer_grouped_parameters
