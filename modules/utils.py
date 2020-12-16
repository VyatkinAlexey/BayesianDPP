from modules.optimality import \
    optimality_A, \
    optimality_C, \
    optimality_D, \
    optimality_V


def _get_optimalty(optimalty_letter: str):
    if optimalty_letter == 'A':
        return optimality_A
    elif optimalty_letter == 'C':
        return optimality_C
    elif optimalty_letter == 'D':
        return optimality_D
    elif optimalty_letter == 'V':
        return optimality_V
    else:
        raise NotImplementedError(f'optimalty {optimalty_letter} is not implemeted')