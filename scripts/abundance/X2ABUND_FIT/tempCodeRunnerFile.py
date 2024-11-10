params = param_group(
    weight_fe=Parameter(value=5, min=3, max=7),
    weight_ti=Parameter(value=1, min=0, max=0.3),
    weight_ca=Parameter(value=9, min=8, max=12),
    weight_si=Parameter(value=21, min=19, max=22),
    weight_al=Parameter(value=14, min=13, max=15),
    weight_mg=Parameter(value=5, min=4, max=6),
    weight_na=Parameter(value=0.5, min=0, max=1),
    weight_o=Parameter(expr='100 - weight_fe - weight_ti - weight_ca - weight_si - weight_al - weight_mg - weight_na')
)