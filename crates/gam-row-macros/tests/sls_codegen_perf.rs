use gam_row_macros::row_program;
use std::time::Instant;

const K: usize = 9;

#[derive(Clone, Copy)]
struct Kernel {
    w: f64,
    d: f64,
    u0: [f64; 5],
    censored_u1: [f64; 5],
    event_u1: [f64; 5],
    event_g: [f64; 5],
}

#[derive(Clone, Copy)]
struct Plan {
    u0: [f64; 5],
    u1: Option<[f64; 5]>,
    g: Option<[f64; 5]>,
}

#[inline(always)]
fn add_scaled(target: &mut [f64; 5], source: [f64; 5], scale: f64) {
    for i in 0..5 {
        target[i] += scale * source[i];
    }
}

#[inline(always)]
fn outer_plan(kernel: &Kernel) -> Plan {
    let mut u0 = [0.0; 5];
    add_scaled(&mut u0, kernel.u0, kernel.w);

    let censored_weight = kernel.w * (1.0 - kernel.d);
    let event_weight = kernel.w * kernel.d;
    let mut u1 = [0.0; 5];
    if censored_weight != 0.0 {
        add_scaled(&mut u1, kernel.censored_u1, -censored_weight);
    }
    if event_weight != 0.0 {
        add_scaled(&mut u1, kernel.event_u1, -event_weight);
    }
    let g = (event_weight != 0.0).then(|| {
        let mut stack = [0.0; 5];
        add_scaled(&mut stack, kernel.event_g, -event_weight);
        stack
    });
    Plan {
        u0,
        u1: (censored_weight != 0.0 || event_weight != 0.0).then_some(u1),
        g,
    }
}

#[inline(always)]
fn outer_plan_order2(kernel: &Kernel) -> Plan {
    let u0 = [
        kernel.w * kernel.u0[0],
        kernel.w * kernel.u0[1],
        kernel.w * kernel.u0[2],
        0.0,
        0.0,
    ];
    let censored_weight = kernel.w * (1.0 - kernel.d);
    let event_weight = kernel.w * kernel.d;
    let u1 = (censored_weight != 0.0 || event_weight != 0.0).then(|| {
        let mut stack = [0.0; 5];
        if censored_weight != 0.0 {
            for i in 0..3 {
                stack[i] -= censored_weight * kernel.censored_u1[i];
            }
        }
        if event_weight != 0.0 {
            for i in 0..3 {
                stack[i] -= event_weight * kernel.event_u1[i];
            }
        }
        stack
    });
    let g = (event_weight != 0.0).then(|| {
        [
            -event_weight * kernel.event_g[0],
            -event_weight * kernel.event_g[1],
            -event_weight * kernel.event_g[2],
            0.0,
            0.0,
        ]
    });
    Plan { u0, u1, g }
}

#[inline(always)]
fn exp_stack(value: f64) -> [f64; 5] {
    let exp = value.exp();
    [exp; 5]
}

#[inline(always)]
fn preserve_composition_domain(point: f64, stack: [f64; 5]) -> [f64; 5] {
    if point.is_nan() { [f64::NAN; 5] } else { stack }
}

#[inline(always)]
fn outer_stack(
    composition_point: f64,
    value: f64,
    first: f64,
    second: f64,
    third: f64,
    fourth: f64,
) -> [f64; 5] {
    preserve_composition_domain(composition_point, [value, first, second, third, fourth])
}

row_program! {
    fn generated_sls(
        h0,
        h1,
        hdot,
        eta_t_exit,
        eta_t_entry,
        eta_t_deriv,
        eta_ls_exit,
        eta_ls_entry,
        eta_ls_deriv;
        u0_active,
        u0_value,
        u0_first,
        u0_second,
        u0_third,
        u0_fourth,
        u1_active,
        u1_value,
        u1_first,
        u1_second,
        u1_third,
        u1_fourth,
        g_active,
        g_value,
        g_first,
        g_second,
        g_third,
        g_fourth
    )
    emit [generic, order2, third, fourth];
    leaves {
        exponential => exp_stack => exp_stack_cuda,
        outer => outer_stack => outer_stack_cuda,
    }
    witnesses [];
    {
        let neg_eta_ls_entry = neg(eta_ls_entry);
        let inv_sigma_entry = compose(exponential, neg_eta_ls_entry);
        let u0 = add(h0, neg(mul(eta_t_entry, inv_sigma_entry)));

        let neg_eta_ls_exit = neg(eta_ls_exit);
        let inv_sigma_exit = compose(exponential, neg_eta_ls_exit);
        let u1 = add(h1, neg(mul(eta_t_exit, inv_sigma_exit)));
        let event_inner = add(mul(eta_t_exit, eta_ls_deriv), neg(eta_t_deriv));
        let g = add(hdot, mul(inv_sigma_exit, event_inner));

        let mut nll = zero();
        if (u0_active != 0.0) {
            nll = compose(
                outer,
                u0,
                u0_value,
                u0_first,
                u0_second,
                u0_third,
                u0_fourth
            );
        }
        if (u1_active != 0.0) {
            nll = add(
                nll,
                compose(
                    outer,
                    u1,
                    u1_value,
                    u1_first,
                    u1_second,
                    u1_third,
                    u1_fourth
                )
            );
        }
        if (g_active != 0.0) {
            nll = add(
                nll,
                compose(
                    outer,
                    g,
                    g_value,
                    g_first,
                    g_second,
                    g_third,
                    g_fourth
                )
            );
        }
        return nll;
    }
}

type Channels = (f64, [f64; K], [[f64; K]; K]);

#[inline(always)]
fn stack_active(stack: &[f64; 5]) -> f64 {
    if stack.iter().all(|value| *value == 0.0) {
        0.0
    } else {
        1.0
    }
}

#[inline(always)]
fn generated(p: &[f64; K], kernel: &Kernel) -> Channels {
    let plan = outer_plan_order2(kernel);
    let u1 = plan.u1.unwrap_or([0.0; 5]);
    let g = plan.g.unwrap_or([0.0; 5]);
    let (value, gradient, hessian, []) = generated_sls_order2(
        p[0],
        p[1],
        p[2],
        p[3],
        p[4],
        p[5],
        p[6],
        p[7],
        p[8],
        stack_active(&plan.u0),
        plan.u0[0],
        plan.u0[1],
        plan.u0[2],
        plan.u0[3],
        plan.u0[4],
        stack_active(&u1),
        u1[0],
        u1[1],
        u1[2],
        u1[3],
        u1[4],
        stack_active(&g),
        g[0],
        g[1],
        g[2],
        g[3],
        g[4],
    );
    (value, gradient, hessian)
}

#[inline(always)]
fn generated_third(p: &[f64; K], kernel: &Kernel, direction: &[f64; K]) -> [[f64; K]; K] {
    let plan = outer_plan(kernel);
    let u1 = plan.u1.unwrap_or([0.0; 5]);
    let g = plan.g.unwrap_or([0.0; 5]);
    generated_sls_third_contracted(
        p[0],
        p[1],
        p[2],
        p[3],
        p[4],
        p[5],
        p[6],
        p[7],
        p[8],
        stack_active(&plan.u0),
        plan.u0[0],
        plan.u0[1],
        plan.u0[2],
        plan.u0[3],
        plan.u0[4],
        stack_active(&u1),
        u1[0],
        u1[1],
        u1[2],
        u1[3],
        u1[4],
        stack_active(&g),
        g[0],
        g[1],
        g[2],
        g[3],
        g[4],
        direction,
    )
}

#[inline(always)]
fn generated_fourth(
    p: &[f64; K],
    kernel: &Kernel,
    direction_u: &[f64; K],
    direction_v: &[f64; K],
) -> [[f64; K]; K] {
    let plan = outer_plan(kernel);
    let u1 = plan.u1.unwrap_or([0.0; 5]);
    let g = plan.g.unwrap_or([0.0; 5]);
    generated_sls_fourth_contracted(
        p[0],
        p[1],
        p[2],
        p[3],
        p[4],
        p[5],
        p[6],
        p[7],
        p[8],
        stack_active(&plan.u0),
        plan.u0[0],
        plan.u0[1],
        plan.u0[2],
        plan.u0[3],
        plan.u0[4],
        stack_active(&u1),
        u1[0],
        u1[1],
        u1[2],
        u1[3],
        u1[4],
        stack_active(&g),
        g[0],
        g[1],
        g[2],
        g[3],
        g[4],
        direction_u,
        direction_v,
    )
}

#[inline(always)]
fn jet_third(p: &[f64; K], kernel: &Kernel, direction: &[f64; K]) -> [[f64; K]; K] {
    use gam_math::jet_scalar::OneSeed;

    let plan = outer_plan(kernel);
    let u1 = plan.u1.unwrap_or([0.0; 5]);
    let g = plan.g.unwrap_or([0.0; 5]);
    let vars: [OneSeed<K>; K] =
        std::array::from_fn(|axis| OneSeed::seed_direction(p[axis], axis, direction[axis]));
    let (value, []) = generated_sls(
        &vars[0],
        &vars[1],
        &vars[2],
        &vars[3],
        &vars[4],
        &vars[5],
        &vars[6],
        &vars[7],
        &vars[8],
        stack_active(&plan.u0),
        plan.u0[0],
        plan.u0[1],
        plan.u0[2],
        plan.u0[3],
        plan.u0[4],
        stack_active(&u1),
        u1[0],
        u1[1],
        u1[2],
        u1[3],
        u1[4],
        stack_active(&g),
        g[0],
        g[1],
        g[2],
        g[3],
        g[4],
    );
    value.contracted_third()
}

#[inline(always)]
fn jet_fourth(
    p: &[f64; K],
    kernel: &Kernel,
    direction_u: &[f64; K],
    direction_v: &[f64; K],
) -> [[f64; K]; K] {
    use gam_math::jet_scalar::TwoSeed;

    let plan = outer_plan(kernel);
    let u1 = plan.u1.unwrap_or([0.0; 5]);
    let g = plan.g.unwrap_or([0.0; 5]);
    let vars: [TwoSeed<K>; K] = std::array::from_fn(|axis| {
        TwoSeed::seed(p[axis], axis, direction_u[axis], direction_v[axis])
    });
    let (value, []) = generated_sls(
        &vars[0],
        &vars[1],
        &vars[2],
        &vars[3],
        &vars[4],
        &vars[5],
        &vars[6],
        &vars[7],
        &vars[8],
        stack_active(&plan.u0),
        plan.u0[0],
        plan.u0[1],
        plan.u0[2],
        plan.u0[3],
        plan.u0[4],
        stack_active(&u1),
        u1[0],
        u1[1],
        u1[2],
        u1[3],
        u1[4],
        stack_active(&g),
        g[0],
        g[1],
        g[2],
        g[3],
        g[4],
    );
    value.contracted_fourth()
}

const PERMUTATIONS_3: [[usize; 3]; 6] = [
    [0, 1, 2],
    [0, 2, 1],
    [1, 0, 2],
    [1, 2, 0],
    [2, 0, 1],
    [2, 1, 0],
];

const PERMUTATIONS_4: [[usize; 4]; 24] = [
    [0, 1, 2, 3],
    [0, 1, 3, 2],
    [0, 2, 1, 3],
    [0, 2, 3, 1],
    [0, 3, 1, 2],
    [0, 3, 2, 1],
    [1, 0, 2, 3],
    [1, 0, 3, 2],
    [1, 2, 0, 3],
    [1, 2, 3, 0],
    [1, 3, 0, 2],
    [1, 3, 2, 0],
    [2, 0, 1, 3],
    [2, 0, 3, 1],
    [2, 1, 0, 3],
    [2, 1, 3, 0],
    [2, 3, 0, 1],
    [2, 3, 1, 0],
    [3, 0, 1, 2],
    [3, 0, 2, 1],
    [3, 1, 0, 2],
    [3, 1, 2, 0],
    [3, 2, 0, 1],
    [3, 2, 1, 0],
];

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn hand_analytic_term<const ORDER: usize, const N: usize>(
    output: &mut [[f64; K]; K],
    active: [usize; N],
    stack: [f64; 5],
    direction_u: &[f64; K],
    direction_v: &[f64; K],
    d1: impl Fn(usize) -> f64,
    d2: impl Fn(usize, usize) -> f64,
    third_terms: &[([usize; 3], f64)],
    fourth_terms: &[([usize; 4], f64)],
) {
    let mut first = [0.0; N];
    let mut second = [[0.0; N]; N];
    let mut second_u = [0.0; N];
    let mut second_v = [0.0; N];
    let mut third_u = [[0.0; N]; N];
    let mut third_v = [[0.0; N]; N];
    let mut third_uv = [0.0; N];
    let mut fourth_uv = [[0.0; N]; N];
    let mut zu = 0.0;
    let mut zv = 0.0;
    let mut zuv = 0.0;
    for i in 0..N {
        first[i] = d1(active[i]);
        zu += first[i] * direction_u[active[i]];
        if ORDER == 4 {
            zv += first[i] * direction_v[active[i]];
        }
        for j in 0..N {
            second[i][j] = d2(active[i], active[j]);
            second_u[i] += second[i][j] * direction_u[active[j]];
            if ORDER == 4 {
                second_v[i] += second[i][j] * direction_v[active[j]];
                zuv += second[i][j] * direction_u[active[i]] * direction_v[active[j]];
            }
        }
    }
    for &(indices, coefficient) in third_terms {
        let mut seen = [[usize::MAX; 3]; 6];
        let mut seen_count = 0;
        for permutation in PERMUTATIONS_3 {
            let ordered = [
                indices[permutation[0]],
                indices[permutation[1]],
                indices[permutation[2]],
            ];
            if seen[..seen_count].contains(&ordered) {
                continue;
            }
            seen[seen_count] = ordered;
            seen_count += 1;
            let [i, j, k] = ordered;
            third_u[i][j] += coefficient * direction_u[active[k]];
            if ORDER == 4 {
                third_v[i][j] += coefficient * direction_v[active[k]];
                third_uv[i] += coefficient * direction_u[active[j]] * direction_v[active[k]];
            }
        }
    }
    if ORDER == 4 {
        for &(indices, coefficient) in fourth_terms {
            let mut seen = [[usize::MAX; 4]; 24];
            let mut seen_count = 0;
            for permutation in PERMUTATIONS_4 {
                let ordered = [
                    indices[permutation[0]],
                    indices[permutation[1]],
                    indices[permutation[2]],
                    indices[permutation[3]],
                ];
                if seen[..seen_count].contains(&ordered) {
                    continue;
                }
                seen[seen_count] = ordered;
                seen_count += 1;
                let [i, j, k, l] = ordered;
                fourth_uv[i][j] += coefficient * direction_u[active[k]] * direction_v[active[l]];
            }
        }
    }

    for i in 0..N {
        let a = active[i];
        let za = first[i];
        for j in 0..N {
            let b = active[j];
            let zb = first[j];
            let zab = second[i][j];
            if ORDER == 3 {
                output[a][b] += stack[3] * zu * za * zb
                    + stack[2] * (second_u[i] * zb + za * second_u[j] + zu * zab)
                    + stack[1] * third_u[i][j];
            } else {
                let f2_hessian = stack[4] * za * zb + stack[3] * zab;
                let f2_gradient_a = stack[3] * za;
                let f2_gradient_b = stack[3] * zb;
                let f2_zu_zv_hessian = f2_hessian * zu * zv
                    + stack[2] * third_u[i][j] * zv
                    + stack[2] * zu * third_v[i][j]
                    + f2_gradient_a * second_u[j] * zv
                    + f2_gradient_b * second_u[i] * zv
                    + f2_gradient_a * zu * second_v[j]
                    + f2_gradient_b * zu * second_v[i]
                    + stack[2] * second_u[i] * second_v[j]
                    + stack[2] * second_u[j] * second_v[i];
                let f1_hessian = stack[3] * za * zb + stack[2] * zab;
                let f1_zuv_hessian = f1_hessian * zuv
                    + stack[1] * fourth_uv[i][j]
                    + stack[2] * za * third_uv[j]
                    + stack[2] * zb * third_uv[i];
                output[a][b] += f2_zu_zv_hessian + f1_zuv_hessian;
            }
        }
    }
}

#[inline(always)]
fn hand_analytic_contracted<const ORDER: usize>(
    p: &[f64; K],
    kernel: &Kernel,
    direction_u: &[f64; K],
    direction_v: &[f64; K],
) -> [[f64; K]; K] {
    let plan = outer_plan(kernel);
    let mut output = [[0.0; K]; K];

    if stack_active(&plan.u0) != 0.0 {
        let exponential = (-p[7]).exp();
        let product = p[4] * exponential;
        hand_analytic_term::<ORDER, 3>(
            &mut output,
            [0, 4, 7],
            plan.u0,
            direction_u,
            direction_v,
            |axis| match axis {
                0 => 1.0,
                4 => -exponential,
                7 => product,
                _ => 0.0,
            },
            |a, b| match [a.min(b), a.max(b)] {
                [4, 7] => exponential,
                [7, 7] => -product,
                _ => 0.0,
            },
            &[([1, 2, 2], -exponential), ([2, 2, 2], product)],
            &[([1, 2, 2, 2], exponential), ([2, 2, 2, 2], -product)],
        );
    }

    if let Some(stack) = plan.u1 {
        let exponential = (-p[6]).exp();
        let product = p[3] * exponential;
        hand_analytic_term::<ORDER, 3>(
            &mut output,
            [1, 3, 6],
            stack,
            direction_u,
            direction_v,
            |axis| match axis {
                1 => 1.0,
                3 => -exponential,
                6 => product,
                _ => 0.0,
            },
            |a, b| match [a.min(b), a.max(b)] {
                [3, 6] => exponential,
                [6, 6] => -product,
                _ => 0.0,
            },
            &[([1, 2, 2], -exponential), ([2, 2, 2], product)],
            &[([1, 2, 2, 2], exponential), ([2, 2, 2, 2], -product)],
        );
    }

    if let Some(stack) = plan.g {
        let exponential = (-p[6]).exp();
        let inner = p[3] * p[8] - p[5];
        let product = exponential * inner;
        hand_analytic_term::<ORDER, 5>(
            &mut output,
            [2, 3, 5, 6, 8],
            stack,
            direction_u,
            direction_v,
            |axis| match axis {
                2 => 1.0,
                3 => exponential * p[8],
                5 => -exponential,
                6 => -product,
                8 => exponential * p[3],
                _ => 0.0,
            },
            |a, b| match [a.min(b), a.max(b)] {
                [3, 6] => -exponential * p[8],
                [3, 8] => exponential,
                [5, 6] => exponential,
                [6, 6] => product,
                [6, 8] => -exponential * p[3],
                _ => 0.0,
            },
            &[
                ([1, 3, 3], exponential * p[8]),
                ([1, 3, 4], -exponential),
                ([2, 3, 3], -exponential),
                ([3, 3, 3], -product),
                ([3, 3, 4], exponential * p[3]),
            ],
            &[
                ([1, 3, 3, 3], -exponential * p[8]),
                ([1, 3, 3, 4], exponential),
                ([2, 3, 3, 3], exponential),
                ([3, 3, 3, 3], product),
                ([3, 3, 3, 4], -exponential * p[3]),
            ],
        );
    }

    output
}

#[inline(always)]
fn hand(p: &[f64; K], kernel: &Kernel) -> Channels {
    let entry_exp = (-p[7]).exp();
    let exit_exp = (-p[6]).exp();

    let u0_point = p[0] - p[4] * entry_exp;
    let u0_stack = [
        kernel.w * kernel.u0[0],
        kernel.w * kernel.u0[1],
        kernel.w * kernel.u0[2],
        0.0,
        0.0,
    ];
    let u0_active = !u0_stack.iter().all(|value| *value == 0.0);
    let u0_stack = if u0_active {
        preserve_composition_domain(u0_point, u0_stack)
    } else {
        u0_stack
    };
    let mut value = u0_stack[0];
    let u0_first = u0_stack[1];
    let u0_second = u0_stack[2];

    let censored_weight = kernel.w * (1.0 - kernel.d);
    let event_weight = kernel.w * kernel.d;
    let mut u1_value = 0.0;
    let mut u1_first = 0.0;
    let mut u1_second = 0.0;
    if censored_weight != 0.0 {
        u1_value -= censored_weight * kernel.censored_u1[0];
        u1_first -= censored_weight * kernel.censored_u1[1];
        u1_second -= censored_weight * kernel.censored_u1[2];
    }
    if event_weight != 0.0 {
        u1_value -= event_weight * kernel.event_u1[0];
        u1_first -= event_weight * kernel.event_u1[1];
        u1_second -= event_weight * kernel.event_u1[2];
    }
    let u1_point = p[1] - p[3] * exit_exp;
    let [u1_value, u1_first, u1_second, _, _] = if censored_weight != 0.0 || event_weight != 0.0 {
        preserve_composition_domain(u1_point, [u1_value, u1_first, u1_second, 0.0, 0.0])
    } else {
        [u1_value, u1_first, u1_second, 0.0, 0.0]
    };
    value += u1_value;

    let inner = p[3] * p[8] - p[5];
    let g_point = p[2] + exit_exp * inner;
    let [g_value, g_first, g_second, _, _] = if event_weight != 0.0 {
        preserve_composition_domain(
            g_point,
            [
                -event_weight * kernel.event_g[0],
                -event_weight * kernel.event_g[1],
                -event_weight * kernel.event_g[2],
                0.0,
                0.0,
            ],
        )
    } else {
        [
            -event_weight * kernel.event_g[0],
            -event_weight * kernel.event_g[1],
            -event_weight * kernel.event_g[2],
            0.0,
            0.0,
        ]
    };
    value += g_value;

    let u0_g4 = -entry_exp;
    let u0_g7 = p[4] * entry_exp;
    let u1_g3 = -exit_exp;
    let u1_g6 = p[3] * exit_exp;
    let g3 = exit_exp * p[8];
    let g5 = -exit_exp;
    let g6 = -exit_exp * inner;
    let g8 = exit_exp * p[3];

    let mut gradient = [0.0; K];
    if u0_active {
        gradient[0] = u0_first;
        gradient[4] = u0_first * u0_g4;
        gradient[7] = u0_first * u0_g7;
    }
    if censored_weight != 0.0 || event_weight != 0.0 {
        gradient[1] = u1_first;
        gradient[3] = u1_first * u1_g3;
        gradient[6] = u1_first * u1_g6;
    }
    if event_weight != 0.0 {
        gradient[2] = g_first;
        gradient[3] += g_first * g3;
        gradient[5] = g_first * g5;
        gradient[6] += g_first * g6;
        gradient[8] = g_first * g8;
    }

    let mut hessian = [[0.0; K]; K];
    macro_rules! symmetric {
        ($i:expr, $j:expr, $channel:expr) => {{
            let channel = $channel;
            hessian[$i][$j] += channel;
            if $i != $j {
                hessian[$j][$i] += channel;
            }
        }};
    }

    if u0_active {
        symmetric!(0, 0, u0_second);
        symmetric!(0, 4, u0_second * u0_g4);
        symmetric!(0, 7, u0_second * u0_g7);
        symmetric!(4, 4, u0_second * u0_g4 * u0_g4);
        symmetric!(4, 7, u0_second * u0_g4 * u0_g7 + u0_first * entry_exp);
        symmetric!(7, 7, u0_second * u0_g7 * u0_g7 - u0_first * u0_g7);
    }

    if censored_weight != 0.0 || event_weight != 0.0 {
        symmetric!(1, 1, u1_second);
        symmetric!(1, 3, u1_second * u1_g3);
        symmetric!(1, 6, u1_second * u1_g6);
        symmetric!(3, 3, u1_second * u1_g3 * u1_g3);
        symmetric!(3, 6, u1_second * u1_g3 * u1_g6 + u1_first * exit_exp);
        symmetric!(6, 6, u1_second * u1_g6 * u1_g6 - u1_first * u1_g6);
    }

    if event_weight != 0.0 {
        symmetric!(2, 2, g_second);
        symmetric!(2, 3, g_second * g3);
        symmetric!(2, 5, g_second * g5);
        symmetric!(2, 6, g_second * g6);
        symmetric!(2, 8, g_second * g8);
        symmetric!(3, 3, g_second * g3 * g3);
        symmetric!(3, 5, g_second * g3 * g5);
        symmetric!(3, 6, g_second * g3 * g6 - g_first * exit_exp * p[8]);
        symmetric!(3, 8, g_second * g3 * g8 + g_first * exit_exp);
        symmetric!(5, 5, g_second * g5 * g5);
        symmetric!(5, 6, g_second * g5 * g6 + g_first * exit_exp);
        symmetric!(5, 8, g_second * g5 * g8);
        symmetric!(6, 6, g_second * g6 * g6 + g_first * exit_exp * inner);
        symmetric!(6, 8, g_second * g6 * g8 - g_first * exit_exp * p[3]);
        symmetric!(8, 8, g_second * g8 * g8);
    }

    (value, gradient, hessian)
}

fn fixture() -> ([f64; K], Kernel) {
    (
        [0.4, -0.7, 0.2, 0.8, -0.35, 0.11, -0.25, 0.31, -0.17],
        Kernel {
            w: 1.3,
            d: 1.0,
            u0: [-0.8, -0.7, 0.3, -0.12, 0.05],
            censored_u1: [-1.1, -0.9, 0.4, -0.18, 0.08],
            event_u1: [-1.4, -0.6, -1.0, 0.0, 0.0],
            event_g: [-0.2, 1.4, -1.96, 5.488, -23.0496],
        },
    )
}

fn assert_close(got: Channels, want: Channels) {
    let close = |a: f64, b: f64| {
        let tolerance = 1e-12 * a.abs().max(b.abs()).max(1.0);
        assert!((a - b).abs() <= tolerance, "{a:+.16e} vs {b:+.16e}");
    };
    close(got.0, want.0);
    for i in 0..K {
        close(got.1[i], want.1[i]);
        for j in 0..K {
            close(got.2[i][j], want.2[i][j]);
        }
    }
}

fn assert_same_channels(got: Channels, want: Channels) {
    let same = |a: f64, b: f64| {
        if a.is_nan() || b.is_nan() {
            assert!(a.is_nan() && b.is_nan(), "{a:+.16e} vs {b:+.16e}");
        } else {
            let tolerance = 1e-12 * a.abs().max(b.abs()).max(1.0);
            assert!((a - b).abs() <= tolerance, "{a:+.16e} vs {b:+.16e}");
        }
    };
    same(got.0, want.0);
    for i in 0..K {
        same(got.1[i], want.1[i]);
        for j in 0..K {
            same(got.2[i][j], want.2[i][j]);
        }
    }
}

fn assert_matrix_close(got: [[f64; K]; K], want: [[f64; K]; K]) {
    for i in 0..K {
        for j in 0..K {
            let tolerance = 2e-11 * got[i][j].abs().max(want[i][j].abs()).max(1.0);
            assert!(
                (got[i][j] - want[i][j]).abs() <= tolerance,
                "[{i}][{j}] {:+.16e} vs {:+.16e}",
                got[i][j],
                want[i][j],
            );
        }
    }
}

fn best_ns(evaluate: impl Fn(&[f64; K], &Kernel) -> Channels) -> f64 {
    let (p, kernel) = fixture();
    let iterations = 2_000_000;
    let mut best = f64::INFINITY;
    for _ in 0..5 {
        let mut checksum = 0.0;
        let started = Instant::now();
        for _ in 0..iterations {
            let mut perturbed = p;
            perturbed[7] += checksum * 1e-18;
            let (value, gradient, hessian) = std::hint::black_box(evaluate(&perturbed, &kernel));
            checksum += value + gradient[4] + hessian[4][4] + hessian[4][7];
        }
        assert!(checksum.is_finite());
        best = best.min(started.elapsed().as_secs_f64());
    }
    best * 1e9 / iterations as f64
}

fn matrix_sample_ns(
    iterations: usize,
    p: &[f64; K],
    kernel: &Kernel,
    mut evaluate: impl FnMut(&[f64; K], &Kernel) -> [[f64; K]; K],
) -> f64 {
    let mut checksum = 0.0;
    let started = Instant::now();
    for _ in 0..iterations {
        let mut perturbed = *p;
        perturbed[7] += checksum * 1e-20;
        let matrix = std::hint::black_box(evaluate(&perturbed, kernel));
        checksum += matrix
            .iter()
            .flat_map(|row| row.iter())
            .copied()
            .sum::<f64>();
    }
    assert!(checksum.is_finite());
    started.elapsed().as_secs_f64() * 1e9 / iterations as f64
}

#[test]
fn generated_sls_vgh_matches_and_beats_inlined_strongest_hand_932() {
    let (p, kernel) = fixture();
    assert_close(generated(&p, &kernel), hand(&p, &kernel));
    for d in [0.0, 1.0, 0.37] {
        let endpoint = Kernel { d, ..kernel };
        assert_close(generated(&p, &endpoint), hand(&p, &endpoint));
    }
    for axis in [0, 1, 2, 6, 7] {
        let mut nonfinite = p;
        nonfinite[axis] = f64::NAN;
        assert_same_channels(generated(&nonfinite, &kernel), hand(&nonfinite, &kernel));
    }
    let inactive = Kernel { w: 0.0, ..kernel };
    let mut nonfinite = p;
    nonfinite[0] = f64::NAN;
    nonfinite[1] = f64::NAN;
    nonfinite[2] = f64::NAN;
    assert_same_channels(
        generated(&nonfinite, &inactive),
        hand(&nonfinite, &inactive),
    );

    let generated_ns = best_ns(generated);
    let hand_ns = best_ns(hand);
    eprintln!(
        "SLS-MACRO-CODEGEN-932 generated={generated_ns:.2} ns/row \
         strongest_hand={hand_ns:.2} ns/row hand_over_production={:.6}",
        hand_ns / generated_ns,
    );
    assert!(
        generated_ns < hand_ns,
        "generated {generated_ns:.2} ns/row must beat strongest hand {hand_ns:.2} ns/row"
    );
}

#[test]
fn generated_sls_contracted_orders_match_canonical_jets_932() {
    let (p, kernel) = fixture();
    let direction_u = [0.7, -1.3, 0.4, 0.6, -0.5, 0.9, -0.2, 0.3, -0.8];
    let direction_v = [-0.4, 0.6, 1.1, -0.2, 0.8, -0.7, 0.5, -0.9, 0.1];
    for d in [0.0, 1.0, 0.37] {
        let endpoint = Kernel { d, ..kernel };
        let hand_third = hand_analytic_contracted::<3>(&p, &endpoint, &direction_u, &direction_v);
        let hand_fourth = hand_analytic_contracted::<4>(&p, &endpoint, &direction_u, &direction_v);
        assert_matrix_close(
            generated_third(&p, &endpoint, &direction_u),
            jet_third(&p, &endpoint, &direction_u),
        );
        assert_matrix_close(generated_third(&p, &endpoint, &direction_u), hand_third);
        assert_matrix_close(
            generated_fourth(&p, &endpoint, &direction_u, &direction_v),
            jet_fourth(&p, &endpoint, &direction_u, &direction_v),
        );
        assert_matrix_close(
            generated_fourth(&p, &endpoint, &direction_u, &direction_v),
            hand_fourth,
        );
    }
}

#[test]
fn release_measure_generated_sls_contractions_vs_strongest_hand_932() {
    let (p, kernel) = fixture();
    let direction_u = [0.7, -1.3, 0.4, 0.6, -0.5, 0.9, -0.2, 0.3, -0.8];
    let direction_v = [-0.4, 0.6, 1.1, -0.2, 0.8, -0.7, 0.5, -0.9, 0.1];
    let iterations = 100_000;
    let mut third_generated_ns = f64::INFINITY;
    let mut third_hand_ns = f64::INFINITY;
    let mut fourth_generated_ns = f64::INFINITY;
    let mut fourth_hand_ns = f64::INFINITY;

    for round in 0..7 {
        if round % 2 == 0 {
            third_generated_ns =
                third_generated_ns.min(matrix_sample_ns(iterations, &p, &kernel, |values, row| {
                    generated_third(values, row, &direction_u)
                }));
            third_hand_ns =
                third_hand_ns.min(matrix_sample_ns(iterations, &p, &kernel, |values, row| {
                    hand_analytic_contracted::<3>(values, row, &direction_u, &direction_v)
                }));
            fourth_generated_ns = fourth_generated_ns.min(matrix_sample_ns(
                iterations,
                &p,
                &kernel,
                |values, row| generated_fourth(values, row, &direction_u, &direction_v),
            ));
            fourth_hand_ns =
                fourth_hand_ns.min(matrix_sample_ns(iterations, &p, &kernel, |values, row| {
                    hand_analytic_contracted::<4>(values, row, &direction_u, &direction_v)
                }));
        } else {
            fourth_hand_ns =
                fourth_hand_ns.min(matrix_sample_ns(iterations, &p, &kernel, |values, row| {
                    hand_analytic_contracted::<4>(values, row, &direction_u, &direction_v)
                }));
            fourth_generated_ns = fourth_generated_ns.min(matrix_sample_ns(
                iterations,
                &p,
                &kernel,
                |values, row| generated_fourth(values, row, &direction_u, &direction_v),
            ));
            third_hand_ns =
                third_hand_ns.min(matrix_sample_ns(iterations, &p, &kernel, |values, row| {
                    hand_analytic_contracted::<3>(values, row, &direction_u, &direction_v)
                }));
            third_generated_ns =
                third_generated_ns.min(matrix_sample_ns(iterations, &p, &kernel, |values, row| {
                    generated_third(values, row, &direction_u)
                }));
        }
    }

    let mut third_specialized_jet_ns = f64::INFINITY;
    let mut fourth_specialized_jet_ns = f64::INFINITY;
    for _ in 0..5 {
        third_specialized_jet_ns = third_specialized_jet_ns.min(matrix_sample_ns(
            iterations,
            &p,
            &kernel,
            |values, row| jet_third(values, row, &direction_u),
        ));
        fourth_specialized_jet_ns = fourth_specialized_jet_ns.min(matrix_sample_ns(
            iterations,
            &p,
            &kernel,
            |values, row| jet_fourth(values, row, &direction_u, &direction_v),
        ));
    }
    let third_strongest_ns = third_hand_ns.min(third_specialized_jet_ns);
    let fourth_strongest_ns = fourth_hand_ns.min(fourth_specialized_jet_ns);
    eprintln!(
        "SLS-CONTRACTED-HAND-932 order=3 generated={third_generated_ns:.2} ns/row \
         hand_analytic={third_hand_ns:.2} ns/row specialized_jet={third_specialized_jet_ns:.2} ns/row \
         strongest_opponent={third_strongest_ns:.2} ns/row opponent_over_generated={:.6}",
        third_strongest_ns / third_generated_ns,
    );
    eprintln!(
        "SLS-CONTRACTED-HAND-932 order=4 generated={fourth_generated_ns:.2} ns/row \
         hand_analytic={fourth_hand_ns:.2} ns/row specialized_jet={fourth_specialized_jet_ns:.2} ns/row \
         strongest_opponent={fourth_strongest_ns:.2} ns/row opponent_over_generated={:.6}",
        fourth_strongest_ns / fourth_generated_ns,
    );
    assert!(
        third_generated_ns < third_strongest_ns,
        "generated third {third_generated_ns:.2} ns/row must beat strongest hand \
         {third_strongest_ns:.2} ns/row",
    );
    assert!(
        fourth_generated_ns < fourth_strongest_ns,
        "generated fourth {fourth_generated_ns:.2} ns/row must beat strongest hand \
         {fourth_strongest_ns:.2} ns/row",
    );
}
