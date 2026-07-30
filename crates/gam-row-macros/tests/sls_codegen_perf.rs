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
fn exp_stack(value: f64) -> [f64; 5] {
    let exp = value.exp();
    [exp; 5]
}

#[inline(always)]
fn preserve_composition_domain(point: f64, stack: [f64; 5]) -> [f64; 5] {
    if point.is_nan() {
        [f64::NAN; 5]
    } else {
        stack
    }
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
    preserve_composition_domain(
        composition_point,
        [value, first, second, third, fourth],
    )
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
    emit [order2];
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
fn scaled_stack(
    composition_point: f64,
    scale: f64,
    value: f64,
    first: f64,
    second: f64,
    third: f64,
    fourth: f64,
) -> [f64; 5] {
    preserve_composition_domain(
        composition_point,
        [
            scale * value,
            scale * first,
            scale * second,
            scale * third,
            scale * fourth,
        ],
    )
}

type MixedStackLeaf = fn(
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
) -> [f64; 5];

const MIXED_STACK: MixedStackLeaf = |
    composition_point: f64,
    censored_weight: f64,
    event_weight: f64,
    censored_value: f64,
    censored_first: f64,
    censored_second: f64,
    censored_third: f64,
    censored_fourth: f64,
    event_value: f64,
    event_first: f64,
    event_second: f64,
    event_third: f64,
    event_fourth: f64,
| -> [f64; 5] {
    let mut stack = [0.0; 5];
    if censored_weight != 0.0 {
        add_scaled(
            &mut stack,
            [
                censored_value,
                censored_first,
                censored_second,
                censored_third,
                censored_fourth,
            ],
            -censored_weight,
        );
    }
    if event_weight != 0.0 {
        add_scaled(
            &mut stack,
            [
                event_value,
                event_first,
                event_second,
                event_third,
                event_fourth,
            ],
            -event_weight,
        );
    }
    preserve_composition_domain(composition_point, stack)
};

row_program! {
    fn generated_sls_direct(
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
        u1_active,
        g_active,
        w,
        censored_weight,
        event_weight,
        u0_value,
        u0_first,
        u0_second,
        u0_third,
        u0_fourth,
        censored_u1_value,
        censored_u1_first,
        censored_u1_second,
        censored_u1_third,
        censored_u1_fourth,
        event_u1_value,
        event_u1_first,
        event_u1_second,
        event_u1_third,
        event_u1_fourth,
        event_g_value,
        event_g_first,
        event_g_second,
        event_g_third,
        event_g_fourth
    )
    emit [order2];
    leaves {
        exponential => exp_stack => exp_stack_cuda,
        scaled => scaled_stack => scaled_stack_cuda,
        mixed => MIXED_STACK => mixed_stack_cuda,
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
                scaled,
                u0,
                w,
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
                    mixed,
                    u1,
                    censored_weight,
                    event_weight,
                    censored_u1_value,
                    censored_u1_first,
                    censored_u1_second,
                    censored_u1_third,
                    censored_u1_fourth,
                    event_u1_value,
                    event_u1_first,
                    event_u1_second,
                    event_u1_third,
                    event_u1_fourth
                )
            );
        }
        if (g_active != 0.0) {
            nll = add(
                nll,
                compose(
                    scaled,
                    g,
                    -event_weight,
                    event_g_value,
                    event_g_first,
                    event_g_second,
                    event_g_third,
                    event_g_fourth
                )
            );
        }
        return nll;
    }
}

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
    let plan = outer_plan(kernel);
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
fn generated_direct(p: &[f64; K], kernel: &Kernel) -> Channels {
    let censored_weight = kernel.w * (1.0 - kernel.d);
    let event_weight = kernel.w * kernel.d;
    let u0_active = if kernel.w != 0.0 && stack_active(&kernel.u0) != 0.0 {
        1.0
    } else {
        0.0
    };
    let u1_active = if (censored_weight != 0.0 && stack_active(&kernel.censored_u1) != 0.0)
        || (event_weight != 0.0 && stack_active(&kernel.event_u1) != 0.0)
    {
        1.0
    } else {
        0.0
    };
    let g_active = if event_weight != 0.0 && stack_active(&kernel.event_g) != 0.0 {
        1.0
    } else {
        0.0
    };
    let (value, gradient, hessian, []) = generated_sls_direct_order2(
        p[0],
        p[1],
        p[2],
        p[3],
        p[4],
        p[5],
        p[6],
        p[7],
        p[8],
        u0_active,
        u1_active,
        g_active,
        kernel.w,
        censored_weight,
        event_weight,
        kernel.u0[0],
        kernel.u0[1],
        kernel.u0[2],
        kernel.u0[3],
        kernel.u0[4],
        kernel.censored_u1[0],
        kernel.censored_u1[1],
        kernel.censored_u1[2],
        kernel.censored_u1[3],
        kernel.censored_u1[4],
        kernel.event_u1[0],
        kernel.event_u1[1],
        kernel.event_u1[2],
        kernel.event_u1[3],
        kernel.event_u1[4],
        kernel.event_g[0],
        kernel.event_g[1],
        kernel.event_g[2],
        kernel.event_g[3],
        kernel.event_g[4],
    );
    (value, gradient, hessian)
}

#[inline(always)]
fn hand(p: &[f64; K], kernel: &Kernel) -> Channels {
    let entry_exp = (-p[7]).exp();
    let exit_exp = (-p[6]).exp();

    let mut value = kernel.w * kernel.u0[0];
    let u0_first = kernel.w * kernel.u0[1];
    let u0_second = kernel.w * kernel.u0[2];

    let censored_weight = kernel.w * (1.0 - kernel.d);
    let event_weight = kernel.w * kernel.d;
    let mut u1_first = 0.0;
    let mut u1_second = 0.0;
    if censored_weight != 0.0 {
        value -= censored_weight * kernel.censored_u1[0];
        u1_first -= censored_weight * kernel.censored_u1[1];
        u1_second -= censored_weight * kernel.censored_u1[2];
    }
    if event_weight != 0.0 {
        value -= event_weight * (kernel.event_u1[0] + kernel.event_g[0]);
        u1_first -= event_weight * kernel.event_u1[1];
        u1_second -= event_weight * kernel.event_u1[2];
    }
    let g_first = -event_weight * kernel.event_g[1];
    let g_second = -event_weight * kernel.event_g[2];

    let u0_g4 = -entry_exp;
    let u0_g7 = p[4] * entry_exp;
    let u1_g3 = -exit_exp;
    let u1_g6 = p[3] * exit_exp;
    let inner = p[3] * p[8] - p[5];
    let g3 = exit_exp * p[8];
    let g5 = -exit_exp;
    let g6 = -exit_exp * inner;
    let g8 = exit_exp * p[3];

    let mut gradient = [0.0; K];
    gradient[0] = u0_first;
    gradient[4] = u0_first * u0_g4;
    gradient[7] = u0_first * u0_g7;
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

    symmetric!(0, 0, u0_second);
    symmetric!(0, 4, u0_second * u0_g4);
    symmetric!(0, 7, u0_second * u0_g7);
    symmetric!(4, 4, u0_second * u0_g4 * u0_g4);
    symmetric!(4, 7, u0_second * u0_g4 * u0_g7 + u0_first * entry_exp);
    symmetric!(7, 7, u0_second * u0_g7 * u0_g7 - u0_first * u0_g7);

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

#[test]
fn generated_sls_vgh_matches_and_beats_inlined_strongest_hand_932() {
    let (p, kernel) = fixture();
    assert_close(generated(&p, &kernel), hand(&p, &kernel));
    assert_close(generated_direct(&p, &kernel), hand(&p, &kernel));
    for d in [0.0, 1.0, 0.37] {
        let endpoint = Kernel { d, ..kernel };
        assert_close(generated(&p, &endpoint), hand(&p, &endpoint));
        assert_close(generated_direct(&p, &endpoint), hand(&p, &endpoint));
    }

    let generated_ns = best_ns(generated);
    let generated_direct_ns = best_ns(generated_direct);
    let hand_ns = best_ns(hand);
    eprintln!(
        "SLS-MACRO-CODEGEN-932 generated={generated_ns:.2} ns/row \
         generated_direct={generated_direct_ns:.2} ns/row \
         strongest_hand={hand_ns:.2} ns/row hand_over_production={:.6}",
        hand_ns / generated_direct_ns,
    );
    assert!(
        generated_direct_ns < hand_ns,
        "generated {generated_direct_ns:.2} ns/row must beat strongest hand {hand_ns:.2} ns/row"
    );
}
