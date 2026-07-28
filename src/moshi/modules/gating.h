#pragma once

/*****************************************\
 * moshi.modules.gating.ActivationGating
\*****************************************/

struct moshi_activation_gating_t {
    own_ptr<torch_nn_linear_t> linear_in;
    own_ptr<torch_nn_linear_t> linear_out;
};

ggml_tensor * moshi_activation_gating(
        ggml_context * ctx,
        moshi_activation_gating_t * gating,
        ggml_tensor * x ) {
    x = torch_nn_linear( ctx, gating->linear_in, x );

    // Split [2h, T] into two [h, T] halves along dim 0.
    //
    // Plain 2-D views, not the previous 4-D ones that parked T in dim 2. Those had correct
    // strides but ggml_silu walks ggml_nrows(src) rows using nb[1] alone, and for T > 1
    // consecutive rows of that layout are nb[1] apart in dim 2, not nb[1]/2 — so silu read
    // the second half of position p as the first half of position p+1. Invisible while T was
    // always 1.
    //
    // ggml_silu also wants a dim-0-contiguous source, which a strided half-view is not, so
    // each half is made contiguous first. That is one copy of h*T floats per layer; at
    // h=4224 and T<=4 it is ~68 KB, far below the weight traffic this batching saves.
    const int64_t h = x->ne[0] / 2;
    auto x_left = ggml_cont( ctx, ggml_view_2d( ctx, x, h, x->ne[1], x->nb[1], 0 ) );
    auto x_right = ggml_cont( ctx, ggml_view_2d( ctx, x, h, x->ne[1], x->nb[1],
                                                 h * x->nb[0] ) );
    x_left = ggml_silu( ctx, x_left );
    x = ggml_mul( ctx, x_left, x_right );

    x = torch_nn_linear( ctx, gating->linear_out, x );
    return x;
}

void get_weights( WeightLoader * loader, std::string path,
        moshi_activation_gating_t * gating ) {
    get_weights( loader, path + "linear_in.", gating->linear_in );
    get_weights( loader, path + "linear_out.", gating->linear_out );
}

