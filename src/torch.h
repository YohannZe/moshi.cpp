#pragma once

/*******************************************\
 *     Modules in PyTorch
 *
 * these are not complete implementations,
 * only enough to get moshi working.
\*******************************************/

/******************************\
 * torch.nn.Conv1d
\******************************/

struct torch_nn_conv1d_t {
    ggml_tensor * weight;
};

ggml_tensor * torch_nn_conv1d(
        ggml_context * ctx,
        torch_nn_conv1d_t * conv,
        ggml_tensor * x ) {
    // NOTE: these were not testable so not included
    //assert conv.stride[0] == 1
    //assert conv.padding[0] == 0
    //assert conv.dilation[0] == 1
    //assert conv.groups == 1
    //assert not conv.bias
    auto y = ggml_conv_1d( ctx, conv->weight, x, 1, 0, 1 );
    return y;
}

void get_weights( WeightLoader * loader, std::string path,
        torch_nn_conv1d_t * conv ) {
    // NOTE: ggml_conv_1d requires GGML_TYPE_F16 due to im2col requiring it
    auto n = loader->fetch( &conv->weight, path + "weight", (void*)ggml_conv_1d );
    assert( n );
}

/******************************\
 * torch.nn.LayerNorm
\******************************/

struct torch_nn_layer_norm_t {
    float eps;
    ggml_tensor * weight;
    ggml_tensor * bias;
};

ggml_tensor * torch_nn_layer_norm(
        ggml_context * ctx,
        torch_nn_layer_norm_t * norm,
        ggml_tensor * x ) {
    if ( x->type != GGML_TYPE_F32 )
        x = ggml_cast( ctx, x, GGML_TYPE_F32 );
    x = ggml_norm( ctx, x, norm->eps );
    x = ggml_mul( ctx, x, norm->weight );
    if ( norm->bias )
        x = ggml_add( ctx, x, norm->bias );
    return x;
}

void get_weights( WeightLoader * loader, std::string path,
        torch_nn_layer_norm_t * norm ) {
    auto n = loader->fetch( &norm->weight, path + "weight", (void*)ggml_mul );
    assert( n );
    // bias not required
    loader->fetch( &norm->bias, path + "bias", (void*)ggml_add );
}

/******************************\
 * torch.nn.Linear
\******************************/

struct torch_nn_linear_t {
    ggml_tensor * weight;
    ggml_tensor * bias;
};

ggml_tensor * torch_nn_linear(
        ggml_context * ctx,
        torch_nn_linear_t * linear,
        ggml_tensor * x ) {
    ggml_tensor * y = ggml_mul_mat( ctx, linear->weight, x );
    if ( linear->bias )
        y = ggml_add( ctx, y, linear->bias );
    return y;
}

void get_weights( WeightLoader * loader, std::string path,
        torch_nn_linear_t * linear ) {
    if ( loader->quantize ) {
        auto n = loader->fetch( &linear->weight, path + "weight", loader->qtype );
        assert( n );
    } else {
        auto n = loader->fetch( &linear->weight, path + "weight", (void*)ggml_mul_mat );
        assert( n );
    }
    // bias not required
    loader->fetch( &linear->bias, path + "bias", (void*)ggml_add );
}

// utility that only applies a subset of a linear module
ggml_tensor * torch_nn_linear_view(
        ggml_context * ctx,
        torch_nn_linear_t * linear,
        int offset,
        int width,
        ggml_tensor * x ) {
    auto weight = linear->weight;
    auto w_view = ggml_view_2d( ctx, weight,
        weight->ne[0], width,
        weight->nb[1],
        weight->nb[1] * offset );
    auto y = ggml_mul_mat( ctx, w_view, x );
    if ( linear->bias )
        y = ggml_add( ctx, y, linear->bias );
    return y;
}

/*****************************************************************************\
 * torch.nn.functional.scaled_dot_product_attention
\*****************************************************************************/

// scaled_dot_product_attention used -infinity which does not multiply against 0
// so changed to use a very large negative number, would be nice to have a 
// mathematical way to generate the bias from a mask, as opposed to a boolean
// operations as it was before, since ggml does not currently support them
ggml_tensor * torch_nn_functional_scaled_dot_product_attention(
        GraphContext & ctx,
        ggml_tensor * query,
        ggml_tensor * key,
        ggml_tensor * value,
        ggml_tensor * attn_mask ) {
    ggml_tensor * attn_bias = NULL;
    if (attn_mask) {
        // invert mask
        auto one = ctx.constant( 1.f );
        attn_bias = ggml_add( ctx, ggml_neg( ctx, attn_mask ), one );
        // max negative value
        // HACK: can't use infinity, so just use a very large number
        attn_bias = ggml_scale( ctx, attn_bias, -100000.0 );
    } else {
        attn_bias = NULL;
    }
    // if we need -inf, in theory we can just scale it by 2 or higher
    float scale_factor = 1.f / sqrtf( (float) query->ne[0] );
    auto attn_weight = ggml_mul_mat( ctx, key, query );
    attn_weight = ggml_soft_max_ext( ctx, attn_weight, attn_bias, scale_factor, 0.0f );
    value = ggml_cont( ctx, ggml_transpose( ctx, value ) );
    auto x = ggml_mul_mat( ctx, value, attn_weight );
    return x;
}

/*****************************************************************************\
 * custom scaled_dot_product_attention
 * 
 * 1) create a pattern
 * 2) index into the pattern to get the bias
 * 3) pass that bias to the custom scaled_dot_product_attention
\*****************************************************************************/

struct bias_pattern_t {
    int capacity;
    int t;
    int start; // = pattern->capacity * 2 - pattern->t
    own_ctx_tensor tensor;
};

int g_bias_pattern = 0;
void create_bias_pattern(
        ggml_backend * backend,
        bias_pattern_t & pattern,
        int capacity,
        int t,
        float hi = 1.f, float lo = 0.f
) {
    auto & tensor = pattern.tensor;
    int start = capacity * 2 - t;
    int width = start + capacity;
    // F16, not F32. ggml_soft_max_ext accepts either, but
    // ggml_compute_forward_flash_attn_ext_f16 reads the mask as ggml_fp16_t
    // unconditionally (ops.cpp: `const ggml_fp16_t * mp = ...`) while
    // ggml_flash_attn_ext asserts only ggml_is_contiguous(mask) — so an F32 mask is
    // silently reinterpreted as F16 and produces garbage. Both consumers are happy
    // with F16; only one is happy with F32. -INFINITY is representable in F16.
    tensor.new_tensor( GGML_NE( width, t ), GGML_TYPE_F16, backend );
    pattern.capacity = capacity;
    pattern.t = t;
    pattern.start = start;
    auto nelements = ggml_nelements( tensor );
    std::vector<ggml_fp16_t> values( nelements );
    const ggml_fp16_t f16_hi = ggml_fp32_to_fp16( hi );
    const ggml_fp16_t f16_lo = ggml_fp32_to_fp16( lo );
    for ( int j = 0; j < t; j++ ) {
        int toff = j * width;
        int right = start + 1 + j;
        for ( int i = 0; i < right; i++ ) {
            values[ toff + i ] = f16_hi;
        }
        for ( int i = right; i < width; i++ ) {
            values[ toff + i ] = f16_lo;
        }
        int b = t - j - 1;
        toff += capacity - 1;
        for ( int i = 0; i < b; i++ ) {
            values[ toff - i ] = f16_lo;
        }
    }
    ggml_backend_tensor_set( tensor, values.data(), 0, ggml_nbytes( tensor ) );
    g_bias_pattern++;
}

ggml_tensor * bias_pattern_index(
        ggml_context * ctx,
        bias_pattern_t & pattern,
        int offset
) {
    auto & tensor = pattern.tensor;
    if ( offset <= pattern.capacity )
        offset = pattern.start - offset;
    else
        // The wrapped branch must subtract the block width, exactly as the unwrapped
        // branch does: pattern.start is capacity*2 - t (create_bias_pattern), so the
        // two formulas have to agree at offset == capacity, and without `- t` they
        // differ by precisely t.
        //
        // Without it, from offset == capacity+1 onward each step is denied the slot
        // holding the immediately preceding block and allowed a slot that set_rows has
        // already overwritten with the *current* block -- a lookahead of t frames.
        //
        // Only bites when t > 1. For t == 1 the window equals the capacity, so once the
        // ring is full every slot is legitimately in range and the mask is all-ones;
        // any offset into the all-hi region happens to be correct. That is why the LM
        // (t=1, capacity=750) is unaffected while the Mimi encoder and decoder
        // (t=2, capacity=250) are wrong from ~10 s of audio onward.
        offset = pattern.capacity - ( offset % pattern.capacity ) - pattern.t;
    auto view = ggml_view_2d( ctx,
        tensor,
        pattern.capacity,
        pattern.t,
        tensor->nb[1],
        offset * tensor->nb[0] );
    auto cont = ggml_cont( ctx, view );
    return cont;
}

// Set to 0 to fall back to the manual softmax(QK^T)V path (for A/B).
#ifndef MOSHI_USE_FLASH_ATTN
#define MOSHI_USE_FLASH_ATTN 1
#endif

// scaled_dot_product_attention immediately followed by the "b h t d -> b t (h d)"
// rearrange that every call site performed identically.
//
// Fusing them is what makes flash attention worth it here: ggml_flash_attn_ext emits
// [DV, H, T, B], which is exactly what that rearrange produces, so the flash path is
// a bare reshape while the manual path needs a permute + ggml_cont.
//
// The manual path also had to do `ggml_cont(ggml_transpose(value))` because ggml_mul_mat
// structurally requires V^T (result[n,m] = sum_k a[k,n] b[k,m]). With a [D, capacity]
// cache and capacity=750 that transposes the ENTIRE cache on every frame, in every
// layer: nb00 != type_size and dst contiguous sends it down the strided branch of
// ggml_compute_forward_dup_bytes that ggml itself comments "this is not optimal - fix
// me", i.e. 2-byte memcpys — ~49 MB of traffic per frame across 16 layers, plus the
// scratch to hold it. ggml_flash_attn_ext consumes V untransposed and fuses the softmax,
// so all of that disappears.
ggml_tensor * torch_sdpa_rearranged(
        ggml_context * ctx,
        ggml_tensor * query,
        ggml_tensor * key,
        ggml_tensor * value,
        ggml_tensor * attn_bias ) {
    const float scale_factor = 1.f / sqrtf( (float) query->ne[0] );

#if MOSHI_USE_FLASH_ATTN
    // ggml_flash_attn_ext requires dim-0-contiguous q/k/v and a contiguous mask.
    const bool flash_ok =
        query->type == GGML_TYPE_F32 &&
        key->type == value->type &&
        query->nb[0] == ggml_type_size( query->type ) &&
        key->nb[0]   == ggml_type_size( key->type )   &&
        value->nb[0] == ggml_type_size( value->type ) &&
        // ggml_flash_attn_ext does NOT assert the mask dtype but its CPU kernel reads
        // it as ggml_fp16_t regardless, so an F32 mask silently yields garbage.
        ( !attn_bias || ( ggml_is_contiguous( attn_bias ) &&
                          attn_bias->type == GGML_TYPE_F16 ) );
    if ( flash_ok ) {
        // [DV, H, T, B] — already the rearranged layout
        auto x = ggml_flash_attn_ext( ctx, query, key, value, attn_bias,
                                      scale_factor, 0.0f, 0.0f );
        return ggml_reshape_3d( ctx, x, x->ne[0] * x->ne[1], x->ne[2], x->ne[3] );
    }
#endif

    auto attn_weight = ggml_mul_mat( ctx, key, query );
    attn_weight = ggml_soft_max_ext( ctx, attn_weight, attn_bias, scale_factor, 0.0f );
    value = ggml_cont( ctx, ggml_transpose( ctx, value ) );
    auto x = ggml_mul_mat( ctx, value, attn_weight );   // [D, T, H, B]
    // b h t d -> b t h d -> b t (h d)
    auto x2 = ggml_cont( ctx, ggml_permute( ctx, x, 0, 2, 1, 3 ) );
    return ggml_reshape_3d( ctx, x2, x2->ne[0] * x2->ne[1], x2->ne[2], x2->ne[3] );
}
