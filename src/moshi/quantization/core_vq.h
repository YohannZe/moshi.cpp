#pragma once

/*****************************************************************************\
 *   moshi.quantization.core_vq.EuclideanCodebook
 * 
 * model locations:
 *   mimi.quantizer.rvq_first.vq.layers.*._codebook
 *   mimi.quantizer.rvq_rest.vq.layers.*._codebook
\*****************************************************************************/

struct moshi_EuclideanCodebook_t {
    ggml_tensor * embedding;
};

ggml_tensor * moshi_EuclideanCodebook_decode(
        ggml_context * ctx,
        moshi_EuclideanCodebook_t * codebook,
        ggml_tensor * codes ) {
    /*
    Given a tensor of codes of shape `[*]`, returns a tensor of shape `[*, D]`,
    corresponding to the centroids associated to each code index.
    */
    assert( (codes->type == GGML_TYPE_I64 || codes->type == GGML_TYPE_I32) );
    return ggml_get_rows( ctx, codebook->embedding, ggml_cont(ctx, codes) );
}

// Per-centroid ||e||^2 / 2, read from the codebook once at graph-build time.
// Frame-invariant, so it must not live in the graph.
static ggml_tensor * moshi_EuclideanCodebook_half_sqnorm(
        GraphContext & ctx,
        moshi_EuclideanCodebook_t * codebook ) {
    ggml_tensor * emb = codebook->embedding;      // [D, card]
    const int64_t D = emb->ne[0];
    const int64_t card = emb->ne[1];

    // Pull the weights to the host in whatever dtype they were stored in.
    std::vector<uint8_t> raw( ggml_nbytes( emb ) );
    ggml_backend_tensor_get( emb, raw.data(), 0, raw.size() );

    std::vector<float> flat( (size_t)D * card );
    if ( emb->type == GGML_TYPE_F32 ) {
        memcpy( flat.data(), raw.data(), raw.size() );
    } else if ( emb->type == GGML_TYPE_F16 ) {
        ggml_fp16_to_fp32_row( (const ggml_fp16_t*)raw.data(), flat.data(), (int64_t)D * card );
    } else if ( emb->type == GGML_TYPE_BF16 ) {
        ggml_bf16_to_fp32_row( (const ggml_bf16_t*)raw.data(), flat.data(), (int64_t)D * card );
    } else {
        const auto * tt = ggml_get_type_traits( emb->type );
        assert( tt && tt->to_float );
        tt->to_float( raw.data(), flat.data(), (int64_t)D * card );
    }

    std::vector<float> half_sq( card );
    for ( int64_t e = 0; e < card; e++ ) {
        const float * row = flat.data() + (size_t)e * D;
        // double accumulator: D=256 squares summed in f32 loses bits we don't need to lose
        double acc = 0.0;
        for ( int64_t d = 0; d < D; d++ ) acc += (double)row[d] * (double)row[d];
        half_sq[e] = (float)( 0.5 * acc );
    }
    return ctx.constant_f32( card, half_sq.data() );
}

ggml_tensor * moshi_EuclideanCodebook_encode(
        GraphContext & ctx,
        moshi_EuclideanCodebook_t * codebook,
        ggml_tensor * x ) {
    /*
    Given a tensor `x` of shape `[*, D]`, returns a tensor of integer codes of shape `[*]`.
    The codes are defined as the indexes of the centroids nearest to each vector in `x`.

    Uses the standard expansion instead of a brute-force distance matrix:

        argmin_e ||x - e||^2  =  argmin_e ( ||x||^2 - 2 x.e + ||e||^2 )
                              =  argmax_e ( x.e - ||e||^2 / 2 )

    ||x||^2 is constant across the codebook for a given x, so it drops out, and the
    factor 2 is folded into the precomputed norms since argmax is invariant to
    positive scaling. What remains is one [D,card] x [D,T] matvec plus a broadcast
    subtract.

    The previous implementation materialised the full [D, card] difference matrix per
    codebook via ggml_repeat_4d + sub + mul + sum_rows. With card=2048, D=256 that is
    ~2 MiB per intermediate and ~18 MiB of traffic per codebook; times 32 codebooks
    times 12.5 frames/s it dominated the encoder. Worse, ggml_repeat_4d always emits a
    REPEAT node even when the shape is unchanged, so line `b = repeat_4d(b, ...)` was a
    2 MiB memcpy of *static weights* on every frame. Because GraphContext::alloc() uses
    ggml_backend_alloc_ctx_tensors rather than a gallocr, those intermediates also held
    ~256 MiB of permanently-resident graph buffer.

    This form is also numerically better. The old code turned argmin into argmax with
    `1/(1+d)`, whose derivative d(1/(1+d)) = -dd/(1+d)^2 collapses distance gaps below
    ~2e-6 into F32 rounding for the d~50 typical here — precisely the near-tie regime
    where centroid choice is decided. It additionally inherited ggml_vec_argmax_f32's
    last-maximum tie-break, where torch.argmin takes the first.
    */

    ggml_tensor * emb = codebook->embedding;                  // [D, card]

    // x.e for every centroid -> [card, T]
    auto score = ggml_mul_mat( ctx, emb, ggml_cont( ctx, x ) );

    // ... minus ||e||^2/2, broadcast over T
    score = ggml_sub( ctx, score, moshi_EuclideanCodebook_half_sqnorm( ctx, codebook ) );

    return ggml_argmax( ctx, score );
}

void get_weights( WeightLoader * loader, std::string path,
        moshi_EuclideanCodebook_t * codebook ) {
    std::string name = path + "embedding";
    if ( loader->is_gguf ) {
        codebook->embedding = loader->get_tensor( name );
        assert( codebook->embedding );
    } else {
        auto sum_st = loader->find( path + "embedding_sum" );
        assert( sum_st );
        auto usage_st = loader->find( path + "cluster_usage" );
        assert( usage_st );

        NE ne;
        int n_dims = safetensor_get_shape( sum_st, ne );
        loader->add_alloc( &codebook->embedding, n_dims, ne, GGML_TYPE_F32, name );

        loader->add_init( [ sum_st, usage_st, codebook ] ( WeightLoader * loader ) {
            auto & scratch_ctx = *loader->scratch;
            auto embedding_sum = scratch_ctx.load( loader->stf, sum_st );
            auto cluster_usage = scratch_ctx.load( loader->stf, usage_st );
            auto clamp = ggml_clamp( scratch_ctx, cluster_usage, 1e-5f, INFINITY );
            auto cont = ggml_cont( scratch_ctx, ggml_transpose(scratch_ctx, clamp) );
            auto embedding = ggml_div( scratch_ctx, embedding_sum, cont );
            scratch_ctx.build_forward_expand( embedding, codebook->embedding );
            scratch_ctx.compute();
        });
    }
}


/*****************************************************************************\
 *   moshi.quantization.core_vq.VectorQuantization
 * 
 * model locations:
 *   mimi.quantizer.rvq_first.vq.layers.*
 *   mimi.quantizer.rvq_rest.vq.layers.*
\*****************************************************************************/

struct moshi_vq_t {
    own_ptr<moshi_EuclideanCodebook_t> _codebook;
};

ggml_tensor * moshi_vq_decode(
        ggml_context * ctx,
        moshi_vq_t * vq,
        ggml_tensor * codes ) {
    /* Converts integer codes into quantized vectors. */
    auto quantized = moshi_EuclideanCodebook_decode( ctx, vq->_codebook, codes );
    quantized = ggml_permute( ctx, quantized, 1, 0, 2, 3 );
    quantized = ggml_cont( ctx, quantized );
    return quantized;
}

ggml_tensor * moshi_vq_encode(
        GraphContext & ctx,
        moshi_vq_t * vq,
        ggml_tensor * x ) {
    /* Encodes `x` into discrete integer codes. */
    //x = self._rearrange_input(x)
    x = ggml_permute( ctx, x, 1, 0, 2, 3 );
    // x = self.project_in(x) was identity
    auto codes = moshi_EuclideanCodebook_encode( ctx, vq->_codebook, x);
    return codes;
}

void get_weights( WeightLoader * loader, std::string path, moshi_vq_t * vq ) {
    get_weights( loader, path + "_codebook.", vq->_codebook );
}

/*****************************************************************************\
 *   moshi.quantization.core_vq.ResidualVectorQuantization
 * 
 * model locations:
 *   mimi.quantizer.rvq_first.vq
 *   mimi.quantizer.rvq_rest.vq
\*****************************************************************************/

struct moshi_residual_vq_t {
    own_ptr_vector<moshi_vq_t> layers;
};

ggml_tensor * moshi_residual_vq_decode(
        ggml_context * ctx,
        moshi_residual_vq_t * rvq,
        ggml_tensor * codes ) {
    /* Converts the integer codes into quantized vectors. */
    auto T =       codes->ne[0];
    auto B =       codes->ne[1];
    auto K =       codes->ne[2];
    auto Bstride = codes->nb[1];
    auto Kstride = codes->nb[2];

    ggml_tensor * quantized = NULL;
    for ( size_t idx = 0; idx < rvq->layers.size() && idx < (size_t)K; idx++ ) {
        auto layer = rvq->layers[idx];

        // select one K
        auto layer_codes = ggml_view_3d( ctx, codes,
            T, B, 1,
            Bstride, Kstride,
            Kstride * idx );

        auto decoded = moshi_vq_decode( ctx, layer, layer_codes );

        if (quantized)
            quantized = ggml_add( ctx, quantized, decoded );
        else
            quantized = decoded;
    }

    return quantized;
}

ggml_tensor * moshi_residual_vq_encode(
        GraphContext & ctx,
        moshi_residual_vq_t * rvq,
        ggml_tensor * x,
        int n_q ) {
    auto residual = x;
    ggml_tensor * out_indices = NULL;
    if ( ! n_q )
        n_q = (int) rvq->layers.size();
    for ( int i = 0; i < n_q; i++ ) {
        auto layer = rvq->layers[i];
        auto indices = moshi_vq_encode( ctx, layer, residual );
        auto quantized = moshi_vq_decode( ctx, layer, indices );
        indices = ggml_cast( ctx, indices, GGML_TYPE_F32 );
        residual = ggml_sub( ctx, residual, quantized );
        // we don't have torch.stack, but concat on 3rd dimension is the same
        // dimensions being [T, B, K] with B batch as 1, K is our codes
        if ( ! out_indices )
            out_indices = indices;
        else
            out_indices = ggml_concat( ctx, out_indices, indices, 2 );
    }
    return out_indices;
}

void get_weights( WeightLoader * loader, std::string path, moshi_residual_vq_t * rvq ) {
    for ( size_t i = 0; i < rvq->layers.size(); i++ ) {
        get_weights( loader, path + "layers." + std::to_string(i) + ".", rvq->layers[i] );
    }
}
