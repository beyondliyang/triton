// compute softmax + cross-entropy loss
// of logits L \in R^{M x N} for indices IDX \in R^M
// stores negative logprobs in PROBS \in R^{M x N}
__global__ void forward(TYPE *L, TYPE *PROBS, long *IDX, TYPE *LOSS, int N) {
  // row and columns handled by this kernel
  int row = get_program_id(0);
  int cols[BLOCK] = 0 ... BLOCK;
  // loads logit and compute negative log-probability in FP32
  TYPE logits[BLOCK] = (cols < N) ? *(L + row * N + cols) : -INFINITY;
  float shifted[BLOCK] = logits - logits[max];
  float probs[BLOCK] = log(exp(shifted)[+]) - shifted;
  // store negative log-probabilities in PROBS
  // and cross-entropy loss in LOSS
  *? (cols < N)(PROBS + row * N + cols) = probs;
  __debug_barrier();
  int idx = *(IDX + row);
  *(LOSS + row) = *(PROBS + (idx + row * N));
}

__global__ void backward(TYPE *neg_logprobs, long *indices, TYPE *dneg_logprobs, int n_cols) {

  int row = get_program_id(0);
  // pointer arithmetic
  bool check[BLOCK] = ((0 ... BLOCK) < n_cols);
  int offset[BLOCK] = row * n_cols + 0 ... BLOCK;
  TYPE *px[BLOCK] = neg_logprobs + offset;
  long local_ind = *(indices + row);
  TYPE local_dn = *(dneg_logprobs + row);
  // We know d(-log(p[i])/dlogit[k] = -id_mat[i,k] + p[k]
  // and we have -log(p[k]) stored, so this is easy
  TYPE intermediate[BLOCK] = check ? exp(-(float[BLOCK]) * px) : 0;
  // selected_logit_idx is selected logit index for our token
  bool find_one[BLOCK] = ((0 ... BLOCK) == local_ind);
  intermediate = intermediate - ((TYPE[BLOCK])find_one);
  // multiply by dneg_logprobs
  *? (check)px = intermediate * local_dn;
}