__global__ void NAME(TYPE *A __readonly __noalias,
                     TYPE *B __readonly __noalias,
                     TYPE *C __noalias,
                     int lda,
                     int ldb,
                     int ldc,
                     long stride_za,
                     long stride_zb,
                     long stride_zc,
                     long stride_ha,
                     long stride_hb,
                     long stride_hc,
                     int DS0, int DS1,
                     int SDD_K,
                     int SDD_off_width,
                     int *lut, int *locks, int nlocks) {
  /* ---------------- */
  /*    Prologue      */
  /* ---------------- */
  // program ids
  int pid0 = get_program_id(0);
  int pid1 = get_program_id(1);
  int pidz = get_program_id(2);
#ifdef SDD
  // load LUT header
  pid1 = pid1 + SDD_off_width;
  int blockidm[TM] = (0 ... TM) / BLOCK;
  int blockidn[TN] = (0 ... TN) / BLOCK;
  int offlutm[TM] = blockidm * (TN / BLOCK) * 4;
  int offlutn[TN] = blockidn * 4;
  int *header = lut + pid1 * (TM / BLOCK) * (TN / BLOCK) * 4;
  int z = *(header + 0);
  int i[TM] = *(header + 1 + offlutm);
  int j[TN] = *(header + 2 + offlutn);
  int AS1 = SDD_K / TZ;
  int lockid = select(TZ > 1, 1, 0);
  int offka = pid0 * AS1;
  int offkb = pid0 * AS1;
  int offmc = 0;
  int offnc = 0;
  int offpa = 0;
  int offpb = 0;
  int maxid = TZ;
  int offhc = 0;
  int offha = z;
  int offhb = z;
  int ram[TM] = i * BLOCK + ((0 ... TM) % BLOCK);
  int rbn[TN] = j * BLOCK + ((0 ... TN) % BLOCK);
#else
  // load LUT header
  int *header = lut + pid0 * 6;
  int offset = *(header + 0);
  int AS1 = *(header + 1);
  int column = *(header + 2);
  int depth = *(header + 3);
  int lockid = *(header + 4);
  int maxid = *(header + 5);
  int *pinc = lut + offset;
  int offhc = depth;
#ifdef DSD
  // output offset
  int offnc = pid1 * TN;
  int offmc = column * TM;
  int offpc = 0;
  // dense input offset
  int offnb = pid1 * TN;
  int offkb __multipleof(8) = *pinc;
  int offpb = 0;
  // sparse input offset
  int offma = 0;
  int offka = 0;
  long offpa __multipleof(8) = *(pinc + 1);
  offpa = offpa * BLOCK * BLOCK;
  int offha = 0;
  int offhb = depth;
#endif
#ifdef DDS
  // output offset
  int offmc = pid1 * TM;
  int offnc = column * TN;
  int offpc = 0;
  // dense input offset
  int offma = pid1 * TM;
  int offka __multipleof(8) = *pinc;
  int offpa = 0;
  // sparse input offset
  int offnb = 0;
  int offkb = 0;
  long offpb __multipleof(8) = *(pinc + 1);
  offpb = offpb * BLOCK * BLOCK;
  int offha = depth;
  int offhb = 0;
#endif
  int ram[TM] = offma + 0 ... TM;
  int rbn[TN] = offnb + 0 ... TN;
#endif
  // initialize a, b pointers
  int rka[TK] = offka + 0 ... TK;
  int rkb[TK] = offkb + 0 ... TK;
  TYPE *pa[TM, TK] = A + pidz * stride_za + offha * stride_ha + offpa + ram[:, newaxis] * STRIDE_AM + rka [newaxis, :] * STRIDE_AK;
  TYPE *pb[TK, TN] = B + pidz * stride_zb + offhb * stride_hb + offpb + rbn [newaxis, :] * STRIDE_BN + rkb[:, newaxis] * STRIDE_BK;
  // pre-fetch
#ifdef DDS
  bool checkam[TM, TK] = ram[:, newaxis] < DS0;
#else
  bool checkam[TM, TK] = AS1 > 0;
#endif
#ifdef DSD
  bool checkbn[TK, TN] = rbn [newaxis, :] < DS0;
#else
  bool checkbn[TK, TN] = AS1 > 0;
#endif
  TYPE a[TM, TK] = checkam ? *pa : 0;
  TYPE b[TK, TN] = checkbn ? *pb : 0;

  /* ---------------- */
  /*    Inner Loop    */
  /* ---------------- */
  // create result tile
  float acc[TM, TN] = 0;
  int step = TK;
  for (int k = AS1; k > 0; k -= step) {
    acc += a @b;
    // update pointers
#ifdef SDD
    int inc_a = TK * STRIDE_AK;
    int inc_b = TK * STRIDE_BK;
#else
    pinc += 2;
#ifdef DSD
    int inc_b __multipleof(8) = *pinc;
    int inc_a __multipleof(8) = *(pinc + 1);
    inc_b = inc_b * STRIDE_BK;
#endif
#ifdef DDS
    int inc_a __multipleof(8) = *pinc;
    int inc_b __multipleof(8) = *(pinc + 1);
    inc_a = inc_a * STRIDE_AK;
#endif
#endif
    pa += inc_a;
    pb += inc_b;
    // pre-fetch
    bool checkak[TM, TK] = k > TK;
    bool checkbk[TK, TN] = k > TK;
    bool checka[TM, TK] = checkam && checkak;
    bool checkb[TK, TN] = checkbk && checkbn;
    a = *? (checka)pa;
    b = *? (checkb)pb;
  }
  TYPE c[TM, TN] = acc;

  /* ---------------- */
  /*    Epilogue      */
  /* ---------------- */
  // initialize c pointers
#ifdef SDD
  bool checkc[TM, TN] = 1;
  // rematerialize
  int rr_blockidm[TM] = (0 ... TM) / BLOCK;
  int rr_blockidn[TN] = (0 ... TN) / BLOCK;
  int rr_offlutm[TM] = rr_blockidm * (TN / BLOCK) * 4;
  int rr_offlutn[TN] = rr_blockidn * 4;
  int off_bkid[TM, TN] = 3 + rr_offlutm[:, newaxis] + rr_offlutn [newaxis, :];
  int bkid[TM, TN] = *(header + off_bkid);
  long offpc[TM, TN] = bkid * BLOCK * BLOCK;
  // range within blocks
  int rcm[TM] = (0 ... TM) % BLOCK;
  int rcn[TN] = (0 ... TN) % BLOCK;
#else
  int rcm[TM] = offmc + 0 ... TM;
  int rcn[TN] = offnc + 0 ... TN;
#ifdef DSD
  bool checkc[TM, TN] = rcn [newaxis, :] < DS0;
#endif
#ifdef DDS
  bool checkc[TM, TN] = rcm[:, newaxis] < DS0;
#endif
#endif
  TYPE *pc[TM, TN] = C + offpc + offhc * stride_hc + pidz * stride_zc + rcm[:, newaxis] * STRIDE_CM + rcn [newaxis, :] * STRIDE_CN;
  // write-back directly
  if (lockid == 0) {
    *? (checkc)pc = c;
  }
  // accumulate partial result using spin-locks
  else {
    int *plock = locks + get_program_id(2) * nlocks * get_num_programs(1) + get_program_id(1) * nlocks + lockid - 1;
    int *pcount = plock + get_num_programs(2) * get_num_programs(1) * nlocks;
    for (int repeat = 1; repeat == 1; repeat = atomic_cas(plock, 0, 1))
      ;
    int count = *pcount;
    if (count == 0)
      *? (checkc)pc = c;
    else
      *? (checkc)pc = c + *? (checkc)pc;
    atomic_xchg(pcount, (count + 1) % maxid);
    atomic_xchg(plock, 0);
  }
}