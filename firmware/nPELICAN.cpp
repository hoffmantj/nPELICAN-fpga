#include <iostream>
#include <hls_math.h>
#include "nPELICAN.h"
#include "weights/weights.h"

psloglut_t psloglut(int index){
  static psloglut_t _table[N_TABLE_PSLOG];
  lut_pslog_init<psloglut_t,N_TABLE_PSLOG>(_table);
  return _table[index];
}

void dot4(input_t p1[4], input_t p2[4], input_t& dot) {
//#pragma HLS INLINE
//#pragma function instatiate

// Input in the form E, px, py, pz
dot = p1[0]*p2[0]-p1[1]*p2[1]-p1[2]*p2[2]-p1[3]*p2[3];

}

void nPELICAN(
    input_t model_input[(NPARTICLES)*4],
    input_t nobj,
    result_t model_out[1]
) {
    #pragma HLS ARRAY_RESHAPE variable=model_input complete dim=0
    #pragma HLS ARRAY_PARTITION variable=model_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=model_input,model_out 
//    #pragma HLS DATAFLOW 
    #pragma HLS PIPELINE II=1

    //pragmas for model weight arrays
    #pragma HLS ARRAY_PARTITION variable=batch1_2to2 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=w1_2to2 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=b1_2to2 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=b1_diag_2to2 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=batch2_2to0 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=w2_2to0 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=b2_2to0 complete dim=0


    if (nobj != 0 ) {
      if (nobj < NPARTICLES) {
        nobj += (NPARTICLES2 - NPARTICLES);
      }
      else {
        nobj = NPARTICLES2;
      }
    }
    //create array mask from number of particles in the event
    internal_t nobjmask[(NPARTICLES2)][(NPARTICLES2)];
    #pragma HLS ARRAY_PARTITION variable=nobjmask complete dim=0
    for(unsigned int i = 0; i < NPARTICLES2; i++){
      for(unsigned int j = 0; j < NPARTICLES2; j++){
        if(i < nobj && j < nobj){
          nobjmask[i][j] = 1;
        }
        else{
          nobjmask[i][j] = 0;
        }
      }
    }

    input_t dots[(NPARTICLES2)*(NPARTICLES2)];
    #pragma HLS ARRAY_PARTITION variable=dots complete dim=0
    input_t p1[(NPARTICLES2)][4];
    #pragma HLS ARRAY_PARTITION variable=p1 complete dim=0
    P1Prep: for (unsigned int i = 0; i < NPARTICLES; i++) {
    #pragma HLS unroll
      for (unsigned int k = 0; k < 4; k++){
      #pragma HLS unroll
        p1[(i + (NPARTICLES2 - NPARTICLES))][k] = model_input[i*(4)+k]*nobjmask[i][0];
      }
    }
    //add beam spurions 
    p1[0][0]   = 1.; p1[0][1]   = 0.; p1[0][2]   = 0.; p1[0][3]   = 1.;
    p1[1][0] = 1.; p1[1][1] = 0.; p1[1][2] = 0.; p1[1][3] = -1.;

    //fill input array
    //TODO: could run over only the upper triangle and copy since the array is symmetric
    for(unsigned int i = 0; i < NPARTICLES2; i++){
      #pragma HLS unroll
      for(unsigned int j = 0; j < NPARTICLES2; j++){
        #pragma HLS unroll
        Dot: dot4(p1[i], p1[j], dots[i*NPARTICLES2+j]);
      }
    }

   //psuedolog input encoder
   /*
    for(unsigned int i = 0; i < NPARTICLES2; i++){
      #pragma HLS unroll
      for(unsigned int j = 0; j < NPARTICLES2; j++){
        #pragma HLS unroll
        dots[i*NPARTICLES2+j] = (input_t) (psloglut(dots[i*NPARTICLES2+j]>>TABLE_FRACS));
      }
    }
    */

    //Do first batchnorm
    internal_t batch1[(NPARTICLES2)*(NPARTICLES2)];
    #pragma HLS ARRAY_PARTITION variable=batch1 complete dim=0
    for(unsigned int i = 0; i < NPARTICLES2; i++){
      #pragma HLS unroll
      for(unsigned int j = 0; j < NPARTICLES2; j++){
        #pragma HLS unroll
        batch1[i*NPARTICLES2+j] = ((dots[i*NPARTICLES2+j] - batch1_2to2[0]) * batch1_2to2[1] + batch1_2to2[2])*nobjmask[i][j];
      }
    }
    
    internal_t jmass = 0.;
    internal_t jdotp[NPARTICLES2] = {0};
    #pragma HLS ARRAY_PARTITION variable=jdotp complete dim=0

    // M_J = sum(dots)
    // J \cdot p_i = sum(dots row i)
    // these are the only things (besides the dots themselves) that we need from the aggregation
    //TODO: could reform this to only loop over the upper triangle and double off diagonal contributions
    for (unsigned int i = 0; i < NPARTICLES2; i++) {
    #pragma HLS unroll
      for (unsigned int j = 0; j < NPARTICLES2; j++) {
      #pragma HLS unroll
        AggMJ: jmass += batch1[i*NPARTICLES2+j];
        AggJdot: jdotp[j] += batch1[i*NPARTICLES2+j];
      }
    }

    //aggregation normalizations
    //TODO: could reform this to use smaller bitwith due to normalization being O(2^-10)
    jmass = jmass*invnave2;
    for( unsigned int i = 0; i < NPARTICLES2; i++){
    #pragma HLS unroll
      jdotp[i] = jdotp[i]*invnave;
    }

    internal_t T[NPARTICLES2][NPARTICLES2][6];
    #pragma HLS ARRAY_PARTITION variable=T complete dim=0
    for (unsigned int i = 0; i < NPARTICLES2; i++) {
    #pragma HLS unroll
      for (unsigned int j = 0; j < NPARTICLES2; j++) {
    #pragma HLS unroll
        for (unsigned int b = 0; b < 6; b++) {
    #pragma HLS unroll
          T[i][j][b] = 0;
        }
      }
    }

    //TODO: it's possible the following can be simplified to hold fewer arrays
    // J is the full jet four-vector
    // T0 = p_i \cdot p_j
    // T1 = (J \cdot p_i)\delta_{ij}
    // T2 = J \cdot p_j
    // T3 = J \cdot p_i
    // T4 = M_J
    // T5 = M_J \delta_{ij}
    for (unsigned int i = 0; i < NPARTICLES2; i++) {
    #pragma HLS unroll
      for (unsigned int j = 0; j < NPARTICLES2; j++) {
      #pragma HLS unroll
        LinEq2to2_0: T[i][j][0] = batch1[i*NPARTICLES2+j];
        LinEq2to2_1: T[i][j][4] = jmass*nobjmask[i][j];
        LinEq2to2_4: T[i][j][3] = jdotp[i];
        LinEq2to2_5: T[i][j][2] = jdotp[j];
      }
    }
    
    for (unsigned int i = 0; i < NPARTICLES2; i++) {
    #pragma HLS unroll
      LinEq2to2_2: T[i][i][5] = jmass*nobjmask[i][i];
      LinEq2to2_3: T[i][i][1] = jdotp[i];
    }
    
    //"dense" summation over aggregators
    internal_t Tp[NPARTICLES2][NPARTICLES2][NHIDDEN];
    #pragma HLS ARRAY_PARTITION variable=Tp complete dim=0
    
    // initialize with bias
    for (unsigned int i = 0; i < NPARTICLES2; i++) {
    #pragma HLS unroll
      for (unsigned int j = 0; j < NPARTICLES2; j++) {
      #pragma HLS unroll
        for (unsigned int h = 0; h < NHIDDEN; h++) {
        #pragma HLS unroll
          Tp[i][j][h] = b1_2to2[h]*nobjmask[i][j];
          }
        }
      }
      
    for (unsigned int i = 0; i < NPARTICLES2; i++){
    #pragma HLS unroll
      for (unsigned int h = 0; h < NHIDDEN; h++) {
      #pragma HLS unroll
        Tp[i][i][h] += b1_diag_2to2[h]*nobjmask[i][i];
      }
    }

    // 2->2 weights
    for (unsigned int i = 0; i < NPARTICLES2; i++) {
    #pragma HLS unroll
      for (unsigned int j = 0; j < NPARTICLES2; j++) {
      #pragma HLS unroll
        for (unsigned int h = 0; h < NHIDDEN; h++) {
        #pragma HLS unroll
          for (unsigned int b = 0; b < 6; b++) {
          #pragma HLS unroll
            Mult2to2: Tp[i][j][h] += w1_2to2[(h*6)+b]*T[i][j][b];
            
          }
        }
      }
    }

    // ReLU
    for (unsigned int i = 0; i < NPARTICLES2; i++) {
    #pragma HLS unroll
      for (unsigned int j = 0; j < NPARTICLES2; j++) {
      #pragma HLS unroll
        for (unsigned int h = 0; h < NHIDDEN; h++) {
        #pragma HLS unroll
          if (Tp[i][j][h] < 0.){
            Tp[i][j][h] = 0;
            }
        }
      }
    }

    //second batchnorm
    internal_t Tr[NPARTICLES2][NPARTICLES2][NHIDDEN];
    #pragma HLS ARRAY_PARTITION variable=Tr complete dim=0
    for (unsigned int i = 0; i < NPARTICLES2; i++) {
    #pragma HLS unroll
      for (unsigned int j = 0; j < NPARTICLES2; j++) {
      #pragma HLS unroll
        for (unsigned int h = 0; h < NHIDDEN; h++) {
        #pragma HLS unroll
            Tr[i][j][h] = ((Tp[i][j][h] - batch2_2to0[h][0]) * batch2_2to0[h][1] + batch2_2to0[h][2])*nobjmask[i][j];
        }
      }
    }
    
    // two aggregators for 2to0: total sum and trace
    internal_t R[NHIDDEN][2];
    #pragma HLS ARRAY_PARTITION variable=R complete dim=0
    
    for (unsigned int h = 0; h < NHIDDEN; h++) {
    #pragma HLS unroll
      for (unsigned int a = 0; a < 2; a++) {
      #pragma HLS unroll
        R[h][a] = 0;
      }
    }
    
    //total sum
    for (unsigned int h = 0; h < NHIDDEN; h++) {
    #pragma HLS unroll
      for (unsigned int i = 0; i < NPARTICLES2; i++) {
      #pragma HLS unroll
        for (unsigned int j = 0; j < NPARTICLES2; j++) {
        #pragma HLS unroll
            LinEq2to0: R[h][0] += (Tr[i][j][h]);
        }
      }
    }

    //total sum normalization
    for (unsigned int h = 0; h < NHIDDEN; h++) {
    #pragma HLS unroll
      R[h][0] = (R[h][0])*invnave2;
    }

    //trace and normalization
    for (unsigned int h = 0; h < NHIDDEN; h++) {
    #pragma HLS unroll
      for (unsigned int i = 0; i < NPARTICLES2; i++) {
      #pragma HLS unroll
        R[h][1] += Tr[i][i][h];
      }
      R[h][1] = (R[h][1])*invnave;
    }
    
    //Final 1D output
    internal_t Rp[NOUT];
    #pragma HLS ARRAY_PARTITION variable=Rp complete dim=0
    
    // initialize with bias
    for (unsigned int o = 0; o < NOUT; o++) {
    #pragma HLS unroll
      Rp[o] = b2_2to0[o];
    }
    
    // 2->0 weights
    for (unsigned int h = 0; h < NHIDDEN; h++) {
    #pragma HLS unroll
      for (unsigned int a = 0; a < 2; a++) {
      #pragma HLS unroll
        for (unsigned int o = 0; o < NOUT; o++) {
        #pragma HLS unroll
          Mult2to0: Rp[o] += w2_2to0[(h*2)+a*(NOUT)+o]*R[h][a];
        }
      }
    }

    model_out[0] = Rp[0];
}
