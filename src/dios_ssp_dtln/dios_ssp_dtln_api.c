/* Copyright (C) 2017 Beijing Didi Infinity Technology and Development Co.,Ltd.
All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

	http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Description: Noise reduction algorithm is based on MCRA noise estimation method,
which includes the following steps:
1. The data goes through the analysis window and FFT.
2. MCRA estimation of noise.
3. Calculate gain factor according to MMSE criterion.
4. Noise reduction based on gain and IFFT.
Details can be found in "Noise Estimation by Minima Controlled Recursive
Averaging for Robust Speech Enhancement" and "Noise Spectrum Estimation in
Adverse Environments: Improved Minima Controlled Recursive Averaging"
==============================================================================*/

#include "dios_ssp_dtln_api.h"
#include "tflite/c_api.h"

typedef struct {
    int frame_len;
    int m_wav_len2;
    int m_max_pack_len;
    int m_shift_size;
    int m_fft_size;
    int m_frame_sum;
    int m_sp_size;
    float *m_wav_buffer;
    float *m_out_buffer;
    float *m_win_wav;
    float* m_re;
    float* m_im;

    //anaylsis window & synthesis window
    float *m_ana_win;
    float *m_syn_win;
    float *m_norm_win;

    //stft_process & istft_process
    float *fftin_buffer;
    float *fft_out;
    void *rfft_param;

    float *m_dtln_out_data;

    // dtln
    float states[DTLN_MODEL_NUM][DTLN_FRAME_SIZE];

    TfLiteTensor *inDetails[DTLN_MODEL_NUM][2];
    const TfLiteTensor *outDetails[DTLN_MODEL_NUM][2];

    TfLiteInterpreter *interpreter[DTLN_MODEL_NUM];
    TfLiteModel *model[DTLN_MODEL_NUM];

    TfLiteInterpreterOptions *options;

    float *m_mag;
    float *m_phase;
    float *m_dtln_out_freq;
    float *m_dtln_out_time;
} objDTLN;

void* dios_ssp_dtln_init_api(const char *modelpath[], int frame_len)
{
    if (NULL == modelpath ||
            NULL == modelpath[0] || NULL == modelpath[1] ||
            strlen(modelpath[0]) <= 0 || strlen(modelpath[1]) <= 0) {
        return NULL;
    }

    objDTLN *srv = (objDTLN *)malloc(sizeof(objDTLN));
    if (NULL == srv) {
        return NULL;
    }
    memset(srv, 0, sizeof(objDTLN));

    srv->frame_len = frame_len;
    srv->m_max_pack_len = DTLN_SAMPLE_RATE + 512;
    srv->m_shift_size = DTLN_FRAME_SHIFT;
    srv->m_fft_size = DTLN_FRAME_SIZE;
    srv->m_sp_size = srv->m_fft_size / 2 +1;
    srv->m_frame_sum = 0;
    srv->m_wav_len2 = 0;
    // 输入缓存
    srv->m_wav_buffer = (float *)calloc(srv->m_max_pack_len, sizeof(float));
    if (NULL == srv->m_wav_buffer) {
        dios_ssp_dtln_uninit_api(srv);
        return NULL;
    }

    // 输出缓存
    srv->m_out_buffer = (float *)calloc(srv->m_max_pack_len, sizeof(float));
    if (NULL == srv->m_out_buffer) {
        dios_ssp_dtln_uninit_api(srv);
        return NULL;
    }

    // 加窗结果 / fft输入 / ifft输出
    srv->m_win_wav = (float *)calloc(srv->m_fft_size, sizeof(float));
    if (NULL == srv->m_win_wav) {
        dios_ssp_dtln_uninit_api(srv);
        return NULL;
    }

    // fft实部
    srv->m_re = (float *)calloc(srv->m_fft_size, sizeof(float));
    if (NULL == srv->m_re) {
        dios_ssp_dtln_uninit_api(srv);
        return NULL;
    }

    // fft虚部
    srv->m_im = (float *)calloc(srv->m_fft_size, sizeof(float));
    if (NULL == srv->m_im) {
        dios_ssp_dtln_uninit_api(srv);
        return NULL;
    }

    //
    srv->fft_out = (float *)calloc(srv->m_fft_size, sizeof(float));
    if (NULL ==	srv->fft_out) {
        dios_ssp_dtln_uninit_api(srv);
        return NULL;
    }

    srv->fftin_buffer = (float*)calloc(srv->m_fft_size, sizeof(float));
    if (NULL ==	srv->fftin_buffer) {
        dios_ssp_dtln_uninit_api(srv);
        return NULL;
    }

    srv->m_ana_win = (float *)calloc(srv->m_fft_size, sizeof(float));
    if (NULL ==	srv->m_ana_win) {
        dios_ssp_dtln_uninit_api(srv);
        return NULL;
    }

    srv->m_syn_win = (float *)calloc(srv->m_fft_size, sizeof(float));
    if (NULL ==	srv->m_syn_win) {
        dios_ssp_dtln_uninit_api(srv);
        return NULL;
    }

    srv->m_norm_win = (float *)calloc(srv->m_fft_size, sizeof(float));
    if (NULL ==	srv->m_norm_win) {
        dios_ssp_dtln_uninit_api(srv);
        return NULL;
    }

    srv->m_dtln_out_data = (float *)calloc(2 * srv->frame_len, sizeof(float));
    if (NULL == srv->m_dtln_out_data) {
        dios_ssp_dtln_uninit_api(srv);
        return NULL;
    }

    srv->rfft_param = dios_ssp_share_rfft_init(srv->m_fft_size);
    if (NULL ==	srv->rfft_param) {
        dios_ssp_dtln_uninit_api(srv);
        return NULL;
    }

    srv->options = TfLiteInterpreterOptionsCreate();
    if (NULL == srv->options) {
        dios_ssp_dtln_uninit_api(srv);
        return NULL;
    }
    TfLiteInterpreterOptionsSetNumThreads(srv->options, 1);

    srv->m_mag = (float *)calloc(srv->m_sp_size, sizeof(float));
    if (NULL == srv->m_mag) {
        dios_ssp_dtln_uninit_api(srv);
        return NULL;
    }

    srv->m_phase = (float *)calloc(srv->m_sp_size, sizeof(float));
    if (NULL == srv->m_phase) {
        dios_ssp_dtln_uninit_api(srv);
        return NULL;
    }

    srv->m_dtln_out_freq = (float *)calloc(srv->m_sp_size, sizeof(float));
    if (NULL == srv->m_dtln_out_freq) {
        dios_ssp_dtln_uninit_api(srv);
        return NULL;
    }

    srv->m_dtln_out_time = (float *)calloc(srv->m_fft_size, sizeof(float));
    if (NULL ==	srv->m_dtln_out_time) {
        dios_ssp_dtln_uninit_api(srv);
        return NULL;
    }

    for (int i = 0; i < DTLN_MODEL_NUM; i++) {
        // load model
        srv->model[i] = TfLiteModelCreateFromFile(modelpath[i]);
        if (NULL == srv->model[i]) {
            dios_ssp_dtln_uninit_api(srv);
            return NULL;
        }

        // create interpreter
        srv->interpreter[i] = TfLiteInterpreterCreate(srv->model[i], srv->options);
        if (NULL == srv->interpreter[i]) {
            dios_ssp_dtln_uninit_api(srv);
            return NULL;
        }

        // allocate tensor buffers
        if (kTfLiteOk != TfLiteInterpreterAllocateTensors(srv->interpreter[i])) {
            dios_ssp_dtln_uninit_api(srv);
            return NULL;
        }

        srv->inDetails[i][0]  = TfLiteInterpreterGetInputTensor(srv->interpreter[i], 0);
        srv->inDetails[i][1]  = TfLiteInterpreterGetInputTensor(srv->interpreter[i], 1);
        srv->outDetails[i][0] = TfLiteInterpreterGetOutputTensor(srv->interpreter[i], 0);
        srv->outDetails[i][1] = TfLiteInterpreterGetOutputTensor(srv->interpreter[i], 1);

        memset(srv->states[i], 0, sizeof(float)*DTLN_FRAME_SIZE);
    }

    return srv;
}

int dios_ssp_dtln_reset_api(void* ptr)
{
    if (NULL == ptr) {
        return -1;
    }

    objDTLN *srv = (objDTLN*)ptr;
    int i = 0;
    int j = 0;

    srv->m_wav_len2 = 0;
    srv->m_frame_sum = 0;
    memset(srv->m_wav_buffer, 0, sizeof(float)*srv->m_max_pack_len);
    memset(srv->m_out_buffer, 0, sizeof(float)*srv->m_max_pack_len);
    memset(srv->m_dtln_out_data, 0, sizeof(float)*2*srv->frame_len);
    memset(srv->m_win_wav, 0, sizeof(float)*srv->m_fft_size);
    memset(srv->m_re, 0, sizeof(float)*srv->m_fft_size);
    memset(srv->m_im, 0, sizeof(float)*srv->m_fft_size);

    for (i = 0; i < srv->m_fft_size; i++) {
        srv->m_ana_win[i] = 0.54f - 0.46f * (float)cos( (2*i)*PI / (srv->m_fft_size-1) );
        srv->m_norm_win[i] = srv->m_ana_win[i] * srv->m_ana_win[i];
    }

    int m_block_num = srv->m_fft_size / srv->m_shift_size;
    float temp = 0.0f;
    for (i = 0; i <srv->m_fft_size; i++) {
        temp = 0;
        for (j = 0; j < m_block_num; ++j ) {
            temp += srv->m_norm_win[i+j*srv->m_shift_size];
        }
        srv->m_norm_win[i] = 1.0f / temp;
    }
    for (i = 0; i < srv->m_shift_size; ++i ) {
        for (j = 1; j < m_block_num; ++j ) {
            srv->m_norm_win[i+j*srv->m_shift_size] = srv->m_norm_win[i];
        }
    }
    for (i = 0; i < srv->m_fft_size; ++i ) {
        srv->m_syn_win[i] = srv->m_norm_win[i] * srv->m_ana_win[i];
    }

    return 0;
}

void dtln_add_ana_win(objDTLN* srv, float *x, float *x_win )
{
    int i;
    for (i = 0; i < srv->m_fft_size; ++i ) {
        x_win[i] = x[i] * srv->m_ana_win[i];
    }
}

void dtln_add_syn_win(objDTLN* srv, float *x, float *x_win )
{
    int i;
    for (i = 0; i < srv->m_fft_size; ++i ) {
        x_win[i] = x[i] * srv->m_syn_win[i];
    }
}

void dtln_calc_mag(objDTLN* srv, float *real, float *imag, float *mag)
{
    int i;
    for (i = 0; i < srv->m_sp_size; i++) {
        mag[i] = sqrtf(real[i]*real[i] + imag[i]*imag[i]);
    }
}

void dtln_calc_phase(objDTLN* srv, float *real, float *imag, float *phase)
{
    int i;
    for (i = 0; i < srv->m_sp_size; i++) {
        phase[i] = atan2f(imag[i], real[i]);
    }
}

int dtln_process(objDTLN* srv, float *in_data, float *out_data)
{
    int i;

    // input (frame_len是一帧帧长，相当于一个ringbuf，来处理帧长与实际帧长之间的不匹配)
    for (i = 0 ; i < srv->frame_len; ++i ) {
        srv->m_wav_buffer[i+srv->m_wav_len2] = in_data[i];
    }
    srv->m_wav_len2 += srv->frame_len;

    // dtln loop
    int sta;
    for ( sta = 0; sta + srv->m_fft_size <= srv->m_wav_len2; sta += srv->m_shift_size ) {
        srv->m_frame_sum ++;
        // 1. add anaylsis window
        dtln_add_ana_win(srv, srv->m_wav_buffer+sta, srv->m_win_wav);

        // 2. stft
        dios_ssp_share_rfft_process(srv->rfft_param, srv->m_win_wav, srv->fft_out);
        for (i = 0; i < srv->m_sp_size-1; i++) {
            srv->m_re[i] = srv->fft_out[i];
        }

        srv->m_im[0] = srv->m_im[srv->m_sp_size-1] = 0.0;
        for (i = 1; i < srv->m_sp_size-1; i++) {
            srv->m_im[i] = -srv->fft_out[srv->m_fft_size - i];
        }

        // 3. dtln in freq domain
        dtln_calc_mag(srv, srv->m_re, srv->m_im, srv->m_mag);
        dtln_calc_phase(srv, srv->m_re, srv->m_im, srv->m_phase);

        TfLiteTensorCopyFromBuffer(srv->inDetails[0][0], srv->m_mag, srv->m_sp_size*sizeof(float));
        TfLiteTensorCopyFromBuffer(srv->inDetails[0][1], srv->states[0], srv->m_fft_size*sizeof(float));

        if (TfLiteInterpreterInvoke(srv->interpreter[0]) != kTfLiteOk) {
            printf("Error invoking detection model in freq domain\n");
        }

        TfLiteTensorCopyToBuffer(srv->outDetails[0][0], srv->m_dtln_out_freq, srv->m_sp_size*sizeof(float));
        TfLiteTensorCopyToBuffer(srv->outDetails[0][1], srv->states[0], srv->m_fft_size*sizeof(float));

        for (int i = 0; i < srv->m_sp_size; i++) {
            srv->m_re[i] = srv->m_mag[i] * srv->m_dtln_out_freq[i] * cosf(srv->m_phase[i]);
            srv->m_im[i] = srv->m_mag[i] * srv->m_dtln_out_freq[i] * sinf(srv->m_phase[i]);
        }

        // 4. istft
        srv->fftin_buffer[0] = srv->m_re[0];
        srv->fftin_buffer[srv->frame_len] = srv->m_re[srv->frame_len];
        for (i = 1; i < srv->frame_len; i++) {
            srv->fftin_buffer[i] = srv->m_re[i];
            srv->fftin_buffer[srv->m_fft_size - i] = -srv->m_im[i];
        }
        dios_ssp_share_irfft_process(srv->rfft_param, srv->fftin_buffer, srv->m_win_wav);
        for (i = 0; i < srv->m_fft_size; ++i) {
            srv->m_win_wav[i] = srv->m_win_wav[i] / srv->m_fft_size;//FFT coefficient 1/N
        }

        // 5. dtln in time domain
        TfLiteTensorCopyFromBuffer(srv->inDetails[1][0], srv->m_win_wav, srv->m_fft_size*sizeof(float));
        TfLiteTensorCopyFromBuffer(srv->inDetails[1][1], srv->states[1], srv->m_fft_size*sizeof(float));

        if (TfLiteInterpreterInvoke(srv->interpreter[1]) != kTfLiteOk) {
            printf("Error invoking detection model in time domain\n");
        }

        TfLiteTensorCopyToBuffer(srv->outDetails[1][0], srv->m_dtln_out_time, srv->m_fft_size*sizeof(float));
        TfLiteTensorCopyToBuffer(srv->outDetails[1][1], srv->states[1], srv->m_fft_size*sizeof(float));

        // 6. add synthesis window
        dtln_add_syn_win(srv, srv->m_dtln_out_time, srv->m_re);
        // 7. ola
        for (i = 0; i < srv->m_fft_size; ++i ) {
            srv->m_out_buffer[i+sta] += srv->m_re[i];
        }
    }

    for (i = 0; i < sta; ++i ) {
        if ( srv->m_out_buffer[i] > 32767 ) {
            out_data[i] = 32767;
        } else if ( srv->m_out_buffer[i] < -32768 ) {
            out_data[i] = -32768;
        } else {
            out_data[i] = srv->m_out_buffer[i];
        }
    }

    memmove( srv->m_out_buffer, srv->m_out_buffer+sta, sizeof(float)*(srv->m_fft_size-srv->m_shift_size) );
    memset( srv->m_out_buffer+srv->m_fft_size-srv->m_shift_size, 0, sizeof(float)*sta );
    memmove( srv->m_wav_buffer, srv->m_wav_buffer+sta, sizeof(float)*(srv->m_wav_len2-sta) );
    srv->m_wav_len2 -= sta;

    return 0;
}

int dios_ssp_dtln_process(void *ptr, float *in_data)
{
    if (NULL == ptr) {
        return -1;
    }

    int i;
    objDTLN* srv = (objDTLN*)ptr;

    dtln_process(srv, in_data, srv->m_dtln_out_data);

    for(i = 0; i < srv->frame_len; i++) {
        in_data[i] = srv->m_dtln_out_data[i];
    }

    return 0;
}

int dios_ssp_dtln_uninit_api(void* ptr)
{
    int ret;
    if (NULL == ptr) {
        return -1;
    }

    objDTLN *srv = (objDTLN *)ptr;

    if (srv->m_wav_buffer) {
        free(srv->m_wav_buffer);
    }
    if (srv->m_out_buffer) {
        free(srv->m_out_buffer);
    }
    if (srv->m_win_wav) {
        free(srv->m_win_wav);
    }
    if (srv->m_re) {
        free(srv->m_re);
    }
    if (srv->m_im) {
        free(srv->m_im);
    }
    if (srv->m_ana_win) {
        free(srv->m_ana_win);
    }
    if (srv->m_syn_win) {
        free(srv->m_syn_win);
    }
    if (srv->m_norm_win) {
        free(srv->m_norm_win);
    }
    if (srv->fftin_buffer) {
        free(srv->fftin_buffer);
    }
    if (srv->fft_out) {
        free(srv->fft_out);
    }
    if (srv->m_dtln_out_data) {
        free(srv->m_dtln_out_data);
    }
    if (srv->m_mag) {
        free(srv->m_mag);
    }

    if (srv->m_phase) {
        free(srv->m_phase);
    }

    if (srv->m_dtln_out_freq) {
        free(srv->m_dtln_out_freq);
    }

    if (srv->m_dtln_out_time) {
        free(srv->m_dtln_out_time);
    }

    if (srv->rfft_param) {
        ret = dios_ssp_share_rfft_uninit(srv->rfft_param);
        if (0 != ret) {
            srv->rfft_param = NULL;
        }
    }

    for (int i = 0; i < DTLN_MODEL_NUM; i++) {
        if (NULL != srv->interpreter[i]) {
            TfLiteInterpreterDelete(srv->interpreter[i]);
            srv->interpreter[i] = NULL;
        }
        if (NULL != srv->model[i]) {
            TfLiteModelDelete(srv->model[i]);
            srv->model[i] = NULL;
        }
    }
    TfLiteInterpreterOptionsDelete(srv->options);

    free(srv);

    return 0;
}
