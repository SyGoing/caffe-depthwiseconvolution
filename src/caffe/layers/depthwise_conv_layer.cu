#include <algorithm>
#include <cfloat>
#include "caffe/layers/depthwise_conv_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


	/*2017.11.10 
	Author:SyGoing(YangShu)
	*/
	//Depthwise Forward Convolution Kernel Function For GPU version
	template <typename Dtype>
	__global__ void DethwiseForwardGPUkernel(
		const Dtype *input,const int num_input,const int in_width,const int in_height,
		const Dtype *kernel,const int kernel_w,const int kernel_h,const int stride,
		const int pad,const int out_width,const int out_height,const int num_output,
	    Dtype *output, const int outputs, const Dtype* const bias, const bool bias_term_){

		int thread_id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
		if (thread_id >= outputs) return;

		//计算当前output像素点的四维索引索引
		const int width= thread_id % out_width;
		const int height = (thread_id / out_width) % out_height;
		const int channel = (thread_id / out_width / out_height) % num_output;
		const int batchID = thread_id / out_width / out_height / num_output;//batch size

		const int in_d =channel;

		const int input_offset_temp = (batchID * num_input + in_d) * (in_height * in_width);//当前output channel对应的input channel 的指针

		const int input_height_start = height * stride - pad;
		const int input_width_start = width* stride - pad;
		const int input_height_end = input_height_start + kernel_h;
		const int input_width_end = input_width_start + kernel_w;

		float sum = 0;
		if (input_height_start >= 0 && input_width_start >= 0 &&
			input_height_end < in_height && input_width_end < in_width)
		{
            #pragma unroll
			for (int f_r = 0; f_r < kernel_h; ++f_r) {
				const int in_r = input_height_start + f_r;
				#pragma unroll
				for (int f_c = 0; f_c < kernel_w; ++f_c) {
					const int in_c = input_width_start + f_c;

					const int input_offset =
						(input_offset_temp)+(in_r * in_width) + in_c;
					const int filter_offset = f_c + kernel_w * f_r + channel*kernel_w*kernel_h;
					sum += (*(input + input_offset)) * (*(kernel + filter_offset));
				}
			}
		}
		else {
			#pragma unroll
			for (int f_r = 0; f_r < kernel_h; ++f_r) {
				const int in_r = input_height_start + f_r;
				#pragma unroll
				for (int f_c = 0; f_c < kernel_w; ++f_c) {
					const int in_c = input_width_start + f_c;

					if (in_r >= 0 && in_r < in_height && in_c >= 0 && in_c < in_width) {
						const int in_c = input_width_start + f_c;

						const int input_offset =
							(input_offset_temp)+(in_r * in_width) + in_c;

						const int filter_offset = f_c + kernel_w * f_r + channel*kernel_w*kernel_h;
						sum += (*(input + input_offset)) * (*(kernel + filter_offset));
					}
				}
			}
		}

		//是否有偏置
		if (bias_term_) {
			sum += bias[channel];
		}

		output[thread_id] = sum;
	}

	//2017.11.10 SyGoing Add
	//overload the Forward_gpu For DepthwiseLayer
	template<typename Dtype>
	void DepthwiseConvLayer<Dtype>::Forward_gpu(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		const Dtype* weight = this->blobs_[0]->gpu_data();

		int* kernel_shape_data = this->kernel_shape_.mutable_cpu_data();
		int* stride_data = this->stride_.mutable_cpu_data();
		int* pad_data = this->pad_.mutable_cpu_data();

		for (int i = 0; i < bottom.size(); ++i) {
			const Dtype* bottom_data = bottom[i]->gpu_data();
			Dtype* top_data = top[i]->mutable_gpu_data();
			const int count = top[i]->count();
			vector<int> shape_ = bottom[i]->shape();
			const int channels_ = shape_[1];
			const int height_ = shape_[2];
			const int width_ = shape_[3];


			//这一块是用的数据拷贝
			const int kernel_h_ = kernel_shape_data[0];
			const int kernel_w_ = kernel_shape_data[1];
			const int stride = stride_data[0];
			const int pad = pad_data[0];

			const int conved_height = this->output_shape_[0];
			const int conved_width = this->output_shape_[1];

			const bool bias_term_ = this->bias_term_;


			

			if (bias_term_) {
				const Dtype* const bias = this->blobs_[1]->gpu_data();
				DethwiseForwardGPUkernel<Dtype> << <SCAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
					bottom_data, channels_, width_, height_, weight, kernel_h_, kernel_w_, stride,
					pad, conved_width, conved_height, channels_,top_data, count, bias, bias_term_);
			}
			else {
				DethwiseForwardGPUkernel<Dtype> << <SCAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
					bottom_data, channels_, width_, height_, weight, kernel_h_, kernel_w_, stride,
					pad, conved_width, conved_height, channels_,top_data, count, 0, bias_term_);
			}

			//2017.11.10 注释不用
			/*stupid method 
			   
			*/
			//for (int n = 0; n < this->num_; ++n) {
			//	for (int c = 0; c < this->channels_; ++c){
			//		const Dtype* const bottom_slice = bottom_data + (n *  this->channels_ + c) * bottom[i]->shape()[2] * bottom[i]->shape()[3];
			//		const Dtype* const weight_slice = weight + c * kernel_shape_data[0] * kernel_shape_data[1];
			//		Dtype*  top_slice = top_data + (n *  this->channels_ + c) * this->output_shape_[0] * this->output_shape_[1];
			//		
			//		//2017.11.08  
			//		this->mforward_gpu_gemm(bottom_slice, weight_slice, top_slice);
			//	}
			//	if (this->bias_term_) {
			//		const Dtype* bias = this->blobs_[1]->gpu_data();
			//		this->mforward_gpu_bias(top_data + n * this->top_dim_, bias);
			//	}
			//}
		}
	}

	template <typename Dtype>
	__global__ void ConvBackward(const int nthreads,
		const Dtype* const top_diff,
		const int num, const int channels, const int height,
		const int width, const int conved_height, const int conved_width,
		const int kernel_h, const int kernel_w, const int stride_h,
		const int stride_w, const int pad_h, const int pad_w,
		Dtype* const bottom_diff,
		const Dtype* const weight) {

		CUDA_KERNEL_LOOP(index, nthreads) {
			const int w = index % width + pad_w;
			const int h = (index / width) % height + pad_h;
			const int c = (index / width / height) % channels;
			const int n = index / width / height / channels;

			const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
			const int phend = min(h / stride_h + 1, conved_height);
			const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
			const int pwend = min(w / stride_w + 1, conved_width);

			const int khstart = (h >= kernel_h) ? ((h - kernel_h) % stride_h) + (kernel_h - stride_h) : h;
			const int kwstart = (w >= kernel_w) ? ((w - kernel_w) % stride_w) + (kernel_w - stride_w) : w;

			Dtype gradient = 0;
			const Dtype* const top_diff_slice =
				top_diff + (n * channels + c) * conved_height * conved_width;

			const Dtype* const weight_slice = weight + c * kernel_h * kernel_w;

			//		if (index==2) {
			//			printf("w%d h%d c%d n%d \n",w,h,c,n);
			//			printf("phstart%d phend%d pwstart%d pwend%d \n",phstart,phend,pwstart,pwend);
			//		}

			for (int ph = phstart; ph < phend; ++ph) {
				for (int pw = pwstart; pw < pwend; ++pw) {
					int kh = khstart - (ph - phstart)*stride_h;
					int kw = kwstart - (pw - pwstart)*stride_w;
					gradient += top_diff_slice[ph * conved_width + pw] * weight_slice[kh*kernel_w + kw];

					//						if (index==2) {
					//							printf("pos:ph%d pw%d kh%d kw%d\n",ph,pw,kh,kw);
					//							printf("cal:top_diff%f weight%f\n",top_diff_slice[ph * conved_width + pw],weight_slice[kh*kernel_w+kw]);
					//				//			printf("cal:top_diff%f weight%f\n",top_diff_slice[ph * conved_width + pw],weight_slice[kh*kernel_w+kw]);
					//						}
				}
			}
			bottom_diff[index] = gradient;
		}
	}

	__device__ float atomicAddme(float* address, float val)
	{
		return atomicAdd(address, val);
	}

	__device__ double atomicAddme(double* address, double val)
	{
		unsigned long long int* address_as_ull =
			(unsigned long long int*)address;
		unsigned long long int old = *address_as_ull, assumed;
		do {
			assumed = old;
			old = atomicCAS(address_as_ull, assumed,
				__double_as_longlong(val +
				__longlong_as_double(assumed)));
		} while (assumed != old);
		return __longlong_as_double(old);
	}



#define DIVIDE_CEIL(a,b) a/b+((a/b*b)<a)


	template <typename Dtype>
	__global__ void ConvBackwardWeight(const int nthreads,
		const Dtype* const top_diff,
		const int num, const int channels, const int height,
		const int width, const int conved_height, const int conved_width,
		const int kernel_h, const int kernel_w, const int stride_h,
		const int stride_w, const int pad_h, const int pad_w,
		Dtype* const weight_diff,
		const Dtype* const bottom_data) {

		CUDA_KERNEL_LOOP(index, nthreads) {
			const int kw = index % kernel_w;
			const int kh = (index / kernel_w) % kernel_h;
			const int c = index / kernel_w / kernel_h;

			//		if (index==5) {
			//			printf("kh%d kw%d kc%d\n",kh,kw,c);
			//		}
			Dtype gradient = 0;
			for (int n = 0; n<num; n++) {

				const Dtype* const top_diff_slice = top_diff + (n * channels + c) * conved_height * conved_width;
				const Dtype* const bottom_data_slice = bottom_data + (n * channels + c) * height * width;


				const int phstart = max(DIVIDE_CEIL((pad_h - kh), stride_h), 0);
				const int phend = min(DIVIDE_CEIL((height + pad_h - kh), stride_h), conved_height);

				const int pwstart = max(DIVIDE_CEIL((pad_w - kw), stride_w), 0);

				const int pwend = min(DIVIDE_CEIL((width + pad_w - kw), stride_w), conved_width);
				//			if (index==5) {
				//				printf("phstart%d phend%d pwstart%d pwend%d \n",phstart,phend,pwstart,pwend);
				//			}
				//			
				for (int ph = phstart; ph<phend; ph++){
					for (int pw = pwstart; pw<pwend; pw++){
						const int h = ph*stride_h + kh - pad_h;
						const int w = pw*stride_w + kw - pad_w;
						gradient += top_diff_slice[ph * conved_width + pw] * bottom_data_slice[h*width + w];
						//					if (index==5) {
						//						printf("n%d h%d w%d ph%d pw%d topdiff%f bottomdata%f\n",n,h,w,ph,pw,top_diff_slice[ph * conved_width + pw],bottom_data_slice[h*width+w]);
						//			//			printf("phstart%d phend%d pwstart%d pwend%d \n",phstart,phend,pwstart,pwend);
						//					}
					}
				}
			}
			weight_diff[c * kernel_h * kernel_w + kh*kernel_w + kw] += gradient;
		}
	}

	template <typename Dtype>
	__global__ void ConvBackwardBias(const int nthreads,
		const Dtype* const top_diff,
		const int num, const int channels, const int height,
		const int width, const int conved_height, const int conved_width,
		const int kernel_h, const int kernel_w, const int stride_h,
		const int stride_w, const int pad_h, const int pad_w,
		Dtype* const bias_diff) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			const int c = index;
			Dtype gradient = 0;
			for (int n = 0; n<num; n++) {
				const Dtype* const top_diff_slice =
					top_diff + (n * channels + c) * conved_height * conved_width;
				for (int ph = 0; ph<conved_height; ph++) {
					for (int pw = 0; pw<conved_width; pw++) {
						gradient += top_diff_slice[ph * conved_width + pw];
					}
				}
			}
			bias_diff[c] += gradient;
		}
	}
	template<typename Dtype>
	void DepthwiseConvLayer<Dtype>::Backward_gpu(
		const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {


		int* kernel_shape_data = this->kernel_shape_.mutable_cpu_data();
		int* stride_data = this->stride_.mutable_cpu_data();
		int* pad_data = this->pad_.mutable_cpu_data();

		const Dtype* weight = this->blobs_[0]->gpu_data();
		Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();

		const bool bias_term_ = this->bias_term_;
		Dtype* bias_diff = bias_term_ ? this->blobs_[1]->mutable_gpu_diff() : 0;
		const bool bias_propagate_down_ = this->param_propagate_down_[1];
		const bool weight_propagate_down_ = this->param_propagate_down_[0];


		const int kernel_h_ = kernel_shape_data[0];
		const int kernel_w_ = kernel_shape_data[1];
		const int stride_h_ = stride_data[0];
		const int stride_w_ = stride_data[1];
		const int pad_h_ = pad_data[0];
		const int pad_w_ = pad_data[1];

		const int conved_height = this->output_shape_[0];
		const int conved_weight = this->output_shape_[1];

		//	CHECK_EQ(stride_h_, 1)
		//	        << "The backward of the net whose stride is bigger than 1 is not implemented now. ";
		//	CHECK_EQ(stride_w_, 1)
		//	        << "The backward of the net whose stride is bigger than 1 is not implemented now. ";


		for (int i = 0; i < top.size(); ++i) {

			const Dtype* top_diff = top[i]->gpu_diff();
			const Dtype* bottom_data = bottom[i]->gpu_data();
			Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();

			vector<int> shape_ = bottom[i]->shape();
			const int channels_ = shape_[1];
			const int height_ = shape_[2];
			const int width_ = shape_[3];

			// Bias gradient, if necessary.
			if (bias_term_ && bias_propagate_down_) {
				const int count_bias = channels_;
				ConvBackwardBias<Dtype> << <CAFFE_GET_BLOCKS(count_bias), CAFFE_CUDA_NUM_THREADS >> >(
					count_bias, top_diff, bottom[i]->num(), channels_,
					height_, width_, conved_height, conved_weight, kernel_h_,
					kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_,
					bias_diff);
			}
			// gradient w.r.t. weight. Note that we will accumulate diffs.
			if (weight_propagate_down_) {
				const int count_weight = channels_ * kernel_h_ * kernel_w_;
				ConvBackwardWeight<Dtype> << <CAFFE_GET_BLOCKS(count_weight), CAFFE_CUDA_NUM_THREADS >> >(
					count_weight, top_diff, bottom[i]->num(), channels_,
					height_, width_, conved_height, conved_weight, kernel_h_,
					kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_,
					weight_diff,
					bottom_data);
			}
			// gradient w.r.t. bottom data, if necessary.
			if (propagate_down[i]) {
				const int count_bottom = bottom[i]->count();
				ConvBackward<Dtype> << <CAFFE_GET_BLOCKS(count_bottom), CAFFE_CUDA_NUM_THREADS >> >(
					count_bottom, top_diff, bottom[i]->num(), channels_,
					height_, width_, conved_height, conved_weight, kernel_h_,
					kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_,
					bottom_diff,
					weight);
			}
		}

	}

	INSTANTIATE_LAYER_GPU_FUNCS(DepthwiseConvLayer);

}  // namespace caffe
