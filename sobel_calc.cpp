#include "opencv2/imgproc/imgproc.hpp"
#include "arm_neon.h"
#include "sobel_alg.h"
using namespace cv;

/*******************************************
 * Model: grayScale
 * Input: Mat img
 * Output: None directly. Modifies a ref parameter img_gray_out
 * Desc: This module converts the image to grayscale
 ********************************************/
void grayScale(Mat& img, Mat& img_gray_out, int thread) {
	// weightings for grayscale average
	uint8x8_t rweight = vdup_n_u8(77);
	uint8x8_t gweight = vdup_n_u8(150);
	uint8x8_t bweight = vdup_n_u8(29);

	// vector code
	int pixels = img.rows * img.cols;
	int first = 0;
	int increment = 8;
	if(thread == 2) first += 8;
	if(thread != 0) increment +=8;

	for(int i = first; i < pixels; i+=increment) {
		// read from input
		uint8x8x3_t bgr = vld3_u8(&img.data[3*i]);
		// V = 77B + 150G + 29R
		uint16x8_t acc = vmull_u8(bgr.val[0], bweight);
		acc = vmlal_u8(acc, bgr.val[1], gweight);
		acc = vmlal_u8(acc, bgr.val[2], rweight);
		// V = V / 8
		uint8x8_t result = vshrn_n_u16(acc, 8);
		vst1_u8(&img_gray_out.data[i], result);
	}
	// possible TODO: make sure vectors never read/write out of bounds
}

/*******************************************
 * Model: sobelCalc
 * Input: Mat img_in, thread
 * Output: None directly. Modifies a ref parameter img_sobel_out
 * Desc: This module performs a sobel calculation on an image. It first
 *  converts the image to grayscale, calculates the gradient in the x
 *  direction, calculates the gradient in the y direction and sum it with Gx
 *  to finish the Sobel calculation
 *  thread = 0: single threaded: perform all rows
 *  thread = 1 and 2 are multithreaded (2 threads)
 *  thread = 1: first thread: perform odd rows
 *  thread = 2: second thread: perform odd rows
 ********************************************/
void sobelCalc(Mat& img_gray, Mat& img_sobel_out, int thread) {
	// Apply Sobel filter to black & white image
	char unsigned *img_data = img_gray.data;
	int first_row = 1;
	int increment = 1;
	// start with even row if 2nd thread
	if(thread == 2) first_row++;
	// only do half the rows if multithreaded
	if(thread != 0) increment++;

	for (int i=first_row; i<img_gray.rows-1; i+=increment) {
		for (int j=1; j<img_gray.cols-1; j+=8) {
			unsigned char *current = &img_data[IMG_WIDTH*i + j];
			// quantity added in both convolutions
			int16x8_t sum = (int16x8_t) vsubl_u8(vld1_u8(current - IMG_WIDTH - 1), vld1_u8(current + IMG_WIDTH + 1));
			// quantity added in x conv, subtracted in y conv
			int16x8_t diff = (int16x8_t) vsubl_u8(vld1_u8(current - IMG_WIDTH + 1), vld1_u8(current + IMG_WIDTH - 1) );

			// x convolution
			// temp3 = 2*(arr[curr -IMG] - arr[curr + IMG])
			int16x8_t temp = (int16x8_t) vsubl_u8(vld1_u8(current - IMG_WIDTH), vld1_u8(current + IMG_WIDTH) );
			temp = vshlq_n_s16(temp, 1);
			//temp3+= sum + diff;
			temp = vaddq_s16(temp, sum);
			temp = vaddq_s16(temp, diff);
			// absolute value
			int16x8_t sobel = vabsq_s16(temp); 

			// y convolution
			// temp3 = 2*(arr[curr - 1] - arr[curr + 1])
			temp = (int16x8_t) vsubl_u8(vld1_u8(current - 1),vld1_u8(current + 1));
			temp = vshlq_n_s16(temp, 1);
			// temp3 += sum - diff
			temp = vaddq_s16(temp, sum);
			temp = vsubq_s16(temp, diff);
			// absolute value
			temp = vabsq_s16(temp); 

			// add convolutions
			sobel = vaddq_s16(sobel, temp);
			// sobel = min(255, sobel)
			temp = vdupq_n_s16(255);
			sobel = vminq_s16(sobel, temp);
			// bit reduce and store to img out
			vst1_u8(&img_sobel_out.data[IMG_WIDTH*i + j], (uint8x8_t) vmovn_s16(sobel));
		}
	}
}
