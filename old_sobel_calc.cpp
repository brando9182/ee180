#include "opencv2/imgproc/imgproc.hpp"
#include "arm_neon.h"	// vectors: also compile with -mfpu=neon
#include "sobel_alg.h"
using namespace cv;


// weights out of 256
#define RWEIGHT 77
#define GWEIGHT 150
#define BWEIGHT 29
/*******************************************
 * Model: grayScale
 * Input: Mat img
 * Output: None directly. Modifies a ref parameter img_gray_out
 * Desc: This module converts the image to grayscale
 ********************************************/
void grayScale(Mat& img, Mat& img_gray_out) {
	uint8x8_t bweight = vdup_n_u8(29);
	uint8x8_t gweight = vdup_n_u8(150);
	uint8x8_t rweight = vdup_n_u8(77);

	int pixels = img.rows * img.cols;
	int i;	// increment by vector length = 8
	for(i = 0; i < pixels; i+=8) {
		// get blue green red values (in that order)
		uint8x8x3_t bgr = vld3_u8(&img.data[3*i]);
		// acc = 29B + 150G + 77R
		uint16x8_t acc = vmull_u8(bgr.val[0], bweight);
		acc = vmlal_u8(acc, bgr.val[1], gweight);
		acc = vmlal_u8(acc, bgr.val[2], rweight);
		// acc /= 256
		uint8x8_t result = vshrn_n_u16(acc, 8);
		// data[i] = acc
		vst1_u8(&img_gray_out.data[i], result);
	}

	/*
	char unsigned *img_data = img.data;
	//int total = img.rows*img.cols;
	// thrice = 3*i
	int pixels = img.rows * img.cols;
	for(int i = 0, thrice = 0; i < pixels; i++, thrice+=STEP1) {
			img_gray_out.data[i] = (29*img_data[thrice]
					+ 150*img_data[thrice + 1]
					+ 77*img_data[thrice + 2])/256;
	}
	*/
}

/*******************************************
 * Model: sobelCalc
 * Input: Mat img_in
 * Output: None directly. Modifies a ref parameter img_sobel_out
 * Desc: This module performs a sobel calculation on an image. It first
 *  converts the image to grayscale, calculates the gradient in the x
 *  direction, calculates the gradient in the y direction and sum it with Gx
 *  to finish the Sobel calculation
 ********************************************/
void sobelCalc(Mat& img_gray, Mat& img_sobel_out) {
	// Apply Sobel filter to black & white image
	unsigned short sobel;

	char unsigned *img_data = img_gray.data;

	// Calculate the x convolution
	// Added -1 to ignore all edges and avoid memory error
	for (int i=1; i<img_gray.rows-1; i++) {
		/*
		for (int j=1; j<img_gray.cols-1; j+=8) {
			unsigned char *current = &img_data[IMG_WIDTH*i + j];
			// quantity added in both convolutions
			uint8x8_t temp1 = vld1_u8(current - IMG_WIDTH - 1);
			uint8x8_t temp2 = vld1_u8(current + IMG_WIDTH + 1);
			int16x8_t sum = (int16x8_t) vsubl_u8(temp1, temp2);
			// quantity added in x conv, subtracted in y conv
			temp1 = vld1_u8(current - IMG_WIDTH + 1);
			temp2 = vld1_u8(current + IMG_WIDTH - 1);
			int16x8_t diff; = (int16x8_t) vsubl_u8(temp1, temp2);

		}
		*/
		printf("entering");
		for (int j=1; j<img_gray.cols-1; j++) {
			// some commonly used values
			short current = IMG_WIDTH*i + j;
			int sum = img_data[current - IMG_WIDTH - 1] - img_data[current + IMG_WIDTH + 1];
			// positive for x conv, negative for y
			int diff = img_data[current - IMG_WIDTH + 1] - img_data[current + IMG_WIDTH - 1];

			// x conv
			sobel = abs(sum + diff 
					+ 2*img_data[current - IMG_WIDTH]
					- 2*img_data[current + IMG_WIDTH]);
			// y conv
			sobel += abs(sum - diff
					+ 2*img_data[current - 1]
					- 2*img_data[current + 1]);

 			img_sobel_out.data[current] = (sobel > 255) ? 255 : sobel;
		}
		printf("exiting");
	}
}
